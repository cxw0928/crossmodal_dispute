"""Run actual text/vision agent inference on the SEC controlled benchmark.

Real SEC facts are rendered into controlled charts. This is a model-in-the-loop
stress test, not an evaluation on original filing exhibits or natural errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from crossmodal_dispute.client import OpenAICompatibleClient
from crossmodal_dispute.core import (TextTableAgent, VisualChartAgent, run_case,
                                     dispute, _parse_answer)
if __package__:
    from .sec_companyfacts_experiment import metrics
else:
    from sec_companyfacts_experiment import metrics


def render_chart(path: Path, frame: str, value_usd: float, visual_usd: float, company: str = "Apple") -> None:
    image = Image.new("RGB", (960, 540), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=23)
    small = ImageFont.load_default(size=19)
    draw.text((480, 32), f"{company} quarterly revenue | {frame}", fill="#18212b", font=font, anchor="mt")
    draw.text((480, 72), "SEC-reported revenue; USD billions", fill="#53606c", font=small, anchor="mt")
    x0, y0, x1, y1 = 160, 430, 800, 145
    draw.line((x0, y0, x1, y0), fill="#333333", width=3)
    draw.line((x0, y0, x0, y1), fill="#333333", width=3)
    max_value = max(value_usd / 1e9 * 1.5, 1)
    for tick in range(4):
        y = y0 - (y0 - y1) * tick / 3
        amount = max_value * tick / 3
        draw.line((x0 - 8, y, x1, y), fill="#e3e7ea", width=1)
        draw.text((x0 - 18, y), f"{amount:.0f}", fill="#38434d", font=small, anchor="rm")
    bar_height = max(3, (y0 - y1) * (visual_usd / 1e9) / max_value)
    draw.rectangle((390, y0 - bar_height, 570, y0), fill="#287a78")
    draw.text((480, y0 - bar_height - 16), f"{visual_usd / 1e9:.2f}", fill="#15232b", font=font, anchor="mb")
    draw.text((480, 465), frame, fill="#252e36", font=small, anchor="mt")
    image.save(path, format="PNG")


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _arkcli_vision(endpoint: str, image: Path, frame: str) -> dict:
    prompt = f"Read the plotted quarterly revenue value for {frame}. What is the value in USD billions?"
    instructions = ("Read the financial chart only. Return JSON with keys value "
                    "(number or null; USD billions) and explanation. Do not use outside information.")
    args = ["+chat", "--model", endpoint, "--input", "@" + str(image.resolve()),
            "--instructions", instructions, "--temperature", "0", "--max-output-tokens", "256",
            "--format", "json", "--no-progress", prompt]
    command = "& 'C:\\Users\\cxw09\\AppData\\Roaming\\npm\\arkcli.cmd' " + " ".join(_ps_quote(x) for x in args) + "; exit $LASTEXITCODE"
    env = os.environ.copy()
    env.update({"ARKCLI_NO_UPDATE_NOTIFIER": "1", "ARKCLI_CALLER_TYPE": "ai_agent",
                "ARKCLI_CALLER_NAME": "codex", "ARKCLI_SKILL_NAME": "arkcli-chat"})
    result = subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=240, env=env)
    if result.returncode:
        raise RuntimeError(f"arkcli inference failed for {frame}: {result.stderr.strip() or result.stdout.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"arkcli returned non-JSON stdout for {frame}: {result.stdout[:500]}") from exc
    return payload


def evaluate_endpoint(cases_path: Path, out_dir: Path, endpoint: str, label: str,
                      smoke_path: Path, limit: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(cases_path.open(encoding="utf-8", newline="")))
    if limit is not None:
        rows = rows[:limit]
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    predictions = []
    table_agent = TextTableAgent()
    for i, row in enumerate(rows, 1):
        value = float(row["source_value"])
        visual = float(row["visual_claim"])
        image = out_dir / "charts" / f"{row['id']}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        company = row.get("company", "Apple")
        render_chart(image, row["frame"], value, visual, company)
        question = f"What was {company} revenue in {row['frame']}?"
        table = f"period,revenue_usd_billions\n{row['frame']},{value / 1e9:.6f}"
        table_claim = table_agent.answer(table, question)
        if table_claim.value is None:
            raise RuntimeError(f"Table agent could not parse SEC value for {row['frame']}")
        payload = smoke.get(row["id"], {}).get(endpoint)
        if payload is None:
            payload = _arkcli_vision(endpoint, image, row["frame"])
        content = payload.get("content", "")
        visual_claim = _parse_answer(content) if content else _parse_answer("")
        result = dispute(table_claim, visual_claim, tolerance=.01)
        predictions.append({"id": row["id"], "frame": row["frame"], "perturbation": row["perturbation"],
                            "gold_label": int(row["label"]),
                            "predicted_label": int(result["status"] == "disputed"),
                            "status": result["status"], "table_value_usd_bn": table_claim.value,
                            "vision_value_usd_bn": visual_claim.value,
                            "relative_disagreement": result.get("relative_disagreement"),
                            "abstained": int(visual_claim.value is None),
                            "response_id": payload.get("id", payload.get("response_id")),
                            "model": payload.get("model"),
                            "usage_total_tokens": payload.get("usage", {}).get("total_tokens"),
                            "response_status": payload.get("status"),
                            "vision_raw": visual_claim.raw, "image": str(image)})
        with (out_dir / "predictions.partial.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader(); writer.writerows(predictions)
        if i % 5 == 0 or i == len(rows):
            print(f"{label}: evaluated {i}/{len(rows)}", flush=True)

    labels = [x["gold_label"] for x in predictions]
    scores = [float(x["relative_disagreement"] or 0.0) for x in predictions]
    calculated = metrics(scores, labels, threshold=.01)
    summary = {"label": label, "endpoint": endpoint,
               "model": next((x["model"] for x in predictions if x["model"]), None),
               "cases": len(predictions), "positive_cases": sum(labels),
               "abstentions": sum(x["abstained"] for x in predictions),
               "accuracy": calculated["accuracy"], "precision": calculated["precision"],
               "recall": calculated["recall"], "f1": calculated["f1"], "auroc": calculated["auroc"],
               "decision_threshold_relative_disagreement": .01,
               "data_protocol": "SEC CompanyFacts values with generated controlled charts; not original filing exhibits.",
               "detector_protocol": "Deterministic SEC table reader vs independent ArkCLI visual reader; disputed iff relative disagreement > 1%; abstentions count as not detected."}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader(); writer.writerows(predictions)
    (out_dir / "predictions.partial.csv").unlink(missing_ok=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def evaluate(cases_path: Path, out_dir: Path, model: str, base_url: str | None, provider: str,
             limit: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if provider == "doubao":
        model = model or "doubao-seed-2-1-turbo-260628"
        base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3"
        api_key = os.getenv("DOUBAO_API_KEY", "")
        if not api_key:
            raise RuntimeError("DOUBAO_API_KEY is not configured")
    else:
        model = model or "gpt-4.1-mini"
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAICompatibleClient(base_url=base_url, api_key=api_key)
    text_agent = TextTableAgent(client, model)
    vision_agent = VisualChartAgent(client, model)
    rows = list(csv.DictReader(cases_path.open(encoding="utf-8", newline="")))
    if limit is not None:
        rows = rows[:limit]
    predictions = []
    for i, row in enumerate(rows, 1):
        value_usd = float(row["source_value"])
        visual_usd = float(row["visual_claim"])
        image_path = out_dir / "charts" / f"{row['id']}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        render_chart(image_path, row["frame"], value_usd, visual_usd)
        table = f"period,revenue_usd_billions\n{row['frame']},{value_usd / 1e9:.6f}"
        question = f"What was Apple revenue in {row['frame']}? Give the value in USD billions."
        result = run_case(table, image_path, question, text_agent, vision_agent, tolerance=.01)
        status = result["dispute"]["status"]
        label = int(row["label"])
        predicted = int(status == "disputed")
        score = result["dispute"].get("relative_disagreement") or 0.0
        predictions.append({"id": row["id"], "frame": row["frame"], "perturbation": row["perturbation"],
                            "gold_label": label, "predicted_label": predicted, "status": status,
                            "table_value": result["table_agent"]["value"],
                            "vision_value": result["visual_agent"]["value"],
                            "relative_disagreement": score, "abstained": int(status == "inconclusive"),
                            "visual_raw": result["visual_agent"]["raw"], "image": str(image_path)})
        if i % 10 == 0 or i == len(rows):
            print(f"evaluated {i}/{len(rows)}")
    labels = [x["gold_label"] for x in predictions]
    binary = metrics([float(x["predicted_label"]) for x in predictions], labels, threshold=.5)
    ranking = metrics([float(x["relative_disagreement"]) for x in predictions], labels, threshold=.01)
    summary = {"provider": provider, "model": model, "cases": len(predictions), "positive_cases": sum(labels),
               "abstentions": sum(x["abstained"] for x in predictions),
               "classification_at_1pct": {k: binary[k] for k in ("accuracy", "precision", "recall", "f1")},
               "auroc_from_relative_disagreement": ranking["auroc"],
               "protocol": "Real SEC CompanyFacts values; generated controlled PNG charts; actual API inference by both agents; abstentions count as not detected; not original SEC chart exhibits.",
               "threshold": .01}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("results/sec_companyfacts/cases.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--smoke", type=Path, default=Path("results/sec_companyfacts/model_smoke_responses.json"))
    parser.add_argument("--limit", type=int, default=None, help="Optional local smoke limit")
    args = parser.parse_args()
    evaluate_endpoint(args.cases, args.out, args.endpoint, args.label, args.smoke, args.limit)
