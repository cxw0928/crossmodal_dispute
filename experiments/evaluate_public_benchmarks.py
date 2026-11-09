"""Evaluate CrossModal-Dispute on ChartQA/FinChart-Bench.

ChartQA supplies both chart images and underlying CSV tables. DeepSeek reads
the CSV; Doubao reads the image through an Ark endpoint. Metrics are computed
only on numeric gold answers, where cross-modal value comparison is well-defined.
"""
from __future__ import annotations

import argparse, csv, json, os, re, shutil, subprocess
from pathlib import Path

from crossmodal_dispute.core import AgentAnswer, _parse_answer
if __package__:
    from .sec_companyfacts_experiment import metrics
else:
    from sec_companyfacts_experiment import metrics


def number(text: object) -> float | None:
    if text is None: return None
    m = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(text))
    return float(m.group(0).replace(",", "")) if m else None


def chartqa_samples(root: Path, split: str, human: bool = True) -> list[dict]:
    folder = root / split
    if not folder.exists() and (root / "ChartQA Dataset" / split).exists():
        folder = root / "ChartQA Dataset" / split
    names = ([f"{split}_human.json", "human.json"] if human else [f"{split}_augmented.json", "augmented.json"])
    annotation = next((folder / n for n in names if (folder / n).exists()), None)
    if annotation is None: raise FileNotFoundError(f"ChartQA annotation not found under {folder}")
    items = json.loads(annotation.read_text(encoding="utf-8")); out = []
    for x in items:
        imgname = x.get("imgname") or x.get("image") or x.get("image_id")
        image = folder / "png" / imgname if imgname else None
        if image is None or not image.exists(): continue
        table = x.get("table")
        if table is None:
            table_path = folder / "tables" / (Path(str(imgname)).stem + ".csv")
            if table_path.exists(): table = table_path.read_text(encoding="utf-8")
        if table is None: continue
        gold = (x.get("label") or x.get("answer") or [None])[0] if isinstance(x.get("label") or x.get("answer"), list) else (x.get("label") or x.get("answer"))
        out.append({"dataset": "ChartQA", "id": str(imgname) + "::" + str(len(out)), "image": str(image), "table": table, "question": x.get("query") or x.get("question"), "gold": gold})
    return out


def finchart_samples(root: Path, task: str = "QA") -> list[dict]:
    annotation = root / f"{task}_data.json"; items = json.loads(annotation.read_text(encoding="utf-8")); out = []
    for i, x in enumerate(items):
        image_name = x.get("image") or x.get("imgname") or x.get("image_path")
        image = root / image_name if image_name and not Path(image_name).is_absolute() else Path(image_name)
        if not image.exists():
            image = root / f"{task}_images" / Path(str(image_name)).name
        if not image.exists(): continue
        out.append({"dataset": "FinChart-Bench", "id": f"{task}_{i}", "image": str(image), "table": x.get("table", ""), "question": x.get("question") or x.get("query"), "gold": x.get("answer") or x.get("label")})
    return out


def arkcli_chat(args: list[str], env: dict[str, str]) -> dict:
    node = shutil.which("node")
    appdata = Path(os.environ.get("APPDATA", ""))
    entry = appdata / "npm" / "node_modules" / "@volcengine" / "ark-cli" / "scripts" / "run.js"
    if not node or not entry.is_file():
        raise RuntimeError("Could not locate Node.js or the installed ArkCLI entry point")
    p = subprocess.run([node, str(entry), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=240, env=env)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return json.loads(p.stdout)


def deepseek_table_ark(endpoint: str, table: object, question: str) -> tuple[AgentAnswer, dict]:
    rendered = table if isinstance(table, str) else json.dumps(table, ensure_ascii=False)
    if not rendered.strip():
        raise ValueError("Refusing to call the text endpoint with an empty table")
    instruction = "Use only the supplied table. Return only compact JSON: {\"value\": number|null}. Do not include reasoning or explanation; use null if the table does not support an answer."
    prompt = f"Table:\n{rendered}\n\nQuestion: {question}"
    args = ["+chat", "--model", endpoint, "--instructions", instruction, "--temperature", "0", "--max-output-tokens", "1024", "--format", "json", "--no-progress", prompt]
    env = os.environ.copy(); env.update({"ARKCLI_NO_UPDATE_NOTIFIER":"1", "ARKCLI_CALLER_TYPE":"ai_agent", "ARKCLI_CALLER_NAME":"codex", "ARKCLI_SKILL_NAME":"arkcli-chat"})
    payload = arkcli_chat(args, env); raw = payload.get("content", "")
    return (_parse_answer(raw) if raw else AgentAnswer(None, "incomplete response", raw)), payload


def doubao_chart(endpoint: str, image: Path, question: str) -> tuple[AgentAnswer, dict]:
    instruction = "Read the chart only. Return only compact JSON: {\"value\": number|null}. Use the chart's units; use null if uncertain."
    args = ["+chat", "--model", endpoint, "--input", "@" + str(image.resolve()), "--instructions", instruction,
            "--temperature", "0", "--max-output-tokens", "1024", "--format", "json", "--no-progress", question]
    env = os.environ.copy(); env.update({"ARKCLI_NO_UPDATE_NOTIFIER":"1", "ARKCLI_CALLER_TYPE":"ai_agent", "ARKCLI_CALLER_NAME":"codex", "ARKCLI_SKILL_NAME":"arkcli-chat"})
    payload = arkcli_chat(args, env); raw = payload.get("content", "")
    return (_parse_answer(raw) if raw else AgentAnswer(None, "incomplete response", raw)), payload


def run(samples: list[dict], out: Path, vision_endpoint: str, text_endpoint: str, limit: int | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    samples = [s for s in samples if number(s["gold"]) is not None]
    samples = samples[:limit] if limit else samples
    predictions_path = out / "predictions.csv"
    raw_path = out / "responses.jsonl"
    existing = {}
    if predictions_path.exists():
        with predictions_path.open(newline="", encoding="utf-8") as f:
            existing = {r["id"]: r for r in csv.DictReader(f)}
    rows = [existing[s["id"]] for s in samples if s["id"] in existing]
    for i, s in enumerate(samples, 1):
        if s["id"] in existing:
            continue
        gold = number(s["gold"])
        table_claim, text_payload = deepseek_table_ark(text_endpoint, s["table"], s["question"])
        visual_claim, payload = doubao_chart(vision_endpoint, Path(s["image"]), s["question"])
        table_ok = table_claim.value is not None and abs(table_claim.value - gold) <= max(abs(gold) * .01, .01)
        visual_ok = visual_claim.value is not None and abs(visual_claim.value - gold) <= max(abs(gold) * .01, .01)
        gold_hallucination = int(not visual_ok)
        eligible = (table_claim.value is not None and visual_claim.value is not None
                    and text_payload.get("status") == "completed" and payload.get("status") == "completed")
        predicted_dispute = int(eligible and abs(table_claim.value - visual_claim.value) > max(abs(table_claim.value) * .01, .01))
        row = {"dataset": s["dataset"], "id": s["id"], "gold": gold, "table_value": table_claim.value, "vision_value": visual_claim.value, "table_correct": int(table_ok), "vision_correct": int(visual_ok), "gold_hallucination": gold_hallucination, "predicted_dispute": predicted_dispute, "dispute_eligible": int(eligible), "text_response_status": text_payload.get("status"), "vision_response_status": payload.get("status"), "text_response_id": text_payload.get("id"), "vision_response_id": payload.get("id")}
        rows.append(row)
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": s["id"], "text": text_payload, "vision": payload}, ensure_ascii=False) + "\n")
        with predictions_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        if i % 25 == 0: print(f"evaluated {i}/{len(samples)}", flush=True)
    scored = [r for r in rows if int(r["dispute_eligible"])]
    labels = [int(r["gold_hallucination"]) for r in scored]; scores = [float(r["predicted_dispute"]) for r in scored]
    m = metrics(scores, labels, threshold=.5)
    summary = {"datasets": sorted(set(r["dataset"] for r in rows)), "numeric_cases": len(rows), "dispute_scored_cases": len(scored), "dispute_coverage": len(scored) / len(rows) if rows else 0.0, "deepseek_endpoint": text_endpoint, "doubao_endpoint": vision_endpoint, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "auroc": m["auroc"], "protocol": "Both agents use Ark CLI endpoints: DeepSeek reads the underlying table and Doubao reads the chart. Non-numeric questions are excluded; dispute metrics include only completed numeric claims from both agents. Incomplete/abstained responses are reported through dispute_coverage."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with predictions_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=["chartqa", "finchart"], required=True); p.add_argument("--root", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--vision-endpoint", required=True); p.add_argument("--text-endpoint", required=True); p.add_argument("--split", default="test"); p.add_argument("--limit", type=int, default=None); args = p.parse_args()
    samples = chartqa_samples(args.root, args.split) if args.dataset == "chartqa" else finchart_samples(args.root)
    run(samples, args.out, args.vision_endpoint, args.text_endpoint, args.limit)
