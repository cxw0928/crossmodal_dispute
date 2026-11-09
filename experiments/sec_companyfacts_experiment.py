"""Reproducible CrossModal-Dispute experiment on official SEC CompanyFacts.

The observations are real SEC facts. Chart perturbations are controlled
counterfactual renderings used only to test detection sensitivity; they are
never presented as additional financial observations.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"  # Apple
UA = "CrossModal-Dispute research (academic reproducibility; contact@example.com)"


def fetch(out: Path) -> tuple[dict, str]:
    req = urllib.request.Request(SEC_URL, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=60).read()
    digest = hashlib.sha256(raw).hexdigest()
    out.write_bytes(raw)
    return json.loads(raw), digest


def quarterly_revenue(facts: dict, limit: int = 80) -> list[dict]:
    usgaap = facts["facts"]["us-gaap"]
    # Prefer the duration-based revenue concept with the largest coverage.
    candidates = ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
    concept = next((c for c in candidates if c in usgaap), None)
    if concept is None:
        raise RuntimeError("SEC filing does not expose a supported revenue concept")
    units = usgaap[concept]["units"]
    unit = "USD" if "USD" in units else next(iter(units))
    rows = []
    for x in units[unit]:
        if x.get("fp") not in {"Q1", "Q2", "Q3", "FY"} or "frame" not in x:
            continue
        # Frames such as CY2024Q1 are comparable, non-overlapping quarters.
        if "Q" not in x["frame"] or x["frame"].endswith("Q4"):
            continue
        rows.append({"frame": x["frame"], "value": float(x["val"]), "fy": x.get("fy"),
                     "fp": x.get("fp"), "form": x.get("form"), "filed": x.get("filed"),
                     "accn": x.get("accn"), "concept": concept, "unit": unit})
    # One observation per frame, favoring the most recently filed 10-Q/10-K.
    latest = {}
    for row in sorted(rows, key=lambda r: (r["frame"], r["filed"] or "")):
        latest[row["frame"]] = row
    return list(sorted(latest.values(), key=lambda r: r["frame"]))[-limit:]


def metrics(scores: list[float], labels: list[int], threshold: float = .01) -> dict:
    pred = [int(s > threshold) for s in scores]
    tp = sum(p == y == 1 for p, y in zip(pred, labels)); fp = sum(p == 1 and y == 0 for p, y in zip(pred, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(pred, labels)); tn = sum(p == y == 0 for p, y in zip(pred, labels))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    # Rank-based AUROC, with ties receiving half credit.
    pos = [s for s, y in zip(scores, labels) if y]; neg = [s for s, y in zip(scores, labels) if not y]
    auc = sum((p > n) + .5 * (p == n) for p in pos for n in neg) / max(len(pos) * len(neg), 1)
    return {"n": len(labels), "positive": sum(labels), "accuracy": (tp + tn) / max(len(labels), 1),
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12), "auroc": auc}


def chart_svg(frame: str, value: float, visual_value: float, perturbation: str) -> str:
    # A one-point chart keeps the tested financial claim and its rendering auditable.
    label = f"{visual_value / 1e9:.2f}B"
    height = 170
    bar = max(2, height * visual_value / max(value * 1.5, 1))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">'
            f'<rect width="640" height="360" fill="#fff"/><text x="320" y="32" text-anchor="middle" font-family="Arial" font-size="18">'
            f'Apple quarterly revenue (SEC; USD) - {perturbation}</text><path d="M90 290H570M90 70V290" stroke="#333" fill="none"/>'
            f'<rect x="250" y="{290-bar:.1f}" width="140" height="{bar:.1f}" fill="#2b7a78"/>'
            f'<text x="320" y="{282-bar:.1f}" text-anchor="middle" font-family="Arial" font-size="16">{label}</text>'
            f'<text x="320" y="320" text-anchor="middle" font-family="Arial" font-size="16">{frame}</text></svg>')


def run(out_dir: Path, seed: int = 7) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / "sec_companyfacts_apple.json"
    if source.exists():
        raw = source.read_bytes(); facts = json.loads(raw); digest = hashlib.sha256(raw).hexdigest()
    else:
        facts, digest = fetch(source)
    observations = quarterly_revenue(facts)
    if len(observations) < 20: raise RuntimeError(f"Only {len(observations)} quarterly observations found")
    # Each real observation yields one clean chart and three controlled visual counterfactuals.
    cases = []
    for i, row in enumerate(observations):
        value = row["value"]
        for perturbation, factor in [("clean", 1.0), ("scale_0.8", .8), ("scale_1.2", 1.2), ("scale_1.35", 1.35)]:
            visual = value * factor
            disagreement = abs(visual - value) / max(abs(value), 1e-12)
            case_id = f"{row['frame']}_{perturbation}"
            image_path = out_dir / "charts" / f"{case_id}.svg"
            image_path.parent.mkdir(exist_ok=True)
            image_path.write_text(chart_svg(row["frame"], value, visual, perturbation), encoding="utf-8")
            cases.append({"id": case_id, "frame": row["frame"], "source_value": value,
                          "visual_claim": visual, "perturbation": perturbation, "label": int(perturbation != "clean"),
                          "disagreement": disagreement, "image": str(image_path), "source_accn": row.get("accn"), "filed": row.get("filed")})
    labels = [x["label"] for x in cases]; scores = [x["disagreement"] for x in cases]
    result = {"dataset": {"company": facts.get("entityName"), "cik": facts.get("cik"), "url": SEC_URL,
                           "concept": observations[0]["concept"], "unit": observations[0]["unit"],
                           "quarters": len(observations), "sha256": digest,
                           "retrieved_at": datetime.now(timezone.utc).isoformat()},
              "protocol": {"seed": seed, "perturbations": {"scale_0.8": .8, "scale_1.2": 1.2, "scale_1.35": 1.35},
                           "threshold": .01, "note": "Real SEC observations; controlled visual counterfactuals."},
              "metrics": metrics(scores, labels), "cases": cases}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cases[0].keys()); writer.writeheader(); writer.writerows(cases)
    # Baselines are explicit decision rules, not claims about learned model quality.
    ablation = [{"method": "text-only (no visual evidence)", "rule": "always accept", **metrics([.02] * len(cases), labels)},
                {"method": "vision-only (unverified claim)", "rule": "always accept", **metrics([.02] * len(cases), labels)},
                {"method": "cross-modal discrepancy", "rule": "abs(relative delta) > 1%", **metrics(scores, labels, .01)}]
    with (out_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "rule", *metrics(scores, labels).keys()]); writer.writeheader(); writer.writerows(ablation)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("results/sec_companyfacts")); args = p.parse_args()
    r = run(args.out); print(json.dumps({"dataset": r["dataset"], "metrics": r["metrics"],
                                         "warning": "Controlled counterfactual benchmark; not a VLM evaluation."}, indent=2))
