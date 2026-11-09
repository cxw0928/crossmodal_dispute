"""Build a larger multi-company SEC controlled benchmark.

All source values come from SEC CompanyFacts. Charts are controlled renderings,
so the benchmark measures model sensitivity to known visual conflicts rather
than natural errors in original filing exhibits.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, urllib.request
from datetime import datetime, timezone
from pathlib import Path

COMPANIES = {
    "Apple": "0000320193", "Microsoft": "0000789019", "Amazon": "0001018724",
    "Alphabet": "0001652044", "NVIDIA": "0001045810", "Meta": "0001326801",
}
SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
UA = "CrossModal-Dispute research (academic reproducibility; contact@example.com)"


def fetch(cik: str, path: Path) -> tuple[dict, str]:
    raw = path.read_bytes() if path.exists() else None
    if raw is None:
        req = urllib.request.Request(SEC_URL.format(cik=cik), headers={"User-Agent": UA})
        raw = urllib.request.urlopen(req, timeout=60).read(); path.write_bytes(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def facts_to_quarters(facts: dict, limit: int) -> tuple[list[dict], str]:
    gaap = facts["facts"]["us-gaap"]
    concepts = ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
    concept = next((x for x in concepts if x in gaap), None)
    if not concept: raise RuntimeError(f"No supported revenue concept for {facts.get('entityName')}")
    units = gaap[concept]["units"]; unit = "USD" if "USD" in units else next(iter(units))
    rows = [x for x in units[unit] if x.get("frame", "").startswith("CY") and "Q" in x.get("frame", "") and not x["frame"].endswith("Q4")]
    latest = {}
    for x in sorted(rows, key=lambda z: (z["frame"], z.get("filed", ""))): latest[x["frame"]] = x
    out = []
    for x in sorted(latest.values(), key=lambda z: z["frame"])[-limit:]:
        out.append({"frame": x["frame"], "value": float(x["val"]), "filed": x.get("filed"), "accn": x.get("accn"), "concept": concept, "unit": unit})
    return out, concept


def build(out: Path, quarters: int) -> dict:
    out.mkdir(parents=True, exist_ok=True); chart_dir = out / "charts"; chart_dir.mkdir(exist_ok=True)
    cases = []; sources = []
    for company, cik in COMPANIES.items():
        facts, digest = fetch(cik, out / f"companyfacts_{cik}.json")
        rows, concept = facts_to_quarters(facts, quarters)
        sources.append({"company": company, "cik": cik, "entity_name": facts.get("entityName"), "quarters": len(rows), "concept": concept, "sha256": digest, "url": SEC_URL.format(cik=cik)})
        for row in rows:
            for perturbation, factor in [("clean", 1.0), ("scale_0.8", .8), ("scale_1.2", 1.2), ("scale_1.35", 1.35)]:
                value = row["value"]; visual = value * factor
                cases.append({"id": f"{company}_{row['frame']}_{perturbation}", "company": company, "frame": row["frame"], "source_value": value, "visual_claim": visual, "perturbation": perturbation, "label": int(perturbation != "clean"), "disagreement": abs(visual-value)/max(abs(value), 1e-12), "source_accn": row.get("accn"), "filed": row.get("filed")})
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cases[0].keys()); writer.writeheader(); writer.writerows(cases)
    meta = {"dataset": "SEC CompanyFacts multi-company controlled benchmark", "companies": sources, "cases": len(cases), "positive_cases": sum(x["label"] for x in cases), "quarters_per_company": quarters, "retrieved_at": datetime.now(timezone.utc).isoformat(), "warning": "Charts are controlled renderings; not original filing exhibits or natural hallucination labels."}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2)); return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("results/sec_multicompany")); p.add_argument("--quarters", type=int, default=12); args = p.parse_args(); build(args.out, args.quarters)
