"""Command line entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import OpenAICompatibleClient
from .core import TextTableAgent, VisualChartAgent, run_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-modal table/chart dispute")
    parser.add_argument("--table", required=True, help="UTF-8 CSV/Markdown table or JSON file")
    parser.add_argument("--image", required=True, help="Chart image file")
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    online = config.get("provider", "offline") == "openai-compatible"
    client = OpenAICompatibleClient(config.get("base_url"), timeout=config.get("timeout", 60)) if online else None
    text = TextTableAgent(client, config.get("text_model", ""))
    vision = VisualChartAgent(client, config.get("vision_model", ""))
    table_path = Path(args.table)
    table_text = table_path.read_text(encoding="utf-8")
    if table_path.suffix.lower() == ".json": table_text = json.loads(table_text)
    result = run_case(table_text, args.image, args.question, text, vision,
                      config.get("relative_tolerance", 0.01))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__": raise SystemExit(main())
