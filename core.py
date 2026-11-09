"""Table/chart agents and cross-modal dispute orchestration."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], *, model: str,
             temperature: float = 0.0, max_tokens: int = 512) -> str: ...


@dataclass
class AgentAnswer:
    value: float | None
    explanation: str
    raw: str


def _number(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    return float(match.group(0).replace(",", "")) if match else None


class TextTableAgent:
    """Reads a row-oriented CSV/Markdown-like table or structured mapping."""
    def __init__(self, client: ChatClient | None = None, model: str = ""):
        self.client, self.model = client, model

    def answer(self, table: str | dict[str, Any], question: str) -> AgentAnswer:
        rendered = json.dumps(table, ensure_ascii=False) if isinstance(table, dict) else table
        if self.client:
            raw = self.client.chat([
                {"role": "system", "content": "You read financial tables. Answer only from the supplied table. Return JSON: {\"value\": number|null, \"explanation\": string}. Do not guess."},
                {"role": "user", "content": f"Table:\n{rendered}\n\nQuestion: {question}"},
            ], model=self.model)
            return _parse_answer(raw)
        # Offline reader supports CSV/Markdown rows and JSON objects with period/value keys.
        query = question.lower()
        if isinstance(table, dict):
            rows = table.get("rows", table)
            if isinstance(rows, list):
                for row in rows:
                    label = str(row.get("period", row.get("label", ""))).lower()
                    if label and label in query:
                        value = _number(str(row.get("value", row.get("revenue", ""))))
                        if value is not None: return AgentAnswer(value, f"表格行 {label}", str(row))
        else:
            for line in rendered.splitlines():
                fields = [x.strip().strip("| ") for x in re.split(r"[,|\t]", line) if x.strip()]
                if len(fields) < 2: continue
                label = fields[0].lower()
                if label and label in query:
                    value = _number(fields[-1])
                    if value is not None: return AgentAnswer(value, f"表格行 {fields[0]}", line)
            return AgentAnswer(None, "Offline parser could not locate the requested table row", "")


class VisualChartAgent:
    """Reads a chart image using an OpenAI-compatible vision model."""
    def __init__(self, client: ChatClient | None = None, model: str = ""):
        self.client, self.model = client, model

    def answer(self, image: str | Path, question: str) -> AgentAnswer:
        path = Path(image)
        if not path.is_file(): raise FileNotFoundError(path)
        if not self.client:
            return AgentAnswer(None, "未配置视觉模型；离线模式不伪造图像读数", "")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        raw = self.client.chat([
            {"role": "system", "content": "You read financial charts. Answer only from the image. Return JSON: {\"value\": number|null, \"explanation\": string}. Include units; use null if uncertain."},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ]},
        ], model=self.model)
        return _parse_answer(raw)


def _parse_answer(raw: str) -> AgentAnswer:
    text = raw.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        val = obj.get("value")
        if isinstance(val, str):
            val = val.strip().replace(",", "")
        try:
            parsed = float(val) if val is not None and val != "" else None
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            return AgentAnswer(parsed, str(obj.get("explanation", "")), raw)
        if val is None:
            return AgentAnswer(None, str(obj.get("explanation", "")), raw)

    # Recover only an explicit value field; do not mistake numbers in prose for data.
    match = re.search(r'["\']?value["\']?\s*[:=]\s*["\']?([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)', text)
    if match:
        return AgentAnswer(float(match.group(1).replace(",", "")), "Recovered explicit value field", raw)
    return AgentAnswer(None, "Could not parse a numeric value", raw)


def dispute(table_answer: AgentAnswer, visual_answer: AgentAnswer,
            tolerance: float = 0.01) -> dict[str, Any]:
    """Cross-examine the image claim against table evidence; abstention is explicit."""
    if table_answer.value is None or visual_answer.value is None:
        return {"status": "inconclusive", "disagreement": None,
                "challenge": "At least one agent abstained; request human or model review."}
    delta = abs(table_answer.value - visual_answer.value)
    relative = delta / max(abs(table_answer.value), 1e-12)
    challenged = relative > tolerance
    return {
        "status": "disputed" if challenged else "consistent",
        "disagreement": delta,
        "relative_disagreement": relative,
        "challenge": (f"Visual claim {visual_answer.value:g} conflicts with table value "
                      f"{table_answer.value:g} (absolute difference {delta:g}); verify axis, "
                      "period, units, and possible visual perturbation." if challenged else
                      "Visual claim is supported by the table within the configured tolerance."),
        "counterfactual_candidate": challenged,
    }


def run_case(table: str | dict[str, Any], image: str | Path, question: str,
             text_agent: TextTableAgent, visual_agent: VisualChartAgent,
             tolerance: float = 0.01) -> dict[str, Any]:
    table_claim = text_agent.answer(table, question)
    visual_claim = visual_agent.answer(image, question)
    return {"question": question, "table_agent": asdict(table_claim),
            "visual_agent": asdict(visual_claim),
            "dispute": dispute(table_claim, visual_claim, tolerance)}


def generate_counterfactuals(table: str | dict[str, Any], visual_agent: VisualChartAgent,
                             images: list[str | Path], question: str,
                             text_agent: TextTableAgent) -> list[dict[str, Any]]:
    """Score candidate chart variants while holding the table fixed."""
    return [{"image": str(image), **run_case(table, image, question, text_agent, visual_agent)}
            for image in images]
