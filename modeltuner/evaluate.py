"""Score predictions against references, optionally with Claude as an LLM judge."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import EvalSpec, TaskSpec
from .data import Example
from .metrics import METRICS

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def _judge_one(client, spec: EvalSpec, task: TaskSpec, ex: Example, pred: str) -> tuple[float, str]:
    from .llm import create

    prompt = (
        f"You are grading a model's answer for this task: {task.description}\n\n"
        f"Rubric: {spec.judge.rubric}\n\n"
        f"<input>\n{json.dumps(ex.prompt_messages, ensure_ascii=False)}\n</input>\n"
        f"<reference_answer>\n{ex.reference}\n</reference_answer>\n"
        f"<model_answer>\n{pred}\n</model_answer>\n\n"
        "Score 1 (wrong or unusable) to 5 (as good as the reference). Return JSON."
    )
    resp = create(
        client,
        model=spec.judge.model,
        max_tokens=2000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        return 0.0, "judge refused"
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    return (data["score"] - 1) / 4, data["reason"]


def score(
    examples: list[Example], preds: list[str], spec: EvalSpec, task: TaskSpec
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for ex, p in zip(examples, preds):
        rows.append({
            "id": ex.id,
            "input": ex.prompt_messages[-1]["content"],
            "reference": ex.reference,
            "prediction": p,
            "scores": {m: round(METRICS[m](p, ex.reference), 4) for m in spec.metrics if m in METRICS},
        })
    if spec.judge.enabled:
        from .llm import make_client

        client = make_client()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda xp: _judge_one(client, spec, task, *xp), zip(examples, preds)))
        for row, (s, reason) in zip(rows, results):
            row["scores"]["llm_judge"] = round(s, 4)
            row["judge_reason"] = reason
    names = list(rows[0]["scores"]) if rows else []
    agg = {m: round(sum(r["scores"][m] for r in rows) / len(rows), 4) for m in names}
    return agg, rows


def targets_met(agg: dict[str, float], targets: dict[str, float]) -> bool:
    return all(agg.get(k, 0.0) >= v for k, v in targets.items())


def primary_score(agg: dict[str, float], targets: dict[str, float]) -> float:
    """One number for ranking trials: mean of each targeted metric's fraction of its target (capped at 1),
    with raw metric means as a tie-breaker."""
    if not targets:
        return sum(agg.values()) / max(len(agg), 1)
    frac = sum(min(agg.get(k, 0.0) / v, 1.0) if v > 0 else 1.0 for k, v in targets.items()) / len(targets)
    raw = sum(agg.get(k, 0.0) for k in targets) / len(targets)
    return round(frac + 1e-3 * raw, 6)
