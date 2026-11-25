"""Golden dataset loading.

Accepted JSONL record shapes (one per line); all are normalised to chat messages whose
last message is the assistant's reference answer:

  {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}
  {"prompt": "...", "completion": "..."}
  {"instruction": "...", "input": "...optional context...", "output": "..."}
  {"input": "...", "output": "..."}           (also "question"/"answer", "query"/"response")
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig

_PAIRS = [("prompt", "completion"), ("question", "answer"), ("query", "response"), ("input", "output")]


@dataclass
class Example:
    id: str
    messages: list[dict[str, str]]  # last message is the assistant reference

    @property
    def prompt_messages(self) -> list[dict[str, str]]:
        return self.messages[:-1]

    @property
    def reference(self) -> str:
        return self.messages[-1]["content"]

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "messages": self.messages}


def _normalise(rec: dict[str, Any], system_prompt: str | None, idx: int, src: str) -> Example:
    if "messages" in rec:
        msgs = [{"role": m["role"], "content": str(m["content"])} for m in rec["messages"]]
    elif "instruction" in rec and "output" in rec:
        user = rec["instruction"] + (f"\n\n{rec['input']}" if rec.get("input") else "")
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": str(rec["output"])}]
    else:
        for a, b in _PAIRS:
            if a in rec and b in rec:
                msgs = [{"role": "user", "content": str(rec[a])}, {"role": "assistant", "content": str(rec[b])}]
                break
        else:
            raise ValueError(f"{src}:{idx + 1}: unrecognised record keys {sorted(rec)}")
    if not msgs or msgs[-1]["role"] != "assistant":
        raise ValueError(f"{src}:{idx + 1}: last message must be the assistant's reference answer")
    if system_prompt and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return Example(id=str(rec.get("id", f"{Path(src).stem}-{idx}")), messages=msgs)


def load_jsonl(path: Path, system_prompt: str | None) -> list[Example]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                out.append(_normalise(json.loads(line), system_prompt, i, str(path)))
    if not out:
        raise ValueError(f"{path} is empty")
    return out


@dataclass
class Splits:
    train: list[Example]
    val: list[Example]
    test: list[Example]

    def stats(self) -> dict[str, Any]:
        def avg_chars(xs, f):
            return round(sum(len(f(x)) for x in xs) / max(len(xs), 1))

        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "avg_prompt_chars": avg_chars(self.train, lambda e: " ".join(m["content"] for m in e.prompt_messages)),
            "avg_reference_chars": avg_chars(self.train, lambda e: e.reference),
        }


def load_splits(cfg: RunConfig) -> Splits:
    sp = cfg.task.system_prompt
    train = load_jsonl(cfg.resolve(cfg.data.train), sp)
    val = load_jsonl(cfg.resolve(cfg.data.eval), sp) if cfg.data.eval else []
    test = load_jsonl(cfg.resolve(cfg.data.test), sp) if cfg.data.test else []
    rng = random.Random(cfg.data.seed)
    pool = train[:]
    rng.shuffle(pool)
    n = len(pool)
    # Carve missing splits out of the training file, deterministically.
    if not test:
        k = max(1, int(n * cfg.data.test_fraction))
        test, pool = pool[:k], pool[k:]
    if not val:
        k = max(1, int(n * cfg.data.val_fraction))
        val, pool = pool[:k], pool[k:]
    # Guard against leakage: identical prompts in train and val/test.
    held = {json.dumps(e.prompt_messages) for e in val + test}
    pool = [e for e in pool if json.dumps(e.prompt_messages) not in held]
    if len(pool) < 8:
        raise ValueError(f"Only {len(pool)} training examples after splitting; need at least 8")
    return Splits(train=pool, val=val, test=test)


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps(e.to_json()) + "\n")
