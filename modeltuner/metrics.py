"""Reference-based metrics. Each returns a score in [0, 1] for one (prediction, reference) pair."""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Callable


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip() == ref.strip())


def normalized_match(pred: str, ref: str) -> float:
    return float(_norm(pred) == _norm(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = _norm(pred).split(), _norm(ref).split()
    if not p and not r:
        return 1.0
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = pred.split(), ref.split()
    if not p or not r:
        return float(p == r)
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def contains_reference(pred: str, ref: str) -> float:
    return float(_norm(ref) in _norm(pred))


def json_match(pred: str, ref: str) -> float:
    """1.0 if both parse as JSON and are equal; 0.5 if prediction is valid JSON but differs."""
    try:
        p = json.loads(pred)
    except Exception:
        return 0.0
    try:
        return 1.0 if p == json.loads(ref) else 0.5
    except Exception:
        return 0.5


METRICS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "normalized_match": normalized_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "contains_reference": contains_reference,
    "json_match": json_match,
}
