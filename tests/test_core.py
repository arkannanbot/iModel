"""Fast tests: no model download, no API calls. The Claude loop is driven by a scripted fake client."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeltuner import agent as agent_mod
from modeltuner.config import Hyperparams, load_config
from modeltuner.data import _normalise
from modeltuner.evaluate import primary_score, targets_met
from modeltuner.metrics import exact_match, rouge_l, token_f1

ROOT = Path(__file__).resolve().parent.parent


def test_metrics():
    assert exact_match("2020-01-02", " 2020-01-02 ") == 1.0
    assert token_f1("the cat sat", "cat sat") == 1.0
    assert 0 < rouge_l("a b c d", "a b x d") < 1


def test_normalise_formats():
    for rec in ({"prompt": "q", "completion": "a"}, {"input": "q", "output": "a"},
                {"instruction": "do", "input": "q", "output": "a"},
                {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}):
        ex = _normalise(rec, "sys", 0, "x.jsonl")
        assert ex.messages[0] == {"role": "system", "content": "sys"}
        assert ex.reference == "a"
    with pytest.raises(ValueError):
        _normalise({"foo": 1}, None, 0, "x.jsonl")


def test_hparam_validation():
    with pytest.raises(ValueError):
        Hyperparams.from_dict({"lora_r": 12})
    with pytest.raises(ValueError):
        Hyperparams.from_dict({"bogus": 1})


def test_targets():
    assert targets_met({"exact_match": 0.9}, {"exact_match": 0.85})
    assert not targets_met({"exact_match": 0.8}, {"exact_match": 0.85})
    assert primary_score({"exact_match": 0.9}, {"exact_match": 0.85}) > primary_score({"exact_match": 0.5}, {"exact_match": 0.85})


class FakeWorkspace:
    """Stands in for runner.Workspace: trials 'improve' by learning rate."""

    def __init__(self, cfg):
        self.cfg, self.trials = cfg, []

    def budget(self):
        return {"used": len(self.trials)}

    def budget_left(self):
        return len(self.trials) < self.cfg.search.max_trials

    def run_trial(self, hp, notes=""):
        em = min(1.0, hp.learning_rate * 3000)
        rec = {"id": f"t{len(self.trials):02d}-train", "kind": "train", "status": "ok", "hyperparams": hp.to_dict(),
               "metrics": {"exact_match": em}, "targets_met": em >= 0.85, "score": em, "notes": notes}
        self.trials.append(rec)
        return rec

    def summary_row(self, t):
        return t

    def failures(self, tid, n):
        return [{"id": "x", "reference": "1", "prediction": "2"}][:n]

    def get(self, tid):
        for t in self.trials:
            if t["id"] == tid:
                return t
        raise KeyError(tid)

    def best(self):
        return max(self.trials, key=lambda t: t["score"]) if self.trials else None


def _tool_use(i, name, inp):
    return SimpleNamespace(type="tool_use", id=f"tu{i}", name=name, input=inp,
                           model_dump=lambda: {"type": "tool_use", "name": name, "input": inp})


def test_claude_loop_with_scripted_client(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs" / "example.yaml")
    cfg.output_dir = str(tmp_path)
    hp = cfg.search.initial.to_dict()
    script = [
        [_tool_use(0, "inspect_failures", {"trial_id": "t00-baseline", "n": 3})],
        [_tool_use(1, "run_trial", {"hyperparams": {**hp, "learning_rate": 1e-4}, "hypothesis": "start low"})],
        [_tool_use(2, "run_trial", {"hyperparams": {**hp, "learning_rate": 4e-4}, "hypothesis": "raise lr"})],
        [_tool_use(3, "run_trial", {"hyperparams": {**hp, "learning_rate": 5e-4}, "hypothesis": "should be refused"})],
        [_tool_use(4, "finish", {"best_trial_id": "t01-train", "summary": "done"})],
    ]
    sent = []

    def fake_create(client, **kw):
        sent.append(kw)
        content = script.pop(0)
        return SimpleNamespace(content=content, stop_reason="tool_use")

    monkeypatch.setattr("modeltuner.llm.create", fake_create)
    monkeypatch.setattr("modeltuner.llm.make_client", lambda: None)
    ws = FakeWorkspace(cfg)
    ws.trials.append({"id": "t00-baseline", "kind": "baseline", "status": "ok", "metrics": {"exact_match": 0.0},
                      "targets_met": False, "score": 0.0})
    pol = agent_mod.ClaudePolicy(ws, log=lambda s: None)
    monkeypatch.setattr(agent_mod, "_overview", lambda ws, b: {"stub": True})
    out = pol.run(ws.trials[0])
    assert out == {"best_trial_id": "t01-train", "summary": "done"}
    # third run_trial was refused because targets were already met
    results = {r["tool_use_id"]: r for m in pol.messages if m["role"] == "user" and isinstance(m["content"], list)
               for r in m["content"]}
    assert results["tu3"]["is_error"] and "already met" in results["tu3"]["content"]
    assert len([t for t in ws.trials if t["kind"] == "train"]) == 2
    assert sent[0]["model"] == "claude-opus-5-5" and sent[0]["thinking"] == {"type": "adaptive"}
    assert all(t.get("strict") for t in sent[0]["tools"])
    json.dumps(agent_mod.TOOLS)  # serialisable
