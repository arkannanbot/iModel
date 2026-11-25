"""Trial execution and the on-disk ledger of everything that has been tried.

Layout of <output_dir>:
  data/{train,val,test}.jsonl   frozen splits used by every trial
  trials.json                   ledger: one record per trial (hyperparams, metrics, timings)
  trials/<id>/                  spec.json, result.json, predictions.jsonl, scored.jsonl, adapter/
  report.md, best.json          written at the end
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Hyperparams, RunConfig
from .data import Example, Splits, load_jsonl, load_splits, write_jsonl
from .evaluate import primary_score, score, targets_met


class TrialError(RuntimeError):
    pass


@dataclass
class Workspace:
    cfg: RunConfig
    splits: Splits = field(init=False)
    trials: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        out = self.cfg.out
        out.mkdir(parents=True, exist_ok=True)
        data_dir = out / "data"
        if (data_dir / "train.jsonl").exists():  # resume: reuse frozen splits
            self.splits = Splits(*(load_jsonl(data_dir / f"{s}.jsonl", None) for s in ("train", "val", "test")))
        else:
            self.splits = load_splits(self.cfg)
            for name in ("train", "val", "test"):
                write_jsonl(data_dir / f"{name}.jsonl", getattr(self.splits, name))
        ledger = out / "trials.json"
        if ledger.exists():
            self.trials = json.loads(ledger.read_text())
        mx = self.cfg.evals.max_examples
        if mx and len(self.splits.val) > mx:
            self.splits.val = self.splits.val[:mx]
            write_jsonl(data_dir / "val.jsonl", self.splits.val)

    # ---------- budget ----------
    @property
    def n_training_trials(self) -> int:
        # Failed attempts count too, so a crashing config can't loop forever.
        return sum(1 for t in self.trials if t["kind"] == "train" and t["status"] in ("ok", "failed"))

    def hours_used(self) -> float:
        return round(sum(t.get("wall_seconds", 0) for t in self.trials) / 3600, 3)

    def budget(self) -> dict[str, Any]:
        s = self.cfg.search
        return {
            "training_trials_used": self.n_training_trials,
            "training_trials_max": s.max_trials,
            "compute_hours_used": self.hours_used(),
            "compute_hours_max": s.max_hours,
        }

    def budget_left(self) -> bool:
        s = self.cfg.search
        return self.n_training_trials < s.max_trials and self.hours_used() < s.max_hours

    # ---------- trials ----------
    def _save(self) -> None:
        (self.cfg.out / "trials.json").write_text(json.dumps(self.trials, indent=2))

    def _run_worker(self, spec: dict[str, Any], tdir: Path) -> dict[str, Any]:
        tdir.mkdir(parents=True, exist_ok=True)
        spec_path = tdir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        log = open(tdir / "worker.log", "w")
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
        proc = subprocess.run([sys.executable, "-m", "modeltuner.worker", str(spec_path)],
                              stdout=log, stderr=subprocess.STDOUT, env=env,
                              cwd=str(Path(__file__).resolve().parent.parent))
        log.close()
        if proc.returncode != 0:
            tail = (tdir / "worker.log").read_text()[-3000:]
            raise TrialError(f"worker exited with {proc.returncode}:\n{tail}")
        return json.loads((tdir / "result.json").read_text())

    def _base_spec(self, mode: str, eval_split: str) -> dict[str, Any]:
        m, ev = self.cfg.model, self.cfg.evals
        d = self.cfg.out / "data"
        return {
            "mode": mode,
            "base_model": m.base,
            "quantization": m.quantization,
            "dtype": m.dtype,
            "trust_remote_code": m.trust_remote_code,
            "train_path": str(d / "train.jsonl"),
            "eval_path": str(d / f"{eval_split}.jsonl"),
            "max_new_tokens": ev.max_new_tokens,
            "eval_batch_size": ev.batch_size,
            "seed": self.cfg.data.seed,
        }

    def _score(self, tdir: Path, examples: list[Example]) -> tuple[dict[str, float], list[dict]]:
        preds_by_id = {}
        with open(tdir / "predictions.jsonl") as f:
            for line in f:
                r = json.loads(line)
                preds_by_id[r["id"]] = r["prediction"]
        preds = [preds_by_id[e.id] for e in examples]
        agg, rows = score(examples, preds, self.cfg.evals, self.cfg.task)
        with open(tdir / "scored.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return agg, rows

    def _record(self, kind: str, hp: Hyperparams | None, notes: str) -> dict[str, Any]:
        tid = f"t{len(self.trials):02d}-{kind}"
        rec = {"id": tid, "kind": kind, "hyperparams": hp.to_dict() if hp else None,
               "notes": notes, "status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.trials.append(rec)
        self._save()
        return rec

    def _finish(self, rec: dict[str, Any], t0: float, tdir: Path, result: dict, split_examples: list[Example]) -> None:
        agg, _ = self._score(tdir, split_examples)
        targets = self.cfg.evals.targets
        rec.update({
            "status": "ok",
            "metrics": agg,
            "targets_met": targets_met(agg, targets),
            "score": primary_score(agg, targets),
            "wall_seconds": round(time.time() - t0, 1),
            **{k: v for k, v in result.items() if k not in ("mode",)},
        })

    def run_baseline(self) -> dict[str, Any]:
        for t in self.trials:
            if t["kind"] == "baseline" and t["status"] == "ok":
                return t
        rec = self._record("baseline", None, "base model, no fine-tuning")
        tdir = self.cfg.out / "trials" / rec["id"]
        t0 = time.time()
        try:
            result = self._run_worker({**self._base_spec("eval", "val"), "out_dir": str(tdir)}, tdir)
            self._finish(rec, t0, tdir, result, self.splits.val)
        except TrialError as e:
            rec.update({"status": "failed", "error": str(e)[-1500:], "wall_seconds": round(time.time() - t0, 1)})
        self._save()
        return rec

    def run_trial(self, hp: Hyperparams, notes: str = "") -> dict[str, Any]:
        if not self.budget_left():
            raise TrialError("Budget exhausted: no more training trials allowed.")
        rec = self._record("train", hp, notes)
        tdir = self.cfg.out / "trials" / rec["id"]
        t0 = time.time()
        try:
            spec = {**self._base_spec("train_eval", "val"), "hparams": hp.to_dict(), "out_dir": str(tdir)}
            result = self._run_worker(spec, tdir)
            self._finish(rec, t0, tdir, result, self.splits.val)
        except TrialError as e:
            rec.update({"status": "failed", "error": str(e)[-1500:], "wall_seconds": round(time.time() - t0, 1)})
        self._save()
        return rec

    def evaluate_test(self, trial_id: str) -> dict[str, Any]:
        """Score a trial's adapter on the held-out test split (done once, at the end)."""
        rec = self.get(trial_id)
        tdir = self.cfg.out / "trials" / trial_id / "test_eval"
        spec = {**self._base_spec("eval", "test"), "out_dir": str(tdir)}
        if rec["kind"] == "train":
            spec["adapter_path"] = rec["adapter_dir"]
        result = self._run_worker(spec, tdir)
        agg, _ = self._score(tdir, self.splits.test)
        rec["test_metrics"] = agg
        rec["test_targets_met"] = targets_met(agg, self.cfg.evals.targets)
        self._save()
        return {"metrics": agg, "targets_met": rec["test_targets_met"], **result}

    # ---------- queries ----------
    def get(self, trial_id: str) -> dict[str, Any]:
        for t in self.trials:
            if t["id"] == trial_id:
                return t
        raise KeyError(f"No trial {trial_id!r}. Known: {[t['id'] for t in self.trials]}")

    def best(self) -> dict[str, Any] | None:
        ok = [t for t in self.trials if t["status"] == "ok"]
        return max(ok, key=lambda t: t["score"]) if ok else None

    def failures(self, trial_id: str, n: int = 8) -> list[dict[str, Any]]:
        path = self.cfg.out / "trials" / trial_id / "scored.jsonl"
        if not path.exists():
            raise KeyError(f"Trial {trial_id!r} has no scored predictions")
        rows = [json.loads(line) for line in open(path)]
        keys = list(self.cfg.evals.targets) or list(rows[0]["scores"])
        rows.sort(key=lambda r: sum(r["scores"].get(k, 0) for k in keys))
        return rows[:n]

    def summary_row(self, t: dict[str, Any]) -> dict[str, Any]:
        keep = ("id", "kind", "status", "metrics", "targets_met", "score", "hyperparams", "notes",
                "train_loss_first", "train_loss_last", "eval_loss_per_epoch", "steps", "wall_seconds", "error")
        return {k: t[k] for k in keep if k in t}
