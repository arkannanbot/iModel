"""The tuning loop. Two interchangeable policies decide what to try next:

  ClaudePolicy     Claude (Opus 5.5 by default) drives the search through tools: it reads
                   results and failure cases, forms hypotheses, and picks the next config.
  HeuristicPolicy  No API key needed. A fixed under/overfitting playbook; useful as a
                   baseline and for offline smoke tests.

Either way, the harness (not the policy) owns the dataset, metrics, targets and budget.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .config import Hyperparams, RunConfig
from .runner import TrialError, Workspace

# --------------------------------------------------------------------------- shared


def _hardware() -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"device": "cuda", "gpu": p.name, "gpu_mem_gb": round(p.total_memory / 2**30, 1),
                "gpu_count": torch.cuda.device_count()}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"device": "mps"}
    import os

    return {"device": "cpu", "cpu_count": os.cpu_count()}


def _overview(ws: Workspace, baseline: dict[str, Any]) -> dict[str, Any]:
    cfg = ws.cfg
    samples = [{"messages": e.messages} for e in ws.splits.train[:3]]
    base = ws.summary_row(baseline)
    if baseline["status"] == "ok":
        base["worst_examples"] = ws.failures(baseline["id"], 4)
    return {
        "task": {"name": cfg.task.name, "description": cfg.task.description},
        "base_model": cfg.model.base,
        "quantization": cfg.model.quantization,
        "hardware": _hardware(),
        "dataset": ws.splits.stats(),
        "sample_training_examples": samples,
        "metrics": cfg.evals.metrics + (["llm_judge"] if cfg.evals.judge.enabled else []),
        "targets_on_validation": cfg.evals.targets,
        "budget": ws.budget(),
        "suggested_starting_hyperparams": cfg.search.initial.to_dict(),
        "baseline": base,
    }


# --------------------------------------------------------------------------- Claude policy

HP_SCHEMA = {
    "type": "object",
    "properties": {
        "learning_rate": {"type": "number", "description": "1e-7 to 1e-2. LoRA usually 5e-5 to 5e-4."},
        "num_epochs": {"type": "number", "description": "0.1 to 50"},
        "lora_r": {"type": "integer", "enum": [2, 4, 8, 16, 32, 64, 128, 256]},
        "lora_alpha": {"type": "integer", "description": "1 to 1024; commonly r or 2*r"},
        "lora_dropout": {"type": "number", "description": "0 to 0.5"},
        "target_modules": {"type": "string", "description": "'all-linear' or comma list like 'q_proj,v_proj'"},
        "batch_size": {"type": "integer", "description": "per-device batch size, 1 to 256"},
        "grad_accum": {"type": "integer", "description": "1 to 64"},
        "max_seq_len": {"type": "integer", "description": "32 to 32768; longer examples are truncated"},
        "warmup_ratio": {"type": "number"},
        "weight_decay": {"type": "number"},
        "lr_scheduler": {"type": "string", "enum": ["cosine", "linear", "constant", "constant_with_warmup"]},
        "neftune_noise_alpha": {"type": "number", "description": "0 disables; 5 is a common value"},
    },
    "required": ["learning_rate", "num_epochs", "lora_r", "lora_alpha", "lora_dropout", "target_modules",
                 "batch_size", "grad_accum", "max_seq_len", "warmup_ratio", "weight_decay", "lr_scheduler",
                 "neftune_noise_alpha"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "run_trial",
        "description": (
            "Fine-tune a fresh LoRA adapter on the base model with these hyperparameters, then evaluate it on the "
            "validation split. Blocks until done (minutes to hours). Returns validation metrics, whether targets "
            "are met, train/eval loss, step count and wall time. Each call uses one trial of the budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hyperparams": HP_SCHEMA,
                "hypothesis": {"type": "string", "description": "What you expect this trial to show and why."},
            },
            "required": ["hyperparams", "hypothesis"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "inspect_failures",
        "description": "Return the n lowest-scoring validation examples (input, reference, prediction, scores) for a trial, "
                       "including the baseline. Use it to understand *how* the model fails before changing settings.",
        "input_schema": {
            "type": "object",
            "properties": {"trial_id": {"type": "string"}, "n": {"type": "integer", "description": "1 to 20"}},
            "required": ["trial_id", "n"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "list_trials",
        "description": "Summaries of every trial so far, plus remaining budget.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "finish",
        "description": "End the search. Name the trial whose adapter should ship. The harness then scores it once on the "
                       "held-out test split and writes the report. Call this as soon as targets are met, when the budget "
                       "is spent, or when further trials are unlikely to help.",
        "input_schema": {
            "type": "object",
            "properties": {
                "best_trial_id": {"type": "string"},
                "summary": {"type": "string", "description": "What was tried, what worked, and recommended next steps "
                                                             "if targets were not met (e.g. more/better data, a larger model)."},
            },
            "required": ["best_trial_id", "summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

SYSTEM = """You are an expert ML engineer running a parameter-efficient fine-tuning search. Your goal is a LoRA adapter for the given open-weight base model that meets every validation target, using as few trials as possible.

How the harness works:
- The dataset splits, metrics, targets and budget are fixed. You choose hyperparameters; you cannot change data or metrics.
- Each run_trial trains a fresh adapter from the base model and reports validation metrics plus train/eval loss. The held-out test split is scored once, after you call finish, so do not try to overfit validation.
- The hardware and dataset size are in the overview. Keep each trial's cost in mind: on CPU or a small GPU, prefer short runs first and scale up once a direction is confirmed.

How to work:
- Diagnose before changing settings. Compare against the baseline. Use train loss vs eval loss to tell underfitting (both high: raise learning rate, epochs, or rank) from overfitting (eval loss rising across epochs: fewer epochs, more dropout, lower rank or learning rate). Call inspect_failures to see whether errors are format problems, truncation (max_seq_len or max_new_tokens), or genuine knowledge gaps.
- Change one or two things per trial unless the previous result was far off, and state your hypothesis.
- If a trial fails (for example out of memory), reduce batch_size and raise grad_accum, or shorten max_seq_len.
- Stop as soon as targets are met: call finish with that trial. If the budget runs out or progress has stalled, call finish with the best trial and explain in the summary what would most likely close the gap (more or cleaner examples of the failing pattern, a larger base model, a different metric definition, and so on).
- Keep your messages brief; the tool calls are what matter."""


class ClaudePolicy:
    def __init__(self, ws: Workspace, log=print):
        from .llm import make_client

        self.ws = ws
        self.client = make_client()
        self.log = log
        self.messages: list[dict[str, Any]] = []
        self.transcript = open(ws.cfg.out / "agent_transcript.jsonl", "a")

    def _note(self, kind: str, payload: Any) -> None:
        self.transcript.write(json.dumps({"t": time.time(), "kind": kind, "data": payload}, default=str) + "\n")
        self.transcript.flush()

    def _exec(self, name: str, args: dict[str, Any]) -> tuple[str, bool, dict | None]:
        """Returns (content, is_error, finish_payload)."""
        ws = self.ws
        try:
            if name == "run_trial":
                if ws.cfg.search.stop_on_target and any(t.get("targets_met") and t["kind"] == "train" for t in ws.trials):
                    return "Error: targets are already met; call finish.", True, None
                hp = Hyperparams.from_dict(args["hyperparams"])
                self.log(f"[agent] run_trial: {args.get('hypothesis', '')[:160]}")
                rec = ws.run_trial(hp, notes=args.get("hypothesis", ""))
                out = ws.summary_row(rec)
                out["budget"] = ws.budget()
                if rec.get("targets_met"):
                    out["next_step"] = "All validation targets are met. Call finish with this trial."
                elif not ws.budget_left():
                    out["next_step"] = "Budget exhausted. Call finish with the best trial."
                self.log(f"[agent]   -> {rec['status']} {rec.get('metrics')}")
                return json.dumps(out), rec["status"] != "ok", None
            if name == "inspect_failures":
                n = max(1, min(int(args["n"]), 20))
                return json.dumps(ws.failures(args["trial_id"], n), ensure_ascii=False), False, None
            if name == "list_trials":
                return json.dumps({"trials": [ws.summary_row(t) for t in ws.trials], "budget": ws.budget()}), False, None
            if name == "finish":
                ws.get(args["best_trial_id"])  # validate
                return "ok", False, args
            return f"Unknown tool {name}", True, None
        except (ValueError, KeyError, TrialError) as e:
            return f"Error: {e}", True, None

    def run(self, baseline: dict[str, Any]) -> dict[str, Any]:
        cfg = self.ws.cfg
        overview = _overview(self.ws, baseline)
        self.messages = [{"role": "user", "content": "Here is the fine-tuning task. Start the search.\n\n"
                                                     + json.dumps(overview, indent=1, ensure_ascii=False)}]
        from .llm import create

        nudges, turns = 0, 0
        max_turns = cfg.search.max_trials * 4 + 10
        while True:
            turns += 1
            if turns > max_turns:
                best = self.ws.best()
                return {"best_trial_id": best["id"] if best else None, "summary": "Stopped: agent turn limit reached."}
            resp = create(
                self.client,
                model=cfg.search.agent_model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": cfg.search.agent_effort},
                system=SYSTEM,
                tools=TOOLS,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": resp.content})
            texts = [b.text for b in resp.content if b.type == "text" and b.text.strip()]
            for t in texts:
                self.log(f"[agent] {t.strip()[:400]}")
            self._note("assistant", [b.model_dump() for b in resp.content if b.type in ("text", "tool_use")])

            if resp.stop_reason == "refusal":
                self.log("[agent] model declined; falling back to best trial so far")
                return {"best_trial_id": (self.ws.best() or {}).get("id"), "summary": "Agent stopped (refusal)."}
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                nudges += 1
                if nudges > 2:
                    best = self.ws.best()
                    return {"best_trial_id": best["id"] if best else None, "summary": "\n".join(texts)}
                self.messages.append({"role": "user", "content": "Continue: call run_trial, or finish if you are done."})
                continue
            results, finish = [], None
            for c in calls:
                content, is_err, fin = self._exec(c.name, c.input)
                finish = finish or fin
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": content, "is_error": is_err})
            self.messages.append({"role": "user", "content": results})
            self._note("tool_results", results)
            if finish:
                return finish


# --------------------------------------------------------------------------- heuristic policy


class HeuristicPolicy:
    """Deterministic playbook: start from the initial config, then react to under/overfitting."""

    def __init__(self, ws: Workspace, log=print):
        self.ws, self.log = ws, log

    def _next(self, best: dict[str, Any]) -> tuple[Hyperparams, str]:
        hp = Hyperparams.from_dict(best["hyperparams"]).to_dict()
        first, last = best.get("train_loss_first"), best.get("train_loss_last")
        ev = best.get("eval_loss_per_epoch") or []
        tried = {json.dumps(t["hyperparams"], sort_keys=True) for t in self.ws.trials if t.get("hyperparams")}
        moves = []
        overfit = len(ev) >= 2 and ev[-1] > min(ev) * 1.05
        if overfit:
            moves.append(("overfitting: fewer epochs, more dropout",
                          {"num_epochs": max(0.5, hp["num_epochs"] * 0.6), "lora_dropout": min(0.3, hp["lora_dropout"] + 0.05)}))
        # Underfitting: training loss barely moved, or validation is far below target with no sign of overfitting.
        if (first and last and last > 0.3 * first) or (not overfit and best.get("score", 0) < 0.9):
            moves.append(("underfitting: higher learning rate", {"learning_rate": min(1e-3, hp["learning_rate"] * 2)}))
            moves.append(("underfitting: more epochs", {"num_epochs": min(20, hp["num_epochs"] * 2)}))
        moves.append(("more capacity: double LoRA rank", {"lora_r": min(256, hp["lora_r"] * 2), "lora_alpha": min(1024, hp["lora_alpha"] * 2)}))
        moves.append(("more training: more epochs", {"num_epochs": min(20, hp["num_epochs"] * 1.5)}))
        moves.append(("higher learning rate", {"learning_rate": min(1e-3, hp["learning_rate"] * 2)}))
        moves.append(("lower learning rate, longer", {"learning_rate": hp["learning_rate"] / 2, "num_epochs": min(20, hp["num_epochs"] * 2)}))
        for why, delta in moves:
            cand = {**hp, **delta}
            if json.dumps(cand, sort_keys=True) not in tried:
                return Hyperparams.from_dict(cand), why
        return Hyperparams.from_dict({**hp, "learning_rate": hp["learning_rate"] * 1.5}), "perturb learning rate"

    def run(self, baseline: dict[str, Any]) -> dict[str, Any]:
        ws = self.ws
        while ws.budget_left():
            best = ws.best()
            last = ws.trials[-1]
            if last["kind"] == "train" and last["status"] == "failed":
                hp0 = last["hyperparams"]
                hp = Hyperparams.from_dict({**hp0, "batch_size": max(1, hp0["batch_size"] // 2),
                                            "grad_accum": min(64, hp0["grad_accum"] * 2)})
                why = "previous trial failed: halve batch size, double grad accumulation"
            elif best is None or best["kind"] != "train":
                hp, why = ws.cfg.search.initial, "initial config"
            else:
                if best.get("targets_met") and ws.cfg.search.stop_on_target:
                    break
                hp, why = self._next(best)
            self.log(f"[heuristic] trial: {why} -> lr={hp.learning_rate:g} epochs={hp.num_epochs:g} r={hp.lora_r}")
            rec = ws.run_trial(hp, notes=why)
            self.log(f"[heuristic]   -> {rec['status']} {rec.get('metrics')} {rec.get('error', '')[-300:]}")
            if rec.get("targets_met") and ws.cfg.search.stop_on_target:
                break
        best = ws.best()
        return {"best_trial_id": best["id"] if best else None,
                "summary": f"Heuristic search finished after {ws.n_training_trials} training trials."}


# --------------------------------------------------------------------------- entry point


def run_search(cfg: RunConfig, policy: str | None = None, log=print) -> dict[str, Any]:
    from .report import write_report

    ws = Workspace(cfg)
    log(f"[modeltuner] splits: {ws.splits.stats()}  output: {cfg.out}")
    baseline = ws.run_baseline()
    log(f"[modeltuner] baseline: {baseline['status']} {baseline.get('metrics')} {baseline.get('error', '')[-500:]}")
    if baseline["status"] != "ok":
        raise SystemExit("Baseline evaluation failed; see the trial's worker.log")

    policy = policy or cfg.search.policy
    if policy == "claude":
        from .llm import has_credentials

        if not has_credentials():
            log("[modeltuner] No Anthropic credentials found; using the heuristic policy instead.")
            policy = "heuristic"
    pol = ClaudePolicy(ws, log) if policy == "claude" else HeuristicPolicy(ws, log)
    outcome = pol.run(baseline)

    best_id = outcome.get("best_trial_id") or (ws.best() or {}).get("id")
    test = None
    if best_id and ws.splits.test:
        log(f"[modeltuner] scoring {best_id} on held-out test split")
        test = ws.evaluate_test(best_id)
        log(f"[modeltuner] test: {test['metrics']}  targets met: {test['targets_met']}")
    report = write_report(ws, best_id, outcome.get("summary", ""), policy)
    log(f"[modeltuner] report: {report}")
    return {"best_trial_id": best_id, "test": test, "report": str(report), "policy": policy}
