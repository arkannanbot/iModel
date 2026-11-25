"""Markdown report + best.json at the end of a search."""
from __future__ import annotations

import json
from pathlib import Path

from .runner import Workspace


def _fmt(m: dict | None) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in (m or {}).items()) or "-"


def write_report(ws: Workspace, best_id: str | None, summary: str, policy: str) -> Path:
    cfg = ws.cfg
    best = ws.get(best_id) if best_id else None
    lines = [
        f"# ModelTuner report: {cfg.task.name}",
        "",
        f"- Base model: `{cfg.model.base}` ({'QLoRA 4-bit' if cfg.model.quantization == '4bit' else 'LoRA'})",
        f"- Search policy: {policy}",
        f"- Splits: {ws.splits.stats()}",
        f"- Targets (validation): {cfg.evals.targets}",
        f"- Budget used: {ws.budget()}",
        "",
        "## Outcome",
        "",
    ]
    if best:
        val_ok = "met" if best.get("targets_met") else "NOT met"
        lines += [
            f"Best trial: **{best['id']}**. Validation: {_fmt(best.get('metrics'))} (targets {val_ok}).",
            f"Held-out test: {_fmt(best.get('test_metrics'))}"
            + (f" (targets {'met' if best.get('test_targets_met') else 'NOT met'})" if best.get("test_metrics") else ""),
            "",
        ]
        if best.get("adapter_dir"):
            lines += [f"Adapter: `{best['adapter_dir']}`", ""]
        if best.get("hyperparams"):
            lines += ["```json", json.dumps(best["hyperparams"], indent=2), "```", ""]
    else:
        lines += ["No successful trial.", ""]
    lines += ["## Agent summary", "", summary.strip() or "-", "", "## All trials", "",
              "| trial | status | validation metrics | lr | epochs | r | train loss (first→last) | eval loss/epoch | time (s) | notes |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t in ws.trials:
        hp = t.get("hyperparams") or {}
        loss = f"{t.get('train_loss_first')}→{t.get('train_loss_last')}" if t.get("train_loss_first") is not None else "-"
        notes = (t.get("notes") or t.get("error", "") or "").replace("\n", " ").replace("|", "/")[:120]
        lines.append(
            f"| {t['id']} | {t['status']} | {_fmt(t.get('metrics'))} | {hp.get('learning_rate', '-')} | "
            f"{hp.get('num_epochs', '-')} | {hp.get('lora_r', '-')} | {loss} | {t.get('eval_loss_per_epoch', '-')} | "
            f"{t.get('wall_seconds', '-')} | {notes} |"
        )
    path = cfg.out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    (cfg.out / "best.json").write_text(json.dumps(best or {}, indent=2))
    return path
