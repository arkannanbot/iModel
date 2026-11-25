"""Command line entry point.

  modeltuner run    configs/example.yaml [--policy claude|heuristic]
  modeltuner eval   configs/example.yaml --trial t03-train [--split test]
  modeltuner export configs/example.yaml --trial t03-train --to merged/
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="modeltuner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="baseline, then search until targets are met or budget is spent")
    r.add_argument("config")
    r.add_argument("--policy", choices=["claude", "heuristic"])
    r.add_argument("--max-trials", type=int)
    e = sub.add_parser("eval", help="score a trial's adapter on the test split")
    e.add_argument("config")
    e.add_argument("--trial", required=True)
    x = sub.add_parser("export", help="merge a trial's LoRA adapter into the base weights")
    x.add_argument("config")
    x.add_argument("--trial", required=True)
    x.add_argument("--to", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "run":
        if a.max_trials:
            cfg.search.max_trials = a.max_trials
        from .agent import run_search

        out = run_search(cfg, a.policy, log=lambda s: print(s, flush=True))
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "eval":
        from .runner import Workspace

        print(json.dumps(Workspace(cfg).evaluate_test(a.trial), indent=2))
    elif a.cmd == "export":
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        from .runner import Workspace

        rec = Workspace(cfg).get(a.trial)
        if rec["kind"] != "train":
            sys.exit("Only training trials have an adapter to export")
        model = AutoPeftModelForCausalLM.from_pretrained(rec["adapter_dir"]).merge_and_unload()
        model.save_pretrained(a.to)
        AutoTokenizer.from_pretrained(rec["adapter_dir"]).save_pretrained(a.to)
        print(f"Merged model written to {a.to}")


if __name__ == "__main__":
    main()
