# ModelTuner

An agent that takes a **golden dataset** and an **open-weight model**, then fine-tunes (LoRA / QLoRA via Hugging Face TRL + PEFT) and evaluates in a loop until the eval targets are met or the budget runs out.

```
golden dataset ──► frozen splits (train / val / test)
                         │
base model ──► baseline eval on val
                         │
        ┌────────────────▼────────────────┐
        │  policy picks next hyperparams   │  ◄── Claude (Opus 5.5) via tool use,
        │  run_trial: LoRA SFT + val eval  │      or an offline heuristic playbook
        │  inspect_failures / list_trials  │
        └────────────────┬────────────────┘
          targets met, budget spent, or agent calls finish
                         │
         best adapter scored ONCE on held-out test ──► report.md, best.json, adapter/
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # add ".[qlora]" on a CUDA box for 4-bit QLoRA
export ANTHROPIC_API_KEY=... # optional; without it the heuristic policy runs

python scripts/make_sample_data.py              # sample golden set: dates -> ISO 8601
modeltuner run configs/example.yaml             # Claude-driven search
modeltuner run configs/example.yaml --policy heuristic
modeltuner export configs/example.yaml --trial t02-train --to merged/   # merge LoRA into base weights
```

Outputs go to `output_dir` (default `runs/<task>`): `report.md`, `best.json`, `trials.json` (full ledger, so a run can be resumed by re-running the same command), and per-trial `adapter/`, `predictions.jsonl`, `scored.jsonl`, `worker.log`. With the Claude policy, `agent_transcript.jsonl` records every decision.

## Using your own data and model

1. Put your golden data in JSONL. Any of these shapes work (mixed files are fine):
   - `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}`
   - `{"prompt": ..., "completion": ...}`
   - `{"instruction": ..., "input": ..., "output": ...}`  or `{"input": ..., "output": ...}`
   The last assistant message is the reference answer. Loss is computed on the answer only.
2. Copy `configs/example.yaml`, then set `task.description`, `model.base` (any HF causal LM with a chat template, e.g. `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`), the data paths, metrics and targets.
3. If you only have one file, leave `eval`/`test` empty: 15% / 15% are carved out deterministically, and duplicate prompts are removed from train to avoid leakage.

### Metrics

`exact_match`, `normalized_match`, `token_f1`, `rouge_l`, `contains_reference`, `json_match`, plus `llm_judge` (Claude grades each answer 1 to 5 against the reference using your rubric; set `evals.judge.enabled: true`). Targets are thresholds on any of these, all of which must be met on the validation split.

## How the agent works

`modeltuner/agent.py`. The harness owns the data, metrics, targets and budget; the policy only chooses hyperparameters, so it cannot pass by moving the goalposts.

- **ClaudePolicy** runs a tool-use loop on the Messages API with `claude-opus-5-5`, adaptive thinking, and effort `high`. Tools: `run_trial`, `inspect_failures`, `list_trials`, `finish`, all with `strict: true` schemas. The system prompt tells it to diagnose under/overfitting from train vs eval loss, read failure cases before changing settings, change one or two things at a time, and stop once targets are met. Server-side refusal fallbacks are enabled (`fallbacks: "default"`), so a rare safety decline is retried on a fallback model instead of stopping the run.
- **HeuristicPolicy** needs no API key: initial config, then a fixed playbook (overfitting → fewer epochs/more dropout; underfitting → higher LR, more epochs; then more LoRA rank).
- Hard stops enforced by the harness: `max_trials`, `max_hours`, and `stop_on_target` (further `run_trial` calls are refused once targets are met).
- Each trial runs in a fresh subprocess (`modeltuner/worker.py`), so GPU memory is fully released between trials.

## Assumptions in this first version (please check)

| Area | Guess | Change it in |
|---|---|---|
| Dataset format | Chat-style JSONL, single-turn answers, one reference per prompt | `data.py` |
| Eval metrics | Reference-based string metrics, optional Claude judge | `metrics.py`, `evaluate.py` |
| Targets / stopping | Every target met on validation, or `max_trials` / `max_hours` spent; test split scored once at the end | config `evals.targets`, `search.*` |
| Search space | LoRA hyperparameters only (LR, epochs, rank, alpha, dropout, target modules, batch, seq len, scheduler, NEFTune). The agent cannot edit data or switch base model | `config.Hyperparams` |
| Compute | One machine, one GPU (or CPU/MPS for tiny models); no multi-GPU/distributed training yet | `worker.py` |
| Agent model | `claude-opus-5-5` at effort `high` | config `search.agent_model` |

## Likely next steps

- Let the agent propose **targeted synthetic data** for failure clusters (kept separate from the golden set, with the test split untouched).
- Let the agent **choose among several base models** and escalate size when LoRA plateaus.
- Multi-GPU via `accelerate launch`, and running trials on a remote GPU (Modal, RunPod, SageMaker).
- Task-specific evaluators (e.g. code execution, schema validation, a classifier F1).

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```
The tests need no model download or API key; the Claude loop is driven by a scripted fake client.
