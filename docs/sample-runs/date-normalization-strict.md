# ModelTuner report: date-normalization

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct` (LoRA)
- Search policy: heuristic
- Splits: {'train': 400, 'val': 60, 'test': 60, 'avg_prompt_chars': 118, 'avg_reference_chars': 10}
- Targets (validation): {'exact_match': 0.99}
- Budget used: {'training_trials_used': 3, 'training_trials_max': 3, 'compute_hours_used': 0.1, 'compute_hours_max': 2}

## Outcome

Best trial: **t03-train**. Validation: exact_match=0.767, token_f1=0.767 (targets NOT met).
Held-out test: exact_match=0.817, token_f1=0.817 (targets NOT met)

Adapter: `runs/date-normalization-strict/trials/t03-train/adapter`

```json
{
  "learning_rate": 0.0001,
  "num_epochs": 1,
  "lora_r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "target_modules": "all-linear",
  "batch_size": 16,
  "grad_accum": 1,
  "max_seq_len": 256,
  "warmup_ratio": 0.03,
  "weight_decay": 0.0,
  "lr_scheduler": "cosine",
  "neftune_noise_alpha": 0.0
}
```

## Agent summary

Heuristic search finished after 3 training trials.

## All trials

| trial | status | validation metrics | lr | epochs | r | train loss (first→last) | eval loss/epoch | time (s) | notes |
|---|---|---|---|---|---|---|---|---|---|
| t00-baseline | ok | exact_match=0.000, token_f1=0.044 | - | - | - | - | - | 9.5 | base model, no fine-tuning |
| t01-train | ok | exact_match=0.517, token_f1=0.517 | 5e-05 | 1 | 16 | 0.592→0.2233 | [0.2248] | 123.1 | initial config |
| t02-train | ok | exact_match=0.633, token_f1=0.633 | 0.0001 | 1 | 16 | 0.5181→0.1059 | [0.106] | 113.1 | underfitting: higher learning rate |
| t03-train | ok | exact_match=0.767, token_f1=0.767 | 0.0001 | 1 | 32 | 0.411→0.0546 | [0.0539] | 114.7 | more capacity: double LoRA rank |
