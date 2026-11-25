# ModelTuner report: date-normalization

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct` (LoRA)
- Search policy: heuristic
- Splits: {'train': 400, 'val': 60, 'test': 60, 'avg_prompt_chars': 118, 'avg_reference_chars': 10}
- Targets (validation): {'exact_match': 0.85}
- Budget used: {'training_trials_used': 1, 'training_trials_max': 5, 'compute_hours_used': 0.064, 'compute_hours_max': 2}

## Outcome

Best trial: **t01-train**. Validation: exact_match=0.967, token_f1=0.967 (targets met).
Held-out test: exact_match=0.933, token_f1=0.933 (targets met)

Adapter: `runs/date-normalization/trials/t01-train/adapter`

```json
{
  "learning_rate": 0.0003,
  "num_epochs": 2,
  "lora_r": 16,
  "lora_alpha": 32,
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

Heuristic search finished after 1 training trials.

## All trials

| trial | status | validation metrics | lr | epochs | r | train loss (first→last) | eval loss/epoch | time (s) | notes |
|---|---|---|---|---|---|---|---|---|---|
| t00-baseline | ok | exact_match=0.000, token_f1=0.044 | - | - | - | - | - | 9.3 | base model, no fine-tuning |
| t01-train | ok | exact_match=0.967, token_f1=0.967 | 0.0003 | 2 | 16 | 0.4588→0.0078 | [0.0164, 0.0136] | 219.3 | initial config |
