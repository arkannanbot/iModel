"""One trial, run in a fresh subprocess so GPU memory is fully released between trials.

    python -m modeltuner.worker spec.json

Modes:
  train_eval  LoRA/QLoRA SFT on train set, then generate predictions on the eval set
  eval        generate predictions with the base model (or a given adapter) only

Writes <out_dir>/result.json and <out_dir>/predictions.jsonl. Metric scoring happens
in the parent process (see evaluate.py), so this file only needs torch/transformers/trl/peft.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(name: str, device: str):
    if name == "auto":
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_model(spec: dict, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["base_model"], trust_remote_code=spec.get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"trust_remote_code": spec.get("trust_remote_code", False)}
    dtype = _dtype(spec.get("dtype", "auto"), device)
    if spec.get("quantization") == "4bit":
        if device != "cuda":
            raise RuntimeError("4-bit QLoRA needs a CUDA GPU; set model.quantization: none")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(spec["base_model"], **kwargs)
    if spec.get("quantization") != "4bit":
        model.to(device)
    return model, tok, dtype


def _train(spec: dict, model, tok, dtype, device: str, out: Path) -> dict:
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    hp = spec["hparams"]

    def to_pc(rows):  # conversational prompt-completion => loss only on the answer tokens
        return Dataset.from_list(
            [{"prompt": r["messages"][:-1], "completion": [r["messages"][-1]]} for r in rows]
        )

    train_ds, eval_ds = to_pc(_read(spec["train_path"])), to_pc(_read(spec["eval_path"]))
    if spec.get("quantization") == "4bit":
        model = prepare_model_for_kbit_training(model)
    tm = hp["target_modules"]
    lora = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        target_modules=tm if tm == "all-linear" else [s.strip() for s in tm.split(",") if s.strip()],
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_epochs"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        warmup_steps=int(hp["warmup_ratio"] * math.ceil(
            math.ceil(len(train_ds) / (hp["batch_size"] * hp["grad_accum"])) * hp["num_epochs"])),
        weight_decay=hp["weight_decay"],
        lr_scheduler_type=hp["lr_scheduler"],
        neftune_noise_alpha=hp["neftune_noise_alpha"] or None,
        max_length=hp["max_seq_len"],
        completion_only_loss=True,
        bf16=(dtype == torch.bfloat16 and device == "cuda"),
        fp16=(dtype == torch.float16 and device == "cuda"),
        gradient_checkpointing=(device == "cuda"),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=spec.get("seed", 13),
        use_cpu=(device == "cpu"),
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok, peft_config=lora,
    )
    t0 = time.time()
    trainer.train()
    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    hist = trainer.state.log_history
    train_losses = [h["loss"] for h in hist if "loss" in h]
    eval_losses = [round(h["eval_loss"], 4) for h in hist if "eval_loss" in h]
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    return {
        "model": trainer.model,
        "train_seconds": round(time.time() - t0, 1),
        "steps": trainer.state.global_step,
        "train_loss_first": round(train_losses[0], 4) if train_losses else None,
        "train_loss_last": round(train_losses[-1], 4) if train_losses else None,
        "eval_loss_per_epoch": eval_losses,
        "trainable_params": n_trainable,
        "adapter_dir": str(adapter_dir),
    }


@torch.no_grad()
def _generate(model, tok, rows: list[dict], max_new_tokens: int, batch_size: int, device: str) -> list[str]:
    model.eval()
    tok.padding_side = "left"
    preds: list[str] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
        new = out[:, enc["input_ids"].shape[1] :]
        preds.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
    return preds


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = _device()
    torch.manual_seed(spec.get("seed", 13))
    model, tok, dtype = _load_model(spec, device)
    result: dict = {"device": device, "mode": spec["mode"]}

    if spec["mode"] == "train_eval":
        info = _train(spec, model, tok, dtype, device, out)
        model = info.pop("model")
        result.update(info)
    elif spec.get("adapter_path"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec["adapter_path"])

    rows = _read(spec["eval_path"])
    t0 = time.time()
    preds = _generate(model, tok, rows, spec["max_new_tokens"], spec["eval_batch_size"], device)
    result["generate_seconds"] = round(time.time() - t0, 1)
    with open(out / "predictions.jsonl", "w") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "prediction": p}) + "\n")
    for k, v in list(result.items()):
        if isinstance(v, float) and math.isnan(v):
            result[k] = None
    (out / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
