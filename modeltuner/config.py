"""Run configuration, loaded from YAML.

Everything the agent is allowed to change lives in `Hyperparams`. Everything it is
NOT allowed to change (the dataset, the metrics, the targets, the budget) lives in
the rest of the config, so the agent cannot "pass" by moving the goalposts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Hyperparams:
    learning_rate: float = 2e-4
    num_epochs: float = 3.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"  # "all-linear" or comma list, e.g. "q_proj,v_proj"
    batch_size: int = 8
    grad_accum: int = 1
    max_seq_len: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    neftune_noise_alpha: float = 0.0  # 0 disables NEFTune

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Hyperparams":
        d = dict(d or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown hyperparameters: {sorted(unknown)}")
        hp = cls(**d)
        hp.validate()
        return hp

    def validate(self) -> None:
        if not (1e-7 <= self.learning_rate <= 1e-2):
            raise ValueError("learning_rate must be in [1e-7, 1e-2]")
        if not (0.1 <= self.num_epochs <= 50):
            raise ValueError("num_epochs must be in [0.1, 50]")
        if self.lora_r not in (2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError("lora_r must be a power of two in [2, 256]")
        if not (1 <= self.lora_alpha <= 1024):
            raise ValueError("lora_alpha must be in [1, 1024]")
        if not (0.0 <= self.lora_dropout <= 0.5):
            raise ValueError("lora_dropout must be in [0, 0.5]")
        if not (1 <= self.batch_size <= 256) or not (1 <= self.grad_accum <= 64):
            raise ValueError("batch_size must be in [1, 256] and grad_accum in [1, 64]")
        if not (32 <= self.max_seq_len <= 32768):
            raise ValueError("max_seq_len must be in [32, 32768]")
        if self.lr_scheduler not in ("cosine", "linear", "constant", "constant_with_warmup"):
            raise ValueError("lr_scheduler must be cosine, linear, constant or constant_with_warmup")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TaskSpec:
    name: str
    description: str
    system_prompt: str | None = None


@dataclass
class ModelSpec:
    base: str
    quantization: str = "none"  # "none" (LoRA) or "4bit" (QLoRA; needs CUDA + bitsandbytes)
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    trust_remote_code: bool = False


@dataclass
class DataSpec:
    train: str
    eval: str | None = None  # validation set the agent tunes against
    test: str | None = None  # held-out set, only scored once at the end
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 13


@dataclass
class JudgeSpec:
    enabled: bool = False
    model: str = "claude-opus-5-5"
    rubric: str = "Score how well the prediction matches the reference answer in substance and format."


@dataclass
class EvalSpec:
    metrics: list[str] = field(default_factory=lambda: ["exact_match", "token_f1"])
    targets: dict[str, float] = field(default_factory=lambda: {"exact_match": 0.9})
    max_new_tokens: int = 128
    batch_size: int = 16
    max_examples: int | None = None  # cap validation size for speed
    judge: JudgeSpec = field(default_factory=JudgeSpec)


@dataclass
class SearchSpec:
    policy: str = "claude"  # "claude" or "heuristic"
    agent_model: str = "claude-opus-5-5"
    agent_effort: str = "high"
    max_trials: int = 8
    max_hours: float = 6.0
    stop_on_target: bool = True
    initial: Hyperparams = field(default_factory=Hyperparams)


@dataclass
class RunConfig:
    task: TaskSpec
    model: ModelSpec
    data: DataSpec
    evals: EvalSpec
    search: SearchSpec
    output_dir: str
    base_dir: Path = Path(".")

    def resolve(self, p: str | None) -> Path | None:
        if p is None:
            return None
        path = Path(p)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def out(self) -> Path:
        return self.resolve(self.output_dir)  # type: ignore[return-value]


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    ev = dict(raw.get("evals", {}))
    judge = JudgeSpec(**ev.pop("judge", {}) or {})
    srch = dict(raw.get("search", {}))
    initial = Hyperparams.from_dict(srch.pop("initial", None))
    cfg = RunConfig(
        task=TaskSpec(**raw["task"]),
        model=ModelSpec(**raw["model"]),
        data=DataSpec(**raw["data"]),
        evals=EvalSpec(judge=judge, **ev),
        search=SearchSpec(initial=initial, **srch),
        output_dir=raw.get("output_dir", "runs/default"),
        # Relative paths in the YAML resolve against the project root (parent of configs/)
        base_dir=path.parent.parent if path.parent.name == "configs" else path.parent,
    )
    from .metrics import METRICS  # local import to avoid cycles

    for m in list(cfg.evals.metrics) + list(cfg.evals.targets):
        if m not in METRICS and m != "llm_judge":
            raise ValueError(f"Unknown metric {m!r}. Known: {sorted(METRICS) + ['llm_judge']}")
    if "llm_judge" in cfg.evals.targets and not cfg.evals.judge.enabled:
        raise ValueError("A target on llm_judge needs evals.judge.enabled: true")
    return cfg
