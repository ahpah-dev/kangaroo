from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

DATASET_FORMATS = ("auto", "alpaca", "chat", "prompt_completion", "text")
SCHEDULERS = ("cosine", "linear", "constant")
CHECKPOINTING = ("auto", "on", "off")
PRECISIONS = ("auto", "bf16", "fp16", "fp32")


@dataclass
class TrainConfig:
    # --- model & data ---------------------------------------------------
    model: str = ""
    dataset: str = ""
    output_dir: str = "kangaroo-out"
    dataset_format: str = "auto"
    text_field: str = "text"
    dataset_split: str = "train"
    max_seq_len: int = 2048
    train_on_inputs: bool = False   # False = loss on completions only
    val_fraction: float = 0.0
    # --- LoRA -------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    rslora: bool = True
    target_modules: str = "all-linear"   # or comma separated names
    load_in_4bit: bool = False           # QLoRA (CUDA + bitsandbytes)
    # --- optimisation -----------------------------------------------------
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    batch_tokens: int = 4096        # padded-token budget per micro-batch
    grad_accum: int = 4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 3407
    # --- speed / memory ---------------------------------------------------
    gradient_checkpointing: str = "auto"
    loss_chunk: int = 1024          # tokens per loss chunk; 0 = dense logits
    precision: str = "auto"
    # --- bookkeeping --------------------------------------------------------
    logging_steps: int = 1
    eval_every: int = 0             # 0 = only at the end (when val_fraction > 0)
    save_every: int = 0
    stop_file: Optional[str] = None
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.model:
            raise ValueError("`model` is required (Hugging Face id or local path).")
        if not self.dataset:
            raise ValueError("`dataset` is required (local .jsonl/.json/.csv or Hub id).")
        _choice("dataset_format", self.dataset_format, DATASET_FORMATS)
        _choice("lr_scheduler", self.lr_scheduler, SCHEDULERS)
        _choice("gradient_checkpointing", self.gradient_checkpointing, CHECKPOINTING)
        _choice("precision", self.precision, PRECISIONS)
        _positive("max_seq_len", self.max_seq_len)
        _positive("lora_r", self.lora_r)
        _positive("lora_alpha", self.lora_alpha)
        _positive("batch_tokens", self.batch_tokens)
        _positive("grad_accum", self.grad_accum)
        _positive("learning_rate", self.learning_rate)
        _positive("logging_steps", self.logging_steps)
        if self.epochs <= 0 and self.max_steps <= 0:
            raise ValueError("Set `epochs` > 0 or `max_steps` > 0.")
        if not 0 <= self.val_fraction < 0.5:
            raise ValueError("`val_fraction` must be in [0, 0.5).")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("`warmup_ratio` must be in [0, 1).")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("`lora_dropout` must be in [0, 1).")
        if self.loss_chunk < 0:
            raise ValueError("`loss_chunk` must be >= 0.")


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"`{name}` must be one of {', '.join(allowed)} (got {value!r}).")


def _positive(name: str, value: float) -> None:
    if not value > 0:
        raise ValueError(f"`{name}` must be > 0 (got {value!r}).")
