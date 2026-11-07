"""Model / tokenizer loading and LoRA attachment."""

from __future__ import annotations

from typing import Any, Optional

import torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_precision(precision: str) -> torch.dtype:
    if precision != "auto":
        return DTYPES[precision]
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(name_or_path: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    return tok


def load_base_model(
    name_or_path: str,
    dtype: torch.dtype,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = dict(dtype=dtype, trust_remote_code=trust_remote_code)
    cuda = torch.cuda.is_available()
    if load_in_4bit:
        if not cuda:
            raise RuntimeError("4-bit (QLoRA) loading needs a CUDA GPU.")
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("4-bit loading needs bitsandbytes (pip install kangaroo-ft[quant]).") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if cuda:
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, attn_implementation="sdpa", **kwargs)
    except (ValueError, ImportError, NotImplementedError):
        # architecture without SDPA support: let transformers pick its default
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model.config.use_cache = False
    return model


def attach_lora(model, r: int, alpha: int, dropout: float, rslora: bool, target_modules: str):
    from peft import LoraConfig, get_peft_model

    targets: Any = "all-linear"
    if target_modules and target_modules != "all-linear":
        targets = [t.strip() for t in target_modules.split(",") if t.strip()]
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        use_rslora=rslora,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def set_checkpointing(model, enabled: bool) -> None:
    """Toggle activation checkpointing on the underlying HF model."""
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()


def causal_lm(peft_model):
    """The HF CausalLM inside a PeftModel (LoRA layers already injected)."""
    return peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model


def final_logit_softcap(lm) -> Optional[float]:
    cfg = lm.config
    for c in (cfg, getattr(cfg, "text_config", None)):
        cap = getattr(c, "final_logit_softcapping", None) if c is not None else None
        if cap:
            return float(cap)
    return None


def hidden_states(lm, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Final hidden states, *without* running the LM head."""
    out = lm.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.last_hidden_state
