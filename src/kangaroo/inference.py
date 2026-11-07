"""Quick generation with an adapter, and merged-weights export."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

import torch

from .model import load_base_model, load_tokenizer, resolve_precision

Emit = Callable[..., None]


def _adapter_base(adapter: str) -> str:
    cfg = Path(adapter) / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"{adapter}: no adapter_config.json (is this a Kangaroo adapter folder?)")
    return json.loads(cfg.read_text(encoding="utf-8"))["base_model_name_or_path"]


def _load(adapter: Optional[str], base: Optional[str], precision: str = "auto", merge: bool = False):
    base_name = base or (_adapter_base(adapter) if adapter else None)
    if not base_name:
        raise ValueError("Provide an adapter folder or a base model.")
    dtype = resolve_precision(precision)
    tok_src = adapter if adapter and (Path(adapter) / "tokenizer_config.json").exists() else base_name
    tok = load_tokenizer(tok_src)
    model = load_base_model(base_name, dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.config.use_cache = True
    model.eval()
    return model, tok


def build_prompt(tok, system: str, user: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    sys_part = f"### System:\n{system}\n\n" if system else ""
    bos = tok.bos_token or ""
    return f"{bos}{sys_part}### Instruction:\n{user}\n\n### Response:\n"


def generate(cfg: dict, emit: Emit) -> dict:
    from transformers import TextIteratorStreamer

    adapter = cfg.get("adapter") or None
    model, tok = _load(adapter, cfg.get("base") or None, cfg.get("precision", "auto"))
    prompt = build_prompt(tok, cfg.get("system", ""), cfg["prompt"])
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    temperature = float(cfg.get("temperature", 0.7))
    kwargs = dict(
        **inputs, streamer=streamer, max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        do_sample=temperature > 0, pad_token_id=tok.pad_token_id,
    )
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=0.95)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with torch.no_grad():
                model.generate(**kwargs)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            streamer.end()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    pieces = []
    for text in streamer:
        pieces.append(text)
        emit("token", text=text)
    thread.join()
    if errors:
        raise errors[0]
    result = {"text": "".join(pieces)}
    emit("done", **result)
    return result


def export_merged(cfg: dict, emit: Emit) -> dict:
    adapter = cfg["adapter"]
    out = Path(cfg["output_dir"])
    precision = cfg.get("precision", "bf16")
    emit("log", message="Loading base model and adapter ...")
    model, tok = _load(adapter, cfg.get("base") or None, precision, merge=True)
    emit("log", message="Merging LoRA weights and saving ...")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    result = {"output_dir": str(out.resolve()), "size_bytes": size}
    emit("done", **result)
    return result
