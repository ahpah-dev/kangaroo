"""A deliberately small training loop (no Trainer / Accelerate overhead)."""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable

import torch

from . import __version__
from .config import TrainConfig
from .data import (
    IGNORE, build_examples, collate, count_targets, make_batches, read_records, split_validation,
)
from .model import (
    attach_lora, causal_lm, final_logit_softcap, hidden_states, load_base_model,
    load_tokenizer, resolve_precision, set_checkpointing,
)
from .loss import linear_cross_entropy_sum

Emit = Callable[..., None]


class TrainingStopped(Exception):
    pass


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(total * cfg.warmup_ratio)) if cfg.warmup_ratio > 0 else 0
    if warmup and step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    if cfg.lr_scheduler == "constant":
        return cfg.learning_rate
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    if cfg.lr_scheduler == "linear":
        return cfg.learning_rate * (1 - progress)
    return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


class Runner:
    """Holds the model plus the pieces needed to compute a loss on a batch."""

    def __init__(self, peft_model, cfg: TrainConfig, device: torch.device, dtype: torch.dtype):
        self.model = peft_model
        self.lm = causal_lm(peft_model)
        self.cfg = cfg
        self.device = device
        self.softcap = final_logit_softcap(self.lm)
        self.use_amp = dtype in (torch.bfloat16, torch.float16) and (
            device.type == "cuda" or dtype is torch.bfloat16
        )
        self.dtype = dtype

    def autocast(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.dtype)

    def loss_sum(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        ids = batch["input_ids"].to(self.device, non_blocking=True)
        att = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with self.autocast():
            hidden = hidden_states(self.lm, ids, att)
            shift = labels[:, 1:]
            keep = shift != IGNORE
            h = hidden[:, :-1][keep]
            y = shift[keep]
            weight = self.lm.get_output_embeddings().weight
            total = linear_cross_entropy_sum(h, weight, y, self.cfg.loss_chunk, self.softcap)
        return total, int(y.numel())


def _largest_batch(batches: list[list[int]], examples: list[dict]) -> list[int]:
    def key(b: list[int]) -> tuple[int, int]:
        width = max(len(examples[i]["input_ids"]) for i in b)
        return width * len(b), width

    return max(batches, key=key)


def _probe_memory(runner: Runner, examples: list[dict], batch: list[int], pad_id: int, emit: Emit) -> bool:
    """Try one forward/backward on the biggest batch with checkpointing off.

    Returns True if it fits (checkpointing can stay off)."""
    data = collate([examples[i] for i in batch], pad_id)
    try:
        loss, n = runner.loss_sum(data)
        if n:
            loss.backward()
        ok = True
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        ok = False
    finally:
        runner.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return ok


@torch.no_grad()
def evaluate(runner: Runner, examples: list[dict], cfg: TrainConfig, pad_id: int) -> float:
    runner.model.eval()
    lengths = [len(e["input_ids"]) for e in examples]
    total, count = 0.0, 0
    for b in make_batches(lengths, cfg.batch_tokens, shuffle=False):
        loss, n = runner.loss_sum(collate([examples[i] for i in b], pad_id))
        total += float(loss)
        count += n
    runner.model.train()
    return total / max(1, count)


def save_adapter(model, tok, directory: Path, cfg: TrainConfig, extra: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    tok.save_pretrained(str(directory))
    (directory / "kangaroo_run.json").write_text(
        json.dumps({"kangaroo": __version__, "config": cfg.to_dict(), **extra}, indent=2), encoding="utf-8"
    )


def train(cfg: TrainConfig, emit: Emit) -> dict:
    t_start = time.perf_counter()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = resolve_precision(cfg.precision)
    emit("log", message=f"Device: {torch.cuda.get_device_name(0) if cuda else 'CPU'} | precision: {str(dtype).split('.')[-1]}")

    emit("phase", name="data")
    tok = load_tokenizer(cfg.model, cfg.trust_remote_code)
    records = read_records(cfg.dataset, cfg.dataset_split)
    examples, stats = build_examples(
        tok, records, cfg.dataset_format, cfg.text_field, cfg.max_seq_len, cfg.train_on_inputs
    )
    for w in stats.warnings:
        emit("log", message=f"Warning: {w}")
    train_ex, val_ex = split_validation(examples, cfg.val_fraction, cfg.seed)
    emit(
        "data", format=stats.fmt, records=stats.records, examples=len(train_ex), val_examples=len(val_ex),
        tokens=stats.tokens, target_tokens=stats.target_tokens,
    )

    emit("phase", name="model")
    emit("log", message=f"Loading {cfg.model} ...")
    base = load_base_model(cfg.model, dtype, cfg.load_in_4bit, cfg.trust_remote_code)
    model = attach_lora(base, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.rslora, cfg.target_modules)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    emit("model", trainable_params=n_train, total_params=n_all, softcap=final_logit_softcap(causal_lm(model)))
    emit("log", message=f"Trainable parameters: {n_train:,} of {n_all:,} ({100 * n_train / n_all:.2f}%)")

    runner = Runner(model, cfg, device, dtype)
    pad_id = tok.pad_token_id
    lengths = [len(e["input_ids"]) for e in train_ex]
    longest = max(lengths)
    batch_tokens = max(cfg.batch_tokens, longest)
    if batch_tokens != cfg.batch_tokens:
        emit("log", message=f"batch_tokens raised to {batch_tokens} to fit the longest example.")
    targets = [count_targets(e) for e in train_ex]

    first_epoch = make_batches(lengths, batch_tokens, cfg.seed, 0)
    groups_per_epoch = math.ceil(len(first_epoch) / cfg.grad_accum)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else max(1, math.ceil(cfg.epochs * groups_per_epoch))
    padded = sum(max(lengths[i] for i in b) * len(b) for b in first_epoch)
    emit(
        "plan", total_steps=total_steps, batches_per_epoch=len(first_epoch), groups_per_epoch=groups_per_epoch,
        padding_efficiency=sum(lengths) / max(1, padded),
    )

    # --- gradient checkpointing policy -----------------------------------
    mode = cfg.gradient_checkpointing
    if mode == "auto":
        if cuda:
            set_checkpointing(base, False)
            model.train()
            fits = _probe_memory(runner, train_ex, _largest_batch(first_epoch, train_ex), pad_id, emit)
            mode = "off" if fits else "on"
            emit("log", message=f"Auto memory probe: activation checkpointing {mode}.")
        else:
            mode = "off"
    set_checkpointing(base, mode == "on")
    model.train()
    emit("checkpointing", mode=mode)

    opt = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), fused=cuda or None
    )
    scaler = torch.amp.GradScaler("cuda") if (cuda and dtype is torch.float16) else None

    def groups():
        epoch = 0
        while True:
            batches = first_epoch if epoch == 0 else make_batches(lengths, batch_tokens, cfg.seed, epoch)
            for g in range(0, len(batches), cfg.grad_accum):
                yield epoch, batches[g : g + cfg.grad_accum]
            epoch += 1

    out_dir = Path(cfg.output_dir)
    emit("phase", name="train")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    history: list[dict] = []
    real_tokens_total = 0
    train_time = 0.0
    step_times: list[float] = []
    stopped = False
    val_loss = None
    gen = groups()

    for step in range(total_steps):
        if cfg.stop_file and os.path.exists(cfg.stop_file):
            stopped = True
            break
        epoch, group = next(gen)
        lr = lr_at(step, total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        n_target = max(1, sum(targets[i] for b in group for i in b))
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        step_loss = torch.zeros((), device=device)
        real_tokens = 0
        for b in group:
            batch = collate([train_ex[i] for i in b], pad_id)
            real_tokens += int(batch["attention_mask"].sum())
            total, _ = runner.loss_sum(batch)
            scaled = total / n_target
            (scaler.scale(scaled) if scaler else scaled).backward()
            step_loss += total.detach()

        if scaler:
            scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

        loss_val = float(step_loss) / n_target
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if not math.isfinite(loss_val):
            raise FloatingPointError(
                "Loss became NaN/inf. Try a lower learning rate, precision=bf16/fp32, or check the dataset."
            )

        real_tokens_total += real_tokens
        train_time += dt
        step_times.append(dt)
        done = step + 1
        if done % cfg.logging_steps == 0 or done == total_steps:
            recent = step_times[-20:]
            eta = (total_steps - done) * (sum(recent) / len(recent))
            row = dict(
                step=done, total_steps=total_steps, epoch=epoch, loss=loss_val, lr=lr, grad_norm=grad_norm,
                tok_s=real_tokens / dt, eta_s=eta,
                mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if cuda else None,
            )
            history.append(row)
            emit("progress", **row)

        if val_ex and cfg.eval_every and done % cfg.eval_every == 0 and done != total_steps:
            val_loss = evaluate(runner, val_ex, cfg, pad_id)
            emit("eval", step=done, val_loss=val_loss)
        if cfg.save_every and done % cfg.save_every == 0 and done != total_steps:
            save_adapter(model, tok, out_dir / f"checkpoint-{done}", cfg, {"step": done})
            emit("log", message=f"Saved checkpoint-{done}")

    if val_ex:
        val_loss = evaluate(runner, val_ex, cfg, pad_id)
        emit("eval", step=len(history) and history[-1]["step"], val_loss=val_loss)

    adapter_dir = out_dir / "adapter"
    summary = dict(
        steps=history[-1]["step"] if history else 0,
        total_steps=total_steps,
        final_loss=history[-1]["loss"] if history else None,
        val_loss=val_loss,
        avg_tok_s=real_tokens_total / train_time if train_time else 0.0,
        peak_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if cuda else None,
        train_seconds=train_time,
        total_seconds=time.perf_counter() - t_start,
        checkpointing=mode,
        stopped=stopped,
    )
    save_adapter(model, tok, adapter_dir, cfg, {"summary": summary})
    emit("done", adapter_dir=str(adapter_dir.resolve()), **summary)
    return summary
