"""Dataset loading, tokenisation, loss masking and padding-free-ish batching."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch

IGNORE = -100
FALLBACK_PROMPT = "### Instruction:\n{prompt}\n\n### Response:\n"
ROLE_MAP = {
    "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
    "bot": "assistant", "model": "assistant", "system": "system",
}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def read_records(source: str, split: str = "train") -> list[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("data", "train", "examples", "rows"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise ValueError(f"{source}: expected a JSON list of records.")
            return data
        if suffix in (".csv", ".tsv"):
            with path.open(encoding="utf-8", newline="") as fh:
                return list(csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ","))
        raise ValueError(f"{source}: unsupported file type {suffix!r} (use .jsonl, .json, .csv, .tsv).")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"{source!r} is not a local file. To load Hub datasets install `datasets` "
            "(pip install kangaroo-ft[hub])."
        ) from exc
    return [dict(r) for r in load_dataset(source, split=split)]


def detect_format(rec: dict[str, Any], text_field: str = "text") -> str:
    if "messages" in rec or "conversations" in rec:
        return "chat"
    if "instruction" in rec:
        return "alpaca"
    if "prompt" in rec and ("completion" in rec or "response" in rec):
        return "prompt_completion"
    if text_field in rec:
        return "text"
    raise ValueError(
        "Could not detect dataset format from keys "
        f"{sorted(rec)[:8]}. Expected messages/conversations, instruction/output, "
        f"prompt/completion or a `{text_field}` column."
    )


# --------------------------------------------------------------------------
# rendering one record into (prompt_text, completion_text)
# --------------------------------------------------------------------------
def _normalize_messages(rec: dict[str, Any]) -> list[dict[str, str]]:
    raw = rec.get("messages") or rec.get("conversations") or []
    out = []
    for m in raw:
        role = m.get("role", m.get("from"))
        content = m.get("content", m.get("value"))
        role = ROLE_MAP.get(str(role).lower())
        if role is None or content is None:
            raise ValueError(f"Bad chat message: {m!r}")
        out.append({"role": role, "content": str(content)})
    return out


def _alpaca_messages(rec: dict[str, Any]) -> list[dict[str, str]]:
    instr = str(rec["instruction"])
    extra = rec.get("input") or ""
    user = f"{instr}\n\n{extra}" if str(extra).strip() else instr
    answer = rec.get("output", rec.get("response"))
    if answer is None:
        raise ValueError("alpaca record without `output`/`response`.")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": str(answer)}]


def _split_chat(tok, msgs: list[dict[str, str]]) -> tuple[str, str, bool]:
    """Return (prompt_text, completion_text, needs_bos) training only on the last assistant turn."""
    last = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=None)
    if last is None:
        raise ValueError("chat record has no assistant message.")
    history, target = msgs[:last], msgs[last]["content"]
    if getattr(tok, "chat_template", None):
        full = tok.apply_chat_template(msgs[: last + 1], tokenize=False)
        prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        if not full.startswith(prompt):
            prompt = _common_prefix(full, prompt)
        return prompt, full[len(prompt):], False
    parts = []
    for m in history:
        label = {"system": "### System:", "user": "### Instruction:", "assistant": "### Response:"}[m["role"]]
        parts.append(f"{label}\n{m['content']}\n\n")
    parts.append("### Response:\n")
    return "".join(parts), target, True


def _common_prefix(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def render_record(tok, rec: dict[str, Any], fmt: str, text_field: str) -> tuple[str, str, bool, bool]:
    """-> (prompt_text, completion_text, needs_bos, append_eos)."""
    if fmt == "text":
        return "", str(rec[text_field]), True, True
    if fmt == "prompt_completion":
        completion = rec.get("completion", rec.get("response"))
        return str(rec["prompt"]), str(completion), True, True
    msgs = _normalize_messages(rec) if fmt == "chat" else _alpaca_messages(rec)
    prompt, completion, needs_bos = _split_chat(tok, msgs)
    # chat templates already close the turn; the fallback template does not.
    return prompt, completion, needs_bos, needs_bos


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------
@dataclass
class DataStats:
    records: int = 0
    kept: int = 0
    dropped: int = 0
    truncated: int = 0
    tokens: int = 0
    target_tokens: int = 0
    fmt: str = ""
    warnings: list[str] = field(default_factory=list)


def build_examples(
    tok,
    records: Sequence[dict[str, Any]],
    fmt: str = "auto",
    text_field: str = "text",
    max_len: int = 2048,
    train_on_inputs: bool = False,
) -> tuple[list[dict[str, list[int]]], DataStats]:
    if not records:
        raise ValueError("Dataset is empty.")
    if fmt == "auto":
        fmt = detect_format(records[0], text_field)
    stats = DataStats(records=len(records), fmt=fmt)
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    out: list[dict[str, list[int]]] = []

    for rec in records:
        prompt, completion, needs_bos, append_eos = render_record(tok, rec, fmt, text_field)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"] if prompt else []
        c_ids = tok(completion, add_special_tokens=False)["input_ids"] if completion else []
        if needs_bos and bos is not None:
            p_ids = [bos] + p_ids
        if append_eos and eos is not None:
            c_ids = c_ids + [eos]

        if len(p_ids) + len(c_ids) > max_len:
            stats.truncated += 1
            if len(p_ids) >= max_len:  # prompt alone fills the window: nothing to learn
                stats.dropped += 1
                continue
            c_ids = c_ids[: max_len - len(p_ids)]

        ids = p_ids + c_ids
        if fmt == "text":
            labels = list(ids)
            # BOS carries no useful target
            if needs_bos and bos is not None and labels:
                labels[0] = IGNORE
        elif train_on_inputs:
            labels = list(ids)
        else:
            labels = [IGNORE] * len(p_ids) + c_ids
        # position 0 can never be predicted; need at least one supervised target after it
        if sum(1 for x in labels[1:] if x != IGNORE) == 0:
            stats.dropped += 1
            continue
        out.append({"input_ids": ids, "labels": labels})
        stats.tokens += len(ids)
        stats.target_tokens += sum(1 for x in labels[1:] if x != IGNORE)

    stats.kept = len(out)
    if not out:
        raise ValueError("No usable examples after tokenisation (all empty or longer than max_seq_len).")
    if stats.dropped:
        stats.warnings.append(f"{stats.dropped} examples dropped (no supervised tokens).")
    if stats.truncated:
        stats.warnings.append(f"{stats.truncated} examples exceeded max_seq_len={max_len} and were truncated.")
    return out, stats


def count_targets(example: dict[str, list[int]]) -> int:
    return sum(1 for x in example["labels"][1:] if x != IGNORE)


# --------------------------------------------------------------------------
# batching: sort-within-window + padded-token budget (no cross-example
# attention leakage, minimal padding waste)
# --------------------------------------------------------------------------
def make_batches(
    lengths: Sequence[int],
    batch_tokens: int,
    seed: int = 0,
    epoch: int = 0,
    window: int = 512,
    shuffle: bool = True,
) -> list[list[int]]:
    rng = random.Random(seed * 1_000_003 + epoch)
    idx = list(range(len(lengths)))
    if shuffle:
        rng.shuffle(idx)
    batches: list[list[int]] = []
    for s in range(0, len(idx), window):
        chunk = sorted(idx[s : s + window], key=lambda i: lengths[i])
        cur: list[int] = []
        cur_max = 0
        for i in chunk:
            new_max = max(cur_max, lengths[i])
            if cur and new_max * (len(cur) + 1) > batch_tokens:
                batches.append(cur)
                cur, new_max = [], lengths[i]
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(examples: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(e["input_ids"]) for e in examples)
    n = len(examples)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    labels = torch.full((n, width), IGNORE, dtype=torch.long)
    for r, e in enumerate(examples):
        L = len(e["input_ids"])
        ids[r, :L] = torch.tensor(e["input_ids"])
        mask[r, :L] = 1
        labels[r, :L] = torch.tensor(e["labels"])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def split_validation(examples: list, fraction: float, seed: int) -> tuple[list, list]:
    if fraction <= 0 or len(examples) < 10:
        return examples, []
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_val = max(1, math.floor(len(examples) * fraction))
    val = [examples[i] for i in order[:n_val]]
    train = [examples[i] for i in order[n_val:]]
    return train, val
