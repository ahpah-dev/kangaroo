"""Memory-frugal cross entropy.

A normal causal-LM loss materialises `[batch * seq, vocab]` fp32 logits. With
modern 128k-256k vocabularies that tensor is often the single largest
allocation of a LoRA step. Two things avoid it:

1. Only positions that actually carry a label (the completion tokens) are
   pushed through the LM head at all.
2. The projection + cross entropy runs in chunks under activation
   checkpointing, so at most one chunk of logits is alive at a time, in the
   forward *and* the backward pass.

The result is mathematically identical to the dense loss.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ce_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    softcap: Optional[float],
) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits, labels, reduction="sum")


def linear_cross_entropy_sum(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk: int = 1024,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Sum of token cross entropies for `hidden @ weight.T` against `labels`.

    hidden: [N, H] (only positions with a real label), labels: [N].
    chunk <= 0 computes the dense logits in one go (reference behaviour).
    """
    n = hidden.shape[0]
    if n == 0:
        return hidden.sum().float() * 0.0
    if chunk <= 0:
        return _ce_chunk(hidden, weight, labels, softcap)

    needs_grad = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, n, chunk):
        h = hidden[start : start + chunk]
        y = labels[start : start + chunk]
        if needs_grad:
            part = checkpoint(_ce_chunk, h, weight, y, softcap, use_reentrant=False)
        else:
            part = _ce_chunk(h, weight, y, softcap)
        total = total + part
    return total
