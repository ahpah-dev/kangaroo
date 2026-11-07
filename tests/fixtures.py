"""Offline fixtures: a tiny random Llama + word-level tokenizer and a toy dataset.

Used by the pytest suite and by `npm run smoke:train` (python tests/fixtures.py <dir>).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

WORDS = "red green blue cat dog bird one two three four five yes no maybe up down left right".split()


def build_tiny_model(d: Path) -> str:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    specials = ["<pad>", "<s>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]
    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator([" ".join(WORDS)] * 4, trainers.WordLevelTrainer(special_tokens=specials))
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )
    tok.chat_template = (
        "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}"
        "{% if m['role'] == 'assistant' %} </s>{% endif %} {% endfor %}"
        "{% if add_generation_prompt %}<|assistant|> {% endif %}"
    )
    torch.manual_seed(0)
    # A larger-than-default init keeps frozen random embeddings distinguishable, so
    # LoRA alone can learn the toy task (stock HF+PEFT plateaus otherwise).
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        tie_word_embeddings=False, initializer_range=0.3,
    )
    LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


def write_toy_dataset(path: Path, n: int = 96) -> str:
    """Deterministic task: the answer repeats the question's colour twice, then the number."""
    rng = random.Random(1)
    colours = ["red", "green", "blue"]
    with Path(path).open("w", encoding="utf-8") as fh:
        for _ in range(n):
            c, num = rng.choice(colours), rng.choice(["one", "two", "three"])
            fh.write(json.dumps({"instruction": f"say {c} {num}", "output": f"{c} {c} {num}"}) + "\n")
    return str(path)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    print(json.dumps({"model": build_tiny_model(out / "tiny"), "dataset": write_toy_dataset(out / "toy.jsonl")}))
