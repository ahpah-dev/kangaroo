import json

import pytest

pytest.importorskip("torch")

from kangaroo.data import (  # noqa: E402
    IGNORE, build_examples, collate, detect_format, make_batches, read_records, split_validation,
)


def test_detect_formats():
    assert detect_format({"messages": []}) == "chat"
    assert detect_format({"conversations": []}) == "chat"
    assert detect_format({"instruction": "a", "output": "b"}) == "alpaca"
    assert detect_format({"prompt": "a", "completion": "b"}) == "prompt_completion"
    assert detect_format({"text": "a"}) == "text"
    with pytest.raises(ValueError):
        detect_format({"foo": 1})


def test_alpaca_masks_prompt_and_trains_on_completion(tokenizer):
    ex, stats = build_examples(tokenizer, [{"instruction": "say red", "output": "red red"}], "auto")
    ids, labels = ex[0]["input_ids"], ex[0]["labels"]
    assert stats.fmt == "alpaca"
    supervised = [i for i, x in enumerate(labels) if x != IGNORE]
    first = supervised[0]
    assert all(x == IGNORE for x in labels[:first])  # prompt masked
    assert all(labels[i] == ids[i] for i in supervised)
    decoded = tokenizer.decode(ids[first:])
    assert "red red" in decoded and decoded.strip().endswith("</s>")  # turn is closed with eos


def test_train_on_inputs_unmasks_everything(tokenizer):
    ex, _ = build_examples(tokenizer, [{"instruction": "say red", "output": "red"}], "alpaca", train_on_inputs=True)
    assert ex[0]["labels"] == ex[0]["input_ids"]


def test_chat_trains_only_on_last_assistant_turn(tokenizer):
    rec = {"messages": [
        {"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"}, {"role": "assistant", "content": "four"},
    ]}
    ex, _ = build_examples(tokenizer, [rec], "chat")
    sup = [t for t in ex[0]["labels"] if t != IGNORE]
    text = tokenizer.decode(sup)
    assert "four" in text and "two" not in text


def test_sharegpt_roles(tokenizer):
    rec = {"conversations": [{"from": "human", "value": "one"}, {"from": "gpt", "value": "two"}]}
    ex, stats = build_examples(tokenizer, [rec])
    assert stats.fmt == "chat" and stats.kept == 1


def test_text_format_adds_bos_eos(tokenizer):
    ex, _ = build_examples(tokenizer, [{"text": "red green"}], "text")
    ids = ex[0]["input_ids"]
    assert ids[0] == tokenizer.bos_token_id and ids[-1] == tokenizer.eos_token_id
    assert ex[0]["labels"][0] == IGNORE


def test_truncation_and_drop(tokenizer):
    long_out = " ".join(["red"] * 50)
    ex, stats = build_examples(tokenizer, [{"instruction": "say", "output": long_out}], "alpaca", max_len=20)
    assert len(ex[0]["input_ids"]) == 20 and stats.truncated == 1
    with pytest.raises(ValueError):  # prompt alone exceeds the window -> nothing usable
        build_examples(tokenizer, [{"instruction": " ".join(["red"] * 40), "output": "x"}], "alpaca", max_len=10)


def test_batches_respect_token_budget_and_cover_everything():
    lengths = [5, 50, 7, 33, 12, 90, 8, 8, 41, 3] * 20
    batches = make_batches(lengths, batch_tokens=120, seed=1)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(lengths)))
    for b in batches:
        assert len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 120


def test_batches_are_deterministic_and_epoch_dependent():
    lengths = list(range(1, 200))
    assert make_batches(lengths, 300, 1, 0) == make_batches(lengths, 300, 1, 0)
    assert make_batches(lengths, 300, 1, 0) != make_batches(lengths, 300, 1, 1)


def test_bucketing_reduces_padding():
    lengths = [4, 100] * 100
    batches = make_batches(lengths, 400, seed=0, window=200)
    padded = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    assert sum(lengths) / padded > 0.9


def test_collate_pads_right():
    out = collate([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4], "labels": [4]}], pad_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]


def test_read_records_formats(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"x":1}\n\n{"x":2}\n', encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"data": [{"x": 3}]}), encoding="utf-8")
    (tmp_path / "c.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    assert len(read_records(str(tmp_path / "a.jsonl"))) == 2
    assert read_records(str(tmp_path / "b.json")) == [{"x": 3}]
    assert read_records(str(tmp_path / "c.csv")) == [{"x": "1", "y": "2"}]


def test_split_validation():
    train, val = split_validation(list(range(100)), 0.1, seed=0)
    assert len(val) == 10 and len(train) == 90 and not set(train) & set(val)
    assert split_validation(list(range(5)), 0.5, 0) == (list(range(5)), [])
