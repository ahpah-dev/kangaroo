import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kangaroo.cli import main  # noqa: E402
from kangaroo.config import TrainConfig  # noqa: E402
from kangaroo.trainer import lr_at, train  # noqa: E402


def _cfg(tiny_model, toy_dataset, out, **kw):
    base = dict(
        model=tiny_model, dataset=toy_dataset, output_dir=str(out), max_seq_len=64, lora_r=8, lora_alpha=16,
        learning_rate=3e-3, batch_tokens=256, grad_accum=2, epochs=0, max_steps=40, warmup_ratio=0.05,
        precision="fp32", target_modules="all-linear", val_fraction=0.1,
    )
    base.update(kw)
    return TrainConfig.from_dict(base)


def test_training_reduces_loss_and_saves_adapter(tiny_model, toy_dataset, tmp_path):
    events = []
    summary = train(_cfg(tiny_model, toy_dataset, tmp_path / "out"), lambda e, **k: events.append((e, k)))
    prog = [k for e, k in events if e == "progress"]
    assert len(prog) == 40
    assert prog[-1]["loss"] < 0.5 * prog[0]["loss"], (prog[0]["loss"], prog[-1]["loss"])
    assert summary["avg_tok_s"] > 0 and summary["val_loss"] is not None
    assert (tmp_path / "out" / "adapter" / "adapter_model.safetensors").exists()
    assert (tmp_path / "out" / "adapter" / "kangaroo_run.json").exists()
    assert any(e == "done" for e, _ in events)


def test_chunked_and_dense_training_agree(tiny_model, toy_dataset, tmp_path):
    losses = {}
    for name, chunk in (("chunked", 7), ("dense", 0)):
        ev = []
        train(_cfg(tiny_model, toy_dataset, tmp_path / name, max_steps=6, loss_chunk=chunk, val_fraction=0.0),
              lambda e, **k: ev.append((e, k)))
        losses[name] = [k["loss"] for e, k in ev if e == "progress"]
    for a, b in zip(losses["chunked"], losses["dense"]):
        assert a == pytest.approx(b, rel=1e-4)


def test_checkpointing_does_not_change_loss(tiny_model, toy_dataset, tmp_path):
    out = {}
    for mode in ("on", "off"):
        ev = []
        train(_cfg(tiny_model, toy_dataset, tmp_path / mode, max_steps=5, gradient_checkpointing=mode, val_fraction=0.0),
              lambda e, **k: ev.append((e, k)))
        out[mode] = [k["loss"] for e, k in ev if e == "progress"]
    for a, b in zip(out["on"], out["off"]):
        assert a == pytest.approx(b, rel=1e-4)


def test_stop_file_stops_gracefully_and_still_saves(tiny_model, toy_dataset, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("x")
    summary = train(_cfg(tiny_model, toy_dataset, tmp_path / "out", stop_file=str(stop)), lambda *a, **k: None)
    assert summary["stopped"] and summary["steps"] == 0
    assert (tmp_path / "out" / "adapter" / "adapter_config.json").exists()


def test_checkpoints_written(tiny_model, toy_dataset, tmp_path):
    train(_cfg(tiny_model, toy_dataset, tmp_path / "out", max_steps=6, save_every=3, val_fraction=0.0), lambda *a, **k: None)
    assert (tmp_path / "out" / "checkpoint-3" / "adapter_config.json").exists()


def test_generate_and_export_roundtrip(tiny_model, toy_dataset, tmp_path):
    train(_cfg(tiny_model, toy_dataset, tmp_path / "out", max_steps=30, val_fraction=0.0), lambda *a, **k: None)
    adapter = str(tmp_path / "out" / "adapter")

    from kangaroo.inference import export_merged, generate

    tokens = []
    res = generate(
        {"adapter": adapter, "prompt": "say red one", "max_new_tokens": 6, "temperature": 0, "precision": "fp32"},
        lambda e, **k: tokens.append(k.get("text")) if e == "token" else None,
    )
    assert res["text"].strip().startswith("red red")
    assert "".join(t for t in tokens if t) == res["text"]

    out = tmp_path / "merged"
    info = export_merged({"adapter": adapter, "output_dir": str(out), "precision": "fp32"}, lambda *a, **k: None)
    assert info["size_bytes"] > 0 and (out / "config.json").exists()
    from transformers import AutoModelForCausalLM

    merged = AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in n for n, _ in merged.named_parameters())


def test_cli_emits_only_json_lines(tiny_model, toy_dataset, tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(dict(
        model=tiny_model, dataset=toy_dataset, output_dir=str(tmp_path / "o"), max_seq_len=64, batch_tokens=256,
        epochs=0, max_steps=2, precision="fp32",
    )))
    assert main(["train", "--config", str(cfg), "--learning-rate", "0.001"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    parsed = [json.loads(l) for l in lines]
    assert parsed[-1]["event"] == "done"
    assert all("event" in p for p in parsed)


def test_cli_reports_errors_as_events(tmp_path, capsys):
    assert main(["train", "--model", "x", "--dataset", str(tmp_path / "missing.jsonl"), "--epochs", "1"]) == 1
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["event"] == "error" and last["message"]


def test_config_validation():
    with pytest.raises(ValueError, match="Unknown config keys"):
        TrainConfig.from_dict({"model": "m", "dataset": "d", "lerning_rate": 1})
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"model": "", "dataset": "d"})
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"model": "m", "dataset": "d", "lr_scheduler": "wavy"})


def test_lr_schedule_shape():
    cfg = TrainConfig(model="m", dataset="d", learning_rate=1.0, warmup_ratio=0.1, lr_scheduler="cosine")
    lrs = [lr_at(s, 100, cfg) for s in range(100)]
    assert lrs[0] < lrs[9] <= 1.0 and max(lrs) == pytest.approx(1.0, abs=0.02)
    assert lrs[-1] < 0.01 and all(a >= b - 1e-9 for a, b in zip(lrs[10:], lrs[11:]))
