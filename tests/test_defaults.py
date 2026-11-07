"""The desktop form duplicates the engine defaults; make sure they never drift apart."""

import re
from pathlib import Path

import pytest

from kangaroo.config import TrainConfig

APP_JS = Path(__file__).resolve().parents[1] / "app" / "renderer" / "app.js"


def _js_value(raw: str):
    raw = raw.strip()
    if raw in ("true", "false"):
        return raw == "true"
    if raw.startswith("'"):
        return raw.strip("'")
    return float(raw)


def _ui_defaults() -> dict:
    src = APP_JS.read_text(encoding="utf-8")
    block = src[src.index("const TRAIN_GROUPS"): src.index("const TEST_FIELDS")]
    pairs = re.findall(r"key: '(\w+)'.*?def: ('[^']*'|[-\w.]+)", block)
    return {k: _js_value(v) for k, v in pairs}


@pytest.mark.skipif(not APP_JS.exists(), reason="desktop app not present")
def test_ui_defaults_match_engine_defaults():
    ui, engine = _ui_defaults(), TrainConfig().to_dict()
    assert len(ui) >= 20
    for key, value in ui.items():
        assert key in engine, f"UI field {key!r} is not a TrainConfig field"
        if key in ("model", "dataset"):  # intentionally pre-filled / empty in the UI
            continue
        assert value == pytest.approx(engine[key]), f"default drift for {key}: UI {value!r} vs engine {engine[key]!r}"


@pytest.mark.skipif(not APP_JS.exists(), reason="desktop app not present")
def test_every_ui_choice_is_valid_for_engine():
    src = APP_JS.read_text(encoding="utf-8")
    from kangaroo.config import CHECKPOINTING, DATASET_FORMATS, PRECISIONS, SCHEDULERS

    for name, allowed in (("dataset_format", DATASET_FORMATS), ("lr_scheduler", SCHEDULERS),
                          ("gradient_checkpointing", CHECKPOINTING), ("precision", PRECISIONS)):
        m = re.search(rf"key: '{name}'.*?options: \[([^\]]*)\]", src)
        options = re.findall(r"'([^']+)'", m.group(1))
        assert tuple(options) == tuple(allowed), name
