import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import build_tiny_model, write_toy_dataset  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    pytest.importorskip("torch")
    return build_tiny_model(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def tokenizer(tiny_model):
    from kangaroo.model import load_tokenizer

    return load_tokenizer(tiny_model)


@pytest.fixture()
def toy_dataset(tmp_path):
    return write_toy_dataset(tmp_path / "toy.jsonl")
