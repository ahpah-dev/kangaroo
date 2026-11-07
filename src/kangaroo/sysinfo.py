from __future__ import annotations

import importlib.metadata as md
import platform
from typing import Any


def _version(pkg: str):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def collect() -> dict[str, Any]:
    import torch

    from . import __version__

    info: dict[str, Any] = {
        "kangaroo": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "bitsandbytes": _version("bitsandbytes"),
        "datasets": _version("datasets"),
        "cuda": torch.cuda.is_available(),
        "gpu": None,
        "vram_gb": None,
        "bf16": False,
    }
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, vram_gb=round(props.total_memory / 2**30, 1), bf16=torch.cuda.is_bf16_supported())
    elif "+cpu" in torch.__version__ or not torch.version.cuda:
        info["note"] = "This PyTorch build has no CUDA support; training will run on the CPU."
    return info
