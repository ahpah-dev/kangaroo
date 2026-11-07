"""Command line entry point; also the process the desktop app spawns.

    python -m kangaroo train --config run.json
    python -m kangaroo train --model Qwen/Qwen2.5-0.5B-Instruct --dataset data.jsonl
    python -m kangaroo generate --adapter out/adapter --prompt "Hi"
    python -m kangaroo export --adapter out/adapter --output-dir merged
    python -m kangaroo info
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import fields
from typing import Any, Optional

from . import events
from .config import TrainConfig


def _add_config_flags(p: argparse.ArgumentParser) -> None:
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        default = f.default
        if isinstance(default, bool):
            p.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif isinstance(default, int):
            p.add_argument(flag, dest=f.name, type=int, default=argparse.SUPPRESS)
        elif isinstance(default, float):
            p.add_argument(flag, dest=f.name, type=float, default=argparse.SUPPRESS)
        else:
            p.add_argument(flag, dest=f.name, type=str, default=argparse.SUPPRESS)


def _load_json(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kangaroo", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="LoRA / QLoRA fine-tune")
    t.add_argument("--config", help="JSON file with TrainConfig keys (flags override it)")
    _add_config_flags(t)

    g = sub.add_parser("generate", help="Generate text with a trained adapter")
    g.add_argument("--config")
    g.add_argument("--adapter")
    g.add_argument("--base")
    g.add_argument("--prompt")
    g.add_argument("--system", default=argparse.SUPPRESS)
    g.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=argparse.SUPPRESS)
    g.add_argument("--temperature", type=float, default=argparse.SUPPRESS)

    e = sub.add_parser("export", help="Merge the adapter into the base weights and save")
    e.add_argument("--config")
    e.add_argument("--adapter")
    e.add_argument("--base")
    e.add_argument("--output-dir", dest="output_dir")
    e.add_argument("--precision", default=argparse.SUPPRESS)

    sub.add_parser("info", help="Print environment / GPU information")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    events.capture_stdout()
    emit = events.emit
    try:
        if args.command == "info":
            from . import sysinfo

            emit("info", **sysinfo.collect())
            return 0

        raw = _load_json(getattr(args, "config", None))
        overrides = {k: v for k, v in vars(args).items() if k not in ("command", "config") and v is not None}
        raw.update(overrides)

        if args.command == "train":
            from .trainer import train

            train(TrainConfig.from_dict(raw), emit)
        elif args.command == "generate":
            from .inference import generate

            if not raw.get("prompt"):
                raise ValueError("`prompt` is required.")
            generate(raw, emit)
        elif args.command == "export":
            from .inference import export_merged

            for key in ("adapter", "output_dir"):
                if not raw.get(key):
                    raise ValueError(f"`{key}` is required.")
            export_merged(raw, emit)
        return 0
    except KeyboardInterrupt:
        emit("error", message="Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - report everything to the UI
        emit("error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
