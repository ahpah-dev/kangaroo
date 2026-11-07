"""JSON-lines event protocol between the engine and the desktop app.

The engine writes one JSON object per line to the *real* stdout. Anything else
(library prints, warnings) is redirected to stderr so it can never corrupt the
protocol stream.
"""

from __future__ import annotations

import json
import sys
from typing import Any

_out = sys.stdout


def capture_stdout() -> None:
    """Reserve the real stdout for protocol events; send stray prints to stderr."""
    global _out
    _out = sys.stdout
    sys.stdout = sys.stderr


def emit(event: str, **fields: Any) -> None:
    _out.write(json.dumps({"event": event, **fields}, default=str) + "\n")
    _out.flush()


def log(message: str) -> None:
    emit("log", message=message)
