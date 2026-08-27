"""Load committed gold JSON next to the eval package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GOLD = Path(__file__).resolve().parent / "gold"


def load_gold(name: str) -> list[dict[str, Any]]:
    path = _GOLD / name
    return json.loads(path.read_text(encoding="utf-8"))
