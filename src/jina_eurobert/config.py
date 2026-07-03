from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "distill_8xa100.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def matryoshka_dims(config: dict[str, Any]) -> list[int]:
    return list(config.get("matryoshka_dims", [32, 64, 128, 256, 512, 768]))
