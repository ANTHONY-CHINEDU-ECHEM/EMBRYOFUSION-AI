"""Configuration loading with attribute access."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("EMBRYOFUSION_ROOT", Path(__file__).resolve().parents[2]))


class Config(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None) -> Config:
    with open(path or ROOT / "configs" / "config.yaml", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
