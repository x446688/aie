from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Загружает YAML-конфиг, объединяя с дефолтными значениями."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        # Возвращаем минимальный дефолт, если файла нет
        return {
            "model": {"window_size": 28, "default_type": "lstm"},
            "paths": {"artifacts_dir": "artifacts"},
        }

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str, config: dict | None = None) -> Path:
    """Convert relative path to absolute, relative to project root."""
    if config is None:
        config = load_config()

    base = Path(__file__).resolve().parent.parent.parent

    if Path(relative_path).is_absolute():
        return Path(relative_path)

    return base / relative_path
