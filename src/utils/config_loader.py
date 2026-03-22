from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    cfg = _read_yaml(path)
    parent_ref = cfg.pop("extends", None)

    if parent_ref is None:
        return cfg

    parent_path = (path.parent / parent_ref).resolve()
    base_cfg = load_config(parent_path)
    base_cfg.update(cfg)
    return base_cfg
