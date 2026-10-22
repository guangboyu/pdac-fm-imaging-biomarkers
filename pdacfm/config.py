"""Configuration files: local data paths and the reported model configurations."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_paths(path: str | Path | None = None) -> dict:
    """Read ``configs/paths.yaml`` (copy it from ``configs/paths.example.yaml``)."""
    path = Path(path) if path else ROOT / "configs" / "paths.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; copy configs/paths.example.yaml and edit it")
    with open(path) as f:
        return yaml.safe_load(f)


def load_configurations() -> dict:
    """Encoder, pooling and aggregation of each reported endpoint (``configs/endpoints.yaml``)."""
    with open(ROOT / "configs" / "endpoints.yaml") as f:
        return yaml.safe_load(f)
