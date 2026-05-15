"""
src/paths.py
============
Resolve artifact locations for both layouts:
  - artifacts/              (canonical, README / old architecture)
  - output/artifacts/       (Cursor sprint / instruction.md)
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_CANDIDATE_DIRS = (
    ROOT / "artifacts",
    ROOT / "output" / "artifacts",
)


def artifacts_dir() -> Path:
    """Return the first artifacts directory that contains model or label files."""
    for p in _CANDIDATE_DIRS:
        if (p / "model_v2.keras").exists() or (p / "label2idx.json").exists():
            return p
    return ROOT / "artifacts"


def model_path(name: str = "model_v2.keras") -> Path:
    return artifacts_dir() / name


def label2idx_path() -> Path:
    return artifacts_dir() / "label2idx.json"


def results_dir() -> Path:
    """Directory for plots and CSV outputs from evaluate.py."""
    p = ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p
