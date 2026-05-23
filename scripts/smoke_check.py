#!/usr/bin/env python3
"""Quick smoke checks — no webcam required."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def main() -> int:
    from src.paths import artifacts_dir

    ad = artifacts_dir()
    print(f"[smoke] artifacts_dir() -> {ad}")

    mv2 = ad / "model_v2.keras"
    if not mv2.is_file():
        print(f"[smoke] SKIP load_model — missing {mv2}")
        return 0

    try:
        import tensorflow as tf  # noqa: F401
    except ModuleNotFoundError:
        print(
            "[smoke] SKIP load_model - tensorflow not installed. "
            "Use Python 3.10-3.11, run setup_venv.ps1 (Windows) or setup_venv.sh, "
            "then: pip install -r requirements.txt",
        )
        return 0

    from src.inference import load_model, predict_frame  # noqa: WPS433

    load_model(str(mv2), str(ad / "label2idx.json"))
    import numpy as np

    lbl, conf, emo = predict_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    print(f"[smoke] predict_frame(black): {lbl!r} conf={conf} emo={emo}")

    lbl2, conf2, emo2 = predict_frame(
        np.zeros((480, 640, 3), dtype=np.uint8),
        cached_emotion="neutral",
        raw=True,
    )
    print(f"[smoke] predict_frame(raw, neutral): {lbl2!r} conf={conf2} emo={emo2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
