"""
Thin re-export shim — canonical implementation lives in src/inference.py.
See .cursor/instruction.md and artifacts/README.md for layout notes.
"""

from src.inference import (  # noqa: F401
    load_model,
    predict_frame,
    update_emotion_async,
    get_last_mp_results,
    get_raw_landmarks,
    reset_window,
    reset_activation_gate,
    VELOCITY_THRESHOLD,
    WINDOW_SIZE,
    MIN_VOTES,
    MIN_CONFIDENCE,
    FEATURE_DIM,
    EMOTION_DIM,
)
