"""
scripts/prepare_mobile.py
=========================
Exports model_v2.keras → model_v2.tflite, validates accuracy parity,
and prints the full integration spec for the mobile team.

Run this once whenever the model is updated.

Usage:
    python scripts/prepare_mobile.py
    python scripts/prepare_mobile.py --optimize float32   # exact parity, larger file
    python scripts/prepare_mobile.py --out-dir  /path/to/share_folder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.paths import artifacts_dir

AD = artifacts_dir()


# ── Constants that the mobile app must mirror exactly ─────────────────────────
FACE_IDX   = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
NEUTRAL_IDX = 4
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]



def export_tflite(source: Path, dest: Path, optimize: str) -> bool:
    try:
        import tensorflow as tf
    except ImportError:
        print("[mobile] ERROR: tensorflow not installed.")
        return False

    print(f"\n[mobile] ── Step 1: Convert {source.name} → {dest.name}")
    model = tf.keras.models.load_model(str(source))
    print(f"[mobile]   Input  shape : {model.input_shape}")
    print(f"[mobile]   Output shape : {model.output_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[mobile]   Optimization : DYNAMIC QUANTIZATION (2–5× faster, ~half size)")
    else:
        print("[mobile]   Optimization : NONE (float32 — exact parity)")

    tflite_bytes = converter.convert()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(tflite_bytes)
    kb = dest.stat().st_size / 1024
    print(f"[mobile]   Saved        : {dest}  ({kb:.1f} KB)")
    return True


def validate(source: Path, dest: Path, n: int = 20) -> dict:
    import numpy as np
    import tensorflow as tf

    print(f"\n[mobile] ── Step 2: Validate accuracy parity ({n} samples)")
    keras_model = tf.keras.models.load_model(str(source))
    interp = tf.lite.Interpreter(model_path=str(dest))
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    val_path = AD / "X_mlp_val.npy"
    if val_path.is_file():
        X = np.load(str(val_path)).astype(np.float32)[:n]
        print(f"[mobile]   Using real val data: {X.shape}")
    else:
        X = np.random.randn(n, 163).astype(np.float32)
        print(f"[mobile]   Using random data (val arrays not found): {X.shape}")

    k_t0 = time.perf_counter()
    k_out = keras_model.predict(X, verbose=0)
    k_ms  = (time.perf_counter() - k_t0) * 1000 / n

    t_out = np.zeros_like(k_out)
    t_t0  = time.perf_counter()
    for i in range(len(X)):
        interp.set_tensor(in_d["index"], X[i:i+1].astype(in_d["dtype"]))
        interp.invoke()
        t_out[i] = interp.get_tensor(out_d["index"])[0]
    t_ms = (time.perf_counter() - t_t0) * 1000 / n

    max_diff  = float(abs(k_out - t_out).max())
    agreement = float((k_out.argmax(1) == t_out.argmax(1)).mean())

    print(f"[mobile]   max |Keras − TFLite|  : {max_diff:.2e}")
    print(f"[mobile]   top-1 agreement       : {agreement*100:.1f}%")
    print(f"[mobile]   Keras  latency/sample : {k_ms:.2f} ms")
    print(f"[mobile]   TFLite latency/sample : {t_ms:.2f} ms")

    ok = agreement >= 0.99 and max_diff < 1e-2
    if ok:
        print("[mobile]   ✓ Conversion verified.")
    else:
        print("[mobile]   ✗ WARN: Drift too high. Re-run with --optimize float32.")
    return {"agreement": agreement, "max_diff": max_diff, "ok": ok,
            "keras_ms": k_ms, "tflite_ms": t_ms}


def print_integration_spec(label_path: Path, stats: dict | None = None):
    import numpy as np

    with open(label_path, encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}
    n_classes  = len(label2idx)

    print(f"""
[mobile] ══════════════════════════════════════════════════════════
[mobile]   MOBILE INTEGRATION SPEC  —  send this to your teammate
[mobile] ══════════════════════════════════════════════════════════

  Model file  : model_v2.tflite
  Label file  : label2idx.json
  Classes     : {n_classes}
  First 5     : {[idx2label[i] for i in range(min(5, n_classes))]}

  ── Input tensor ──────────────────────────────────────────────────
  Shape  : [1, 163]   dtype: float32
  Layout :
    [  0 :  63]  left  hand landmarks  (21 joints × x,y,z)  — zeros if not detected
    [ 63 : 126]  right hand landmarks  (21 joints × x,y,z)  — zeros if not detected
    [126 : 156]  face  landmarks       (10 points × x,y,z)  — zeros if not detected
    [156 : 163]  emotion one-hot       = [0,0,0,0,1,0,0]    — ALWAYS THIS (neutral)

  Face landmark indices from MediaPipe (must match exactly):
    {FACE_IDX}

  ── Normalization (must match exactly or accuracy degrades) ───────
  For EACH hand (left and right independently):
    1. wrist  = joint[0]  (x, y, z)
    2. shift  = subtract wrist from all 21 joints
    3. scale  = max(norm(joint) for joint in shifted_joints)
    4. if scale > 0: divide all shifted joints by scale

  For FACE (10 landmarks):
    1. centroid = mean of all 10 (x, y, z) points
    2. shift    = subtract centroid from all 10 points
    3. scale    = max(norm(point) for point in shifted_points)
    4. if scale > 0: divide all shifted points by scale

  ── Output tensor ─────────────────────────────────────────────────
  Shape  : [1, {n_classes}]   dtype: float32
  Values : softmax probabilities for each sign class
  Usage  : class_index = argmax(output[0])
           sign_name   = lookup label2idx.json reversed

  ── Recommended sliding window (mobile side) ──────────────────────
  WINDOW_SIZE    = 5    (collect last 5 predictions)
  MIN_VOTES      = 3    (same label must appear 3+ times)
  MIN_CONFIDENCE = 0.60 (mean conf of agreeing frames must be >= 0.60)
  COMMIT_COOLDOWN = 8   (skip 8 frames after each commit)
  VELOCITY_THRESHOLD = 0.02  (skip inference if hands barely moved)

  ── What changed from old model ───────────────────────────────────
  OLD model: likely 156-dim input  (landmarks only)
  NEW model: 163-dim input         (landmarks + 7 emotion)
  CHANGE NEEDED: append [0,0,0,0,1,0,0] to your feature vector
                 before passing to the model.

  Kotlin example:
    val features = FloatArray(163)
    landmarks.copyInto(features, 0, 0, 156)   // copy 156 landmark values
    features[160] = 1.0f                        // neutral emotion (index 4)

  Swift example:
    var features = [Float](repeating: 0, count: 163)
    features.replaceSubrange(0..<156, with: landmarks)
    features[160] = 1.0   // neutral emotion (index 4)
{''.join(f"""
  ── Validation stats ──────────────────────────────────────────────
  top-1 agreement  : {stats['agreement']*100:.1f}%
  max |drift|      : {stats['max_diff']:.2e}
  Keras  latency   : {stats['keras_ms']:.2f} ms / sample
  TFLite latency   : {stats['tflite_ms']:.2f} ms / sample
""" if stats else '')}
[mobile] ══════════════════════════════════════════════════════════
""")


def copy_to_output(files: list[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = out_dir / f.name
        shutil.copy2(f, dest)
        print(f"[mobile]   Copied {f.name} → {dest}")


def main():
    p = argparse.ArgumentParser(description="Export TFLite model and print mobile integration spec.")
    p.add_argument("--optimize", choices=["dynamic", "float32"], default="dynamic")
    p.add_argument("--out-dir", default=None,
                   help="Optional folder to copy tflite + json for sharing (e.g. USB, Drive)")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip validation step (faster)")
    args = p.parse_args()

    source = AD / "model_v2.keras"
    dest   = AD / "model_v2.tflite"
    labels = AD / "label2idx.json"

    if not source.is_file():
        print(f"[mobile] ERROR: {source} not found.")
        sys.exit(1)
    if not labels.is_file():
        print(f"[mobile] ERROR: {labels} not found.")
        sys.exit(1)

    ok = export_tflite(source, dest, args.optimize)
    if not ok:
        sys.exit(1)

    stats = None
    if not args.no_validate:
        stats = validate(source, dest)

    print_integration_spec(labels, stats)

    if args.out_dir:
        print(f"\n[mobile] ── Step 3: Copying files to {args.out_dir}")
        copy_to_output([dest, labels], Path(args.out_dir))
        print("[mobile]   Done. Share the folder with your teammate.")

    print("[mobile] ── All done. Send model_v2.tflite + label2idx.json to mobile team.\n")


if __name__ == "__main__":
    main()
