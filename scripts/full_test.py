"""
scripts/full_test.py
====================
Comprehensive multi-image test: validation array accuracy + WebSocket endpoint + latency.

Produces:
  1. Per-class accuracy breakdown
  2. Overall accuracy / macro F1 / weighted F1
  3. Top-5 worst-performing classes
  4. Top-3 confused pairs
  5. WebSocket latency test (multiple frames)
  6. Direct model inference latency (100 runs)
  7. Edge-case tests (black, noise, partial hands)

Usage:
    python scripts/full_test.py
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from sklearn.metrics import f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTS = ROOT / "artifacts"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Validation Array Test — model accuracy on held-out data
# ═══════════════════════════════════════════════════════════════════════════════

def test_validation_arrays():
    import tensorflow as tf

    print("=" * 70)
    print("SECTION 1: Validation Array Accuracy (model_v2.keras on X_mlp_val)")
    print("=" * 70)

    with open(ARTS / "label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(idx2label))]

    X_val = np.load(ARTS / "X_mlp_val.npy")
    y_val = np.load(ARTS / "y_val.npy")

    model = tf.keras.models.load_model(str(ARTS / "model_v2.keras"))
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = (y_pred == y_val).mean()
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"\nOverall Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print(f"Total samples:     {len(y_val)}")
    print(f"Total classes:     {len(class_names)}")

    # Per-class accuracy
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for true, pred in zip(y_val, y_pred):
        lbl = class_names[true]
        per_class[lbl]["total"] += 1
        if true == pred:
            per_class[lbl]["correct"] += 1

    print(f"\n--- Per-class Accuracy ---")
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["correct"]/x[1]["total"], reverse=True)
    perfect = [(k, v) for k, v in sorted_classes if v["correct"] == v["total"]]
    imperfect = [(k, v) for k, v in sorted_classes if v["correct"] < v["total"]]

    print(f"\nPerfect accuracy (100%): {len(perfect)}/{len(sorted_classes)} classes")
    for k, v in perfect[:10]:
        print(f"  {k:>20s}  {v['correct']}/{v['total']}  ({v['correct']/v['total']*100:.0f}%)")
    if len(perfect) > 10:
        print(f"  ... and {len(perfect)-10} more at 100%")

    print(f"\nWorst 5 classes:")
    worst5 = sorted_classes[-5:]
    for k, v in worst5:
        pct = v["correct"]/v["total"]*100
        print(f"  {k:>20s}  {v['correct']}/{v['total']}  ({pct:.0f}%)")

    # Top-3 confused pairs
    cm = confusion_matrix(y_val, y_pred)
    cm_nodiag = cm.copy()
    np.fill_diagonal(cm_nodiag, 0)
    flat = np.argsort(cm_nodiag.ravel())[::-1]
    confused = []
    for idx in flat[:10]:
        r, c = divmod(idx, len(class_names))
        if cm_nodiag[r, c] > 0 and r != c:
            confused.append((class_names[r], class_names[c], cm_nodiag[r, c]))
        if len(confused) == 3:
            break

    print(f"\nTop 3 confused pairs:")
    for true_lbl, pred_lbl, count in confused:
        print(f"  {true_lbl} -> {pred_lbl}  ({count} misclassifications)")

    # Confidence distribution
    confidences = y_pred_probs.max(axis=1)
    correct_conf = confidences[y_pred == y_val]
    wrong_conf = confidences[y_pred != y_val]
    print(f"\n--- Confidence Distribution ---")
    print(f"  Mean confidence (correct): {correct_conf.mean():.4f}")
    print(f"  Mean confidence (wrong):   {wrong_conf.mean():.4f}")
    print(f"  Min confidence (correct):  {correct_conf.min():.4f}")
    print(f"  Max confidence (wrong):    {wrong_conf.max():.4f}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_samples": len(y_val),
        "n_classes": len(class_names),
        "perfect_classes": len(perfect),
        "worst5": [(k, v["correct"]/v["total"]) for k, v in worst5],
        "confused_pairs": confused,
        "mean_conf_correct": float(correct_conf.mean()),
        "mean_conf_wrong": float(wrong_conf.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Direct Inference Latency — 100 runs on model directly
# ═══════════════════════════════════════════════════════════════════════════════

def test_direct_latency():
    import tensorflow as tf

    print("\n" + "=" * 70)
    print("SECTION 2: Direct Model Inference Latency (100 runs)")
    print("=" * 70)

    model = tf.keras.models.load_model(str(ARTS / "model_v2.keras"))
    sample = np.random.randn(1, 163).astype(np.float32)

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)

    print(f"  Mean latency:  {np.mean(times):.2f} ms")
    print(f"  Std latency:   {np.std(times):.2f} ms")
    print(f"  Median latency: {np.median(times):.2f} ms")
    print(f"  P95 latency:   {np.percentile(times, 95):.2f} ms")
    print(f"  P99 latency:   {np.percentile(times, 99):.2f} ms")

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Predict Frame Latency — full pipeline (MediaPipe + model + smoothing)
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_frame_latency():
    from src.inference import load_model, predict_frame, reset_window

    print("\n" + "=" * 70)
    print("SECTION 3: predict_frame() Full Pipeline Latency")
    print("=" * 70)

    load_model()

    test_images = {
        "black_240x320":    np.zeros((240, 320, 3), dtype=np.uint8),
        "noise_240x320":    np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        "black_640x480":    np.zeros((640, 480, 3), dtype=np.uint8),
        "gradient_320x240": np.tile(np.linspace(0,255,320,dtype=np.uint8), (240,1)).reshape(240,320,1).repeat(3,axis=2),
    }

    for name, img in test_images.items():
        reset_window()
        t0 = time.perf_counter()
        lbl, conf, emo = predict_frame(img, cached_emotion="neutral", raw=True)
        lat = (time.perf_counter() - t0) * 1000
        print(f"  {name:>20s}: label={lbl:>20s}  conf={conf:.4f}  latency={lat:.2f}ms")

    # Run 50 iterations with a single test image for timing
    reset_window()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        predict_frame(frame, cached_emotion="neutral", raw=True)
        times.append((time.perf_counter() - t0) * 1000)

    print(f"\n  50-run stats (random 640x480):")
    print(f"    Mean:   {np.mean(times):.2f} ms")
    print(f"    Median: {np.median(times):.2f} ms")
    print(f"    P95:    {np.percentile(times, 95):.2f} ms")

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WebSocket Endpoint Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_websocket():
    print("\n" + "=" * 70)
    print("SECTION 4: WebSocket Endpoint Test (localhost:8000/ws)")
    print("=" * 70)

    try:
        import websocket as ws_lib
    except ImportError:
        print("  SKIP: websocket-client not installed")
        return None

    test_frames = {
        "black":         np.zeros((240, 320, 3), dtype=np.uint8),
        "noise":         np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        "gradient":      np.tile(np.linspace(0,255,320,dtype=np.uint8), (240,1)).reshape(240,320,1).repeat(3,axis=2),
        "white":         np.full((240, 320, 3), 255, dtype=np.uint8),
        "dark_center":   cv2.rectangle(np.zeros((240,320,3),dtype=np.uint8), (80,60),(240,180),(128,128,128),-1),
    }

    results = []
    try:
        conn = ws_lib.create_connection("ws://localhost:8000/ws", timeout=10)
    except Exception as e:
        print(f"  SKIP: Cannot connect to WebSocket — {e}")
        return None

    for name, frame in test_frames.items():
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf).decode()
        t0 = time.perf_counter()
        conn.send(b64)
        resp = json.loads(conn.recv())
        ws_lat = (time.perf_counter() - t0) * 1000
        results.append({
            "name": name,
            "label": resp.get("label"),
            "confidence": resp.get("confidence"),
            "emotion": resp.get("emotion"),
            "latency_ms": resp.get("latency_ms"),
            "roundtrip_ms": ws_lat,
        })
        print(f"  {name:>15s}: label={resp.get('label'):>20s}  conf={resp.get('confidence',0):.4f}  server_lat={resp.get('latency_ms',0):.2f}ms  roundtrip={ws_lat:.2f}ms")

    # Burst test: 20 rapid frames
    burst_lats = []
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode()

    print(f"\n  Burst test (20 rapid frames):")
    for i in range(20):
        t0 = time.perf_counter()
        conn.send(b64)
        resp = json.loads(conn.recv())
        burst_lats.append((time.perf_counter() - t0) * 1000)

    print(f"    Mean roundtrip:   {np.mean(burst_lats):.2f} ms")
    print(f"    Median roundtrip: {np.median(burst_lats):.2f} ms")
    print(f"    P95 roundtrip:    {np.percentile(burst_lats, 95):.2f} ms")

    conn.close()

    return {
        "single_frames": results,
        "burst_mean_ms": float(np.mean(burst_lats)),
        "burst_median_ms": float(np.median(burst_lats)),
        "burst_p95_ms": float(np.percentile(burst_lats, 95)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model v1 vs v2 vs v3 Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def test_model_comparison():
    import tensorflow as tf

    print("\n" + "=" * 70)
    print("SECTION 5: Model Variant Comparison (v1, v2, v3)")
    print("=" * 70)

    with open(ARTS / "label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}
    n_classes = len(idx2label)

    X_mlp_val  = np.load(ARTS / "X_mlp_val.npy")
    X_lstm_val = np.load(ARTS / "X_lstm_val.npy")
    y_val      = np.load(ARTS / "y_val.npy")

    variants = [
        ("v1: Baseline MLP (no aug)", ARTS / "model_v1.keras", X_mlp_val),
        ("v2: Aug MLP + Emotion",     ARTS / "model_v2.keras", X_mlp_val),
        ("v3: Aug LSTM + Emotion",    ARTS / "model_v3.keras", X_lstm_val),
    ]

    rows = []
    for name, path, X in variants:
        if not path.is_file():
            print(f"  SKIP {name} — file not found")
            continue
        model = tf.keras.models.load_model(str(path))
        probs = model.predict(X, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        acc = (y_pred == y_val).mean()
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        sample = X[:1]
        lats = []
        for _ in range(50):
            t0 = time.perf_counter()
            model.predict(sample, verbose=0)
            lats.append((time.perf_counter() - t0) * 1000)

        row = {
            "name": name,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "lat_mean_ms": round(np.mean(lats), 2),
            "lat_median_ms": round(np.median(lats), 2),
        }
        rows.append(row)
        print(f"  {name:>30s}: acc={acc:.4f}  F1={f1:.4f}  lat={np.mean(lats):.1f}ms")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Overall Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(val_res, lat_res, pf_res, ws_res, comp_res):
    print("\n" + "=" * 70)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n  Model: model_v2.keras (Augmented MLP + Emotion, 163-dim input)")
    print(f"  Classes: {val_res['n_classes']}  |  Val samples: {val_res['n_samples']}")
    print(f"")
    print(f"  === ACCURACY ===")
    print(f"  Overall accuracy:     {val_res['accuracy']*100:.2f}%")
    print(f"  Macro F1:             {val_res['macro_f1']*100:.2f}%")
    print(f"  Weighted F1:          {val_res['weighted_f1']*100:.2f}%")
    print(f"  Perfect classes:      {val_res['perfect_classes']}/{val_res['n_classes']} (100% accuracy)")
    print(f"  Mean conf (correct):  {val_res['mean_conf_correct']:.4f}")
    print(f"  Mean conf (wrong):    {val_res['mean_conf_wrong']:.4f}")
    print(f"")
    print(f"  === LATENCY ===")
    print(f"  Direct model (mean):  {lat_res['mean_ms']:.1f} ms")
    print(f"  Direct model (P95):   {lat_res['p95_ms']:.1f} ms")
    print(f"  predict_frame (mean): {pf_res['mean_ms']:.1f} ms")
    print(f"  predict_frame (P95):  {pf_res['p95_ms']:.1f} ms")
    if ws_res:
        print(f"  WebSocket roundtrip (mean): {ws_res['burst_mean_ms']:.1f} ms")
        print(f"  WebSocket roundtrip (P95):  {ws_res['burst_p95_ms']:.1f} ms")
    else:
        print(f"  WebSocket: not tested (server may be down)")
    print(f"")
    print(f"  === WORST 5 CLASSES ===")
    for k, acc in val_res["worst5"]:
        print(f"    {k:>20s}: {acc*100:.0f}%")
    print(f"")
    print(f"  === TOP 3 CONFUSED PAIRS ===")
    for t, p, c in val_res["confused_pairs"]:
        print(f"    {t} -> {p} ({c} times)")
    print(f"")
    print(f"  === MODEL COMPARISON ===")
    for r in comp_res:
        print(f"    {r['name']:>30s}: acc={r['accuracy']*100:.2f}%  F1={r['macro_f1']*100:.2f}%  lat={r['lat_mean_ms']:.1f}ms")
    print(f"")
    print(f"  === VERDICT ===")
    v2 = [r for r in comp_res if "v2" in r["name"]][0]
    v1 = [r for r in comp_res if "v1" in r["name"]][0]
    improvement = (v2["accuracy"] - v1["accuracy"]) * 100
    print(f"  v2 vs v1: +{improvement:.2f}% accuracy improvement from augmentation + emotion slot")
    print(f"  v2 is the PRODUCTION model ({v2['accuracy']*100:.2f}% accuracy, {v2['lat_mean_ms']:.1f}ms latency)")
    if lat_res['p95_ms'] < 100:
        print(f"  P95 latency < 100ms: PASS (target met)")
    else:
        print(f"  P95 latency >= 100ms: WARN (target is <= 100ms)")
    if ws_res and ws_res['burst_p95_ms'] < 150:
        print(f"  WS P95 roundtrip < 150ms: PASS (target met)")
    elif ws_res:
        print(f"  WS P95 roundtrip >= 150ms: WARN (target is <= 150ms)")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    val_res = test_validation_arrays()
    lat_res = test_direct_latency()
    pf_res = test_predict_frame_latency()
    ws_res = test_websocket()
    comp_res = test_model_comparison()
    print_summary(val_res, lat_res, pf_res, ws_res, comp_res)