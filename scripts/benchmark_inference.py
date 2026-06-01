"""
scripts/benchmark_inference.py
=============================
Comprehensive benchmark of each component in the inference pipeline.
Measures: MediaPipe extraction, normalization, model inference, sliding window.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --frames 100
    python scripts/benchmark_inference.py --model artifacts/model_v3.keras
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.inference import load_model, predict_frame, get_raw_landmarks, reset_window
from src.paths import artifacts_dir

def benchmark_mediaipe_extraction(n_frames: int = 20) -> dict:
    """Measure MediaPipe Holistic extraction time on random frames."""
    print("\n── Benchmarking MediaPipe Holistic Extraction ────────────────────")

    times = []
    for i in range(n_frames):
        # Generate random BGR frame (320x240 as used in predict_frame)
        frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        landmarks = get_raw_landmarks(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))

    print(f"  Samples: {n_frames}")
    print(f"  Mean:    {mean_ms:.2f} ms")
    print(f"  Std:     {std_ms:.2f} ms")
    print(f"  Min:     {min_ms:.2f} ms")
    print(f"  Max:     {max_ms:.2f} ms")

    return {
        "component": "MediaPipe Holistic",
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "samples": n_frames,
    }


def benchmark_model_inference(model_path: str, n_samples: int = 20) -> dict:
    """Measure pure model inference time (excluding extraction/normalization)."""
    print(f"\n── Benchmarking Model Inference ({Path(model_path).name}) ────────")

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    # Use real validation data if available
    val_path = artifacts_dir() / "X_mlp_val.npy"
    if val_path.exists() and model.input_shape[-1] == 163:
        X = np.load(str(val_path))[:n_samples].astype(np.float32)
    else:
        X = np.random.randn(n_samples, 163).astype(np.float32)

    times = []
    for sample in X:
        t0 = time.perf_counter()
        _ = model.predict(sample.reshape(1, -1), verbose=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))

    print(f"  Model:   {Path(model_path).name}")
    print(f"  Input:   {model.input_shape}")
    print(f"  Samples: {n_samples}")
    print(f"  Mean:    {mean_ms:.2f} ms")
    print(f"  Std:     {std_ms:.2f} ms")

    return {
        "component": f"Model Inference ({Path(model_path).name})",
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "samples": n_samples,
        "input_shape": str(model.input_shape),
    }


def benchmark_tflite_inference(tflite_path: str, n_samples: int = 20) -> dict:
    """Measure TFLite model inference time."""
    print(f"\n── Benchmarking TFLite Inference ({Path(tflite_path).name}) ────────")

    import tensorflow as tf

    if not os.path.exists(tflite_path):
        print(f"  SKIP: {tflite_path} not found")
        return None

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    # Use real validation data if available
    val_path = artifacts_dir() / "X_mlp_val.npy"
    if val_path.exists():
        X = np.load(str(val_path))[:n_samples].astype(in_details["dtype"])
    else:
        X = np.random.randn(n_samples, 163).astype(in_details["dtype"])

    times = []
    for sample in X:
        t0 = time.perf_counter()
        interpreter.set_tensor(in_details["index"], sample.reshape(1, -1))
        interpreter.invoke()
        _ = interpreter.get_tensor(out_details["index"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    size_kb = os.path.getsize(tflite_path) / 1024.0

    print(f"  Model:   {Path(tflite_path).name}")
    print(f"  Size:    {size_kb:.1f} KB")
    print(f"  Samples: {n_samples}")
    print(f"  Mean:    {mean_ms:.2f} ms")
    print(f"  Std:     {std_ms:.2f} ms")

    return {
        "component": f"TFLite Inference ({Path(tflite_path).name})",
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "samples": n_samples,
        "size_kb": round(size_kb, 1),
    }


def benchmark_end_to_end(n_frames: int = 20) -> dict:
    """Measure full predict_frame() pipeline."""
    print(f"\n── Benchmarking End-to-End predict_frame() ────────────────────")

    times = []
    for i in range(n_frames):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        reset_window()
        t0 = time.perf_counter()
        label, conf, emotion = predict_frame(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms

    print(f"  Frames:  {n_frames}")
    print(f"  Mean:    {mean_ms:.2f} ms")
    print(f"  Std:     {std_ms:.2f} ms")
    print(f"  FPS:     {fps:.1f}")

    return {
        "component": "End-to-End predict_frame()",
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "frames": n_frames,
    }


def benchmark_model_comparison(n_samples: int = 20) -> list:
    """Compare all available models."""
    print("\n── Model Inference Comparison ────────────────────────────────────")

    results = []
    models = [
        ("model_v1.keras", "Baseline MLP (no aug)"),
        ("model_v2.keras", "Aug MLP + Emotion (PRIMARY)"),
        ("model_v3.keras", "Aug LSTM + Emotion"),
    ]

    for model_file, description in models:
        path = artifacts_dir() / model_file
        if not path.exists():
            print(f"  SKIP {description} — {model_file} not found")
            continue

        result = benchmark_model_inference(str(path), n_samples)
        result["description"] = description
        results.append(result)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ESL inference pipeline")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames for benchmarks")
    parser.add_argument("--model", type=str, help="Override model path for single-model test")
    args = parser.parse_args()

    print("=" * 70)
    print("ESL INFERENCE BENCHMARK")
    print("=" * 70)

    # Load model once
    try:
        load_model()
        print("[benchmark] Model loaded successfully\n")
    except Exception as e:
        print(f"[benchmark] ERROR loading model: {e}", file=sys.stderr)
        return 1

    results = {
        "timestamp": time.time(),
        "benchmarks": [],
    }

    # Run benchmarks
    results["benchmarks"].append(benchmark_mediaipe_extraction(args.frames))
    results["benchmarks"].extend(benchmark_model_comparison(args.frames))
    results["benchmarks"].append(benchmark_tflite_inference(
        str(artifacts_dir() / "model_v2.tflite"), args.frames
    ))
    results["benchmarks"].append(benchmark_end_to_end(args.frames))

    # Remove None entries
    results["benchmarks"] = [r for r in results["benchmarks"] if r is not None]

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & ANALYSIS")
    print("=" * 70)

    e2e = next((r for r in results["benchmarks"] if "End-to-End" in r["component"]), None)
    if e2e:
        print(f"\nCurrent Performance:")
        print(f"  Latency:     {e2e['mean_ms']:.1f} ms")
        print(f"  Real-time FPS: {e2e['fps']:.1f}")
        if e2e['fps'] < 10:
            print(f"  ⚠️  WARNING: FPS < 10 is not real-time. Optimization needed!")

    keras = next((r for r in results["benchmarks"] if "model_v2.keras" in r["component"]), None)
    tflite = next((r for r in results["benchmarks"] if "TFLite" in r["component"]), None)

    if keras and tflite:
        speedup = keras['mean_ms'] / tflite['mean_ms']
        print(f"\nTFLite vs Keras Speedup:")
        print(f"  Keras:   {keras['mean_ms']:.2f} ms")
        print(f"  TFLite:  {tflite['mean_ms']:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

    # Save results
    results_file = Path(__file__).parent.parent / "results" / "benchmark_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())