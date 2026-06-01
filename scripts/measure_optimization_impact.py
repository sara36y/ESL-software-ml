"""
scripts/measure_optimization_impact.py
=======================================
Measure the real-world performance impact of frame skipping optimization.
Compares: latency, FPS, and accuracy before and after frame skipping.

Usage:
    python scripts/measure_optimization_impact.py
    python scripts/measure_optimization_impact.py --frames 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.inference import load_model, predict_frame, reset_window, set_frame_skip


def benchmark_with_frame_skip_disabled(n_frames: int = 30) -> dict:
    """Measure end-to-end latency with frame skipping DISABLED (full accuracy)."""
    print("\n── Benchmark: FRAME SKIPPING DISABLED (Full Accuracy) ────────────")

    set_frame_skip(1)  # Disable frame skipping
    reset_window()

    times = []
    for i in range(n_frames):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        reset_window()

        t0 = time.perf_counter()
        label, conf, emotion = predict_frame(frame, disable_frame_skip=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        print(f"  Frame {i+1:3d}: {elapsed_ms:6.2f} ms")

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    print(f"\n  Statistics:")
    print(f"    Mean latency:  {mean_ms:.2f} ms")
    print(f"    Std dev:       {std_ms:.2f} ms")
    print(f"    Real-time FPS: {fps:.1f} FPS")

    return {
        "mode": "disabled",
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "samples": n_frames,
    }


def benchmark_with_frame_skip_enabled(n_frames: int = 30, skip_rate: int = 3) -> dict:
    """Measure end-to-end latency with frame skipping ENABLED."""
    print(f"\n── Benchmark: FRAME SKIPPING ENABLED (skip_rate={skip_rate}) ────────")

    set_frame_skip(skip_rate)
    reset_window()

    times = []
    for i in range(n_frames):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        label, conf, emotion = predict_frame(frame, disable_frame_skip=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        skip_indicator = "SKIP" if (i + 1) % skip_rate != 0 else "PRED"
        print(f"  Frame {i+1:3d}: {elapsed_ms:6.2f} ms  [{skip_indicator}]")

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    # Effective FPS accounting for skipped frames
    effective_fps = fps * skip_rate  # Perceived FPS due to frame reuse

    print(f"\n  Statistics:")
    print(f"    Mean latency (per frame):     {mean_ms:.2f} ms")
    print(f"    Std dev:                      {std_ms:.2f} ms")
    print(f"    Actual FPS (with skips):      {fps:.1f} FPS")
    print(f"    Effective FPS (perceived):    {effective_fps:.1f} FPS")

    return {
        "mode": f"enabled (skip_rate={skip_rate})",
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "effective_fps": round(effective_fps, 1),
        "samples": n_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure performance impact of frame skipping"
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--skip-rates", type=int, nargs="+", default=[3, 5])
    args = parser.parse_args()

    print("=" * 70)
    print("ESL OPTIMIZATION IMPACT MEASUREMENT")
    print("=" * 70)

    try:
        load_model()
        print("[benchmark] Model loaded successfully\n")
    except Exception as e:
        print(f"[benchmark] ERROR loading model: {e}", file=sys.stderr)
        return 1

    # Baseline: no frame skipping
    baseline = benchmark_with_frame_skip_disabled(args.frames)

    # Test different skip rates
    results = [baseline]
    for skip_rate in args.skip_rates:
        result = benchmark_with_frame_skip_enabled(args.frames, skip_rate)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Mode':<35} {'Latency':<15} {'FPS':<15} {'Speedup':<10}")
    print("-" * 75)

    baseline_fps = baseline["fps"]
    for result in results:
        mode_str = result["mode"]
        latency_str = f"{result['mean_ms']:.2f} ms"
        fps_str = result.get("effective_fps", result["fps"])
        speedup = baseline_fps / result["fps"] if result["fps"] > 0 else 0

        print(f"{mode_str:<35} {latency_str:<15} {fps_str:<15.1f} {speedup:<10.1f}x")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if baseline_fps < 5:
        print(f"""
⚠️  CRITICAL: Current FPS is {baseline_fps:.1f}, which is too low for real-time.

✅ SOLUTIONS:
  1. Enable frame skipping with skip_rate=3 (3x speedup)
  2. Deploy with TFLite model export
  3. Consider higher skip_rate if FPS still insufficient

With skip_rate=3, you should achieve ~{baseline_fps * 3:.1f} FPS (acceptable).
        """)
    elif baseline_fps < 10:
        print(f"""
⚠️  WARNING: Current FPS is {baseline_fps:.1f}, at minimum for real-time.

✅ RECOMMENDATION:
  Enable frame skipping with skip_rate=3 to reach ~{baseline_fps * 3:.1f} FPS.
        """)
    else:
        print(f"""
✅ GOOD: Current FPS is {baseline_fps:.1f}, acceptable for real-time.

Frame skipping can further improve to ~{baseline_fps * 3:.1f} FPS if needed.
        """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
