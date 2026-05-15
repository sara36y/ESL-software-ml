---
name: ESL inference & Phase 4 fixes
overview: Align live inference with the Phase 1 training pipeline (fixes two silent feature-mismatch bugs), restore the emotion-fusion path on the web server, regenerate missing Phase 3 eval artifacts, add TFLite export, and make launchers work from Windows PowerShell. No model retraining required — the training notebook is already correct.
todos:
  - id: face_idx
    content: Restore FACE_IDX order in src/inference.py to match Phase 1 notebook ([0, 1, 13, 14, 17, 33, 61, 199, 263, 291])
    status: completed
  - id: normalize
    content: Rewrite _normalize() in src/inference.py to use max-radial-distance scaling, matching normalize_frame() in Phase 1 Cell 9
    status: completed
  - id: input_shape_assert
    content: Add input-shape sanity assert in load_model() (expect 163-dim for model_v2)
    status: completed
  - id: web_emotion
    content: Add async emotion background task + latest_frame_ref to web/server.py so the browser gets real emotion values
    status: completed
  - id: export_eval
    content: Write scripts/export_eval_arrays.py to produce X_mlp_val_emo.npy, X_lstm_val_emo.npy, y_val.npy (+ train variants) from existing arrays
    status: completed
  - id: tflite
    content: Write scripts/export_tflite.py with --validate flag; produce artifacts/model_v2.tflite and log max diff vs Keras
    status: completed
  - id: ps1_launchers
    content: Add run_demo.ps1 and run_web.ps1 PowerShell equivalents; keep .sh versions; update README
    status: completed
  - id: drop_phase2_prototype
    content: Remove/relocate phase2 (1).py (dead prototype with stale constants)
    status: completed
  - id: share_mp_results
    content: Make desktop inference_thread publish MediaPipe results so display does not re-run Holistic
    status: completed
  - id: readme_fixes
    content: Fix README path typos (data/augmented_landmarks), add PowerShell note, note known web emotion/heartbeat limits
    status: completed
  - id: results_dir
    content: Create empty results/ directory with .gitkeep
    status: completed
  - id: verification
    content: Run verification sequence (load_model sanity, evaluate.py, TFLite diff, desktop run, web run) and record numbers in README
    status: completed
isProject: false
---

## Root cause summary

Two silent regressions in [src/inference.py](src/inference.py) cause the loaded `model_v2.keras` to see feature vectors it never saw during training. The training pipeline in [ESL_Phase1_Complete (1).ipynb](ESL_Phase1_Complete%20%281%29.ipynb) Cell 3 / Cell 8 / Cell 9 is the ground truth; the reference prototype [phase2 (1).py](phase2%20%281%29.py) also uses the correct values.

| Item | Training (Phase 1 notebook) | Live (src/inference.py) | Action |
| --- | --- | --- | --- |
| `FACE_IDX` order | `[0, 1, 13, 14, 17, 33, 61, 199, 263, 291]` | `[1, 33, 61, 199, 263, 291, 17, 0, 13, 14]` | Restore training order |
| Hand / face scale | `np.max(np.linalg.norm(seg, axis=1))` (max radial) | `np.linalg.norm(seg.max(0) - seg.min(0))` (bbox diagonal) | Restore max-radial |

Everything else (model format, label2idx loading, activation gate, confidence threshold, sliding window) is correct.

---

## Critical fixes — [src/inference.py](src/inference.py)

1. Change `FACE_IDX` on line 25 to the training order:

```python
FACE_IDX = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
```

2. Rewrite `_normalize` (lines 235-254) so it mirrors `normalize_frame()` from Cell 9 of the notebook exactly — hands centered on wrist with `origin = seg[0]`, face centered on `face.mean(axis=0)`, scale in every case is `np.max(np.linalg.norm(seg, axis=1))`. Pattern:

```python
def _normalize(ff):
    raw = ff.astype(np.float64)
    left  = raw[0:63].reshape(21, 3).copy()
    right = raw[63:126].reshape(21, 3).copy()
    face  = raw[126:].reshape(-1, 3).copy()
    for seg, origin in [(left, left[0].copy()), (right, right[0].copy())]:
        if seg.any():
            seg -= origin
            s = np.max(np.linalg.norm(seg, axis=1))
            if s > 0: seg /= s
    if face.any():
        face -= face.mean(axis=0)
        s = np.max(np.linalg.norm(face, axis=1))
        if s > 0: face /= s
    return np.concatenate([left.flatten(), right.flatten(), face.flatten()]).astype(np.float32)
```

3. After the fix, add a tiny self-check to `load_model()`: assert `_model.input_shape[-1] == FEATURE_DIM + EMOTION_DIM` (= 163), fail fast with a clear message if someone points it at `model_v1.keras` (156-dim) by mistake.

---

## High-priority fixes

### 4. Web emotion thread — [web/server.py](web/server.py)

Currently `update_emotion_async()` is never invoked, so the browser's emotion always reads "neutral". Add a background task started on FastAPI `@app.on_event("startup")` that consumes the most recent frame from a module-level `latest_frame_ref = [None]` (same pattern as desktop). Inside `websocket_endpoint`, after decoding `frame`, set `latest_frame_ref[0] = frame`. Run the background task in a thread pool (DeepFace is blocking) every `DEEPFACE_INTERVAL` frames.

Sketch:

```python
import asyncio
from src.inference import update_emotion_async

latest_frame_ref = [None]
async def _emotion_worker():
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(0.25)
        f = latest_frame_ref[0]
        if f is not None:
            await loop.run_in_executor(None, update_emotion_async, f)

@app.on_event("startup")
async def _start(): asyncio.create_task(_emotion_worker())
```

### 5. Regenerate missing Phase 3 artifacts

[src/evaluate.py](src/evaluate.py) imports `X_mlp_val_emo.npy`, `X_lstm_val_emo.npy`, `y_val.npy` which are absent. Add a short helper `scripts/export_eval_arrays.py` (or a new cell at the end of the notebook) that:

- Loads the existing `X_mlp_val.npy` + `X_lstm_val.npy`.
- Applies `add_neutral_emotion()` from [src/augmentation.py](src/augmentation.py) to produce the `_emo` variants.
- Exports `y_val.npy` from the notebook split (currently only in-memory during training).
- Also export `_train_emo` pairs while we are there.

No model retraining needed.

### 6. TFLite export — new `scripts/export_tflite.py`

Five-line converter matching README snippet; target `artifacts/model_v2.tflite`. Add one CLI flag `--validate` that runs 20 val samples through both Keras and TFLite and prints max |diff|, so we can show < 1% drift in the ablation table as required by plan §5.2.

### 7. PowerShell launcher — `run_demo.ps1`, `run_web.ps1`

Equivalents of the two `.sh` files, with `py -3.11` usage. Keep the `.sh` versions for WSL/Git Bash. README gets a short "On Windows, use `.\run_demo.ps1`" line.

---

## Medium polish (quick wins)

8. Delete [phase2 (1).py](phase2%20%281%29.py) — it's a dead prototype that references missing `model_v2_config.json`/`model_v2_weights.npy`. Keeping it around is what made these bugs easy to spot, but it is also how they drifted. Move to `scratch/` or remove.

9. Reuse MediaPipe results across the desktop inference → display boundary. Today [demo.py](demo.py) runs Holistic twice per frame (once in `inference_thread`, once in `run` for landmark drawing). Extend `inference_thread` to publish `mediapipe_results` in the result tuple and update `draw_landmarks` to use it. Saves ~10-20 ms/frame.

10. Fix README typo: `data/augmented/` → `data/augmented_landmarks/` in the File Structure section.

11. Create an empty `results/` directory + `.gitkeep` so `evaluate.py` does not need to guard `os.makedirs` (it already does, but the expected outputs should be pre-declared for the viva checklist).

---

## Verification plan

After fixes, run in this order:

1. `python -c "from src.inference import load_model, predict_frame; import numpy as np; load_model(); print(predict_frame(np.zeros((480,640,3), np.uint8)))"` — should print `("No hand detected", 0.0, "neutral")` with the dim-assertion passing.
2. `python scripts/export_eval_arrays.py` — regenerate missing `.npy` files.
3. `python -m src.evaluate` — expect `results/confusion_matrix.png`, `ablation_table.csv`, `classification_report.txt`, `failure_analysis.png` to all be produced without errors.
4. `python scripts/export_tflite.py --validate` — expect max |diff| < 1e-3.
5. `.\run_demo.ps1` — run desktop app, confirm a trained sign is recognised correctly (this is the real proof that the Critical fixes worked; the model was *always* loading, but it wasn't *predicting right*).
6. `.\run_web.ps1` — connect from browser, confirm sign + non-neutral emotion are both updating.
7. Record numbers in README §Performance Targets table (val acc, latency, round-trip).

## Out of scope (but flagged)

- Cross-signer testing (plan §0.3 CRITICAL) — still a manual Phase 3 step, not code.
- WebSocket heartbeat / ngrok long-session stability — add only if viva is remote.
- 10-min stability memory log — Phase 3 Task 3.2, not part of this PR.