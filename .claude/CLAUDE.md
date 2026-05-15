# ESL Real-Time Sign Language Recognition — Project Context

## Project Overview

Real-time Egyptian Sign Language (ESL) recognition system that runs locally on a laptop/PC with a webcam,
**plus** a Progressive Web App (PWA) layer reachable from a phone browser over Wi-Fi.

The system watches a person sign and displays the recognised sign as text — no human interpreter, no cloud, no internet required for the desktop version.

**Team:** 5 members — 1 team lead (also does AI work), 2 AI members, 2 software members
**Timeline:** 4 phases, ~87 hours total (including 8-hour buffer)
**Status:** Phase 1 mostly complete. Two required additions (1A, 1B, 1C) remain before Phase 2 starts.

---

## ⚠️ Critical Risks — Read First

| Priority | Risk | Consequence If Ignored | Fix |
|----------|------|----------------------|-----|
| **CRITICAL** | Activation gate skipped "for now" | Demo shows random predictions constantly — kills viva | Task 2.2 must be done before first demo run |
| **CRITICAL** | No cross-signer testing | Model learned one person's hand shape, not the sign | Record 2–3 extra videos with a 2nd person in Phase 1B |
| **HIGH** | GPU-less exam environment | FPS drops below 10 on examiner's laptop | TFLite export in Task 2.6 is mandatory, not optional |
| **HIGH** | Emotion ground truth absent at training time | Neutral placeholder may reduce emotion feature value | Document this limitation explicitly in ablation table and viva |
| **MEDIUM** | No error handling in threading scaffold | One DeepFace timeout crashes the display thread | Wrap all DeepFace calls in try/except, fall back to "neutral" |
| **MEDIUM** | Stability test is only 10 minutes | Memory grows during live viva demo | Run overnight test once, log memory every 5 minutes |
| **LOW** | Git workflow depends on team discipline | Merge conflicts on demo day | Set branch protection on main in GitHub settings today |

---

## What the System Does

1. Webcam captures live video
2. MediaPipe Holistic extracts hand/face landmarks every frame
3. Activation gate checks hand velocity — skips inference when hands are still
4. Sign classifier (MLP or LSTM) predicts sign from normalised landmarks
5. DeepFace runs every 5 frames on face crop to detect dominant emotion
6. Emotion is concatenated as 7-dim one-hot vector to landmark feature before prediction
7. Sliding window (last 5 frames, majority vote) smooths predictions
8. Desktop result: `"Sign: HELLO (0.89)  ·  Emotion: happy"`
9. Web result: same JSON `{ label, confidence, emotion, fps }` sent over WebSocket to browser

**Primary deliverable:** Local Python desktop app (OpenCV window).
**Secondary deliverable:** FastAPI WebSocket server + HTML/JS PWA frontend.

---

## Runtime Architecture

### Desktop — Thread Model

| Thread | Job |
|--------|-----|
| Thread 1 — Capture | `cv2.VideoCapture(0)`, pushes to `queue.Queue(maxsize=2)`, drops old frames |
| Thread 2 — Inference | MediaPipe → activation gate → normalise → `model.predict()` → result queue |
| Thread 3 — Emotion | `DeepFace.analyze()` every K=5 frames, caches result behind `threading.Lock` |
| Main thread — Display | Reads result + cached emotion, overlays on frame, `cv2.imshow` |

### Web — Component Model

| Component | Technology | Responsibility |
|-----------|-----------|---------------|
| WebSocket Server | FastAPI + uvicorn | Receives JPEG frames from browser, calls `predict_frame()`, returns JSON |
| Inference Backend | `src/inference.py` (unchanged) | Same `predict_frame()` used by desktop — zero changes to AI pipeline |
| Frontend | Plain HTML + Vanilla JS | Camera capture, frame encoding, WebSocket comms, canvas overlay |
| PWA Layer | `manifest.json` + service worker | Makes site installable on Android/iOS home screen over Wi-Fi |
| HTTPS (mobile) | ngrok tunnel | Required by `getUserMedia()` on phone browsers — one command, no config |

### Interface Contract (between AI and software teams)

```python
predict_frame(frame: np.ndarray) -> tuple[str, float, str]
# returns: (label, confidence, emotion_str)
# stub: return ("HELLO", 0.91, "happy")
```

Software members build the entire UI against the stub from day one. AI members build the real model behind it. **This contract never changes — the web layer calls the exact same function.**

---

## Dataset

- 55 sign classes, 10 videos per class
- MVP vocabulary: 8–12 signs chosen for first-pass training
- Train/val split: **video-level 80/20** (never frame-level — that causes data leakage)
- Landmark format: MediaPipe Holistic — 21 keypoints per hand (42 total), plus optional face

### On Pre-Extracted Landmarks

**Golden rule:** Training landmarks and live inference landmarks must come from the same extractor with the same keypoint count and coordinate system. Break this and the model silently predicts wrong answers.

| Scenario | Situation | Action |
|----------|-----------|--------|
| A | Dataset has `.npy`/`.csv` landmarks confirmed from MediaPipe Holistic | Skip re-extraction for training. Still need MediaPipe at inference time. |
| B | Already ran MediaPipe extraction in Phase 1 | Nothing to do. |
| C | Landmarks from unknown/different tool (OpenPose, etc.) | Re-extract with MediaPipe. 3–4 hrs is cheaper than debugging silent failures. |

Spot-check: load 5–10 files, confirm shape (21 joints per hand), values in 0.0–1.0 range.

---

## Models

### Landmark Normalisation (required before any model input)

```python
# 1. Wrist-centred translation
lm -= lm[0]  # wrist = (0,0,0)
# 2. Scale normalisation
lm /= np.max(np.linalg.norm(lm, axis=1))
# 3. Flatten to 1-D vector
features = lm.flatten()  # shape: (126,) for 2 hands × 21 pts × 3 dims
```

### MLP (for static signs)

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(D,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, callbacks=[EarlyStopping(patience=5)])
```

### LSTM (for dynamic signs)

```python
model = Sequential([
    LSTM(64, input_shape=(N_frames, D_per_frame), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

### Emotion Input (required for model_v2 onwards)

Concatenate a 7-dim one-hot vector to landmark features before prediction.
Classes (order is fixed — must match training exactly): `angry, disgust, fear, happy, neutral, sad, surprise`
At training time (no ground-truth emotion labels): use placeholder `neutral` (index 4) for all samples.

```python
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def emotion_to_onehot(dominant_emotion: str) -> np.ndarray:
    vec = np.zeros(7)
    idx = EMOTION_CLASSES.index(dominant_emotion) if dominant_emotion in EMOTION_CLASSES else 4
    vec[idx] = 1.0
    return vec

# At inference:
features = np.concatenate([landmark_features, emotion_to_onehot(cached_emotion)])
```

### Saved Model Naming

| File | Description |
|------|-------------|
| `artifacts/model_v1.h5` | Baseline MLP, no augmentation |
| `artifacts/model_v2.h5` | Augmented MLP + emotion concat (primary demo model) |
| `artifacts/model_v3.h5` | Augmented LSTM + emotion concat |

Keep all variants — they are the ablation table.

---

## Data Augmentation (Phase 1A — required before Phase 2)

Apply to training set only. **Never augment validation or test sets.**

```python
# 1. Horizontal flip — valid for most ESL signs
lm_flip = lm.copy()
lm_flip[:, 0] = 1 - lm_flip[:, 0]

# 2. Temporal speed jitter (80% and 120% speed)
from scipy.interpolate import interp1d
def resample_sequence(seq, factor):
    N = len(seq)
    old_t = np.linspace(0, 1, N)
    new_N = int(N * factor)
    new_t = np.linspace(0, 1, new_N)
    f = interp1d(old_t, seq, axis=0)
    return f(new_t)

# 3. Gaussian noise — simulates hand tremor
lm_noisy = lm + np.random.normal(0, 0.005, lm.shape)

# 4. Landmark dropout — simulates occlusion (hand joints only, not wrist)
mask = np.random.rand(*lm.shape[:1]) < 0.1
lm_drop = lm.copy()
lm_drop[mask, 1:] = 0
```

Target: ~2,200+ training sequences from original ~440 (5× multiplier).

---

## Activation Gate (Phase 2 — non-negotiable)

Without this, the model predicts a class every frame even when the signer is resting. The demo shows a constant stream of wrong predictions. This single feature is worth two weeks of accuracy tuning in terms of demo quality.

```python
VELOCITY_THRESHOLD = 0.02  # tune empirically

lm_prev = None

def activation_gate(lm_current):
    global lm_prev
    if lm_prev is None:
        lm_prev = lm_current
        return False
    velocity = np.linalg.norm(lm_current - lm_prev)
    lm_prev = lm_current
    return velocity >= VELOCITY_THRESHOLD
```

UI states (must implement all four):
- No hand detected → `"No hand detected — move closer"`
- Hand still (below threshold) → `"Ready — show a sign"`
- Hand moving (above threshold) → run classifier → `"Detecting..."`
- Confident prediction committed → `"Sign: HELLO (0.89)"`

---

## Sliding Window Smoothing (Phase 2)

```python
from collections import deque

WINDOW_SIZE = 5
MIN_VOTES = 3
MIN_CONFIDENCE = 0.65

prediction_window = deque(maxlen=WINDOW_SIZE)

def smooth_prediction(label, confidence):
    prediction_window.append((label, confidence))
    if len(prediction_window) < WINDOW_SIZE:
        return None, 0.0
    labels = [p[0] for p in prediction_window]
    confs  = [p[1] for p in prediction_window]
    top_label = max(set(labels), key=labels.count)
    if labels.count(top_label) >= MIN_VOTES and np.mean(confs) >= MIN_CONFIDENCE:
        return top_label, np.mean(confs)
    return None, 0.0
```

---

## Emotion Confidence Modifier

```python
EMOTION_CONFLICTS = {
    ('happy', 'angry'): 0.75,
    ('happy', 'disgust'): 0.75,
    ('sad', 'happy'): 0.75,
}

def apply_emotion_modifier(label, confidence, emotion):
    key = (label.lower(), emotion.lower())
    modifier = EMOTION_CONFLICTS.get(key, 1.0)
    return confidence * modifier
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Display FPS | ≥ 15 FPS |
| Inference resolution | 320×240 (resize before MediaPipe) |
| DeepFace frequency | Every K=5 frames |
| Confidence threshold | ≥ 0.65 to display prediction |
| Model size (MLP) | ~50 KB |
| Web round-trip latency | < 150ms on localhost |

If FPS < 10: export to TFLite — this is **mandatory**, not optional.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('artifacts/model_v2.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Evaluation (Phase 3)

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Per-class metrics
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)
plt.savefig('results/confusion_matrix.png')
```

Ablation table required columns: model name, accuracy, macro F1, inference latency (ms mean ± std).

For the top 3 confused pairs: visualise their landmark sequences side by side and explain **why** they are visually similar. This is the strongest part of your results in the viva.

---

## Phase 4 — Web App Architecture Detail

### FastAPI WebSocket Server (web/server.py)

```python
import base64, cv2, numpy as np, json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.inference import predict_frame

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'])

@app.websocket('/ws')
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(data), np.uint8),
                cv2.IMREAD_COLOR
            )
            label, conf, emotion = predict_frame(frame)
            await websocket.send_text(
                json.dumps({"label": label, "confidence": conf, "emotion": emotion})
            )
    except WebSocketDisconnect:
        pass  # client disconnected — do not crash
```

**Add `/health` GET endpoint** returning `{ "status": "ok" }` — used to verify server is running before demo.

### Frontend Frame Capture Loop (web/index.html)

```javascript
// Every 100ms: capture → encode → send → receive → overlay
setInterval(() => {
    ctx.drawImage(video, 0, 0, 320, 240);
    const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    ws.send(base64);
}, 100);

ws.onmessage = (event) => {
    const { label, confidence, emotion } = JSON.parse(event.data);
    // update overlay canvas
};
```

**Auto-reconnect on disconnect:**
```javascript
function connect() {
    ws = new WebSocket('ws://localhost:8000/ws');
    ws.onclose = () => setTimeout(connect, 2000);  // exponential backoff
}
```

### Stub WebSocket Server for Frontend Dev (parallel work)

Software Member 1 can start frontend on day one of Phase 4 using this stub — no real model needed:

```python
from fastapi import FastAPI, WebSocket
import asyncio, json

app = FastAPI()

@app.websocket('/ws')
async def ws(websocket):
    await websocket.accept()
    while True:
        await websocket.receive_text()
        await websocket.send_text(
            json.dumps({"label": "HELLO", "confidence": 0.91, "emotion": "happy", "fps": 18})
        )
```

### ngrok for Mobile HTTPS

`getUserMedia()` requires HTTPS on all browsers. `localhost` is the only exception. For phone testing:

```bash
ngrok http 8000
# copy: https://xxxxx.ngrok-free.app
# update WebSocket URL in frontend to: wss://xxxxx.ngrok-free.app/ws
```

---

## File Structure

```
project/
├── artifacts/
│   ├── model_v1.h5           # baseline MLP
│   ├── model_v2.h5           # augmented + emotion (primary)
│   ├── model_v2.tflite       # TFLite export (if FPS < 10)
│   └── model_v3.h5           # LSTM variant
├── data/
│   ├── landmarks/            # .npy files, shape (num_frames, feature_dim)
│   └── augmented/            # augmented sequences
├── results/
│   └── confusion_matrix.png
├── src/
│   ├── inference.py          # predict_frame() — owned by AI Member 1
│   ├── mediapipe_utils.py
│   ├── augmentation.py       # owned by AI Member 1
│   └── evaluate.py           # owned by AI Member 2
├── web/
│   ├── server.py             # FastAPI WebSocket server — owned by Software Member 2
│   ├── index.html            # PWA frontend — owned by Software Member 1
│   ├── manifest.json         # PWA manifest
│   └── sw.js                 # service worker
├── demo.py                   # desktop app entry point
├── run_demo.sh               # activate venv + python demo.py
├── run_web.sh                # activate venv + uvicorn web.server:app
└── requirements.txt          # includes fastapi, uvicorn[standard]
```

---

## Team Roles — Complete Assignment Table

### Team Lead (Abdullah)

| Phase | Task | What You Own | Definition of Done |
|-------|------|-------------|-------------------|
| 1 | 1B review | Review & merge AI Member 1 PR, run final model comparison | All 4 model variants in `artifacts/`, ablation numbers recorded |
| 2 | 2.1 Threading scaffold | Write Thread 1–3 + main, code review all PRs | App runs at 15+ FPS, no crashes for 5 minutes |
| 2 | 2.2 Activation gate | Implement and tune velocity threshold | No predictions shown when hands are still |
| 2 | 2.3 Emotion fusion | Implement live emotion → one-hot → concat pipeline | Emotion updates every 5 frames, correct encoding confirmed |
| 2 | 2.6 Performance tuning | Profile bottleneck, TFLite if needed, benchmark 100 frames | Mean inference latency documented in README |
| 3 | 3.4 Viva slide deck (9 slides) | Own all slides; gather results content from AI Member 2 | Reviewed by all 5 members before viva |
| 4 | 4.5 Integration test | End-to-end test of web app, 10 signs, round-trip latency | < 150ms, auto-reconnect working, mobile test passing |
| 4 | 4.8 Viva slide update | Add web/mobile slide, update future work slide | 10-slide deck complete |

### AI Member 1 — Model & Inference Pipeline

| Phase | Task | What You Own | Definition of Done |
|-------|------|-------------|-------------------|
| 1 | 1A Augmentation | `src/augmentation.py` with all 4 augmentation types | ~2,200+ training sequences confirmed in logs |
| 1 | 1B Retrain | `model_v1.h5` through `v3.h5`, training logs saved | All models load, predict correctly on 3 test signs |
| 2 | `src/inference.py` | `predict_frame()`, normalisation, feature concat | Stub returns correct types; real version matches output format |
| 2 | TFLite export (if FPS < 10) | `artifacts/model_v2.tflite` | TFLite matches Keras accuracy within 1% |
| 3 | Latency benchmarking | 100-frame timing test, mean ± std in ms | Numbers in README and ablation table slide |

### AI Member 2 — Evaluation & Results

| Phase | Task | What You Own | Definition of Done |
|-------|------|-------------|-------------------|
| 1 | 1C Static/dynamic labeling | Table in README: each MVP sign labeled Static or Dynamic | All 8–12 signs categorised, rationale documented |
| 2 | Ablation study (`src/evaluate.py`) | Run all model variants on held-out test set | Accuracy + macro F1 + latency for v1, v2, v3 in CSV |
| 2 | Confusion matrix | `results/confusion_matrix.png` (seaborn heatmap) | File saved, top 3 confused pairs identified |
| 2 | Failure analysis | Landmark visualisations of top 3 confused pairs side by side | Visual explanation of each confusion, ready for viva |
| 3 | Results slide content | Ablation table values + confusion matrix image delivered to Team Lead | Delivered 2 days before viva |

### Software Member 1 — UI / Display / Frontend

| Phase | Task | What You Own | Definition of Done |
|-------|------|-------------|-------------------|
| 2 | 2.5 OpenCV overlays | Status bar, L/D/Q key toggles, FPS counter | All 4 UI states shown; FPS displayed |
| 3 | 3.3 Desktop demo video (60–90s) | `demo_desktop.mp4` | 5 signs recognised, emotion visible, one deliberate failure |
| 4 | 4.2 HTML/JS frontend | `web/index.html` — camera, canvas overlay, status, FPS, mobile layout | Works on desktop Chrome; mobile layout tested in DevTools |
| 4 | 4.7 Web + mobile demo videos | `demo_web.mp4`, `demo_mobile.mp4` | Both clips < 45s, predictions visible, mobile clip shows phone |

### Software Member 2 — Packaging, Server, Reproducibility

| Phase | Task | What You Own | Definition of Done |
|-------|------|-------------|-------------------|
| 2 | 2.5 Video fallback | `--source` arg in `demo.py`, `run_demo.sh` | `python demo.py --source test.mp4` runs without errors |
| 3 | 3.5 Final packaging | Repo tag `v1-final`, deliverables zip, clean env test | Fresh venv install + demo in under 10 steps |
| 3 | `README.md` | Full README: install, run desktop, run web, known issues | Another team member can set up from scratch using only README |
| 4 | 4.1 FastAPI server | `web/server.py`, WebSocket, CORS, health check, disconnect handling | `/health` returns 200; 5 signs predicted end-to-end |
| 4 | 4.3 PWA manifest + service worker | `web/manifest.json`, `web/sw.js`, app icon | "Add to Home Screen" prompt appears on Android Chrome via ngrok |
| 4 | 4.4 Mobile test via ngrok | ngrok setup, mobile test, README ngrok section | 2 signs predicted correctly on physical phone |
| 4 | 4.6 `run_web.sh` + requirements update | `run_web.sh`, `fastapi`/`uvicorn[standard]` in requirements | Server starts in under 5 seconds on clean install |

---

## Hour Estimate (Revised)

| Phase | Original | Revised | Status |
|-------|---------|---------|--------|
| Phase 1 — Data & Model | ~22 hrs | ~22 hrs | Mostly done |
| Phase 2 — Real-Time Integration | ~24 hrs | ~28 hrs | Not started |
| Phase 3 — Polish, Test & Deliver | ~12 hrs | ~14 hrs | Not started |
| Phase 4 — Web App (NEW) | — | ~15 hrs | Not started |
| Buffer | — | ~8 hrs | Non-negotiable |
| **TOTAL** | **~54 hrs** | **~87 hrs** | ~17.4 hrs/person |

**Where the original estimate is optimistic:**
- Task 2.1 (threading): 3 hrs estimated → likely 4–5 hrs for first-timers. Race conditions are subtle.
- Task 3.2 (stability): 2 hrs estimated → add 2–3 hrs if memory leaks appear.
- Task 3.4 (viva slides): 2 hrs estimated → underestimated if results are not compiled first.

---

## Viva Slide Structure (10 slides after Phase 4)

| # | Slide | Owner |
|---|-------|-------|
| 1 | Problem statement + motivation (1.5M deaf/HoH in Egypt) | Team Lead |
| 2 | Dataset — 55 classes, 10 videos/class, MVP vocabulary rationale | AI Member 2 |
| 3 | System architecture diagram | Team Lead |
| 4 | Model design — MLP vs LSTM, static/dynamic sign split | AI Member 1 |
| 5 | Ablation table — baseline vs augmented vs emotion-fused | AI Member 2 |
| 6 | Confusion matrix + failure analysis (top 2–3 confused pairs) | AI Member 2 |
| 7 | Real-time system — threading, FPS, latency numbers | Team Lead |
| 8 | Live demo or demo video links (desktop + web + mobile) | Software Member 1 |
| 9 | Web & mobile interface — FastAPI + browser + PWA architecture | Team Lead |
| 10 | Limitations + future work | Team Lead |

**Key viva answers to prepare:**
- "Why LSTM over MLP?" → Because [X, Y] signs are dynamic — motion encodes meaning. Static signs use MLP. We categorised all 12 MVP signs.
- "Why landmark-based over raw CNN?" → Small dataset (10 videos/class), CPU-only demo, background invariance, 5-minute training vs hours.
- "What are your limitations?" → Emotion placeholder at training time. Single signer in training videos. MVP vocabulary only (8–12 of 55 classes). Point to the confusion matrix.
- "Why not a mobile app?" → We have a PWA — installable on Android/iOS home screen via the same codebase, no app store required.

---

## Key Decisions Already Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Desktop UI | OpenCV window | `st.camera_input` is snapshot-only, not real-time |
| Web UI | Plain HTML + Vanilla JS | No build toolchain; runs on any browser; deployable in < 1 day |
| Primary model | LSTM for dynamic signs, MLP for static signs | Sign type determines temporal requirement |
| Emotion integration | 7-dim one-hot concat to feature vector | Integrates with model input, not just display |
| Landmark source | MediaPipe Holistic for both training and inference | Must match — this is the golden rule |
| Web backend | FastAPI + WebSockets | Async, lightweight, reuses `predict_frame()` unchanged |
| Mobile delivery | PWA over ngrok (demo), or HTTPS server (production) | No native SDK needed; one codebase |
| Git workflow | Feature branches → PRs → Team Lead merges to `main` | `main` is always demo-ready |

---

## Pre-Viva Checklist (Team Lead Owns — Run 48 Hours Before)

| ☐ | Check | Owner |
|---|-------|-------|
| ☐ | Desktop app runs on a **clean machine** from `run_demo.sh` alone | Software Member 2 |
| ☐ | Web app runs from `run_web.sh`, predictions shown in browser | Software Member 2 |
| ☐ | Mobile demo works on a physical phone via ngrok | Software Member 2 |
| ☐ | `model_v2.h5` (not v1) loads without errors | AI Member 1 |
| ☐ | Activation gate is ON — no predictions when hands are still | Team Lead |
| ☐ | Confidence threshold is ON — "Detecting..." shown below 0.65 | Team Lead |
| ☐ | DeepFace errors are caught — falls back to "neutral", doesn't crash | Team Lead |
| ☐ | Confusion matrix PNG saved in `results/` | AI Member 2 |
| ☐ | Ablation table values filled in viva slides | Team Lead |
| ☐ | Failure analysis slide has landmark visualisations | AI Member 2 |
| ☐ | All 3 demo videos recorded: `demo_desktop.mp4`, `demo_web.mp4`, `demo_mobile.mp4` | Software Member 1 |
| ☐ | At least one team member NOT in training videos has tested the system | Team Lead |
| ☐ | Viva slides reviewed by all 5 members | Team Lead |
| ☐ | Repo tagged `v1-final`, deliverables zip created | Software Member 2 |
| ☐ | `requirements.txt` includes `fastapi` and `uvicorn[standard]` | Software Member 2 |
| ☐ | Overnight memory stability test completed (10+ minutes, memory logged) | Team Lead |

---

## Git Discipline

- `main` is always demo-ready. Nobody commits directly to `main`. Use feature branches: `feat/augmentation`, `feat/threading`, `feat/ui-overlay`, `feat/web-server`.
- Small, frequent commits. "Add horizontal flip augmentation" is a good message. "Fix stuff" is not.
- Team Lead owns final integration. Others make PRs. Team Lead reviews and merges.

---

## Future Work (Know for Viva — Do Not Implement)

- **Sentence-level recognition:** current system recognises isolated signs. Real communication needs temporal segmentation to find sign boundaries in continuous signing, plus a language model to fill gaps.
- **Full 55-class vocabulary:** MVP uses 8–12 signs. Scaling to all 55 requires 50+ videos per class and possibly a hierarchical classifier.
- **Continuous on-device learning:** fine-tune on-device as new signers use it, adapting to individual hand proportions.
- **Native mobile app:** TFLite + MediaPipe Android/iOS SDK. Reaches far more users than the PWA but requires native development — out of scope for this project timeline.
