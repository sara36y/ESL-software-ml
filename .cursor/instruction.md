# ESL Real-Time Sign Language Recognition — Cursor Instructions
> Revised 10-hour sprint plan · 2–3 days · 5 people · CPU-only · no GPU required
> Drop this file at the project root. In Cursor, this file acts as persistent context for every AI interaction.

**Note:** Canonical Python code lives under `src/`. The `output/` tree is an optional **compatibility mirror** for Colab handoff (`output/artifacts/` + `output/src/inference.py` re-exports). Prefer `artifacts/` at repo root — see [artifacts/README.md](artifacts/README.md).

---

## What Cursor needs to know about this project

This is a graduation project: a real-time Egyptian Sign Language (ESL) recognition system.

**Phase 1 is already complete.** The Phase 1 notebook produced:
- `output/artifacts/model_v2.keras` — primary MLP model (augmented + emotion input)
- `output/artifacts/label2idx.json` — class index mapping
- `output/src/inference.py` — ready-to-import `predict_frame()` function

**Phase 2–4 are what we are building now**, simplified for a 10-hour sprint.

---

## Architecture decisions — always apply these

| Decision | Choice | Do NOT suggest the alternative |
|---|---|---|
| Emotion integration | Pass `"neutral"` always | Do NOT add DeepFace — model trained with neutral placeholder, zero benefit |
| Threading | Single main loop | Do NOT add threads unless FPS < 10 and you've measured it |
| Primary model | `model_v2.keras` (MLP) | Do NOT use LSTM for demo — 2s buffer lag kills the live demo |
| Web server | FastAPI + WebSocket | Do NOT use Streamlit — `st.camera_input` is snapshot-only, not real-time |
| PWA | `manifest.json` only | Do NOT implement `sw.js` service worker — not needed for viva |
| UI | OpenCV window | Do NOT redesign the UI — functionality only |

### The interface contract — never change this

```python
predict_frame(frame: np.ndarray, cached_emotion: str = "neutral") -> tuple[str, float, str]
# returns: (label, confidence, emotion_str)
# label:      predicted sign class name (or "__no_hands__" if no hands)
# confidence: softmax probability in [0.0, 1.0]
# emotion_str: the cached_emotion passed through unchanged
```

All code must call `predict_frame()` from `src/inference.py`. Never call the model directly.

---

## File structure — where everything lives

```
project/
├── CURSOR_INSTRUCTIONS.md     ← this file
├── output/
│   ├── artifacts/
│   │   ├── model_v2.keras     ← PRIMARY MODEL — use this
│   │   ├── model_v1.keras     ← ablation only
│   │   ├── model_v3.keras     ← ablation only
│   │   └── label2idx.json     ← class index map
│   ├── src/
│   │   └── inference.py       ← predict_frame() lives here — DO NOT EDIT
│   └── results/
│       ├── confusion_matrix.png
│       ├── training_curves.png
│       └── sign_types.csv
├── demo.py                    ← desktop app entry point (build this)
├── web/
│   ├── server.py              ← FastAPI WebSocket server (build this)
│   └── index.html             ← browser frontend (build this)
├── run_demo.sh                ← activate venv + python demo.py
├── run_web.sh                 ← activate venv + uvicorn web.server:app
└── requirements.txt           ← includes fastapi, uvicorn[standard]
```

---

## What to build — task list by owner

### Task 1 — `demo.py` (Team Lead: Abdullah)
**Status:** Not started  
**Time budget:** 2 hours  
**Definition of done:** OpenCV window opens, webcam shows, activation gate works, predictions appear with confidence >= 0.65, 5 consecutive correct signs confirmed

Build a single-file, single-loop desktop app. Do NOT add threading unless asked explicitly.

```python
# Skeleton — fill in the gaps
import cv2, numpy as np, argparse
from collections import deque
from output.src.inference import predict_frame

VELOCITY_THRESHOLD = 0.02   # tune if signs don't trigger: try 0.01
WINDOW_SIZE        = 5
MIN_VOTES          = 3
MIN_CONFIDENCE     = 0.65
EMOTION            = "neutral"  # always neutral — no DeepFace

window  = deque(maxlen=WINDOW_SIZE)
lm_prev = None   # for velocity gate

def smooth(lbl, conf):
    window.append((lbl, conf))
    if len(window) < WINDOW_SIZE:
        return None, 0.0
    labels = [p[0] for p in window]
    confs  = [p[1] for p in window]
    top = max(set(labels), key=labels.count)
    if labels.count(top) >= MIN_VOTES and np.mean(confs) >= MIN_CONFIDENCE:
        return top, float(np.mean(confs))
    return None, 0.0

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=0)
args = parser.parse_args()

cap = cv2.VideoCapture(
    int(args.source) if str(args.source).isdigit() else args.source
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lbl, conf, _ = predict_frame(frame, EMOTION)

    # Status logic
    if lbl == "__no_hands__":
        status = "No hand detected — move closer"
    elif conf < MIN_CONFIDENCE:
        status = "Detecting..."
    else:
        committed, avg_conf = smooth(lbl, conf)
        if committed:
            status = f"Sign: {committed}  ({avg_conf:.0%})"
        else:
            status = "Detecting..."

    # Overlays — add FPS counter, status bar, emotion line
    cv2.putText(frame, status,          (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(frame, f"Emotion: {EMOTION}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.imshow("ESL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        pass  # TODO: toggle landmark overlay

cap.release()
cv2.destroyAllWindows()
```

**Activation gate — wire this in after getting the raw feature vector:**
```python
def activation_gate(feat_126: np.ndarray) -> bool:
    """feat_126: first 126 values of the raw landmark array (both hands only)."""
    global lm_prev
    if lm_prev is None:
        lm_prev = feat_126.copy()
        return True
    velocity = np.linalg.norm(feat_126 - lm_prev)
    lm_prev  = feat_126.copy()
    return velocity >= VELOCITY_THRESHOLD
```

> Note for Cursor: `inference.py` exposes `_extract(frame)` internally — you can call it to get the raw 156-dim feature before normalisation, then check the first 126 values for velocity. Or add a `get_raw_landmarks(frame)` helper to `inference.py` that returns the raw array.

---

### Task 2 — OpenCV overlays (Software Member 1)
**Status:** Not started  
**Time budget:** 1 hour  
**Definition of done:** All 4 UI states displayed correctly, L/D/Q keys work, FPS shown

Add to `demo.py` display layer:

| Element | Code hint |
|---|---|
| FPS counter | `deque(maxlen=30)` of `time.perf_counter()` timestamps, compute rolling FPS |
| Status bar | Bottom of frame: `cv2.rectangle` background + `cv2.putText` |
| Landmark skeleton | `mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)` |
| L key | Toggle `show_landmarks` bool |
| D key | Toggle `debug_mode` bool — show raw label + conf every frame |
| Q key | Break main loop |

4 status states (match exactly):
1. `"No hand detected — move closer"` — gray text
2. `"Ready — show a sign"` — white text (hands still, below velocity threshold)  
3. `"Detecting..."` — yellow text  
4. `"Sign: HELLO  (89%)"` — green text, large font

---

### Task 3 — FastAPI WebSocket server (Software Member 2)
**Status:** Not started  
**Time budget:** 1.5 hours  
**Definition of done:** `/health` returns 200, 5 signs predicted correctly via browser

**File:** `web/server.py`

```python
import base64, cv2, numpy as np, json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from output.src.inference import predict_frame

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data  = await websocket.receive_text()
            img   = np.frombuffer(base64.b64decode(data), np.uint8)
            frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
            label, conf, emotion = predict_frame(frame, "neutral")
            await websocket.send_text(json.dumps({
                "label":      label,
                "confidence": round(conf, 3),
                "emotion":    emotion
            }))
    except WebSocketDisconnect:
        pass  # client disconnected — do not crash server
```

Run with: `uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload`

**Stub for frontend-first development** (use this while real model loads):
```python
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.receive_text()
        await websocket.send_text(
            json.dumps({"label": "HELLO", "confidence": 0.91, "emotion": "neutral"})
        )
```

---

### Task 4 — HTML/JS frontend (Software Member 1)
**Status:** Not started  
**Time budget:** 1 hour  
**Definition of done:** Works in desktop Chrome, predictions visible, mobile layout tested in DevTools

**File:** `web/index.html` — single file, no build toolchain, no npm, plain HTML + CSS + JS

Key implementation points:
```javascript
// Camera setup
navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
  .then(stream => { video.srcObject = stream; });

// Frame capture loop (every 100ms)
setInterval(() => {
    ctx.drawImage(video, 0, 0, 320, 240);
    const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(base64);
}, 100);

// WebSocket with auto-reconnect
function connect() {
    ws = new WebSocket('ws://localhost:8000/ws');
    ws.onmessage = (event) => {
        const { label, confidence, emotion } = JSON.parse(event.data);
        // update overlay
    };
    ws.onclose = () => setTimeout(connect, 2000);
}
connect();
```

Status states — match exactly with desktop app:
1. Connecting...
2. Waiting — no hand detected
3. Detecting...
4. Sign: HELLO (89%) + confidence bar

---

### Task 5 — Evaluation (AI Member 2)
**Status:** Not started  
**Time budget:** 1.5 hours  
**Definition of done:** `results/confusion_matrix.png` saved, ablation table CSV ready

Run this after the Phase 1 notebook has completed (models are trained, val set is available):

```python
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# These come from the Phase 1 notebook scope:
# model_v1, model_v2, model_v3, X_mlp_val, X_lstm_val, y_val, class_names

y_pred_v2 = np.argmax(model_v2.predict(X_mlp_val), axis=1)
print(classification_report(y_val, y_pred_v2, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_v2)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues')
plt.title("model_v2 — validation set")
plt.tight_layout()
plt.savefig("output/results/confusion_matrix.png", dpi=100)

# Top 3 confused pairs
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
for idx in np.argsort(cm_no_diag.flatten())[-3:][::-1]:
    r, c = divmod(idx, len(class_names))
    print(f"Confused: {class_names[r]} → {class_names[c]}  ({cm_no_diag[r,c]} times)")

# Ablation table
rows = []
for model, name, X in [(model_v1, "v1 baseline MLP", X_mlp_val),
                        (model_v2, "v2 aug MLP+emo",  X_mlp_val),
                        (model_v3, "v3 Bi-LSTM+emo",  X_lstm_val)]:
    pred  = np.argmax(model.predict(X), axis=1)
    acc   = (pred == y_val).mean()
    rows.append({"model": name, "val_accuracy": round(acc, 4)})
pd.DataFrame(rows).to_csv("output/results/ablation_table.csv", index=False)
print(pd.DataFrame(rows))
```

---

### Task 6 — Run scripts + packaging (Software Member 2)
**Status:** Not started  
**Time budget:** 1 hour  

**`run_demo.sh`:**
```bash
#!/bin/bash
source venv/bin/activate
python demo.py "$@"
```

**`run_web.sh`:**
```bash
#!/bin/bash
source venv/bin/activate
uvicorn web.server:app --host 0.0.0.0 --port 8000
```

**`requirements.txt` — must include:**
```
mediapipe==0.10.20
tensorflow
opencv-python
numpy
scipy
fastapi
uvicorn[standard]
python-multipart
seaborn
matplotlib
scikit-learn
```

**One-command test (clean machine verification):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run_demo.sh --source 0     # should open webcam window
```

---

## Configuration constants — never change these

These values are set by Phase 1 and must match exactly between training and inference:

```python
FACE_IDX        = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
FEATURE_DIM     = 156       # 63 (left hand) + 63 (right hand) + 30 (face)
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NEUTRAL_IDX     = 4
EMOTION_DIM     = 7
N_FRAMES        = 30        # LSTM only
INPUT_DIM_MLP   = 163       # 156 landmarks + 7 emotion
```

---

## Normalisation — must match training exactly

```python
def normalize(raw: np.ndarray) -> np.ndarray:
    f = raw.astype(np.float64)
    l  = f[0:63].reshape(21, 3).copy()
    r  = f[63:126].reshape(21, 3).copy()
    fc = f[126:].reshape(-1, 3).copy()
    for seg in (l, r):
        if seg.any():
            seg -= seg[0]
            s = np.max(np.linalg.norm(seg, axis=1))
            seg /= s if s > 0 else 1.0
    if fc.any():
        fc -= fc.mean(axis=0)
        s = np.max(np.linalg.norm(fc, axis=1))
        fc /= s if s > 0 else 1.0
    return np.concatenate([l.flatten(), r.flatten(), fc.flatten()]).astype(np.float32)
```

> If Cursor suggests a different normalisation: reject it. This exact implementation is in `inference.py` and must match what was used during training.

---

## What NOT to do — common AI assistant mistakes on this project

1. **Do not suggest adding DeepFace** — the model was trained with neutral placeholder. DeepFace provides zero accuracy improvement and risks crashing the demo.
2. **Do not suggest threading** unless explicitly asked and FPS has been measured below 10.
3. **Do not modify `inference.py`** — it is the AI team's deliverable and the interface contract. Add helper functions in separate files.
4. **Do not use `st.camera_input` or Streamlit** for the real-time video feed — it is snapshot-only.
5. **Do not load `model_v1.keras` or `model_v3.keras`** for the demo — `model_v2.keras` is the primary model.
6. **Do not change `EMOTION_CLASSES` order** — this breaks the one-hot encoding.
7. **Do not split data at frame level** — always video-level. This is a known data leakage issue already fixed.
8. **Do not use relative imports** like `from .inference import` in `demo.py` — use `from output.src.inference import predict_frame` or adjust `sys.path`.

---

## Viva Q&A — answers the whole team must know

**"Why no real emotion detection?"**
> We designed the architecture to support emotion integration — the 7-dim feature is in the model input. We used a neutral placeholder during training because our dataset has no emotion ground truth. Shipping a DeepFace integration that provides no accuracy improvement (since the model hasn't been trained to use live emotion) would be misleading. The production upgrade path is: collect emotion-labeled data → retrain model_v4 → swap in live DeepFace output. Zero changes to the architecture.

**"Why MLP not LSTM for the live demo?"**
> We train and evaluate both. The LSTM requires a 30-frame buffer before making any prediction — 2 seconds of latency. For a live demo this makes the system feel broken. The MLP predicts from a single frame and the sliding window handles temporal smoothing. We use the appropriate model for the use case. See the ablation table: both models are evaluated and compared.

**"Why landmark-based over raw CNN?"**
> 10 videos per class. A CNN needs 10,000+. MediaPipe extracts 156-dim features that are background-invariant, lighting-invariant, and run in under 5ms on CPU. The model is 50KB vs 14MB for MobileNet. Training takes minutes not hours. For our dataset size and deployment constraints, landmarks dominate on every axis.

**"What is the data leakage bug you found?"**
> Frame-level splitting lets frames from the same video appear in both train and val. Since frames from the same video share the signer's proportions, the model memorises the person not the sign — inflated accuracy. We split at video level: 80% of videos per class to train, 20% to val. No video appears in both sets.

**"Why FastAPI WebSocket not REST?"**
> REST would require polling from the browser — either slow (1 req/sec) or expensive (10 req/sec). WebSocket is a persistent connection. The browser sends a frame and immediately receives a response. Round-trip under 150ms on localhost. REST polling at the same frequency would add connection overhead per frame.

---

## Sprint timeline

| Day | Hours | Milestone |
|-----|-------|-----------|
| Day 1 | 1–2h | Phase 1 notebook runs on Colab → model artifacts delivered to shared folder |
| Day 1 | 2–3h | `demo.py` skeleton running with stub, webcam window open |
| Day 1 | 3–4h | FastAPI stub server running, `/health` returns 200 |
| Day 2 | 1–2h | Real `inference.py` wired into `demo.py`, 5 signs correct live |
| Day 2 | 2–3h | Browser frontend showing predictions from real model |
| Day 2 | 3–4h | Confusion matrix + ablation table delivered |
| Day 3 | 1–2h | Viva slides complete (10 slides) |
| Day 3 | 2–3h | Demo videos recorded, clean machine test passed |
| Day 3 | 3h   | Pre-viva checklist complete, repo tagged `v1-final` |

---

## Pre-viva checklist

- [ ] `demo.py` runs on clean machine from `run_demo.sh` alone  
- [ ] Web app runs from `run_web.sh`, predictions visible in browser  
- [ ] `model_v2.keras` (not v1) loads without errors  
- [ ] Activation gate ON — no predictions when hands are still  
- [ ] Confidence threshold ON — "Detecting..." shown below 0.65  
- [ ] `"neutral"` always passed for emotion — no DeepFace crash risk  
- [ ] `results/confusion_matrix.png` saved  
- [ ] Ablation table values in viva slides  
- [ ] Top 3 confused pairs documented with one-sentence explanations  
- [ ] `demo_desktop.mp4` recorded (5 signs + 1 deliberate failure shown honestly)  
- [ ] `demo_web.mp4` recorded (browser showing live predictions)  
- [ ] At least one team member NOT in training videos has tested the system  
- [ ] All 5 members reviewed the viva slides  
- [ ] Repo tagged `v1-final`, deliverables zip created  
- [ ] `requirements.txt` includes `fastapi`, `uvicorn[standard]`, `mediapipe==0.10.20`  
- [ ] Team can answer the 5 viva questions above without reading notes  

---

*Last updated: Revised sprint plan · simplified architecture · CPU-only · 10-hour budget*
