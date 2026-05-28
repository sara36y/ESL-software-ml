# ESL Real-Time Sign Language Recognition — Master Project Context
> This file is the single source of truth for Claude Code.
> Read this entire file before writing a single line of code or running any command.

---

## Current Project Status (Read This First)

| Item | Status |
|------|--------|
| Phase 1 — Data & Model | ✅ COMPLETE. `artifacts/model_v2.keras` + `label2idx.json` exist. |
| Phase 2 — Desktop App | ✅ COMPLETE. `demo.py` runs with full threading, activation gate, overlays. |
| Phase 3 — Evaluation | ✅ COMPLETE. `src/evaluate.py` ready. Run after confirming val arrays exist. |
| Phase 4 — Web Server | ✅ COMPLETE. `web/server.py` is written and working. |
| Phase 5 — Live Deployment | 🔴 NOT DONE. Server only runs on localhost. Mobile app can't reach it. |
| Phase 6 — CI/CD Pipeline | 🔴 NOT DONE. No auto-redeploy when model improves. |

**The two things blocking the project right now:**
1. The server is not publicly accessible — the mobile app (on teammate's laptop) cannot connect.
2. There is no deployment pipeline — improving the model requires manual steps.

**What to build next, in order:**
1. `scripts/download_model.py` — downloads model from HuggingFace Hub at server startup
2. `Dockerfile` — packages the server for Railway deployment
3. `.github/workflows/deploy.yml` — triggers Railway redeploy on every push to `main`
4. `scripts/promote_model.py` — promotes a new model version to production in one command

---

## The System — Plain English Explanation

Every time someone signs in front of a camera, this chain runs:

```
Camera frame (BGR image)
    ↓
Resize to 320×240 (speed)
    ↓
MediaPipe Holistic extracts 156 numbers
  — 63 numbers: left hand joint positions (21 joints × x,y,z)
  — 63 numbers: right hand joint positions (21 joints × x,y,z)
  — 30 numbers: 10 face landmark positions (× x,y,z)
    ↓
Activation gate: measure how much hands moved since last frame
  — if still → show "Ready — show a sign", skip model
  — if moving → continue
    ↓
Normalize landmarks (wrist → origin, scale by max radial distance)
    ↓
Concatenate 7-dim emotion one-hot → 163-dim feature vector
    ↓
MLP model predicts: outputs probability for each of the sign classes
    ↓
Sliding window: collect last 5 predictions, commit when 3+ agree and mean conf ≥ 0.65
    ↓
Output: ("HELLO", 0.89, "neutral")
```

The desktop app runs this in Thread 2. The web server runs this on every WebSocket frame.
They call the exact same function: `predict_frame()` in `src/inference.py`.

---

## Interface Contract — Never Change This

```python
# src/inference.py
predict_frame(
    frame: np.ndarray,           # BGR image from cv2 or WebSocket decode
    cached_emotion: str | None,  # None = use DeepFace cache; "neutral" = sprint/web mode
    *,
    raw: bool = False,           # True = per-frame label, no sliding window
) -> tuple[str, float, str]
# Returns: (label, confidence, emotion_str)
```

**The web server always calls:** `predict_frame(frame, cached_emotion="neutral")`
**The desktop app calls:** `predict_frame(frame)` (uses DeepFace thread cache)
**The mobile app receives JSON:** `{"label": "HELLO", "confidence": 0.89, "emotion": "neutral", "latency_ms": 45.2}`

The mobile app integration point is `/ws` WebSocket. It sends base64 JPEG frames, receives JSON.

---

## Emotion — The Honest Truth (Important for Viva)

The model was trained with `neutral` emotion placeholder for ALL training samples because we had no emotion ground-truth labels. This means:

- The model learned to mostly ignore the 7 emotion dimensions
- Passing live DeepFace emotion at inference time has minimal effect on accuracy TODAY
- **The architecture is correct** — the slot exists for when we have real emotion labels
- For MVP: always pass `"neutral"`. No DeepFace dependency. No crash risk.

**Viva answer:** "We designed the architecture to support emotion fusion — the 7-dimensional feature is in the model input. We used neutral placeholder during training because our dataset has no emotion ground truth. The production upgrade path is: collect emotion-labeled signing data → retrain model_v4 → swap in live DeepFace output. Zero architecture changes needed."

---

## Critical Constants — Never Change Without Retraining

```python
# src/inference.py — these values are burned into model_v2.keras's weights
FACE_IDX        = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]  # EXACT ORDER FROM PHASE 1
FEATURE_DIM     = 156       # 63 left hand + 63 right hand + 30 face
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_DIM     = 7
NEUTRAL_IDX     = 4         # index of "neutral" in EMOTION_CLASSES
INPUT_DIM_MLP   = 163       # 156 + 7 — what model_v2.keras expects
N_FRAMES        = 30        # LSTM only (model_v3)
VELOCITY_THRESHOLD = 0.02   # tune only if activation gate is too sensitive
WINDOW_SIZE     = 5
MIN_VOTES       = 3
MIN_CONFIDENCE  = 0.65
```

Changing `FACE_IDX` order silently shuffles the model's input and breaks predictions without any error message. This was a previously fixed bug — do not reintroduce it.

---

## Deployment Architecture (Phase 5 — What to Build)

### Current state (broken for mobile)
```
Mobile app (teammate's laptop)  →  ???  →  localhost:8000 (your laptop)
                                                    ↑
                               internet can't see this
```

### Target state (MVP)
```
Mobile app (anywhere)  →  wss://esl-app.up.railway.app/ws
                                        ↓
                          Railway container (your code)
                                        ↓
                          downloads model_v2.keras from HuggingFace Hub
                                        ↓
                          predict_frame() → JSON response
```

### Why this architecture

| Choice | Why |
|--------|-----|
| Railway (not Heroku) | Free tier, Docker support, auto-redeploy from GitHub push |
| HuggingFace Hub (not Git LFS) | model_v2.keras is ~200KB–2MB. GitHub has 100MB file limit. HuggingFace is designed for ML model storage, free, permanent. |
| WebSocket (not REST) | Mobile app sends 10 frames/sec. REST would add connection overhead per frame. WebSocket is one persistent connection — <150ms round-trip. |
| Docker (not buildpack) | MediaPipe needs system libraries (libGL, libglib). Buildpacks don't handle this reliably. Dockerfile gives full control. |

---

## Files to Create for Deployment

### 1. `scripts/download_model.py`

Downloads model from HuggingFace at container startup. Uses `HF_TOKEN` env var (set in Railway dashboard).

```python
"""
scripts/download_model.py
Downloads model artifacts from HuggingFace Hub.
Called by Dockerfile RUN step and can be re-run locally.

Usage:
    python scripts/download_model.py
    HF_TOKEN=xxx python scripts/download_model.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_REPO_ID = os.environ.get("HF_REPO_ID", "YOUR_USERNAME/esl-model")
HF_TOKEN   = os.environ.get("HF_TOKEN")

FILES_NEEDED = [
    ("model_v2.keras",  "artifacts/model_v2.keras"),
    ("label2idx.json",  "artifacts/label2idx.json"),
]

def download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs("artifacts", exist_ok=True)
    for hf_filename, local_path in FILES_NEEDED:
        if os.path.exists(local_path):
            print(f"[download] Already exists: {local_path}")
            continue
        print(f"[download] Downloading {hf_filename} from {HF_REPO_ID}...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            local_dir="artifacts",
            token=HF_TOKEN,
        )
        print(f"[download] Saved to {local_path}")
    print("[download] All artifacts ready.")

if __name__ == "__main__":
    download()
```

### 2. `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

# Copy source code
COPY src/ src/
COPY web/ web/
COPY scripts/ scripts/

# Download model at build time (cached in image layer)
# HF_TOKEN must be set as a build arg or env var
ARG HF_TOKEN
ARG HF_REPO_ID=YOUR_USERNAME/esl-model
ENV HF_TOKEN=$HF_TOKEN
ENV HF_REPO_ID=$HF_REPO_ID
RUN python scripts/download_model.py

EXPOSE 8000
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**Important:** `--workers 1` is required. MediaPipe and the Keras model are not safe to share across multiple processes. One worker handles all WebSocket connections asynchronously — this is fine for demo load.

### 3. `.github/workflows/deploy.yml`

```yaml
name: Deploy to Railway

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'web/**'
      - 'scripts/**'
      - 'requirements.txt'
      - 'Dockerfile'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Railway CLI
        run: npm install -g @railway/cli

      - name: Deploy
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: railway up --service esl-server --detach
```

Get `RAILWAY_TOKEN` from Railway dashboard → Account Settings → Tokens. Add it as a GitHub secret.

### 4. `scripts/promote_model.py`

```python
"""
scripts/promote_model.py
Promote a new model version to production.

Usage:
    python scripts/promote_model.py model_v3.keras
    # Then: git commit -m "promote model_v3" && git push
    # Railway auto-redeploys with the new model.
"""
import sys
import os

def promote(version: str):
    from huggingface_hub import HfApi
    repo_id = os.environ.get("HF_REPO_ID", "YOUR_USERNAME/esl-model")
    token   = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable first.")
        sys.exit(1)

    local_path = f"artifacts/{version}"
    if not os.path.exists(local_path):
        print(f"ERROR: {local_path} not found locally.")
        sys.exit(1)

    api = HfApi()
    print(f"Uploading {version} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=version,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Updating current.txt → {version}")
    api.upload_file(
        path_or_fileobj=version.encode(),
        path_in_repo="current.txt",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Done. Production model is now: {version}")
    print("Trigger redeploy: git commit --allow-empty -m 'deploy: promote to {version}' && git push")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/promote_model.py <model_filename>")
        print("Example: python scripts/promote_model.py model_v3.keras")
        sys.exit(1)
    promote(sys.argv[1])
```

---

## Step-by-Step Deployment Checklist

Run these in order. Do not skip steps.

### Step 1 — Verify local server works (5 min)
```bash
python scripts/smoke_check.py
# Expected: predict_frame(black): "__no_hands__" conf=0.0 emo=neutral

python -m uvicorn web.server:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}
```
If `model_loaded` is false: check that `artifacts/model_v2.keras` exists. `src/paths.py` looks in `artifacts/` then `output/artifacts/`.

### Step 2 — ngrok for immediate mobile testing (10 min)
```bash
# Install ngrok: https://ngrok.com/download
ngrok config add-authtoken YOUR_NGROK_TOKEN

# Terminal 1: start server
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000

# Terminal 2: expose it
ngrok http 8000
# Output: Forwarding https://abc123.ngrok-free.app -> localhost:8000
```
Give teammate: `wss://abc123.ngrok-free.app/ws`
This works for demo/viva. URL changes every restart. Replace with Railway URL after.

### Step 3 — Upload model to HuggingFace Hub (20 min)
```bash
pip install huggingface_hub

# 1. Create free account at huggingface.co
# 2. Create a new model repository named "esl-model" (can be private)
# 3. Get token from huggingface.co/settings/tokens (write permission)

python -c "
from huggingface_hub import HfApi
api = HfApi()
token = 'YOUR_TOKEN_HERE'
repo  = 'YOUR_USERNAME/esl-model'

api.upload_file(path_or_fileobj='artifacts/model_v2.keras',  path_in_repo='model_v2.keras',  repo_id=repo, repo_type='model', token=token)
api.upload_file(path_or_fileobj='artifacts/label2idx.json', path_in_repo='label2idx.json', repo_id=repo, repo_type='model', token=token)
api.upload_file(path_or_fileobj=b'model_v2.keras',           path_in_repo='current.txt',     repo_id=repo, repo_type='model', token=token)
print('Upload complete')
"
```

Update `HF_REPO_ID` in `scripts/download_model.py` and `scripts/promote_model.py` to your actual repo ID.

### Step 4 — Test the download script locally (5 min)
```bash
# Rename your local model temporarily to test
mv artifacts/model_v2.keras artifacts/model_v2.keras.bak
HF_TOKEN=your_token python scripts/download_model.py
# Should download and restore the file
mv artifacts/model_v2.keras.bak artifacts/model_v2.keras  # restore
```

### Step 5 — Create Dockerfile and test it locally (20 min)
```bash
# Build the Docker image
docker build --build-arg HF_TOKEN=your_token -t esl-server .

# Run it locally to verify
docker run -p 8000:8000 esl-server

# Test
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}
```
If Docker is not installed: install from docker.com. This step is required before Railway.

### Step 6 — Deploy to Railway (30 min)
```bash
# 1. Go to railway.app → sign up (free) → New Project → Deploy from GitHub
# 2. Select your repository
# 3. Railway detects the Dockerfile automatically
# 4. Go to your service → Variables tab → add:
#      HF_TOKEN = your_huggingface_token
#      HF_REPO_ID = your_username/esl-model
# 5. Go to Settings → Networking → Generate Domain
#    You get: https://esl-xxxxx.up.railway.app

# Install Railway CLI for later use
npm install -g @railway/cli
railway login
```

### Step 7 — Update mobile app with permanent URL (5 min)
Tell teammate the permanent WebSocket URL: `wss://esl-xxxxx.up.railway.app/ws`
This never changes. The mobile app is configured once.

### Step 8 — Add GitHub Actions for auto-redeploy (15 min)
```bash
# Get Railway token: railway.app → Account Settings → Tokens → Create token
# Add to GitHub: your-repo → Settings → Secrets → Actions → New secret
#   Name: RAILWAY_TOKEN
#   Value: your_railway_token

# Create the workflow file:
mkdir -p .github/workflows
# (write .github/workflows/deploy.yml — content in section above)

git add .github/workflows/deploy.yml
git commit -m "feat: add Railway auto-deploy on push to main"
git push
```

After this: every push to `main` that touches `src/`, `web/`, or `Dockerfile` triggers a Railway redeploy automatically.

---

## CI/CD — Accuracy Improvement Loop (Post-Launch)

This is the loop you run every time you want to ship better predictions:

```
1. Collect new data (more signers, more signs, or emotion-labeled data)
       ↓
2. Run Phase 1 Colab notebook → produces model_v3.keras (or v4, v5...)
       ↓
3. Run evaluation: python scripts/export_eval_arrays.py && python -m src.evaluate
   Confirm new model beats old: check results/ablation_table.csv
       ↓
4. Promote new model:
   python scripts/promote_model.py model_v3.keras
       ↓
5. Trigger redeploy (empty commit is enough):
   git commit --allow-empty -m "deploy: promote model_v3" && git push
       ↓
6. Railway rebuilds container, downloads model_v3.keras from HuggingFace
   Mobile app users get better predictions — zero manual steps, zero downtime
```

### Accuracy improvement roadmap (in order of impact)

| Priority | Action | Expected gain | Effort |
|----------|--------|---------------|--------|
| 1 | Add 4+ new signers to training data | +15–25% on unseen people | 2 hrs recording + 3 hrs retraining |
| 2 | Expand from 8–12 to 25–30 MVP signs | Broader vocabulary | 4 hrs recording + 4 hrs retraining |
| 3 | Use LSTM (model_v3) for dynamic signs | +5–10% on motion-dependent signs | Already trained, just evaluate and promote |
| 4 | Collect emotion-labeled data | Unlocks emotion feature | Large effort — Phase 2 of future work |
| 5 | Scale to all 55 classes | Full vocabulary | 50+ videos/class needed |

---

## File Structure (Current, Complete)

```
project/
├── .github/
│   └── workflows/
│       └── deploy.yml              ← CREATE THIS (Step 8)
├── artifacts/
│   ├── model_v1.keras              ✅ exists — baseline MLP
│   ├── model_v2.keras              ✅ exists — primary model (augmented MLP + emotion)
│   ├── model_v3.keras              ✅ exists — LSTM variant
│   ├── model_v2.tflite             (create if FPS < 10: python scripts/export_tflite.py)
│   └── label2idx.json              ✅ exists — class name mapping
├── data/
│   ├── landmarks/                  raw .npy per-video landmark files
│   └── augmented_landmarks/        augmented sequences
├── results/
│   ├── confusion_matrix.png        (run: python -m src.evaluate)
│   ├── ablation_table.csv          (run: python -m src.evaluate)
│   ├── classification_report.txt   (run: python -m src.evaluate)
│   └── failure_analysis.png        (run: python -m src.evaluate)
├── scripts/
│   ├── download_model.py           ← CREATE THIS (Step 3)
│   ├── promote_model.py            ← CREATE THIS (for post-launch)
│   ├── export_eval_arrays.py       ✅ exists
│   ├── export_tflite.py            ✅ exists
│   └── smoke_check.py              ✅ exists
├── src/
│   ├── inference.py                ✅ exists — predict_frame() — DO NOT MODIFY
│   ├── augmentation.py             ✅ exists
│   ├── evaluate.py                 ✅ exists
│   ├── landmark_gate.py            ✅ exists
│   └── paths.py                    ✅ exists
├── web/
│   ├── server.py                   ✅ exists — FastAPI WebSocket server
│   ├── index.html                  ✅ exists — PWA browser frontend
│   └── manifest.json               ✅ exists
├── demo.py                         ✅ exists — full desktop app
├── Dockerfile                      ← CREATE THIS (Step 5)
├── requirements.txt                ✅ exists
├── run_demo.sh / run_demo.ps1      ✅ exists
├── run_web.sh  / run_web.ps1       ✅ exists
└── pyproject.toml                  ✅ exists
```

---

## What NOT to Do — Rules for Claude Code

1. **Do not modify `src/inference.py`** unless fixing a verified bug. It is the core contract. Every other file depends on it.
2. **Do not change `FACE_IDX` order** — reordering silently breaks the model's input distribution. This was already fixed once.
3. **Do not add DeepFace to the web server** — pass `"neutral"` always in web/mobile mode. DeepFace requires 500MB download and crashes on frames without faces.
4. **Do not use `--workers 2+` in uvicorn** — MediaPipe and Keras model are not multi-process safe.
5. **Do not commit `artifacts/*.keras` or `artifacts/*.npy` to git** — the `.gitignore` already blocks this. Model files go to HuggingFace Hub only.
6. **Do not retrain the model from Claude Code** — retraining happens in Google Colab (Phase 1 notebook). Claude Code only handles serving/deployment.
7. **Do not change the WebSocket message format** without notifying the mobile app team. Current format is: `{"label": str, "confidence": float, "emotion": str, "latency_ms": float}`.

---

## Models — Reference

| File | Type | Input dim | Use for |
|------|------|-----------|---------|
| `model_v1.keras` | Baseline MLP, no augmentation | 163 | Ablation comparison only |
| `model_v2.keras` | Augmented MLP + emotion | 163 | **Production. Use this.** |
| `model_v3.keras` | Augmented LSTM + emotion | (30, 163) | Evaluate for dynamic signs — may outperform v2 |
| `model_v2.tflite` | TFLite export of v2 | 163 | Use if FPS < 10 on exam laptop |

### Normalisation (must match training exactly)
```python
def _normalize(ff: np.ndarray) -> np.ndarray:
    raw   = ff.astype(np.float64)
    left  = raw[0:63].reshape(21, 3).copy()
    right = raw[63:126].reshape(21, 3).copy()
    face  = raw[126:].reshape(-1, 3).copy()
    # Hands: origin = wrist (joint 0), scale = max radial distance from wrist
    for seg in [left, right]:
        if seg.any():
            seg -= seg[0]
            s = np.max(np.linalg.norm(seg, axis=1))
            if s > 0: seg /= s
    # Face: origin = centroid, scale = max radial distance from centroid
    if face.any():
        face -= face.mean(axis=0)
        s = np.max(np.linalg.norm(face, axis=1))
        if s > 0: face /= s
    return np.concatenate([left.flatten(), right.flatten(), face.flatten()]).astype(np.float32)
```

---

## Performance Targets

| Metric | Target | How to check |
|--------|--------|-------------|
| Desktop FPS | ≥ 15 | FPS counter in OpenCV window |
| Inference latency | ≤ 100ms | `python -m src.evaluate` timing section |
| Web round-trip | ≤ 150ms | `latency_ms` field in WebSocket JSON response |
| Railway cold start | ≤ 30s | First request after deploy |
| Model size (v2) | ~200KB | `ls -lh artifacts/model_v2.keras` |

---

## Team Assignments (Current State)

| Person | Owns | Status |
|--------|------|--------|
| Abdullah (Team Lead) | Deployment pipeline, `Dockerfile`, `download_model.py`, Railway setup, `deploy.yml`, viva slides | 🔴 TODO |
| AI Member 1 | `src/inference.py`, model variants, TFLite export, latency benchmark | ✅ Done |
| AI Member 2 | `src/evaluate.py`, confusion matrix, ablation table, failure analysis | Run `python -m src.evaluate` |
| Software Member 1 | `web/index.html`, demo videos | ✅ Done |
| Software Member 2 | `web/server.py`, `README.md`, `run_web.sh`, repo packaging | ✅ Done |

---

## Viva — Key Questions and Exact Answers

**"Why landmark-based instead of CNN on raw images?"**
> With 10 videos per sign class, a CNN would overfit severely — it needs thousands of images. MediaPipe gives us 156 clean numbers per frame representing every hand joint position, stripped of background and lighting. Our MLP is 200KB, trains in minutes, and runs at 50+ FPS on CPU. A CNN would be 14MB, need GPU, and still perform worse at this dataset size.

**"Why LSTM over MLP?"**
> We categorised all MVP signs as static or dynamic. Static signs — where only hand shape matters — use MLP and predict from a single frame. Dynamic signs — where motion encodes meaning — use LSTM, which reads a sequence of 30 frames. The ablation table shows LSTM outperforms MLP on dynamic signs specifically.

**"What is the data leakage bug you fixed?"**
> Frame-level train/val splitting lets multiple frames from the same video appear in both sets. Since those frames share the signer's hand proportions, the model memorises the person not the sign — artificially inflated accuracy. We fixed this by splitting at video level: 80% of videos per class go to train, 20% to val. No video ever appears in both sets.

**"Why is emotion accuracy limited?"**
> We trained with neutral emotion as a placeholder because our signing dataset has no emotion ground-truth labels. The architecture supports live emotion — the 7-dim slot is in the input — but the model hasn't learned to use variation there. Fix: collect emotion-labeled data, retrain model_v4. Zero architecture changes needed.

**"Why not a native mobile app?"**
> We have a PWA — the web app is installable on Android and iOS home screen via the same codebase. No app store, no native SDK, no cross-compilation. The same `predict_frame()` function powers the desktop app, the browser frontend, and the mobile interface. Adding a native app is future work requiring TFLite + MediaPipe iOS/Android SDKs.

**"What are your limitations?"**
> Single signer in training data — accuracy drops on unseen signers. MVP vocabulary is 8–12 of 55 total signs. Emotion trained on neutral placeholder only. No sentence-level recognition — only isolated signs. The confusion matrix shows the top 3 misclassified pairs.

---

## Git Workflow

```
main              ← always deployable. Railway watches this branch.
feat/deployment   ← Dockerfile + download_model.py + deploy.yml
feat/evaluation   ← evaluation artifacts and ablation numbers
feat/accuracy-v3  ← promote model_v3 after testing
```

Never commit directly to `main`. PR → review → merge. Railway redeploys automatically on merge.

Small commit messages: "Add Dockerfile for Railway deployment" not "update stuff".

---

## Environment Variables (Required for Deployment)

| Variable | Where set | Value |
|----------|-----------|-------|
| `HF_TOKEN` | Railway dashboard → Variables | HuggingFace write token |
| `HF_REPO_ID` | Railway dashboard → Variables | `your_username/esl-model` |
| `RAILWAY_TOKEN` | GitHub Secrets | Railway API token for CI |

Never put tokens in code or commit them to git.

---

## Known Issues

- `mediapipe==0.10.20` is pinned — newer versions removed `mp.solutions.holistic`. Do not upgrade.
- DeepFace downloads ~500MB of model weights on first run — this is why we skip it in web/mobile mode.
- Python 3.12+ is not supported by TensorFlow on Windows — use 3.10 or 3.11 only.
- `--workers 1` is required for uvicorn — MediaPipe Holistic is not safe to share between processes.
- On Railway free tier: container sleeps after inactivity. First request after sleep takes ~10–15s (cold start). Add a `/health` ping from the mobile app on launch to wake it up.

