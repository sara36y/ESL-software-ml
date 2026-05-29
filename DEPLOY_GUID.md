# ESL Deployment — Complete Step-by-Step Guide

# ============================================================

# 

# All commands assume your PowerShell 5.1 environment and the project venv.
# Replace `.venv\Scripts\python.exe` with the venv python path shown below.
# Replace `YOUR_USERNAME` with your actual HuggingFace username.
# Replace `YOUR_NGROK_TOKEN` / `YOUR_TOKEN` / etc with your actual values.
# ============================================================

# Prerequisites:
#   - .venv must to activated (Python 3.13 with all deps installed)
#   - HuggingFace account with a model repo created
#   - ngrok installed (for Step 2)
#   - Docker Desktop installed (for Step 5)
#   - Railway account (for Step 6)
# ============================================================

# 
# STEP 1 — Test Web Server Locally
5 minutes]
# Verify the server starts and the model loads:

 
#   .venv\Scripts\python.exe -m uvicorn web.server:app --host 0.0.0.0 --port 8000
 --reload
# 
# Open browser: http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}
# 
# If model_loaded is false: check artifacts/model_v2.keras exists.
# 
# Open browser: http://localhost:8000 — should show the PWA frontend
# Test WebSocket in browser console:
#   const ws = new WebSocket('ws://localhost:8000/ws');
#   ws.onmessage = (e) => console.log(JSON.parse(e.data));
# 
# STEP 2 — ngrok for Immediate Mobile Testing [10 minutes]
# Install ngrok: https://ngrok.com/download
# 
# Configure:
#   ngrok config add-authtoken YOUR_NGROK_TOKEN
# 
# Terminal 1 (start server):
#   $env:TF_ENABLE_ONEDNN_OPTS='0'; .venv\Scripts\python.exe -m uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
# 
# Terminal 2 (expose via ngrok):
#   ngrok http 8000
# 
# Output will show: Forwarding https://abc123.ngrok-free.app -> localhost:8000
# 
# Give your teammate: wss://abc123.ngrok-free.app/ws
# Note: ngrok URL changes every restart. Good for demo/viva only.
# 
# STEP 3 — Upload Model to HuggingFace Hub [20 minutes]
# Create HuggingFace repo:
#   1. Go to huggingface.co and sign up (free)
#   2. Create a new Model repository named "esl-model" (can be private)
#   3. Get a token from huggingface.co/settings/tokens (need Write permission)
# 
# Upload the 3 artifact files:
#   $env:TF_ENABLE_ONEDNN_OPTS='0'
#   .venv\Scripts\python.exe -c "
#   from huggingface_hub import HfApi
#   api = HfApi()
#   token = 'YOUR_TOKEN_HERE'
#   repo  = 'YOUR_USERNAME/esl-model'
#   
#   api.upload_file(
#       path_or_fileobj='artifacts/model_v2.keras',
#       path_in_repo='model_v2.keras',
#       repo_id=repo, repo_type='model', token=token
#   )
#   api.upload_file(
#       path_or_fileobj='artifacts/label2idx.json',
#       path_in_repo='label2idx.json',
#       repo_id=repo, repo_type='model', token=token
#   )
#   api.upload_file(
#       path_or_fileobj='artifacts/holistic_landmarker.task',
#       path_in_repo='holistic_landmarker.task',
#       repo_id=repo, repo_type='model', token=token
#   )
#   api.upload_file(
#       path_or_fileobj=b'model_v2.keras',
#       path_in_repo='current.txt',
#       repo_id=repo, repo_type='model', token=token
#   )
#   print('Upload complete')
#   "
# 
# Update repo ID in scripts:
#   Edit scripts/download_model.py — change YOUR_USERNAME/esl-model to your actual repo ID
#   Edit scripts/promote_model.py — change YOUR_USERNAME/esl-model to your actual repo ID
#   Edit Dockerfile -- change ARG HF_REPO_ID=YOUR_USERNAME/esl-model to your actual repo ID
# 
# STEP 4 — Test Download Script Locally [5 minutes]
#   Temporarily rename local model to test the download:
#   Move-Item -Path artifacts\model_v2.keras -Destination artifacts\model_v2.keras.bak
#   
#   $env:TF_ENABLE_ONEDNN_OPTS='0'; $env:HF_TOKEN='your_token'; .venv\Scripts\python.exe scripts/download_model.py
#   Should download and restore model_v2.keras from HuggingFace.
#   
#   Restore local file:
#   Move-Item -Path artifacts\model_v2.keras.bak -Destination artifacts/model_v2.keras
# 
# STEP 5 — Build Docker Image and Test Locally [20 minutes]
#   Make sure Docker Desktop is installed: https://docker.com/products/docker-desktop
# 
#   Build the image:
#   docker build --build-arg HF_TOKEN=your_huggingface_token -t esl-server .
# 
#   Run it:
#   docker run -p 8000:8000 esl-server
# 
#   Test:
#   curl http://localhost:8000/health
#   Expected: {"status":"ok","model_loaded":true}
# 
# STEP 6 — Deploy to Railway [30 minutes]
#   1. Go to railway.app — sign up (free tier)
#   2. New Project -> Deploy from GitHub repo
#   3. Select your repository
#   4. Railway detects the Dockerfile automatically
#   5. Go to service -> Variables tab -> add:
#       HF_TOKEN = your_huggingface_token
#       HF_REPO_ID = your_username/esl-model
#   6. Go to Settings -> Networking -> Generate Domain
#      You get: https://esl-xxxxx.up.railway.app
# 
#   STEP 7 — Update Mobile App URL [5 minutes]
#   Tell teammate the permanent WebSocket URL: wss://esl-xxxxx.up.railway.app/ws
#   This URL never changes.
# 
# STEP 8 — Add GitHub Actions Auto-Redeploy [15 minutes]
#   1. Get Railway token: railway.app -> Account Settings -> Tokens -> Create token
#   2. Add to GitHub: your-repo -> Settings -> Secrets and variables -> Actions -> New secret
#      Name: RAILWAY_TOKEN
#      Value: your_railway_token
#   3. Push the workflow file:
#       git add .github/workflows/deploy.yml
#       git commit -m "feat: add Railway auto-deploy"
#       git push
# 
#   After this: every push to main that touches src/, web/, or Dockerfile triggers Railway redeploy.
# 
# ============================================================
# 
# Notes:
#   - Always use --workers 1 for uvicorn (MediaPipe + Keras are not multi-process safe)
#   - Railway free tier: container sleeps after inactivity, first request takes 10-15s
#   - Set TF_ENABLE_ONEDNN_OPTS=0 before running Python to suppress oneDNN warnings
#   - Never commit tokens/secrets to git