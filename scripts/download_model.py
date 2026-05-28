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
    ("model_v2.keras",          "artifacts/model_v2.keras"),
    ("label2idx.json",          "artifacts/label2idx.json"),
    ("holistic_landmarker.task", "artifacts/holistic_landmarker.task"),
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