# Artifacts — model and evaluation artifacts

This folder stores trained models, label maps and pre-exported NumPy arrays used by the demo and evaluation scripts.

Quick list of files you'll find here:

- `label2idx.json` — JSON map from string label to integer index used during training and inference. Load with `json.load()`.
- `model_v1.keras`, `model_v2.keras`, `model_v3.keras` — Keras model files produced during Phase 1 training. `model_v2.keras` is the primary demo model.
- `X_lstm_train.npy`, `X_lstm_val.npy` — NumPy arrays prepared for LSTM training (typically shape: `(N_samples, N_frames, D_per_frame)`).
- `X_mlp_train.npy`, `X_mlp_val.npy` — NumPy arrays prepared for MLP training (typically shape: `(N_samples, D)` where `D` is flattened landmarks + optional emotion features).
- `y_train.npy`, `y_val.npy` — Label arrays (integer indices) aligned with the `X_*.npy` arrays.
- `README.md` — this file.

How to inspect & load these artifacts (Python examples):

```python
import json
import numpy as np
import tensorflow as tf

# load label map
label2idx = json.load(open('artifacts/label2idx.json'))
idx2label = {int(v): k for k, v in label2idx.items()}  # if values are strings, adapt

# load model
model = tf.keras.models.load_model('artifacts/model_v2.keras')

# load arrays
X = np.load('artifacts/X_mlp_train.npy')
y = np.load('artifacts/y_train.npy')
print(X.shape, y.shape)
```

Notes on shapes and formats:
- The arrays are plain NumPy `.npy` files. Always inspect `arr.shape` to confirm expected dimensionality before feeding into models.
- LSTM arrays are 3D: `(samples, time_steps, features_per_frame)`; MLP arrays are 2D: `(samples, features)`.

How an AI agent should use these files:
- For quick inference checks: load `model_v2.keras` and call `model.predict()` on preprocessed features.
- For re-training or evaluation: use the `X_*` and `y_*` arrays as direct inputs to model training/evaluation scripts.
- If an agent needs label names, read `[artifacts/label2idx.json](artifacts/label2idx.json#L1)`.

Exporting to TFLite (example):

```python
import tensorflow as tf

model = tf.keras.models.load_model('artifacts/model_v2.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open('artifacts/model_v2.tflite', 'wb').write(tfmodel)
```

Related scripts and where to find them:
- Export evaluation arrays: see `scripts/export_eval_arrays.py` (run from repo root).
- Export TFLite: see `scripts/export_tflite.py`.

Practical tips:
- If a downstream component expects a specific order of emotion features or landmark flattening, verify the preprocessing in `src/inference.py` and `src/augmentation.py` before reusing arrays.
- If you alter the label set, update `[artifacts/label2idx.json](artifacts/label2idx.json#L1)` and regenerate `y_*.npy` accordingly.

If you want, I can also add small helper loader functions to `src/` to standardise loading across agents.
