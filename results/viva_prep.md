# Viva preparation checklist

Complete after `python scripts/export_eval_arrays.py` and `python -m src.evaluate`.

## Recording

- [ ] `demo_desktop.mp4` — 5 signs + 1 honest failure (hands out of frame).
- [ ] `demo_web.mp4` — browser live predictions (optional if Phase 4 shipped).

## Numbers (copy into [README.md](../README.md) Performance Targets)

| Metric | Your value | Date |
|--------|------------|------|
| Val accuracy (model_v2) | | |
| Inference latency (evaluate.py) | ms | |
| TFLite max abs diff (`export_tflite.py --validate`) | | |
| Desktop display FPS | | |
| Web round-trip (optional) | ms | |

## Cross-signer testing (required risk mitigation)

Test with **at least one signer not in training videos**. Document:

- Signs tested:
- Observed behaviour / accuracy drop:

## Top 3 confused pairs

After evaluation, copy from console output or fill from `results/failure_analysis.png`:

1. **→**  (count: ) — one sentence: why they look similar kinematically.
2. **→**  (count: )
3. **→**  (count: )

## Pre-viva (.cursor/instruction.md alignment)

- [ ] Activation gate ON (full mode) OR sprint mode matches instruction strings
- [ ] Confidence threshold & sliding window documented
- [ ] Emotion: explain neutral-at-training + optional DeepFace in full mode
- [ ] Ablation table + confusion matrix in slides
