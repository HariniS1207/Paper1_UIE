# Paper 1 V7 enhancement-only V1 experiment

## Objective and scope

This experiment warm-started the existing V7 model from the verified epoch-19 baseline and continued optimizing its existing enhancement objective. Only V7's existing trainable `selection_network` was updated. The condition encoder, candidate generator, information-preservation module, and consequence-check module stayed frozen. Detection/classification losses and metrics were not used.

## Method

- Initialization: `checkpoints/v7/paper1_v7_best_epoch_019.pth`; SHA-256 before and after the experiment: `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`.
- Runtime: Python 3.12.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 3050 6GB Laptop GPU. The repository `.venv` still blocks torch with Windows Application Control, so model execution used the pre-existing `TestTorchEnv`. The image-metric pass ran separately in `.venv` using NumPy/Pillow/scikit-image only, without importing torch.
- Configuration: seed 42; 2,960 training and 370 validation pairs; batch size 8; fresh Adam optimizer at `1e-4`; maximum 20 epochs; gradient clipping 1.0; CUDA float16 AMP with initial scale 2048; no scheduler; best checkpoint selected by minimum validation total.
- Training used paired RGB tensors in `[0,1]` with shared random horizontal flip. Validation used RGB tensors in `[0,1]` with no resize or random augmentation. Train/validation/test manifests contained 2,960/370/370 identifiers with zero pairwise overlap. Test identifiers were read only for the pre-run disjointness audit; test images were not loaded or used for training, tuning, or checkpoint selection.
- The unchanged objective was `L = L_reconstruction + 0.2 L_preservation + 0.2 L_consequence + 0.5 L_selection`.
- All 20 epochs completed in 2,000.85 seconds of recorded epoch time, with 7,400 successful batch updates. There were two AMP non-finite-gradient replay events (epochs 14 and 18); neither failed attempt applied an optimizer update, and both batches completed on retry. No batch was ultimately dropped. Peak allocated GPU memory per epoch was 743,825,408 bytes.
- The pre-training batch-8 smoke check produced finite loss/gradients, a nonzero selector update, unchanged frozen-module hashes, valid `[0,1]` output, and 675,266,048 bytes peak CUDA allocation.

## Validation results

The selected checkpoint is epoch 19 at `checkpoints/v7_enhancement_only_v1/paper1_v7_enhancement_only_best_epoch_019.pth` (8,493,579 bytes; SHA-256 `119A2C58DCC3274F0D87E19442A4B7CF41DDE48447C009E8BDA5AAA06186E502`). It strict-loaded into a fresh `Paper1ModelV7` with zero missing/unexpected keys and passed a finite forward check with shape `[8,3,256,256]` and range `[0,1]`.

The baseline was re-evaluated after training under the same validation protocol and matched the previously reproduced Phase 1 values exactly. The selected updated checkpoint was evaluated twice; its second-pass pixels matched the archived first-pass outputs exactly (maximum absolute difference `0.0`), and all five losses agreed exactly (maximum absolute difference `0.0`, allowed tolerance `1e-7`). Both comparisons accounted for all 370 validation pairs.

| Measurement | Baseline | Updated | Updated − baseline | Relative change |
|---|---:|---:|---:|---:|
| Validation total loss (lower) | 0.250245914749 | 0.245716525413 | -0.004529389336 | -1.810% |
| Reconstruction loss (lower) | 0.073375016733 | 0.073502864910 | +0.000127848177 | +0.174% |
| Preservation loss (lower) | 0.035322277228 | 0.034983630619 | -0.000338646608 | -0.959% |
| Consequence loss (lower) | 0.077731877888 | 0.077970067794 | +0.000238189907 | +0.306% |
| Selection loss (lower) | 0.308520128049 | 0.299245843859 | -0.009274284190 | -3.006% |

The total objective decreased mainly through the selection component. Reconstruction and consequence components regressed slightly; preservation improved slightly.

## Image-quality evaluation

PSNR and SSIM use `data_range=1.0` on RGB images. UIQM and UCIQE use the existing `calculate_metrics` implementation in `src/evaluate_test.py`, on clipped RGB `[0,1]` arrays. The canonical metric function definitions were extracted directly from that source and run by the existing `.venv` NumPy/Pillow/scikit-image environment; no metric formulas were reimplemented. The source SHA-256 was `728F00EE0D80BD98766D5CB155D1BC1AD96693B7C48A3EEC147F4B6DF75C0705`.

| Metric (higher is better) | Baseline mean ± SD | Updated mean ± SD | Mean difference | Updated better / worse / unchanged samples |
|---|---:|---:|---:|---:|
| PSNR | 21.605943 ± 3.466248 | 21.513313 ± 3.507379 | -0.092630 (-0.429%) | 3 / 8 / 359 |
| SSIM | 0.793094 ± 0.066858 | 0.791344 ± 0.067966 | -0.001750 (-0.221%) | 3 / 8 / 359 |
| UIQM | 1.659742 ± 0.826961 | 1.656616 ± 0.823926 | -0.003126 (-0.188%) | 4 / 7 / 359 |
| UCIQE | 5.660413 ± 2.149138 | 5.651389 ± 2.134213 | -0.009024 (-0.159%) | 6 / 5 / 359 |

All four image-quality means moved slightly downward. The majority of samples were unchanged, reflecting the frozen candidate space and selector-only updates; among changed samples, quality gains were not consistent. A lower enhancement objective therefore does not demonstrate improved image quality in this run.

## Downstream task boundary and conclusion

Detection loss, classification loss, and classification accuracy are **NOT EVALUATED BY THIS EXPERIMENT**. The project did not contain the supervisor's models or aligned task annotations. The historical staff values are not inserted into the objective. The supervisor must evaluate the updated checkpoint using the same task models, preprocessing, labels, split, and metrics used for the baseline.

The run is reproducible and the updated checkpoint is valid as a separate experimental artifact. However, the evidence does **not** support accepting it as an image-quality improvement or replacing the original V7 baseline: validation total loss decreased by 1.81%, while all four measured image-quality means decreased slightly and two enhancement components regressed. Keep the epoch-19 checkpoint as an experimental candidate pending supervisor task evaluation and scientific review.

## Artifacts

- Pre-training audit: `PRETRAINING_AUDIT.md`
- Effective configuration and runtime: `TRAINING_CONFIGURATION.json`
- Epoch training log: `TRAINING_LOG.csv`
- Validation loss components: `VALIDATION_COMPONENT_LOSSES.csv`
- Baseline comparison: `BASELINE_VS_UPDATED_COMPARISON.md`
- Image-quality metrics and per-metric counts: `IMAGE_QUALITY_METRICS.csv`, `IMAGE_QUALITY_METRICS.json`
- Prediction arrays used for metric calculation: `validation_predictions_baseline.npz`, `validation_predictions_updated.npz`
- Checkpoint integrity: `CHECKPOINT_INTEGRITY_REPORT.md`
- Machine-readable evaluation: `FINAL_EVALUATION.json`
- Training entry point: `src/train_v7_enhancement_only.py`
- Isolated canonical-metric runner: `src/compute_v7_quality_metrics.py`
