# Paper 1 UIE V7 final training report

## Decision

**B ? V7 improves the trade-off but does not yet establish genuine adaptive selection; do not claim adaptivity.** Selection is severity dependent and uses all three distinct candidates, but V7 does not maintain V4/V6 fidelity, and its confidence is sharply concentrated on Aggressive (mean probability 0.723, entropy 0.225 nats). Treat as an adaptive-selection research candidate, not a frozen final model. No V8 is started.

## Forensic diagnosis

The validation candidate-space Pareto analysis identified the selector as the primary bottleneck; retain the V4 candidate generator. Under strict Pareto dominance across reference L1, preservation distortion, and consequence error, V6 had C/B/A Pareto-optimal rates 81.35% / 34.32% / 100%; every image had C or B non-dominated. V4 rates were 84.9% / 87.3% / 100%, V5 rates 45.9% / 100% / 53.5%. Multiple alternatives exist, so the generator was retained.

V7 adds a candidate-conditioned utility scorer using degradation condition, candidate features, preservation, and consequence features. Training targets rank the three candidates equally on fidelity, preservation distortion, and consequence per image; there is no quota, resampling, or hard-coded severity rule. Frozen V4 encoder/generator/preservation/consequence modules were reused; only the selector was trained.

## Training

- Seed 42; batch size 8; Adam learning rate 1e-4; max 20 epochs; train/validation only; best checkpoint chosen by validation total loss (epoch 19).
- Best validation loss: 0.250246.
- AMP overflow handling event count: 1. A non-finite gradient at epoch 18 was skipped/replayed; no optimizer step was applied for the event.
- Checkpoint: `checkpoints\v7\paper1_v7_best_epoch_019.pth`
- SHA-256: `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`
- Test split accessed: **NO**.

## Validation quality

| Model | PSNR | SSIM | UIQM | UCIQE | Preservation score | Consequence error |
|---|---:|---:|---:|---:|---:|---:|
| Original | 16.9136 | 0.7371 | 1.8214 | 5.6640 |  |  |
| V3 | 20.6375 | 0.7814 | 1.7816 | 5.4329 | 0.9577407619437656 | 0.06035463031683419 |
| V4 | 23.3725 | 0.8133 | 1.7115 | 5.8733 |  |  |
| V5 | 21.6596 | 0.7963 | 1.6685 | 5.8258 | 0.966339514223305 | 0.04110551217973635 |
| V6 | 23.3963 | 0.8140 | 1.7412 | 5.8089 | 0.9544575526907637 | 0.053126187088924484 |
| V7 | 21.6060 | 0.7931 | 1.6597 | 5.6602 | 0.9643555141784049 | 0.07686445134112964 |

V7 per-image selected-output improvement over original (mean metric delta):
- PSNR: +4.6923
- SSIM: +0.0560
- UIQM: -0.1617
- UCIQE: -0.0038

V7 preservation score averages 0.9644; consequence error averages 0.0769. Fidelity improves over original and V3, but V7 PSNR/SSIM are below V4/V6. UIQM is below original; UCIQE is approximately unchanged from original. These are validation measurements only.

## Selector diagnostics

- Training target counts: Conservative 339, Balanced 517, Aggressive 2104 (train-only audit).
- Validation target counts: {'Conservative': 48, 'Balanced': 59, 'Aggressive': 263}
- Validation selections: {'Conservative': 42, 'Balanced': 56, 'Aggressive': 272}
- Target/selector agreement: 91.622%
- Mean selection entropy: 0.2251 nats; mean utility margin: 5.2123.
- Pairwise candidate mean absolute distances: {'Conservative_vs_Balanced': 0.011103761438406199, 'Balanced_vs_Aggressive': 0.10783049769296839, 'Conservative_vs_Aggressive': 0.11362383519676891}
- Pairwise residual cosine: {'Conservative_vs_Balanced': 0.7598270159995032, 'Balanced_vs_Aggressive': 0.34405236883542023, 'Conservative_vs_Aggressive': -0.0463193481297207}

### Selection by input degradation severity quartile

| Quartile | N | Mean severity | Conservative | Balanced | Aggressive |
|---|---:|---:|---:|---:|---:|
| Q1 | 93 | 0.0816 | 11.8% | 24.7% | 63.4% |
| Q2 | 92 | 0.1070 | 19.6% | 26.1% | 54.3% |
| Q3 | 92 | 0.1263 | 7.6% | 4.3% | 88.0% |
| Q4 | 93 | 0.1650 | 6.5% | 5.4% | 88.2% |

Maximum pairwise total variation between quartile selection distributions is 0.338; selection changes most between Q2 and the more severe groups. All three branches are selected and branch outputs are distinct. However, low entropy/high mean margin indicate concentrated confidence, and fidelity regression relative to V4/V6 argues against freezing V7 as the final model.

## Integrity and scope

V2?V6 checkpoint SHA-256 inventories match the pre-run manifest with zero mismatches. Frozen V4 modules remained unchanged during V7 training. Only train and validation splits were used. `TEST SET ACCESSED: NO`.
