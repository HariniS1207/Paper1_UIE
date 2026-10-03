# Final frozen V7 test evaluation

**Final frozen held-out evaluation. V7 was not retrained, tuned, or modified.**

Execution note: the first full scoring pass completed inference on all 370 test pairs but hit a report-aggregation KeyError. After correcting only report aggregation, the same frozen evaluation was rerun to write these artifacts. The checkpoint, preprocessing, metrics, and test manifest were unchanged; no test-driven tuning occurred.

- Test images: 370
- Checkpoint: `checkpoints\v7\paper1_v7_best_epoch_019.pth` (epoch 19)
- SHA-256 before/after: `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`
- Test data used for training/model selection: **No**.
- Preprocessing: official paired test split with `PairedTransform(training=False)`, RGB tensor [0,1], no resize or random augmentation.

## Aggregate metrics

| Output | PSNR | SSIM | UIQM | UCIQE | Preservation | Consequence error |
|---|---:|---:|---:|---:|---:|---:|
| Original | 16.8987 | 0.737308 | 1.8555 | 5.6085 | 1.000000 | N/A |
| Conservative | 16.4324 | 0.719748 | 1.7163 | 5.4714 | 0.996815 | 0.13217311835772283 |
| Balanced | 16.7567 | 0.733091 | 1.6083 | 5.3665 | 0.995782 | 0.13019984136964824 |
| Aggressive | 23.3363 | 0.817886 | 1.7161 | 5.8820 | 0.954935 | 0.05309764989525885 |
| V7 Selected | 21.2027 | 0.789558 | 1.6703 | 5.6111 | 0.965145 | 0.08136838316615369 |

## Frozen baseline comparison

V3 and V4 values below are the frozen figures supplied for this final comparison and were not changed.

| Model | PSNR | SSIM | UIQM | UCIQE | Preservation |
|---|---:|---:|---:|---:|---:|
| Original | 16.8987 | 0.737308 | 1.8555 | 5.6085 | 1.000000 |
| V3 Selected | 20.4374 | 0.783765 | 1.8003 | 5.3948 | 0.957900 |
| V4 Selected | 23.3363 | 0.817886 | 1.7163 | 5.8820 | 0.954935 |
| V7 Selected (test) | 21.2027 | 0.789558 | 1.6703 | 5.6111 | 0.965145 |

### V7 Selected - V4 Selected

| Metric | Difference |
|---|---:|
| psnr | -2.133574 |
| ssim | -0.028328 |
| uiqm | -0.045987 |
| uciqe | -0.270929 |
| preservation | +0.010210 |

## Selection analysis

| Candidate | Count | Percent | Mean probability |
|---|---:|---:|---:|
| Conservative | 51 | 13.78% | 0.141672 |
| Balanced | 62 | 16.76% | 0.162657 |
| Aggressive | 257 | 69.46% | 0.695671 |

- Mean entropy: 0.220526 nats.
- Mean maximum selection probability: 0.915555.
- Mean utility margin: 5.401951.

### Fixed validation severity quartiles

The boundaries below are reused unchanged from the V7 validation analysis; test images outside the established edge range are reported separately.

| Quartile | n | Mean severity | C | B | A | Entropy | Max probability | Utility margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 98 | 0.081865 | 16.33% | 38.78% | 44.90% | 0.38723 | 0.85118 | 3.26374 |
| Q2 | 98 | 0.107244 | 12.24% | 17.35% | 70.41% | 0.27919 | 0.88576 | 4.96813 |
| Q3 | 73 | 0.125995 | 9.59% | 4.11% | 86.30% | 0.11450 | 0.95695 | 6.61307 |
| Q4 | 101 | 0.166556 | 15.84% | 3.96% | 80.20% | 0.07849 | 0.97702 | 7.02222 |

Images outside the fixed severity interval: 0.

## Selected versus Original improvement counts

| Metric | Improved images | Percent | Mean change |
|---|---:|---:|---:|
| psnr | 265 | 71.62% | +4.304032 |
| ssim | 257 | 69.46% | +0.052250 |
| uiqm | 138 | 37.30% | -0.185173 |
| uciqe | 170 | 45.95% | +0.002600 |
| preservation | 0 | 0.00% | -0.034855 |

## Final assessment

**B. V7 shows some test-set candidate variation, but remains strongly biased toward one candidate.**

The measurements are reported as observed; the test set was not used for cherry-picking, tuning, training, or checkpoint selection.

Checkpoint integrity: all `.pth` files under `checkpoints/` match their before-evaluation hashes. The V7 checkpoint hash before and after evaluation is identical.

**TEST SET ACCESSED: YES - FINAL FROZEN EVALUATION ONLY**
