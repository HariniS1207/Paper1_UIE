# Final condition-aware selector report

## Validation decision

**C - FAILURE to establish meaningful adaptive selection. Stop model development. Use V4 as the strongest enhancement baseline and retain the final selector/V7 results as adaptive-selection investigations. No further architecture experiment.**

- Best epoch: 19; validation total loss: 0.585527.
- Checkpoint SHA-256: `5CB2ACD32F599709C7B5AB0E472FA9E879D9BEB7E20F8C375111FBE86CA743C9`.
- Test set accessed: **NO**.

## Validation metrics

| Model | PSNR | SSIM | UIQM | UCIQE | Preservation | Consequence |
|---|---:|---:|---:|---:|---:|---:|
| FinalSelector | 23.3058 | 0.8125 | 1.7064 | 5.8486 | 0.955379 | 0.053233 |
| V3 | 20.6375 | 0.7814 | 1.7816 | 5.4329 | 0.9577407619437656 | 0.06035463031683419 |
| V4 | 23.3725 | 0.8133 | 1.7115 | 5.8733 | 0.954983 | 0.052315 |
| V5 | 21.6596 | 0.7963 | 1.6685 | 5.8258 | 0.966339514223305 | 0.04110551217973635 |
| V6 | 23.3963 | 0.8140 | 1.7412 | 5.8089 | 0.9544575526907637 | 0.053126187088924484 |
| V7 | 21.6060 | 0.7931 | 1.6597 | 5.6602 | 0.964356 | 0.076864 |

## Selector behavior

- Hard target counts: {'Conservative': 48, 'Balanced': 59, 'Aggressive': 263}; mean soft target mass: {'Conservative': 0.18636282126645784, 'Balanced': 0.28090324758275137, 'Aggressive': 0.532733959603954}.
- Selected counts: {'Conservative': 2, 'Balanced': 2, 'Aggressive': 366}; target agreement: 72.162%.
- Mean probabilities: {'Conservative': 0.17329061712968993, 'Balanced': 0.2512836593530468, 'Aggressive': 0.5754257243227314}.
- Mean entropy: 0.9257 nats; mean utility margin: 0.6854.

| Quartile | Mean severity | C | B | A | Mean entropy | Mean confidence | Mean margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.0816 | 1.1% | 0.0% | 98.9% | 0.9331 | 0.5588 | 0.6053 |
| Q2 | 0.1070 | 1.1% | 1.1% | 97.8% | 0.9217 | 0.5569 | 0.5516 |
| Q3 | 0.1263 | 0.0% | 1.1% | 98.9% | 0.9275 | 0.5894 | 0.7708 |
| Q4 | 0.1650 | 0.0% | 0.0% | 100.0% | 0.9206 | 0.5974 | 0.8132 |

### Representative selected examples (validation only; two selections each for C and B)

| Selected class | Image | Probabilities (C/B/A) |
|---|---|---|
| Conservative | `n01914609_12065.jpg` | 0.483/0.049/0.468 |
| Balanced | `n01917289_652.jpg` | 0.043/0.499/0.458 |
| Aggressive | `n01917289_440.jpg` | 0.137/0.234/0.629 |


## Final assessment

The final selector had higher mean entropy than V7, but its argmax decisions were Aggressive 366/370, Conservative 2/370, Balanced 2/370. Per-quartile Aggressive selection ranged from 97.8% to 100%; this does not establish meaningful condition-dependent selection. Mean PSNR/SSIM were close to but slightly below V4, while UIQM was also lower. Decision: **C**. V4 remains the enhancement baseline; preserve this run as a negative adaptive-selection result. `TEST SET ACCESSED: NO`.


### Candidate quality and diversity

| Candidate | Fidelity L1 (lower better) | Preservation | Consequence error (lower better) | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| Conservative | 0.1273 | 0.9968 | 0.1314 | 16.4567 | 0.7191 |
| Balanced | 0.1224 | 0.9959 | 0.1292 | 16.7814 | 0.7311 |
| Aggressive | 0.0549 | 0.9550 | 0.0523 | 23.3725 | 0.8133 |

Candidate mean absolute distances and residual cosine are included in `diagnostics/final_selector/selector_diagnosis.json` under `candidate_diversity_from_frozen_v7_candidate_space`. Target-to-selected confusion matrix is also in that diagnosis JSON.

Previous checkpoint integrity: V2-V6 all match the pre-experiment SHA-256 manifest, with zero mismatches. The V7 best checkpoint remains unchanged at `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`.
