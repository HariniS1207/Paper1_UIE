# Paper 1 V3 vs V4 — Final Held-Out Test Evaluation

## 1. Evaluation Configuration

- Split: `data/splits/test.txt only` (370 images)
- Preprocessing: EUVPPairedDataset + PairedTransform(training=False): RGB conversion, ToTensor-equivalent [0,1], no random augmentation
- Metric implementation: `src/evaluate_test.py:calculate_metrics, uiqm, uciqe`; RGB HWC float `[0,1]`.
- V3 checkpoint: `checkpoints/v3/paper1_v3_best.pth` (epoch 8)
- V4 checkpoint: `checkpoints/v4/paper1_v4_best.pth` (epoch 10)
- V2 SHA recorded for safety only: `BBC8027FF23DCD477CD2E4BD0C95D7A42222AAAAE2E6A6197AF3D198149C53D1`
- No training, tuning, or epoch selection occurred during this evaluation.

## 2. Main Metrics

| Method | PSNR | SSIM | UIQM | UCIQE | Preservation |
|---|---:|---:|---:|---:|---:|
| Original | 16.8987 | 0.737308 | 1.8555 | 5.6085 | 1.000000 |

Original preservation is 1.0 by identity (input compared with itself).
| V3 Selected | 20.4374 | 0.783765 | 1.8003 | 5.3948 | 0.957900 |

Original preservation is 1.0 by identity (input compared with itself).
| V4 Conservative | 16.4324 | 0.719748 | 1.7163 | 5.4714 | 0.996815 |

Original preservation is 1.0 by identity (input compared with itself).
| V4 Balanced | 16.7567 | 0.733091 | 1.6083 | 5.3665 | 0.995782 |

Original preservation is 1.0 by identity (input compared with itself).
| V4 Aggressive | 23.3363 | 0.817886 | 1.7161 | 5.8820 | 0.954935 |

Original preservation is 1.0 by identity (input compared with itself).
| V4 Selected | 23.3363 | 0.817886 | 1.7161 | 5.8820 | 0.954935 |

Original preservation is 1.0 by identity (input compared with itself).

## 3. Delta From Original

| Method | Δ PSNR | Δ SSIM | Δ UIQM | Δ UCIQE | Δ Preservation |
|---|---:|---:|---:|---:|---:|
| V3 Selected | +3.5387 | +0.046457 | -0.0552 | -0.2136 | -0.042100 |
| V4 Conservative | -0.4663 | -0.017560 | -0.1391 | -0.1370 | -0.003185 |
| V4 Balanced | -0.1420 | -0.004218 | -0.2472 | -0.2420 | -0.004218 |
| V4 Aggressive | +6.4376 | +0.080578 | -0.1394 | +0.2735 | -0.045065 |
| V4 Selected | +6.4376 | +0.080578 | -0.1394 | +0.2735 | -0.045065 |

## 4. V4 Candidate Comparison

| Candidate | PSNR | SSIM | UIQM | UCIQE | Preservation | Reference L1 | Consequence L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conservative | 16.4324 | 0.719748 | 1.7163 | 5.4714 | 0.996815 | 0.128085 | 0.132178 |
| Balanced | 16.7567 | 0.733091 | 1.6083 | 5.3665 | 0.995782 | 0.123354 | 0.130205 |
| Aggressive | 23.3363 | 0.817886 | 1.7161 | 5.8820 | 0.954935 | 0.055772 | 0.053098 |

## 5. V4 Selection Distribution

- Conservative: 0 (0.00%)
- Balanced: 0 (0.00%)
- Aggressive: 370 (100.00%)

Reference-L1 target counts: {'conservative': {'count': 0, 'percent': 0.0}, 'balanced': {'count': 7, 'percent': 1.8918918918918919}, 'aggressive': {'count': 363, 'percent': 98.10810810810811}}
Selector/reference-L1 agreement: 363/370 (98.11%).
Consequence-best agreement: 370/370 (100.00%).

## 6. V3 vs V4 Selected

| Metric | Mean V4−V3 | Median V4−V3 | Std paired Δ | V4 > V3 | V4 < V3 | Tied (±1e-6) |
|---|---:|---:|---:|---:|---:|---:|
| PSNR | +2.898965 | +3.018415 | 1.750604 | 94.05% | 5.95% | 0.00% |
| SSIM | +0.034121 | +0.029100 | 0.023111 | 97.03% | 2.97% | 0.00% |
| UIQM | -0.084201 | -0.079349 | 0.191412 | 28.92% | 71.08% | 0.00% |
| UCIQE | +0.487139 | +0.338529 | 1.229068 | 61.62% | 38.38% | 0.00% |
| PRESERVATION | -0.002965 | -0.002548 | 0.010442 | 38.38% | 61.62% | 0.00% |

## 7. Per-Image Improvements vs Original

| Model | Metric | Improved | Percentage | Mean paired Δ | Std paired Δ |
|---|---|---:|---:|---:|---:|
| V3 Selected | PSNR | 357 | 96.49% | +3.538662 | 1.837086 |
| V3 Selected | SSIM | 325 | 87.84% | +0.046457 | 0.045342 |
| V3 Selected | UIQM | 193 | 52.16% | -0.055211 | 0.434024 |
| V3 Selected | UCIQE | 153 | 41.35% | -0.213633 | 0.813505 |
| V3 Selected | PRESERVATION | 0 | 0.00% | -0.042100 | 0.008063 |
| V4 Selected | PSNR | 366 | 98.92% | +6.437627 | 2.281789 |
| V4 Selected | SSIM | 361 | 97.57% | +0.080578 | 0.048015 |
| V4 Selected | UIQM | 149 | 40.27% | -0.139411 | 0.473799 |
| V4 Selected | UCIQE | 205 | 55.41% | +0.273506 | 1.419914 |
| V4 Selected | PRESERVATION | 0 | 0.00% | -0.045065 | 0.009443 |

## 8. Statistical Summary

| Method | Metric | Mean | Median | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| Original | PSNR | 16.898694 | 17.010382 | 2.302236 | 10.149174 | 23.237372 |
| Original | SSIM | 0.737308 | 0.747613 | 0.073439 | 0.494946 | 0.878436 |
| Original | UIQM | 1.855487 | 1.756156 | 0.702378 | 0.438411 | 4.647998 |
| Original | UCIQE | 5.608471 | 5.417822 | 1.951911 | 1.888390 | 13.029162 |
| Original | PRESERVATION | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| V3 Selected | PSNR | 20.437356 | 20.557228 | 2.518522 | 12.122509 | 27.326143 |
| V3 Selected | SSIM | 0.783765 | 0.792987 | 0.061349 | 0.524871 | 0.933071 |
| V3 Selected | UIQM | 1.800276 | 1.666408 | 0.808027 | 0.371291 | 5.242932 |
| V3 Selected | UCIQE | 5.394838 | 5.132521 | 1.973639 | 1.640367 | 13.312033 |
| V3 Selected | PRESERVATION | 0.957900 | 0.958959 | 0.008063 | 0.906531 | 0.986653 |
| V4 Conservative | PSNR | 16.432374 | 16.704870 | 2.030695 | 10.172010 | 22.950053 |
| V4 Conservative | SSIM | 0.719748 | 0.733883 | 0.080907 | 0.383098 | 0.871035 |
| V4 Conservative | UIQM | 1.716337 | 1.588755 | 0.703185 | 0.406134 | 4.526424 |
| V4 Conservative | UCIQE | 5.471425 | 5.286427 | 1.919214 | 1.675600 | 12.932305 |
| V4 Conservative | PRESERVATION | 0.996815 | 0.999301 | 0.004430 | 0.978107 | 0.999772 |
| V4 Balanced | PSNR | 16.756659 | 16.937669 | 2.138078 | 10.461437 | 24.009707 |
| V4 Balanced | SSIM | 0.733091 | 0.747366 | 0.076832 | 0.412310 | 0.876732 |
| V4 Balanced | UIQM | 1.608250 | 1.498072 | 0.692276 | 0.388023 | 4.485564 |
| V4 Balanced | UCIQE | 5.366520 | 5.280947 | 1.833178 | 1.760727 | 11.526789 |
| V4 Balanced | PRESERVATION | 0.995782 | 0.997255 | 0.003480 | 0.980346 | 0.998609 |
| V4 Aggressive | PSNR | 23.336321 | 23.593928 | 2.561842 | 14.424338 | 30.655434 |
| V4 Aggressive | SSIM | 0.817886 | 0.829089 | 0.055337 | 0.600243 | 0.948128 |
| V4 Aggressive | UIQM | 1.716075 | 1.608892 | 0.748500 | 0.338294 | 4.908113 |
| V4 Aggressive | UCIQE | 5.881977 | 5.457985 | 2.291133 | 1.800586 | 13.232511 |
| V4 Aggressive | PRESERVATION | 0.954935 | 0.954922 | 0.009443 | 0.918155 | 0.977144 |
| V4 Selected | PSNR | 23.336321 | 23.593928 | 2.561842 | 14.424338 | 30.655434 |
| V4 Selected | SSIM | 0.817886 | 0.829089 | 0.055337 | 0.600243 | 0.948128 |
| V4 Selected | UIQM | 1.716075 | 1.608892 | 0.748500 | 0.338294 | 4.908113 |
| V4 Selected | UCIQE | 5.881977 | 5.457985 | 2.291133 | 1.800586 | 13.232511 |
| V4 Selected | PRESERVATION | 0.954935 | 0.954922 | 0.009443 | 0.918155 | 0.977144 |

## 9. Quality Trade-Offs

- V4 Selected vs Original, PSNR: 366 improved, 4 declined, 0 tied.
- V4 Selected vs Original, SSIM: 361 improved, 9 declined, 0 tied.
- V4 Selected vs Original, UIQM: 149 improved, 221 declined, 0 tied.
- V4 Selected vs Original, UCIQE: 205 improved, 165 declined, 0 tied.
- V4 Selected vs Original, PRESERVATION: 0 improved, 370 declined, 0 tied.
- UIQM/UCIQE are no-reference underwater quality measures, not ground-truth reconstruction accuracy.
- Preservation is a separate model score and is not established by changes in UIQM/UCIQE.

## 10. Extreme Cases
- max_psnr_improvement: `n01496331_4137.jpg`; Original 17.971388, V4 Selected 30.655434, Δ +12.684046.
- max_psnr_degradation: `n01664065_12383.jpg`; Original 18.317634, V4 Selected 15.996577, Δ -2.321057.
- max_ssim_improvement: `n01914609_1112.jpg`; Original 0.498939, V4 Selected 0.752881, Δ +0.253942.
- max_ssim_degradation: `n02607072_11391.jpg`; Original 0.818463, V4 Selected 0.719103, Δ -0.099360.

## 11. Scale and Diversity Diagnostics

V4 learned gains: {'conservative': 0.06605783849954605, 'balanced': 0.13242442905902863, 'aggressive': 0.26391690969467163}
Mean residual L2: {'conservative': {'mean': 5.5906215203774945, 'median': 1.9030518531799316, 'std': 6.37027426708888, 'min': 0.3789343535900116, 'max': 22.548276901245117}, 'balanced': {'mean': 9.49665776265634, 'median': 5.9874961376190186, 'std': 7.143755967847774, 'min': 2.53653883934021, 'max': 31.535381317138672}, 'aggressive': {'mean': 54.24485231347986, 'median': 54.47873497009277, 'std': 9.410750362589747, 'min': 21.16628074645996, 'max': 78.06531524658203}}
Mean absolute residual: {'conservative': {'mean': 0.01011665077908342, 'median': 0.0022636346984654665, 'std': 0.01246230842162029, 'min': 0.0005218746373429894, 'max': 0.04678463563323021}, 'balanced': {'mean': 0.015734893559302028, 'median': 0.01062100101262331, 'std': 0.01181265450898346, 'min': 0.004421507939696312, 'max': 0.05595299229025841}, 'aggressive': {'mean': 0.1058087963510204, 'median': 0.1072472333908081, 'std': 0.019435735229659256, 'min': 0.03523510694503784, 'max': 0.16021084785461426}}
Residual cosine similarity: {'conservative_vs_balanced': {'mean': 0.7702908213980294, 'median': 0.8959588408470154, 'std': 0.2883929303338006, 'min': -0.5482488870620728, 'max': 0.9891144037246704}, 'balanced_vs_aggressive': {'mean': 0.3533417444223085, 'median': 0.6508150398731232, 'std': 0.5906272120263765, 'min': -0.6279568076133728, 'max': 0.9832583069801331}, 'conservative_vs_aggressive': {'mean': -0.02537809930666274, 'median': -0.28838589787483215, 'std': 0.6821375132900269, 'min': -0.8874837160110474, 'max': 0.9641557931900024}}

## 12. Configuration, Hashes, and Safety

- V3 SHA before/after: `80163DBA423CCDE672318A3DD4F93B68982ADE6A0CD3ED737FA8D326948D0555` / `80163DBA423CCDE672318A3DD4F93B68982ADE6A0CD3ED737FA8D326948D0555`
- V4 SHA before/after: `5C877EDD0094E0C3F35AD977B5EA7651FFA7E22FBF4C38C15006AECE19E7C8D7` / `5C877EDD0094E0C3F35AD977B5EA7651FFA7E22FBF4C38C15006AECE19E7C8D7`
- V2 SHA before/after: `BBC8027FF23DCD477CD2E4BD0C95D7A42222AAAAE2E6A6197AF3D198149C53D1` / `BBC8027FF23DCD477CD2E4BD0C95D7A42222AAAAE2E6A6197AF3D198149C53D1`
- Only `test.txt` was evaluated. No training, test tuning, or checkpoint changes occurred.
- Evaluator: `diagnostics/v3_v4_final_test_eval.py`; metric functions from `src/evaluate_test.py:calculate_metrics, uiqm, uciqe`.

## 13. Scientific Interpretation and Limitations

On 370 held-out test images, V4 Selected mean PSNR/SSIM were 23.3363/0.817886, versus V3 Selected 20.4374/0.783765. V4 selected Aggressive on 370 images and matched the reference-L1-best candidate on 363/370. These are measurements, not a winner claim; UIQM/UCIQE are no-reference metrics, and preservation is a separate model score.

## 14. Exact Next Step

Freeze these test results. Review the metric trade-offs and validation-selected epoch, then plan a controlled ablation on the same train/validation protocol. Do not tune V4 using this test set.
