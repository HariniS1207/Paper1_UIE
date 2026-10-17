# Baseline, V1 and V2 validation comparison

All three checkpoints use the same 370-pair validation split, V7 preprocessing and loss implementation. V1/V2 modify only the selector. No test images were used for selection or these evaluations.

- Original V7: `checkpoints\v7\paper1_v7_best_epoch_019.pth`; SHA-256 `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`.
- V1 selected epoch 19: `checkpoints\v7_enhancement_only_v1\paper1_v7_enhancement_only_best_epoch_019.pth`; SHA-256 `119A2C58DCC3274F0D87E19442A4B7CF41DDE48447C009E8BDA5AAA06186E502`.
- V2 selected epoch 19: `checkpoints\v7_enhancement_only_v2\paper1_v7_enhancement_only_best_epoch_019.pth`; SHA-256 `1822A9D11D80BB46C0D4EAD88BF64C1478E4B84A1C77D050046A9556625AA35D`.

## Validation losses

| Loss (lower is better) | Original V7 | V1 | V2 |
|---|---:|---:|---:|
| Total | 0.250245914749 | 0.245716525413 | 0.252804185893 |
| Reconstruction | 0.073375016733 | 0.073502864910 | 0.073151643091 |
| Preservation | 0.035322277228 | 0.034983630619 | 0.035391546051 |
| Consequence | 0.077731877888 | 0.077970067794 | 0.077649545267 |
| Selection | 0.308520128049 | 0.299245843859 | 0.314088648928 |

## Image-quality metrics (higher is better)

| Metric | Original V7 | V1 | V2 | V2 minus V7 |
|---|---:|---:|---:|---:|
| PSNR | 21.60594305 | 21.51331265 | 21.64381029 | +0.03786724 |
| SSIM | 0.79309426 | 0.79134396 | 0.79363146 | +0.00053720 |
| UIQM | 1.65974209 | 1.65661616 | 1.65826323 | -0.00147885 |
| UCIQE | 5.66041319 | 5.65138916 | 5.65893781 | -0.00147538 |

V2 meets the image-quality target for PSNR and SSIM, but not UIQM or UCIQE. Its best total validation loss is above 0.2 and also slightly above original V7. V1 results are shown for context; neither retraining is established as a uniform image-quality improvement.

V1: 20 epochs, best epoch 19, final LR 1.00e-04, non-finite retry events 2. V2: 20 epochs, best epoch 19, final LR 1.25e-5, 0 non-finite retry events and 0 AMP interventions. V2 applied 7400 optimizer updates; rejected attempts, if any, are not counted as updates.

Detection loss, classification loss and classification accuracy: **NOT EVALUATED**; approved downstream evaluation remains necessary.
