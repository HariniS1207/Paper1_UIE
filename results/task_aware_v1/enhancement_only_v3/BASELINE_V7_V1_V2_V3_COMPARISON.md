# Original V7, V1, V2 and V3 validation comparison

All checkpoints were independently evaluated on the same 370 validation pairs with RGB [0,1] preprocessing and the unchanged Paper1LossV7. Image quality uses the canonical project metric functions. Test data was not used.

| Result | V7 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| Total loss (lower) | 0.250245914749 | 0.245716525413 | 0.252804185893 | 0.180641507901 |
| Reconstruction loss (lower) | 0.073375016733 | 0.073502864910 | 0.073151643091 | 0.058406659639 |
| Preservation loss (lower) | 0.035322277228 | 0.034983630619 | 0.035391546051 | 0.052129711332 |
| Consequence loss (lower) | 0.077731877888 | 0.077970067794 | 0.077649545267 | 0.059971925375 |
| Selection loss (lower) | 0.308520128049 | 0.299245843859 | 0.314088648928 | 0.199629042552 |
| PSNR (higher) | 21.60594305 | 21.51331265 | 21.64381029 | 22.82398283 |
| SSIM (higher) | 0.79309426 | 0.79134396 | 0.79363146 | 0.81109402 |
| UIQM (higher) | 1.65974209 | 1.65661616 | 1.65826323 | 1.75147532 |
| UCIQE (higher) | 5.66041319 | 5.65138916 | 5.65893781 | 5.66623224 |

V3 best epoch: 9; checkpoint `checkpoints\v7_enhancement_only_v3\paper1_v7_enhancement_only_v3_best_epoch_009.pth`; SHA-256 `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA`.
V7 primary validation objective target <0.2: ACHIEVED (V3=0.180641507901).
Image-quality target status versus V7: PSNR IMPROVED, SSIM IMPROVED, UIQM IMPROVED, UCIQE IMPROVED.
Detection loss, classification loss, and classification accuracy: **NOT EVALUATED**; this project lacks the approved differentiable task interfaces and aligned labels.
