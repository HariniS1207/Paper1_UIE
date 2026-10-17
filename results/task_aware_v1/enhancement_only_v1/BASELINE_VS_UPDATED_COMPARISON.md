# Baseline versus enhancement-only V1

Both models were evaluated on the same 370 EUVP validation pairs using the same V7 loss, preprocessing, model mode, batch size, and image-quality metric implementations. Test data was not loaded.

- Baseline: `checkpoints\v7\paper1_v7_best_epoch_019.pth`; SHA-256 `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`.
- Updated: `checkpoints\v7_enhancement_only_v1\paper1_v7_enhancement_only_best_epoch_019.pth`; SHA-256 `119A2C58DCC3274F0D87E19442A4B7CF41DDE48447C009E8BDA5AAA06186E502`; selected epoch 19.
- Lower is better for loss components; higher is better for PSNR, SSIM, UIQM, and UCIQE. Image-quality means use all 370 validation pairs.

| Measurement | Baseline | Updated | Updated − baseline |
|---|---:|---:|---:|
| Validation total loss (lower) | 0.250245914749 | 0.245716525413 | -0.004529389336 |
| Validation reconstruction loss (lower) | 0.073375016733 | 0.073502864910 | +0.000127848177 |
| Validation preservation loss (lower) | 0.035322277228 | 0.034983630619 | -0.000338646608 |
| Validation consequence loss (lower) | 0.077731877888 | 0.077970067794 | +0.000238189907 |
| Validation selection loss (lower) | 0.308520128049 | 0.299245843859 | -0.009274284190 |
| PSNR (higher) | 21.60594305 | 21.51331265 | -0.09263040 |
| SSIM (higher) | 0.79309426 | 0.79134396 | -0.00175030 |
| UIQM (higher) | 1.65974209 | 1.65661616 | -0.00312593 |
| UCIQE (higher) | 5.66041319 | 5.65138916 | -0.00902402 |

Detection loss, classification loss, and classification accuracy: **NOT EVALUATED BY THIS EXPERIMENT**. The supervisor must run both checkpoints through the same task models, preprocessing, labels, split, and metric definitions for a valid downstream comparison.
