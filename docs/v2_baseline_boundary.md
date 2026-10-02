# Historical V2 Baseline Boundary

## Terminology

- **V2** is the previously recorded baseline experiment, represented here by its historical validation results.
- **V2 checkpoint** is a frozen historical artifact. The checkpoint is not itself proof that the historical experiment can be reproduced from the current repository.
- **Fixed-preservation experiment** is the new preservation mechanism documented in `preservation_method.md`; it is not V2.
- **V3** is the future trained model using the fixed-preservation mechanism. V3 training has not started.

## Previously recorded V2 baseline

The following values are the **Previously recorded V2 baseline**. They are historical results, not a fresh evaluation from the current repository.

| Output | PSNR | SSIM | UIQM | UCIQE |
|---|---:|---:|---:|---:|
| Original | 16.8987 | 0.7373 | 1.8555 | 5.6085 |
| Conservative | 19.0630 | 0.7753 | 1.4816 | 5.7313 |
| Balanced | 20.2636 | 0.7781 | 1.6061 | 5.3645 |
| Aggressive | 17.5195 | 0.7059 | 2.2838 | 5.0216 |
| Selected | 20.3166 | 0.7786 | 1.5943 | 5.3520 |

Historical selection distribution:

| Candidate | Selection rate |
|---|---:|
| Conservative | 5.95% |
| Balanced | 89.46% |
| Aggressive | 4.59% |

These results must not be attributed to the current fixed-preservation implementation.

## Frozen checkpoint artifact

The protected artifact is `checkpoints/v2/paper1_v2_epoch_006.pth`. Do not overwrite or modify it.

The checkpoint contains the legacy trainable preservation branch, including state-dictionary entries under `information_preservation.feature_extractor.*` and `information_preservation.preservation_head.*`. The exact original preservation forward implementation is not recoverable from the current repository. Strict loading into the current model fails structurally because the current implementation has a different preservation module and parameter names.

Therefore, this checkpoint cannot currently be used to claim exact reproducibility of the original V2 experiment. Do not infer or reconstruct the missing mathematical forward from checkpoint architecture alone.

The existing generated files under `results/v2_baseline/` and `results/inference/` were produced before the strict-load cleanup through an earlier non-strict loading path into the current model. They are not the historical V2 results and must not be presented as an exact V2 reproduction. The current evaluation scripts now use `strict=True` and fail clearly on the frozen checkpoint. These generated files are retained unchanged.

## New fixed-preservation experiment

The current fixed multi-scale grayscale and Sobel-gradient formulation is a separate experiment. Its monotonicity test checks and reports the scores for identity, mild, strong, and severe transformations, and requires:

`identity > mild > strong > severe`

These test scores characterize the new mechanism only. They are not historical V2 results and must not be merged into the V2 baseline record.

## Reproducibility status

- **Previously recorded V2 baseline:** historical metrics are available above.
- **Exact V2 reproduction from this repository:** not established; the original preservation forward implementation is missing.
- **V2 checkpoint artifact:** frozen and retained, but structurally incompatible with the current model under strict loading.
- **Fixed-preservation experiment:** documented separately and monotonicity-tested.
- **V3:** not trained.