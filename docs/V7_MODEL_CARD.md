# V7 Frozen Model Card

## Identity

- **Model:** `Paper1ModelV7`
- **Checkpoint:** `checkpoints/v7/paper1_v7_best_epoch_019.pth`
- **Epoch:** 19
- **SHA-256:** `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`
- **Purpose:** Research model for paired underwater image enhancement with candidate selection.

## Inputs and outputs

- **Input:** One RGB underwater image. The V7 forward method does not require a reference image.
- **Candidates:** Conservative, Balanced, and Aggressive.
- **Selection:** Condition-aware utility from candidate features, input condition, preservation score, and consequence error; utilities are converted to softmax probabilities and the maximum is selected.
- **Output:** All three candidates, selection probabilities, and the selected candidate/image. The demo optionally accepts an aligned reference for PSNR/SSIM evaluation.
- **Demo preprocessing:** RGB conversion and Lanczos resize to 256 × 256 in `demo/app_v7.py`.

## Training and evaluation context

- **Training data:** Paired EUVP Underwater ImageNet subset.
- **Frozen test set:** 370 pairs, held out from V7 training and checkpoint selection according to the recorded final test report.
- **Test selection distribution:** Conservative 51 (13.78%), Balanced 62 (16.76%), Aggressive 257 (69.46%).
- **Reported V7 selected metrics:** PSNR 21.2027, SSIM 0.789558, UIQM 1.6703, UCIQE 5.6111, preservation 0.965145, consequence error 0.081368.

## Limitations and appropriate use

V7 selected Aggressive for most test images. Candidate variation is present, but the distribution is strongly biased. V4 had higher reported test PSNR and SSIM, so V7 is not claimed to be universally superior to V4 or optimal for every input. UIQM and UCIQE are no-reference quality measures. The frozen results are research evidence, not a guarantee of quality for arbitrary real-world images.

For new single-image inference, the EUVP dataset and reference image are unnecessary. The paired dataset and split manifests are needed to reproduce quantitative evaluation or train new experiments. Preserve the epoch 19 checkpoint and recorded test artifacts; label new experiments separately.
