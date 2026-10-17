# Paper 1 V7 enhancement-only V2 experiment

## Outcome

V2 completed all 20 configured epochs. The validation-selected checkpoint is `checkpoints/v7_enhancement_only_v2/paper1_v7_enhancement_only_best_epoch_019.pth` (epoch 19), selected by minimum validation total under the unchanged V7 objective. Its SHA-256 is `1822A9D11D80BB46C0D4EAD88BF64C1478E4B84A1C77D050046A9556625AA35D`. Epoch 20 is the final resumable checkpoint; its payload agrees with the 20-epoch CSV and component history. No additional training was required.

## Protocol and integrity

V2 initialized from the original V7 epoch-19 checkpoint (SHA-256 `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`), which remains unchanged. Training scope was the existing 53,665-parameter selection network only. The original V7 enhancement loss coefficients were retained. LR began at 5e-5 and scheduler halving reached the configured 1.25e-5 floor. There were 7,400 successful optimizer updates, 0 non-finite gradient retries, and 0 AMP interventions. The validation split has 370 pairs. Test images were not loaded for training, tuning, selection or evaluation.

Strict checkpoint loading, finite forward/loss checks, frozen-module digest comparison, and repeat evaluation passed. Repeat V2 loss values matched exactly (max absolute difference 0.0). See `CHECKPOINT_INTEGRITY_REPORT.md` and `PRETRAINING_AUDIT.md`.

## Original V7, V1 and V2 results

See `BASELINE_VS_V1_V2.md` for all loss components and image-quality means. V2 best validation total was 0.252804185893, above the 0.2 target and slightly above original V7's 0.250245914749. PSNR and SSIM improved slightly over V7; UIQM and UCIQE decreased slightly. Therefore the loss threshold was not achieved and not all image-quality targets were achieved. Epoch 4's value 0.263539 was not the best; epoch 19 was.

## Downstream boundary and remaining work

Detection loss, classification loss and classification accuracy are **NOT EVALUATED**. The supervisor must evaluate the selected checkpoint with the same approved task models, preprocessing, labels and validation split used for the original model.

Training, validation comparison and integrity checks are complete. No test-data leakage was found. Remaining work is the downstream task evaluation and scientific review; V2 is not established as a uniform image-quality improvement.
