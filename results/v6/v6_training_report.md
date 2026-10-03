# V6 Training Report

## Result

**Failed adaptive-selection experiment.** The selector chose Aggressive for all 370 validation images, including the 3 Conservative/Balanced targets. Do not freeze V6 as an adaptive selector or run the held-out test.

## Method and threshold

Before V6 training, training-only analysis evaluated relative reference-L1 gaps for each candidate on 2,960 train images using frozen V4 best. Means / medians / SDs: Conservative 1.48977 / 1.47689 / 0.63798; Balanced 1.38952 / 1.36609 / 0.61588; Aggressive 0.00194 / 0 / 0.02866. Examined tolerances: 1%, 2%, 3%, 5%, 7.5%, 10%, 15%. Froze 5% as an interpretable maximum relative fidelity penalty; at 5%, 25 images had >1 eligible candidate and target counts were 12/21/2927. See [complete training-only sweep](../diagnostics/v6_selection/v6_training_only_threshold_analysis.md).

Target: mark c eligible iff `(L1(c)-best_L1)/(best_L1+1e-8) <= 0.05`; among eligible candidates minimize equal mean train-median/IQR normalized preservation distortion and consequence L1. Medians [0.00441405, 0.11060812], IQRs [0.03656310, 0.07987094]. V4 generator/base loss retained.

## Training

Seed 42; batch 8; Adam, lr 1e-4; 20 epochs; AMP; V4 best initialization; same train/validation split and preprocessing; loss weights reconstruction/preservation/consequence/selection = 1.0/0.2/0.2/0.5. Best epoch 1; best validation total 0.084553 (recon 0.055026, preservation 0.018730, consequence 0.053130, selection 0.030310). Epoch 20 validation total 0.117320.

AMP overflow at epoch 19 batch 351: no optimizer step applied; replay gradients became finite at scaler 4096 and below; resumed deterministically from completed epoch 18 at 4096 and finished all 20 epochs.

## Validation findings

Target counts C/B/A: 2/1/367. Selector counts: 0/0/370. Agreement 99.19%, mean entropy 0.01810 nats, mean utility margin 8.1371. Selected mean metrics: PSNR 23.3963, SSIM 0.8140, UIQM 1.7412, UCIQE 5.8089. Candidate PSNR/SSIM: Conservative 16.402/0.7171, Balanced 16.005/0.7116, Aggressive 23.396/0.8140. Preservation scores C/B/A: 0.9952/0.9944/0.9545. Branches remained distinct (candidate distances C-B/B-A/C-A 0.01450/0.12617/0.11755; residual cosine 0.3814/-0.5218/0.0795; residual L2 7.973/11.826/54.965).

Severity quartiles all selected Aggressive, so condition-dependent class choice is absent. Per-group quality, candidate quality, V3-V6 comparison and hashes are in [selector diagnosis](../diagnostics/v6_selection/v6_selector_diagnosis.md) and its JSON. V4 selected Aggressive everywhere, V5 Balanced everywhere; V6 target diversified three cases but trained selector remained collapsed.

## Checkpoint and disposition

Best checkpoint: `checkpoints/v6/paper1_v6_best_epoch_001.pth`; SHA-256 `6E4FFC2DFED363BB5748D90925F78D4D748A0805A7B46B77F81998432E205E0F`.
Keep this as a negative result. Diagnose the selector?s failure to reproduce minority targets before a new experiment. V6 is not frozen for final evaluation.

**TEST SET ACCESSED: NO**
