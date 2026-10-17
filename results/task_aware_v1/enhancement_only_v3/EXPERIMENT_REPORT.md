# Paper 1 V7 controlled fine-tuning V3

V3 completed 20 epochs, initialized from original V7, with best validation epoch 9. Selected checkpoint: `checkpoints\v7_enhancement_only_v3\paper1_v7_enhancement_only_v3_best_epoch_009.pth`; SHA-256 `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA`.

Trainable scope: existing candidate generator and selector; condition encoder, fixed information-preservation module and consequence-check module frozen. Primary objective is unchanged Paper1LossV7: reconstruction + 0.2 preservation + 0.2 consequence + 0.5 selection. No optional similarity terms were added.

Best V3 V7-objective total: 0.180641507901; target below 0.2 achieved. See `BASELINE_V7_V1_V2_V3_COMPARISON.md` for component losses and image-quality metrics.

Integrity and repeatability checks passed; original V7, V1 and V2 artifacts remain unchanged. Test images were not used. Detection/classification losses and accuracy are NOT EVALUATED. The validation results do not establish downstream task improvement.
