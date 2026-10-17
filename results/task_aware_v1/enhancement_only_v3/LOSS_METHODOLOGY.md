# V3 loss methodology

## Primary comparable objective

V3 uses the existing `Paper1LossV7` without changes:

`L_V7 = L_reconstruction + 0.2 L_preservation + 0.2 L_consequence + 0.5 L_selection`.

All five components (including total) are logged separately for training and validation. The validation total under this exact formula selects the checkpoint and is the only value compared to the historical V7 0.2 target.

## Candidate auxiliary losses considered but disabled

A pixel-distance term `mean(abs(I_V3 - I_V7))` would constrain the output to the original model; it would have direct gradients through the new candidate generator, but similarity alone does not establish perceptual quality or preserve task-relevant cues. An SSIM term `1 - SSIM(I_V3, I_GT)` would need a differentiable SSIM implementation, window and data-range definition, and a coefficient chosen on validation data; enabling it would alter the objective and complicate comparison to V7. Neither term is enabled. No coefficients are tuned and no differently defined experimental total is reported.

Inputs and outputs are RGB float tensors in `[0,1]`; the unchanged V7 losses use the existing model outputs and reference tensors. Candidate-generation gradients flow through reconstruction and selector utility paths. Preservation and consequence measurements remain the fixed V7 definitions; the model computes these measurements under `no_grad`, so this experiment does not claim a gradient from those measurements into the candidate generator.

## Optimization

A fresh Adam optimizer starts from original V7 weights (no V1/V2 weights or optimizer state). Candidate generator parameters use `1e-5`; the already V7-trained selection network uses `5e-6`, reflecting their distinct roles and the observed V1/V2 selector sensitivity. Weight decay is zero. A validation-total ReduceLROnPlateau scheduler halves each group's rate after two bad checks, with documented per-group floors. Gradient norm clipping is 1.0. The run is capped at 20 epochs.

No detector/classifier loss is included. The supervisor's historical task losses are not used as constants. Downstream performance remains NOT EVALUATED pending approved models, interfaces, labels and evaluation protocol.
