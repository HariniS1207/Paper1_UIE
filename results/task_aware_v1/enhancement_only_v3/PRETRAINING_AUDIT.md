# V3 pre-training architecture and experiment audit

Audit completed before training. The baseline loaded strictly; the existing TestTorchEnv runtime and split manifests passed the preflight checks.

## Architecture and output path

`Paper1ModelV7.forward` computes a condition vector, generates conservative/balanced/aggressive candidates, computes fixed preservation and consequence measurements, scores candidates in `selection_network`, then selects one candidate with an argmax index. The selected candidate tensor is returned directly as `selected_image`. There is no separate final refinement or output layer.
The V7 training implementation freezes all parameters, then enables only `selection_network`; its nominal `requires_grad=True` defaults in model construction are not the training policy. V1/V2 follow the same selector-only scope. V3 instead updates the existing pretrained `candidate_generator` plus `selection_network`; this is a bounded architecture-supported scope, not an invented refinement layer.

## Parameter inventory

The table lists every parameter tensor in the strict-loaded V7 model. V3 trainable parameters are the candidate generator and selection network only; all condition, preservation, and consequence parameters are frozen.

| Parameter | Shape | Count | V3 status |
|---|---|---:|---|
| `condition_encoder.encoder.0.block.0.weight` | `[32, 3, 3, 3]` | 864 | FROZEN |
| `condition_encoder.encoder.0.block.1.weight` | `[32]` | 32 | FROZEN |
| `condition_encoder.encoder.0.block.1.bias` | `[32]` | 32 | FROZEN |
| `condition_encoder.encoder.1.block.0.weight` | `[64, 32, 3, 3]` | 18432 | FROZEN |
| `condition_encoder.encoder.1.block.1.weight` | `[64]` | 64 | FROZEN |
| `condition_encoder.encoder.1.block.1.bias` | `[64]` | 64 | FROZEN |
| `condition_encoder.encoder.2.block.0.weight` | `[128, 64, 3, 3]` | 73728 | FROZEN |
| `condition_encoder.encoder.2.block.1.weight` | `[128]` | 128 | FROZEN |
| `condition_encoder.encoder.2.block.1.bias` | `[128]` | 128 | FROZEN |
| `condition_encoder.encoder.3.block.0.weight` | `[256, 128, 3, 3]` | 294912 | FROZEN |
| `condition_encoder.encoder.3.block.1.weight` | `[256]` | 256 | FROZEN |
| `condition_encoder.encoder.3.block.1.bias` | `[256]` | 256 | FROZEN |
| `condition_encoder.projection.0.weight` | `[128, 256]` | 32768 | FROZEN |
| `condition_encoder.projection.0.bias` | `[128]` | 128 | FROZEN |
| `candidate_generator.feature_extractor.0.weight` | `[32, 3, 3, 3]` | 864 | TRAINABLE |
| `candidate_generator.feature_extractor.0.bias` | `[32]` | 32 | TRAINABLE |
| `candidate_generator.feature_extractor.2.weight` | `[64, 32, 3, 3]` | 18432 | TRAINABLE |
| `candidate_generator.feature_extractor.2.bias` | `[64]` | 64 | TRAINABLE |
| `candidate_generator.feature_extractor.4.weight` | `[128, 64, 3, 3]` | 73728 | TRAINABLE |
| `candidate_generator.feature_extractor.4.bias` | `[128]` | 128 | TRAINABLE |
| `candidate_generator.condition_projections.conservative.0.weight` | `[32, 128]` | 4096 | TRAINABLE |
| `candidate_generator.condition_projections.conservative.0.bias` | `[32]` | 32 | TRAINABLE |
| `candidate_generator.condition_projections.balanced.0.weight` | `[32, 128]` | 4096 | TRAINABLE |
| `candidate_generator.condition_projections.balanced.0.bias` | `[32]` | 32 | TRAINABLE |
| `candidate_generator.condition_projections.aggressive.0.weight` | `[32, 128]` | 4096 | TRAINABLE |
| `candidate_generator.condition_projections.aggressive.0.bias` | `[32]` | 32 | TRAINABLE |
| `candidate_generator.candidate_heads.conservative.0.weight` | `[64, 160, 3, 3]` | 92160 | TRAINABLE |
| `candidate_generator.candidate_heads.conservative.0.bias` | `[64]` | 64 | TRAINABLE |
| `candidate_generator.candidate_heads.conservative.2.weight` | `[32, 64, 3, 3]` | 18432 | TRAINABLE |
| `candidate_generator.candidate_heads.conservative.2.bias` | `[32]` | 32 | TRAINABLE |
| `candidate_generator.candidate_heads.conservative.4.weight` | `[3, 32, 3, 3]` | 864 | TRAINABLE |
| `candidate_generator.candidate_heads.conservative.4.bias` | `[3]` | 3 | TRAINABLE |
| `candidate_generator.candidate_heads.balanced.0.weight` | `[64, 160, 3, 3]` | 92160 | TRAINABLE |
| `candidate_generator.candidate_heads.balanced.0.bias` | `[64]` | 64 | TRAINABLE |
| `candidate_generator.candidate_heads.balanced.2.weight` | `[32, 64, 3, 3]` | 18432 | TRAINABLE |
| `candidate_generator.candidate_heads.balanced.2.bias` | `[32]` | 32 | TRAINABLE |
| `candidate_generator.candidate_heads.balanced.4.weight` | `[3, 32, 3, 3]` | 864 | TRAINABLE |
| `candidate_generator.candidate_heads.balanced.4.bias` | `[3]` | 3 | TRAINABLE |
| `candidate_generator.candidate_heads.aggressive.0.weight` | `[64, 160, 3, 3]` | 92160 | TRAINABLE |
| `candidate_generator.candidate_heads.aggressive.0.bias` | `[64]` | 64 | TRAINABLE |
| `candidate_generator.candidate_heads.aggressive.2.weight` | `[32, 64, 3, 3]` | 18432 | TRAINABLE |
| `candidate_generator.candidate_heads.aggressive.2.bias` | `[32]` | 32 | TRAINABLE |
| `candidate_generator.candidate_heads.aggressive.4.weight` | `[3, 32, 3, 3]` | 864 | TRAINABLE |
| `candidate_generator.candidate_heads.aggressive.4.bias` | `[3]` | 3 | TRAINABLE |
| `candidate_generator.gain_logits.aggressive_increment` | `[]` | 1 | TRAINABLE |
| `candidate_generator.gain_logits.balanced_increment` | `[]` | 1 | TRAINABLE |
| `candidate_generator.gain_logits.conservative` | `[]` | 1 | TRAINABLE |
| `selection_network.candidate_encoder.0.weight` | `[32, 3, 3, 3]` | 864 | TRAINABLE |
| `selection_network.candidate_encoder.0.bias` | `[32]` | 32 | TRAINABLE |
| `selection_network.candidate_encoder.2.weight` | `[64, 32, 3, 3]` | 18432 | TRAINABLE |
| `selection_network.candidate_encoder.2.bias` | `[64]` | 64 | TRAINABLE |
| `selection_network.candidate_projection.0.weight` | `[64, 64]` | 4096 | TRAINABLE |
| `selection_network.candidate_projection.0.bias` | `[64]` | 64 | TRAINABLE |
| `selection_network.condition_projection.0.weight` | `[64, 128]` | 8192 | TRAINABLE |
| `selection_network.condition_projection.0.bias` | `[64]` | 64 | TRAINABLE |
| `selection_network.utility_head.0.weight` | `[96, 194]` | 18624 | TRAINABLE |
| `selection_network.utility_head.0.bias` | `[96]` | 96 | TRAINABLE |
| `selection_network.utility_head.2.weight` | `[32, 96]` | 3072 | TRAINABLE |
| `selection_network.utility_head.2.bias` | `[32]` | 32 | TRAINABLE |
| `selection_network.utility_head.4.weight` | `[1, 32]` | 32 | TRAINABLE |
| `selection_network.utility_head.4.bias` | `[1]` | 1 | TRAINABLE |
| `consequence_check.encoder.0.weight` | `[32, 3, 3, 3]` | 864 | FROZEN |
| `consequence_check.encoder.0.bias` | `[32]` | 32 | FROZEN |
| `consequence_check.encoder.2.weight` | `[64, 32, 3, 3]` | 18432 | FROZEN |
| `consequence_check.encoder.2.bias` | `[64]` | 64 | FROZEN |
| `consequence_check.encoder.4.weight` | `[128, 64, 3, 3]` | 73728 | FROZEN |
| `consequence_check.encoder.4.bias` | `[128]` | 128 | FROZEN |
| `consequence_check.encoder.6.weight` | `[256, 128, 3, 3]` | 294912 | FROZEN |
| `consequence_check.encoder.6.bias` | `[256]` | 256 | FROZEN |
| `consequence_check.reconstruction_head.0.weight` | `[256, 128, 4, 4]` | 524288 | FROZEN |
| `consequence_check.reconstruction_head.0.bias` | `[128]` | 128 | FROZEN |
| `consequence_check.reconstruction_head.2.weight` | `[128, 64, 4, 4]` | 131072 | FROZEN |
| `consequence_check.reconstruction_head.2.bias` | `[64]` | 64 | FROZEN |
| `consequence_check.reconstruction_head.4.weight` | `[64, 32, 4, 4]` | 32768 | FROZEN |
| `consequence_check.reconstruction_head.4.bias` | `[32]` | 32 | FROZEN |
| `consequence_check.reconstruction_head.6.weight` | `[32, 3, 4, 4]` | 1536 | FROZEN |
| `consequence_check.reconstruction_head.6.bias` | `[3]` | 3 | FROZEN |

Trainable count: **493,965** (440,300 candidate-generator; 53,665 selector). Frozen module state hashes are checked each epoch.

## Prior experiment evidence and rationale

V1 used selector-only Adam at 1e-4 and recorded two non-finite gradient attempts at GradScaler scale 8192; its validation curve fluctuated and its image-quality means declined. V2 used selector-only Adam starting at 5e-5 with validation-driven halving to 1.25e-5, recorded zero non-finite retries, and still had a volatile validation curve; best validation total was 0.252804, with small PSNR/SSIM gains and UIQM/UCIQE declines. These observations do not prove a causal mechanism. V3 tests the explicit hypothesis that tuning the existing spatial candidate generator can alter image output quality beyond selector-only choice.
V3 uses a smaller 1e-5 rate for the pretrained candidate generator and 5e-6 for the previously optimized selector, Adam without weight decay, gradient clipping at 1.0, and a validation-total plateau scheduler. The original V7 objective remains unchanged. No V1/V2 initialization is used.

## Integrity and data gates

- Original V7 epoch-19 checkpoint strict load: passed; SHA-256 `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`.
- Runtime: Python 3.12.0, torch 2.13.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Manifest counts train/validation/test identifiers: 2960/370/370; all intersections zero. Test identifiers were read only to audit disjointness; test images are never loaded.
- One-batch preflight produced finite losses/gradients, updated both intended trainable modules, and left frozen modules and the baseline checkpoint unchanged.

## Limitations

The V7 objective does not provide differentiable detector/classifier feedback: its preservation/consequence measurements are fixed and evaluated under no-grad inside the model. This V3 run does not optimize approved downstream task losses. PSNR/SSIM/UIQM/UCIQE and the V7 objective are validation measurements only; no target is guaranteed.
