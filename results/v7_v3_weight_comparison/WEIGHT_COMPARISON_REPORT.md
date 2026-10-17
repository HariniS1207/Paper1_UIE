# V7–V3 checkpoint weight comparison

Both checkpoint state dictionaries were loaded strictly into separate `Paper1ModelV7` instances. The script only reads checkpoints and verifies their expected SHA-256 values.

## Checkpoints

| Version | Path | Epoch | SHA-256 | Size |
|---|---|---:|---|---:|
| V7 | `checkpoints/v7/paper1_v7_best_epoch_019.pth` | 19 | `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E` | 8,937,437 bytes |
| V3 | `checkpoints/v7_enhancement_only_v3/paper1_v7_enhancement_only_v3_best_epoch_009.pth` | 9 | `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA` | 12,047,106 bytes |

## Summary

| Category | Tensors | Exactly identical | Changed tensors | Elements | Changed elements | Changed elements (%) | Global L2 difference | Relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All State | 91 | 44 | 47 | 1,995,046 | 411,459 | 20.624036% | 1.1320829 | 0.00015297989476539245 |
| Parameters | 77 | 30 | 47 | 1,994,064 | 411,459 | 20.634192% | 1.1320829 | 0.027833294234684763 |
| Buffers | 14 | 14 | 0 | 982 | 0 | 0.000000% | 0 | 0.0 |
| Changed Trainable Parameters | 47 | 0 | 47 | 493,965 | 411,459 | 83.297197% | 1.1320829 | 0.05233900865942608 |

Changed trainable parameter elements: **411,459** of **493,965** trainable elements defined by the V3 script.
Frozen modules exactly identical: **True**.

## Changed tensors by model module

| Module | Parameter tensors changed | Changed parameter elements | Buffer tensors changed | Changed buffer elements |
|---|---:|---:|---:|---:|
| `candidate_generator` | 33 | 372,665 | 0 | 0 |
| `condition_encoder` | 0 | 0 | 0 | 0 |
| `consequence_check` | 0 | 0 | 0 | 0 |
| `information_preservation` | 0 | 0 | 0 | 0 |
| `selection_network` | 14 | 38,794 | 0 | 0 |

### Changed parameter and buffer names

- `candidate_generator`
  - Parameter: `candidate_generator.candidate_heads.aggressive.0.bias`
  - Parameter: `candidate_generator.candidate_heads.aggressive.0.weight`
  - Parameter: `candidate_generator.candidate_heads.aggressive.2.bias`
  - Parameter: `candidate_generator.candidate_heads.aggressive.2.weight`
  - Parameter: `candidate_generator.candidate_heads.aggressive.4.bias`
  - Parameter: `candidate_generator.candidate_heads.aggressive.4.weight`
  - Parameter: `candidate_generator.candidate_heads.balanced.0.bias`
  - Parameter: `candidate_generator.candidate_heads.balanced.0.weight`
  - Parameter: `candidate_generator.candidate_heads.balanced.2.bias`
  - Parameter: `candidate_generator.candidate_heads.balanced.2.weight`
  - Parameter: `candidate_generator.candidate_heads.balanced.4.bias`
  - Parameter: `candidate_generator.candidate_heads.balanced.4.weight`
  - Parameter: `candidate_generator.candidate_heads.conservative.0.bias`
  - Parameter: `candidate_generator.candidate_heads.conservative.0.weight`
  - Parameter: `candidate_generator.candidate_heads.conservative.2.bias`
  - Parameter: `candidate_generator.candidate_heads.conservative.2.weight`
  - Parameter: `candidate_generator.candidate_heads.conservative.4.bias`
  - Parameter: `candidate_generator.candidate_heads.conservative.4.weight`
  - Parameter: `candidate_generator.condition_projections.aggressive.0.bias`
  - Parameter: `candidate_generator.condition_projections.aggressive.0.weight`
  - Parameter: `candidate_generator.condition_projections.balanced.0.bias`
  - Parameter: `candidate_generator.condition_projections.balanced.0.weight`
  - Parameter: `candidate_generator.condition_projections.conservative.0.bias`
  - Parameter: `candidate_generator.condition_projections.conservative.0.weight`
  - Parameter: `candidate_generator.feature_extractor.0.bias`
  - Parameter: `candidate_generator.feature_extractor.0.weight`
  - Parameter: `candidate_generator.feature_extractor.2.bias`
  - Parameter: `candidate_generator.feature_extractor.2.weight`
  - Parameter: `candidate_generator.feature_extractor.4.bias`
  - Parameter: `candidate_generator.feature_extractor.4.weight`
  - Parameter: `candidate_generator.gain_logits.aggressive_increment`
  - Parameter: `candidate_generator.gain_logits.balanced_increment`
  - Parameter: `candidate_generator.gain_logits.conservative`
- `selection_network`
  - Parameter: `selection_network.candidate_encoder.0.bias`
  - Parameter: `selection_network.candidate_encoder.0.weight`
  - Parameter: `selection_network.candidate_encoder.2.bias`
  - Parameter: `selection_network.candidate_encoder.2.weight`
  - Parameter: `selection_network.candidate_projection.0.bias`
  - Parameter: `selection_network.candidate_projection.0.weight`
  - Parameter: `selection_network.condition_projection.0.bias`
  - Parameter: `selection_network.condition_projection.0.weight`
  - Parameter: `selection_network.utility_head.0.bias`
  - Parameter: `selection_network.utility_head.0.weight`
  - Parameter: `selection_network.utility_head.2.bias`
  - Parameter: `selection_network.utility_head.2.weight`
  - Parameter: `selection_network.utility_head.4.bias`
  - Parameter: `selection_network.utility_head.4.weight`

## Interpretation

The V3 training script marks `candidate_generator` and `selection_network` trainable and freezes `condition_encoder`, `information_preservation`, and `consequence_check`. Frozen-module state tensors are required by this script to match V7 exactly; a discrepancy causes the comparison to stop. Weight differences verify final checkpoint changes but do not alone prove which tensors were trainable; that conclusion comes from the training script and pretraining audit.

Changed parameter names are enumerated in `tensor_differences.csv` and `weight_comparison.json`; raw tensor values are not included.

## Metric definitions

Per-tensor mean absolute difference is the absolute elementwise difference sum divided by tensor size. Per-tensor L2 is the Euclidean norm of the difference; relative L2 is that norm divided by the V7 tensor norm and is N/A for a zero V7 norm. Aggregate L2 combines squares across the category before taking a square root. Integer/bool buffers use exact value comparisons and are converted only for magnitude statistics.

Run from the project root with `python src/compare_v7_v3_weights.py`. Output files can be regenerated without changing either checkpoint.
