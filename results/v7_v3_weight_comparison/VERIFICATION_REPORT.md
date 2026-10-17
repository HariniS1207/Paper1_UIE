# V7/V3 handoff verification report

## Audit baseline

- Repository: `Paper1_UIE`; active branch at audit: `master`; configured `origin`: `https://github.com/HariniS1207/Paper1_UIE.git`.
- Initial `git status --short`: no staged changes and no modified tracked files. The workspace already contained untracked V1/V2/V3 checkpoint directories, V3 training/evaluation files, the AquaEnhance app, local demo result folders, and generated working-image outputs. These were preserved; only files enumerated in the reviewed stage set are included in this handoff.
- `.venv/` and `data/EUVP/` are excluded by `.gitignore`; the dataset directory was not staged. No `.gitattributes` or repository Git LFS rules are configured. The original V7 checkpoint is already tracked.
- Initial `git ls-remote origin refs/heads/master` could not connect because sandbox network access to GitHub is restricted. Remote branch state and push therefore require a later approved networked Git operation.

## Checkpoints and distribution decision

| Artifact | Relative path | Bytes | SHA-256 | Distribution |
|---|---|---:|---|---|
| Original V7 baseline | `checkpoints/v7/paper1_v7_best_epoch_019.pth` | 8,937,437 | `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E` | Already tracked; remain frozen |
| V1 selected | `checkpoints/v7_enhancement_only_v1/paper1_v7_enhancement_only_best_epoch_019.pth` | 8,493,579 | `119A2C58DCC3274F0D87E19442A4B7CF41DDE48447C009E8BDA5AAA06186E502` | Preserved locally; not needed for V7/V3 notebook |
| V2 selected | `checkpoints/v7_enhancement_only_v2/paper1_v7_enhancement_only_best_epoch_019.pth` | 8,492,235 | `1822A9D11D80BB46C0D4EAD88BF64C1478E4B84A1C77D050046A9556625AA35D` | Preserved locally; not needed for V7/V3 notebook |
| V3 selected epoch 9 | `checkpoints/v7_enhancement_only_v3/paper1_v7_enhancement_only_v3_best_epoch_009.pth` | 12,047,106 | `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA` | Included as ordinary Git file; no external artifact service required |

GitHub documents a hard 100 MiB per-file limit for ordinary Git repositories; the V3 checkpoint is 12,047,106 bytes, and the V7 checkpoint is already tracked. No LFS installation or configuration was performed. V1/V2 weights and full local training artifacts were not staged. EUVP images, test images, uploaded images, and generated local image outputs are excluded.

## Verified experiment facts

- Split manifests contain 2,960 train, 370 validation, and 370 test identifiers; intersections are zero. V3 preflight audited identifiers to verify disjointness, but did not load test images. Training and selection used train/validation only.
- V3 completed 20 epochs, selected epoch 9 by minimum validation objective `0.180641507901`, applied 7,400 successful optimizer updates, and recorded zero non-finite retries/AMP interventions.
- Loss: `L_V7 = reconstruction + 0.2 × preservation + 0.2 × consequence + 0.5 × selection`.
- Training scope: candidate generator and selection network (493,965 parameters). Frozen: condition encoder, information-preservation module, and consequence-check module.
- Training used seed 42, batch size 8, Adam with zero weight decay, candidate generator LR `1e-5`, selector LR `5e-6`, validation-total ReduceLROnPlateau (factor 0.5, patience 2, threshold `1e-4`), gradient clipping 1.0, and CUDA float16 AMP when available. No task detection/classification losses were evaluated.
- V3 validation means improved the total objective and PSNR/SSIM/UIQM/UCIQE. Preservation loss regressed from `0.035322277228` to `0.052129711332`; the result is not a uniform improvement across loss components.

## Weight comparison

`src/compare_v7_v3_weights.py` strictly loaded separate `Paper1ModelV7` instances and compared all matching state tensors. The generated CSV, JSON, and Markdown report are in this directory.

- 91 total state tensors: 77 parameters and 14 buffers.
- 44 exactly identical tensors; 47 changed tensors.
- 1,995,046 total elements; 411,459 changed elements (20.624036%).
- Changed parameters: 47 tensors and 411,459 elements; 411,459 / 493,965 trainable-scope elements changed.
- All 14 buffers and all three frozen modules are exactly identical. Changed tensors are in `candidate_generator` (33 parameter tensors) and `selection_network` (14 parameter tensors).
- Global aggregate mean absolute difference: `0.0002518695224713979`; global L2 difference: `1.132082935878477`; relative global L2 to V7: `0.00015297989476539245`. The parameter-only relative L2 is `0.027833294234684763`; trainable-scope relative L2 is `0.05233900865942608`. Relative L2 is the aggregate difference norm divided by the V7 norm; it is undefined when that denominator is zero.
- Two full executions produced byte-identical JSON, CSV, and Markdown outputs. The script verified both checkpoint hashes before loading and does not write checkpoint paths.

## Colab and inference checks

- The notebook JSON is valid, all 18 cells are present, and Python cells parse after removing the IPython `%pip` magic.
- Local model check: both actual checkpoints hash-verified and strict-loaded in the existing PyTorch/CUDA environment; V7 and V3 ran on the same synthetic 48×32 RGB image with replicate edge padding/cropping; all three V3 candidate outputs and the actual selector output were present and dimension-correct.
- Local metrics check: canonical functions from `src/evaluate_test.py` (source SHA-256 `728F00EE0D80BD98766D5CB155D1BC1AD96693B7C48A3EEC147F4B6DF75C0705`) produced finite UIQM/UCIQE and finite PSNR/SSIM with a same-size synthetic aligned reference. No research test split was accessed.
- The local PyTorch environment lacks scikit-image and the existing `.venv` blocks PyTorch import under the pre-existing Application Control policy. Model inference and canonical metric checks were therefore verified separately in the two existing environments. No packages or security settings were changed.
- No Google Colab runtime was available in this session. The notebook was not executed in Colab; only JSON/code syntax and the equivalent local model/metric steps were checked. Its cells explicitly verify checkpoint hashes/epochs, strict-load, use `eval()` and `torch.inference_mode()`, and use the real selected output.
- AquaEnhance V3 remains a separate local app. Its local verification is documented in `demo/aqua_enhance_v3/VERIFICATION_REPORT.md`; this handoff does not claim a Colab UI deployment.

## Git publication status

- Initial stage was empty. The exact staged file list and diff review are recorded in the handoff task session before commit.
- Commit: **pending**.
- Push and remote-branch verification: **pending** due sandbox network restriction. No remote, branch, or Git history has been changed by this report.

After a successful commit/push, replace the pending fields with the verified commit and remote branch state. Do not claim Colab execution unless someone runs the notebook in a Google Colab runtime.
