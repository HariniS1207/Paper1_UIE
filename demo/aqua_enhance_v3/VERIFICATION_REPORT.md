# AquaEnhance V3 verification report

## Checkpoint and runtime

- Checkpoint: `checkpoints/v7_enhancement_only_v3/paper1_v7_enhancement_only_v3_best_epoch_009.pth`
- Expected and verified SHA-256: `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA`
- Metadata epoch: 9; strict `Paper1ModelV7.load_state_dict(..., strict=True)` completed with all keys matched.
- Inference runtime: Python 3.12.0, PyTorch 2.13.0+cu130, CUDA available; NVIDIA GeForce RTX 3050 6GB Laptop GPU selected.
- UI runtime: Gradio 6.29.0 in the existing project `.venv`.
- Original V7 checkpoint SHA-256: `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`.
- V1 checkpoint SHA-256: `119A2C58DCC3274F0D87E19442A4B7CF41DDE48447C009E8BDA5AAA06186E502`.
- V2 checkpoint SHA-256: `1822A9D11D80BB46C0D4EAD88BF64C1478E4B84A1C77D050046A9556625AA35D`.
- V3 checkpoint SHA-256: `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA`.
- All four checkpoint hashes were re-read after app verification and remained unchanged.
- Environment note: Gradio is unavailable in `TestTorchEnv`; PyTorch import in `.venv` is blocked by the pre-existing Application Control policy. The app therefore separates the Gradio front end and PyTorch backend across the two already-existing environments. No packages or environment/security settings were changed.

## Inference path

Uses `Paper1ModelV7` from `src/models/paper1_model_v7.py` and the checkpoint's `model_state_dict`, loaded strictly. Inference uses `eval()` and `torch.inference_mode()`. The interface shows the three real `candidates` from the forward output and the real `selected_image`; the latter is not substituted with a guessed candidate. The backend checks the selector index and probabilities and verifies finite, in-range RGB output tensors.

Input handling applies EXIF orientation, converts to RGB, scales to float32 `[0,1]`, adds no mean/std normalization, and preserves original dimensions. No resize is applied. To satisfy the actual V7 consequence-check decoder's four stride-2 downsampling/upsampling stages, the backend pads only the right/bottom edge to a multiple of 16 and crops output tensors back to the unpadded image size. This small edge-context change is disclosed in the UI status and README.

## Per-upload metric panel

The Gradio UI imports `load_canonical_metric_functions` from `src/compute_v7_quality_metrics.py`. That helper extracts the existing function definitions from `src/evaluate_test.py`; it does not duplicate formulas or import/load PyTorch. UIQM and UCIQE are called directly on both images after RGB conversion to float32 `[0,1]`. PSNR/SSIM use the canonical combined `calculate_metrics` function with `data_range=1.0` and `channel_axis=2` for both images against the same reference.

Canonical metric source SHA-256 (`src/evaluate_test.py`): `728F00EE0D80BD98766D5CB155D1BC1AD96693B7C48A3EEC147F4B6DF75C0705` (unchanged).

The optional reference is EXIF-oriented and converted to RGB using the same path as the source image. The user must confirm it is the paired, spatially aligned ground truth and same color space. Its post-orientation dimensions must exactly match the source; the app does not resize or register it. Without a reference, PSNR/SSIM display `N/A — ground-truth reference required`. UIQM/UCIQE changes are descriptive no-reference scores, not a guarantee of perceptual improvement. Relative change uses `(enhanced − original) / original × 100` and is unavailable when the original value is zero.

## Checks performed

- Verified configured checkpoint existence, expected SHA-256, epoch metadata, strict state-dictionary load, and finite model state tensors.
- Ran in-memory synthetic inference using a 47×31 RGB image. The model input was edge-padded to 48×32; original, all three candidate images, and selected output were finite RGB images at 47×31.
- Confirmed the selected output matches the actual candidate indicated by the selector, returned valid selection probabilities, and did not modify the input image bytes.
- Synthetic inference wrote no files. No dataset or test images were read.
- Started both services and confirmed the Gradio page responds with HTTP 200 at `http://127.0.0.1:7860/`; backend `/health` reports ready with the expected epoch-9 checkpoint, hash, and RTX 3050 device.
- Called the live Gradio `preview_upload` and `enhance` APIs with synthetic input and a pre-existing 256×256 local qualitative demo image. Both returned the original preview, before/after slider images, downloadable enhanced image, and all three named candidate previews at the input dimensions. Input bytes and source SHA-256 were unchanged.
- Both listeners are loopback-only: UI `127.0.0.1:7860` and backend `127.0.0.1:8768`. No uploaded-image result was saved in the project; Gradio's configured temporary cache is cleaned after one hour.
- Added the per-upload metrics panel and restarted the UI; Gradio page again returned HTTP 200. The existing checkpoint remained at SHA-256 `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA`.
- Tested a deterministic synthetic 32×32 RGB input twice: UIQM/UCIQE values were finite and exactly repeatable. With no reference, PSNR/SSIM displayed the required N/A text. A generated, same-size aligned RGB reference produced finite PSNR/SSIM. A 32×31 reference was rejected with an explicit dimensions error.
- Checked the exact-match reference edge case: canonical PSNR infinity is displayed as `∞ (exact match)` with an unavailable delta when both images are exact matches.
- Compared inference output across no-reference and paired-reference calls: before/after original and selected images and all three candidate outputs were pixel-identical, confirming the metric feature does not change inference or selector results. Temporary test images were deleted after testing.
- Queried the live Gradio API after restart: `/enhance` exposes the original upload, optional reference, alignment confirmation, and metrics table; actual no-reference and reference requests returned all five outputs and the expected table states.
- No research test split was accessed. No uploaded image/result was saved permanently by the feature; Gradio continues to use its one-hour temporary cache.

## Run status and limitations

The app is running locally at `http://127.0.0.1:7860/` (UI listener PID 2688; backend listener PID 18280). The app uses two local processes as documented in `README.md`; both bind only to `127.0.0.1`, and Gradio sharing is disabled. Validation-set values shown in the UI are checkpoint-level results (V3 best epoch 9), distinct from the per-upload panel. Reference scores depend on a true aligned ground truth; no-reference UIQM/UCIQE are heuristic proxies. Downstream detection/classification performance is not evaluated.
