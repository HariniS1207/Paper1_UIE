# AquaEnhance — Underwater Image Enhancement

Local qualitative preview app for the Paper 1 V3 model. It loads the selected best epoch-9 V3 checkpoint, shows the model's actual Conservative/Balanced/Aggressive candidate images, and displays the V7 selector's actual selected output.

## Runtime layout

The existing environments split the required packages: `TestTorchEnv` has working PyTorch/CUDA but no Gradio; the project `.venv` has Gradio 6.29.0 but its PyTorch import is blocked by the existing Windows Application Control policy. To avoid installing packages or changing either environment, the app runs as two loopback-only processes:

- Inference backend: `TestTorchEnv` (`127.0.0.1:8768`)
- Gradio interface: project `.venv` (`127.0.0.1:7860`)

No firewall, security policy, package, or Python environment changes are needed.

## Launch

Open two PowerShell terminals in the project root.

Terminal 1 — model backend:

```powershell
Set-Location '<project-root>'
$torchPython = '<existing-PyTorch-environment>\Scripts\python.exe'
& $torchPython 'demo\aqua_enhance_v3\inference_backend.py'
```

Wait for the message that the V3 backend is ready. Terminal 2 — web interface:

```powershell
Set-Location '<project-root>'
& '.\.venv\Scripts\python.exe' 'demo\aqua_enhance_v3\app.py'
```

Open [http://127.0.0.1:7860/](http://127.0.0.1:7860/). Stop each process with **Ctrl+C** in its terminal. Both services bind only to `127.0.0.1`; Gradio sharing is disabled.

In this workspace, `TestTorchEnv` supplies the compatible PyTorch/CUDA runtime and `.venv` supplies Gradio. On another machine, set `$torchPython` to a local Python executable with compatible PyTorch installed, and use the existing `.venv` only if it contains Gradio. The processes can be stopped with **Ctrl+C** in their respective terminals.

## Inputs and outputs

Upload a JPG, JPEG, PNG, or WebP image. Images are EXIF-oriented, converted to RGB, and scaled to float32 `[0,1]`, matching the project transform. The app does not resize or stretch the image. Since the model's consequence decoder has four stride-2 stages, the backend edge-pads only the right and bottom edges to a multiple of 16, runs inference, then crops outputs back to the original dimensions. Maximum image size is 2,500,000 pixels and maximum side length is 4,096 pixels; an oversized image is rejected with guidance to resize it yourself.

The app previews the input, provides a before/after slider, an enhanced RGB image with a download control, and all three genuine candidate outputs. The selected output is the image returned by `Paper1ModelV7` after its actual adaptive argmax selector. Selector probabilities and the selected candidate name are shown.

### Per-upload metrics

After inference, the app compares the original RGB input and final selected V3 output using canonical functions extracted from `src/evaluate_test.py` through `src/compute_v7_quality_metrics.py`. Both are evaluated as RGB float32 arrays in `[0,1]` at the same original dimensions. UIQM combines colorfulness, sharpness, and contrast components; UCIQE combines chroma spread, luminance contrast, and saturation. These are heuristic no-reference measures, and a positive change does not guarantee a perceptual improvement. Relative percentage change is `(enhanced - original) / original × 100`; it is omitted when the original value is zero.

PSNR and SSIM appear as `N/A — ground-truth reference required` unless an optional paired reference is supplied. The reference must be a valid supported image, EXIF-oriented, convertible to RGB, exactly the same dimensions as the input, and confirmed by the user as spatially aligned and in the same color space. No resizing or registration is performed. PSNR measures pixel error and SSIM structural similarity; both can be misleading when the reference is not a true aligned ground truth. They are calculated for both original and enhanced images against that same reference. A zero-error PSNR is displayed as infinity, its mathematical value.

The backend hashes and strict-loads the configured checkpoint at startup. `app_config.json` sets the project-relative checkpoint, expected SHA-256 and epoch. The backend rejects checkpoint paths outside the project's `checkpoints` directory and refuses a hash or epoch mismatch.

Uploaded image data is sent only over loopback to the inference service. The app creates no run folders or permanent result files. Gradio uses its temporary upload/output cache and is configured to remove cached files older than one hour.

## Research context

The checkpoint-level V7/V3 objectives and validation metrics are separate from the per-upload panel and are explicitly identified as validation-set results. No research test split is read by this app. Detection loss, classification loss, and accuracy were not evaluated; image metrics do not establish downstream task improvement.

See `VERIFICATION_REPORT.md` for the checkpoint, runtime, preprocessing, and local inference verification record.
