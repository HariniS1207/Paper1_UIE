"""AquaEnhance Gradio UI; calls the loopback PyTorch inference backend."""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import math
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((APP_DIR / "app_config.json").read_text(encoding="utf-8"))
if CONFIG.get("ui_host") != "127.0.0.1" or CONFIG.get("backend_host") != "127.0.0.1":
    raise RuntimeError("AquaEnhance V3 is restricted to loopback; both configured hosts must be 127.0.0.1.")
BACKEND = f"http://{CONFIG['backend_host']}:{int(CONFIG['backend_port'])}"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from compute_v7_quality_metrics import load_canonical_metric_functions

CANONICAL_METRIC_FUNCTIONS, METRIC_SOURCE_SHA256 = load_canonical_metric_functions(PROJECT_ROOT)
CANONICAL_METRICS = CANONICAL_METRIC_FUNCTIONS["calculate_metrics"]


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def checkpoint_panel() -> str:
    relative = Path(CONFIG["checkpoint"])
    if relative.is_absolute() or ".." in relative.parts:
        return "**Checkpoint status:** unsafe checkpoint path in local configuration."
    checkpoint = (PROJECT_ROOT / relative).resolve()
    if not checkpoint.is_relative_to((PROJECT_ROOT / "checkpoints").resolve()):
        return "**Checkpoint status:** configured path is outside the project checkpoints directory."
    expected = CONFIG["expected_sha256"]
    if not checkpoint.is_file():
        return "**Checkpoint status:** unavailable. Start the inference backend after restoring the configured checkpoint."
    actual = hash_file(checkpoint)
    if actual != expected:
        return f"**Checkpoint status:** SHA-256 mismatch; inference backend will refuse to load it. Expected `{expected}`, found `{actual}`."
    return (f"**Checkpoint:** `{CONFIG['checkpoint']}`  \n**Best validation epoch:** {CONFIG['expected_epoch']}  \n"
            f"**SHA-256:** `{actual}`  \n**Validation-set results (not per-upload scores):** V7 objective 0.250245914749 → V3 objective 0.180641507901; "
            "PSNR 22.82398283; SSIM 0.81109402; UIQM 1.75147532; UCIQE 5.66623224.")


def image_from_b64(value: str) -> Image.Image:
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(BytesIO(raw)) as image:
            image.load()
            return image.convert("RGB").copy()
    except Exception as error:
        raise gr.Error(f"The local inference service returned an invalid image: {error}") from error


def load_upload(path: str | None) -> Image.Image:
    if not path:
        raise gr.Error("Choose a JPG, JPEG, PNG, or WebP image first.")
    try:
        with Image.open(path) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise gr.Error("Unsupported image format. Use JPG, JPEG, PNG, or WebP.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"Could not open this image: {type(error).__name__}: {error}") from error
    if image.width < 1 or image.height < 1:
        raise gr.Error("The uploaded image has an empty dimension.")
    if max(image.size) > int(CONFIG["max_image_side"]):
        raise gr.Error(f"Image sides must be at most {CONFIG['max_image_side']} pixels.")
    if image.width * image.height > int(CONFIG["max_image_pixels"]):
        raise gr.Error(f"The local limit is {int(CONFIG['max_image_pixels']):,} pixels. Resize the image yourself and upload it again.")
    return image


def preview_upload(path: str | None):
    return load_upload(path) if path else None


def _metric_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise gr.Error("Metrics require a finite RGB image in the [0,1] range.")
    return array


def _format_comparison(original: Image.Image, enhanced: Image.Image, reference_path: str | None,
                       aligned_confirmed: bool, progress) -> list[list[str]]:
    progress(0.62, desc="Calculating canonical UIQM and UCIQE")
    original_np, enhanced_np = _metric_array(original), _metric_array(enhanced)
    if original_np.shape != enhanced_np.shape:
        raise gr.Error("The original and enhanced images have different dimensions; metrics were not calculated.")
    original_clipped = np.clip(original_np, 0.0, 1.0)
    enhanced_clipped = np.clip(enhanced_np, 0.0, 1.0)
    scores = {
        "UIQM": (float(CANONICAL_METRIC_FUNCTIONS["uiqm"](original_clipped)),
                 float(CANONICAL_METRIC_FUNCTIONS["uiqm"](enhanced_clipped))),
        "UCIQE": (float(CANONICAL_METRIC_FUNCTIONS["uciqe"](original_clipped)),
                  float(CANONICAL_METRIC_FUNCTIONS["uciqe"](enhanced_clipped))),
    }
    if reference_path:
        if not aligned_confirmed:
            raise gr.Error("Confirm that the optional reference is the paired ground truth, spatially aligned, and in the same color space.")
        reference = load_upload(reference_path)
        if reference.size != original.size:
            raise gr.Error(f"Reference dimensions are {reference.width}×{reference.height}, but the uploaded image is {original.width}×{original.height}. No resizing or alignment was applied.")
        reference_np = _metric_array(reference)
        if reference_np.shape != original_np.shape:
            raise gr.Error("Reference RGB dimensions do not match the original image. No reference metrics were calculated.")
        progress(0.78, desc="Calculating PSNR and SSIM against the paired reference")
        original_ref = CANONICAL_METRICS(original_np, reference_np)
        enhanced_ref = CANONICAL_METRICS(enhanced_np, reference_np)
        scores["PSNR"] = (float(original_ref[0]), float(enhanced_ref[0]))
        scores["SSIM"] = (float(original_ref[1]), float(enhanced_ref[1]))
    else:
        scores["PSNR"] = (None, None)
        scores["SSIM"] = (None, None)

    rows = []
    for name in ("UIQM", "UCIQE", "PSNR", "SSIM"):
        before, after = scores[name]
        if before is None:
            rows.append([name, "N/A — ground-truth reference required", "N/A — ground-truth reference required", "N/A", "N/A"])
            continue
        if math.isnan(before) or math.isnan(after) or before == -math.inf or after == -math.inf:
            raise gr.Error(f"Canonical {name} calculation returned a non-finite value; no result is shown.")
        if math.isinf(before) or math.isinf(after):
            if name != "PSNR":
                raise gr.Error(f"Canonical {name} calculation returned an unexpected non-finite value; no result is shown.")
            before_text = "∞ (exact match)" if math.isinf(before) else f"{before:.6f}"
            after_text = "∞ (exact match)" if math.isinf(after) else f"{after:.6f}"
            delta_text = "N/A — both are exact matches" if math.isinf(before) and math.isinf(after) else ("+∞" if math.isinf(after) else "−∞")
            rows.append([name, before_text, after_text, delta_text, "N/A — infinite value"])
            continue
        delta = after - before
        relative = f"{(delta / before * 100):+.2f}%" if before != 0 else "N/A — original is zero"
        rows.append([name, f"{before:.6f}", f"{after:.6f}", f"{delta:+.6f}", relative])
    return rows


def enhance(path: str | None, reference_path: str | None, aligned_confirmed: bool, progress=gr.Progress()):
    if not path:
        raise gr.Error("Upload a JPG, JPEG, PNG, or WebP image first.")
    try:
        progress(0.1, desc="Preparing image locally")
        original = load_upload(path)
        if reference_path:
            if not aligned_confirmed:
                raise gr.Error("Confirm that the optional reference is the paired ground truth, spatially aligned, and in the same color space.")
            reference_check = load_upload(reference_path)
            if reference_check.size != original.size:
                raise gr.Error(f"Reference dimensions are {reference_check.width}×{reference_check.height}, but the uploaded image is {original.width}×{original.height}. No resizing or alignment was applied.")
        payload = BytesIO()
        original.save(payload, format="PNG")
        if payload.tell() > int(CONFIG["max_upload_bytes"]):
            raise gr.Error("The encoded image exceeds the 25 MB local upload limit.")
        request = Request(BACKEND + "/enhance", data=payload.getvalue(),
                          headers={"Content-Type": "image/png", "Content-Length": str(payload.tell())}, method="POST")
        progress(0.25, desc="Running V3 selector and candidate generator")
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try: detail = json.loads(error.read().decode("utf-8")).get("error", str(error))
        except Exception: detail = str(error)
        raise gr.Error(detail) from error
    except URLError as error:
        raise gr.Error("The V3 inference backend is not responding. Start inference_backend.py in TestTorchEnv, then retry.") from error
    except TimeoutError as error:
        raise gr.Error("Inference timed out. Try a smaller image or check the local accelerator.") from error
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"Could not process this image: {type(error).__name__}: {error}") from error
    progress(0.55, desc="Preparing previews")
    enhanced = image_from_b64(result["selected"])
    candidates = result.get("candidates", {})
    gallery = [(image_from_b64(candidates[key]), key.title()) for key in ("conservative", "balanced", "aggressive")]
    comparison_original = image_from_b64(result["original"])
    metric_rows = _format_comparison(comparison_original, enhanced, reference_path, aligned_confirmed, progress)
    weights = result["selection_weights"]
    weight_summary = " · ".join(f"{name}: {value:.3f}" for name, value in weights.items())
    dims = result["original_dimensions"]
    padded = result["padded_dimensions"]
    meta = result["checkpoint"]
    status = (f"**Selected enhancement:** {result['selected_candidate']}  \n"
              f"**Selector weights:** {weight_summary}  \n"
              f"**Input/output dimensions:** {dims[0]} × {dims[1]} pixels (no resize); internal edge padding {padded[0]} × {padded[1]}  \n"
              f"**Device:** {meta['device']} · **Checkpoint:** V3 epoch {meta['epoch']} · `{meta['sha256'][:16]}…`")
    progress(1.0, desc="Enhancement complete")
    return ((comparison_original, enhanced), enhanced, gallery, status, metric_rows)


CSS = """
:root { --aqua-ink:#12333a; --aqua-teal:#087e83; --aqua-soft:#e8f5f2; }
body, .gradio-container { background: linear-gradient(145deg,#f4faf8,#edf5f4 48%,#f8fbfa)!important; color:var(--aqua-ink)!important; }
.gradio-container { max-width:1180px!important; }
.hero { padding:28px 30px; border:1px solid #d5e8e5; border-radius:24px; background:linear-gradient(120deg,#ffffff,#eaf6f3); box-shadow:0 12px 34px #17444a12; }
.hero h1 { margin:0 0 8px; font-size:2.15rem; color:#123b42; }
.hero p { margin:0; color:#4d6b70; }
.model-card { border:1px solid #d5e8e5; border-radius:18px; padding:18px; background:#fff; }
button.primary { background:#087e83!important; border-color:#087e83!important; }
"""

with gr.Blocks(title="AquaEnhance — Underwater Image Enhancement", delete_cache=(900, 3600)) as demo:
    gr.Markdown("""
    <div class="hero">
      <h1>AquaEnhance</h1>
      <p>Underwater Image Enhancement · Paper 1 V3 · Local qualitative research preview</p>
    </div>
    """)
    gr.Markdown("Upload an image to view the V3 enhancement. Images are processed locally and are not written to the project results folders.")
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Underwater image · JPG, PNG, or WebP", file_types=[".jpg", ".jpeg", ".png", ".webp"],
                             type="filepath", file_count="single")
            reference_upload = gr.File(label="Optional paired ground-truth/reference image · same dimensions", file_types=[".jpg", ".jpeg", ".png", ".webp"],
                                       type="filepath", file_count="single")
            reference_confirm = gr.Checkbox(label="I confirm this is the paired ground truth, spatially aligned, and in the same color space", value=False)
            original_preview = gr.Image(label="Original image preview", type="pil", image_mode="RGB",
                                        interactive=False, buttons=["fullscreen"], height=280)
            run = gr.Button("Enhance image", variant="primary", size="lg")
            status = gr.Markdown("Upload an image, then choose **Enhance image**.")
        with gr.Column(scale=1):
            compare = gr.ImageSlider(label="Before / after comparison", type="pil", height=340,
                                     slider_position=50, buttons=["fullscreen"])
            enhanced = gr.Image(label="Enhanced image · use the image download control to save", type="pil",
                                image_mode="RGB", buttons=["download", "fullscreen"], height=340)
    with gr.Accordion("Before vs. after image quality metrics", open=True):
        metric_table = gr.Dataframe(headers=["Metric", "Original", "Enhanced", "Absolute change (enhanced − original)", "Relative change"],
                                    datatype=["str", "str", "str", "str", "str"], interactive=False,
                                    value=[[name, "Run enhancement to calculate", "Run enhancement to calculate", "—", "—"] for name in ("UIQM", "UCIQE", "PSNR", "SSIM")],
                                    label="Per-upload comparison")
        gr.Markdown("""UIQM combines underwater colorfulness, sharpness, and contrast; UCIQE combines chroma spread, luminance contrast, and saturation. These no-reference measures are heuristic and a positive change does not guarantee better perceived quality. PSNR measures pixel error and SSIM measures structural similarity against the supplied paired reference; both depend on correct alignment and are unavailable without ground truth. Relative change is shown only when the original value is nonzero. Canonical project metrics use RGB float32 values in [0,1].""")
    with gr.Accordion("Candidate images produced by the V3 model", open=False):
        gallery = gr.Gallery(label="Actual V7 candidate-generator outputs", columns=3, rows=1,
                             height=280, object_fit="contain", type="pil")
    upload.change(preview_upload, inputs=[upload], outputs=[original_preview], show_progress="hidden")
    run.click(enhance, inputs=[upload, reference_upload, reference_confirm], outputs=[compare, enhanced, gallery, status, metric_table], show_progress="full")
    gr.Markdown("""
    <div class="model-card">
    <strong>Checkpoint-level validation results</strong><br>
    These are measured on the research validation set, not on the uploaded image. Uploaded-image results are shown separately in the metrics panel below.
    </div>
    """)
    gr.Markdown(checkpoint_panel())
    gr.Markdown("Detection loss, classification loss, and classification accuracy were **not evaluated** for this checkpoint. Visual results do not establish downstream task improvement.")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name=CONFIG["ui_host"], server_port=int(CONFIG["ui_port"]),
        share=False, inbrowser=False, show_error=True, max_file_size=CONFIG["max_upload_bytes"], quiet=True,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"), css=CSS)
