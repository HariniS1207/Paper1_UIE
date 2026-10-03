from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from evaluate_test import calculate_metrics, uciqe, uiqm
from models.paper1_model_v7 import Paper1ModelV7


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "v7" / "paper1_v7_best_epoch_019.pth"
EXPECTED_SHA256 = "A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E"
RESULTS_ROOT = PROJECT_ROOT / "results" / "demo_v7"
INPUT_SIZE = (256, 256)
CANDIDATE_NAMES = ("Conservative", "Balanced", "Aggressive")
CANDIDATE_KEYS = tuple(name.lower() for name in CANDIDATE_NAMES)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"V7 checkpoint not found: {CHECKPOINT_PATH}")
    actual_hash = checkpoint_sha256(CHECKPOINT_PATH)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"V7 checkpoint SHA-256 mismatch: expected {EXPECTED_SHA256}, found {actual_hash}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != 19:
        raise RuntimeError(f"Expected V7 best checkpoint from epoch 19, found epoch {epoch}.")
    model = Paper1ModelV7().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    nonfinite = [name for name, tensor in model.state_dict().items()
                 if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item()]
    if nonfinite:
        raise RuntimeError(f"V7 checkpoint has non-finite model tensors: {nonfinite[:5]}")
    model.eval()
    return model, device, epoch, actual_hash


MODEL, DEVICE, BEST_EPOCH, CHECKPOINT_SHA256 = load_model()
DEVICE_LABEL = f"CUDA · {torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else "CPU"


def prepare_image(image: Image.Image, label: str):
    if not isinstance(image, Image.Image):
        raise gr.Error(f"{label}: upload a valid JPG or PNG image.")
    try:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 1 or image.height < 1:
            raise ValueError("image has an empty dimension")
        prepared = image.resize(INPUT_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        return prepared, tensor
    except Exception as error:
        raise gr.Error(f"{label}: could not decode or prepare the image ({error}).") from error


def image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().cpu()[0].permute(1, 2, 0).numpy()
    if not np.isfinite(array).all():
        raise gr.Error("The model produced a non-finite image; no output was saved.")
    if array.min() < -1e-5 or array.max() > 1.00001:
        raise gr.Error("The model output was outside [0,1]; no output was saved.")
    array = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def tensor_to_metric_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu()[0].permute(1, 2, 0).numpy()
    if not np.isfinite(array).all():
        raise gr.Error("A model image contains non-finite values; metrics are unavailable.")
    return np.clip(array, 0.0, 1.0)


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    values = {"uiqm": float(uiqm(image)), "uciqe": float(uciqe(image))}
    if not np.isfinite(list(values.values())).all():
        raise gr.Error("The existing UIQM/UCIQE implementation returned a non-finite result.")
    return values


def cell(value, signed=False):
    return "Unavailable" if value is None else (f"{value:+.4f}" if signed else f"{value:.4f}")


def quality_table(metrics):
    rows = []
    for key in ("original", *CANDIDATE_KEYS, "selected"):
        item = metrics[key]
        label = "Original" if key == "original" else ("Selected" if key == "selected" else key.title())
        dpsnr = "-" if key == "original" else cell(item["delta_psnr"], signed=True)
        dssim = "-" if key == "original" else cell(item["delta_ssim"], signed=True)
        rows.append(f"| {label} | {cell(item['psnr'])} | {dpsnr} | {cell(item['ssim'])} | {dssim} | "
                    f"{item['uiqm']:.4f} | {item['uciqe']:.4f} | {item['preservation']:.6f} |")
    return "\n".join(["| Output | PSNR | Δ PSNR | SSIM | Δ SSIM | UIQM | UCIQE | Preservation |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|", *rows])


def run_v7(input_image: Image.Image | None, reference_image: Image.Image | None):
    if input_image is None:
        raise gr.Error("Upload an underwater image before running V7.")
    original_display, input_tensor = prepare_image(input_image, "Input image")
    reference_tensor = None
    if reference_image is not None:
        _, reference_tensor = prepare_image(reference_image, "Reference image")

    try:
        with torch.inference_mode():
            outputs = MODEL(input_tensor)
        if not torch.isfinite(outputs["selected_image"]).all().item():
            raise RuntimeError("non-finite selected image")
        selected_index = int(outputs["selected_index"][0].item())
        if selected_index not in range(len(CANDIDATE_NAMES)):
            raise RuntimeError(f"selector returned invalid candidate index {selected_index}")
        weights = outputs["selection_weights"][0].detach().float().cpu().numpy()
        utilities = outputs["utilities"][0].detach().float().cpu().numpy()
        if not np.isfinite(weights).all() or not np.isfinite(utilities).all():
            raise RuntimeError("non-finite selector values")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-5):
            raise RuntimeError("invalid selector probabilities")
        if int(np.argmax(weights)) != selected_index:
            raise RuntimeError("selected index does not match the highest actual selection weight")

        candidate_images = {key: image_from_tensor(outputs["candidates"][key]) for key in CANDIDATE_KEYS}
        selected_image = image_from_tensor(outputs["selected_image"])
        preservation = {name: float(outputs["preservation_scores"][key][0].item())
                        for name, key in zip(CANDIDATE_NAMES, CANDIDATE_KEYS)}
        if not np.isfinite(list(preservation.values())).all():
            raise RuntimeError("non-finite preservation scores")
        selected_name = CANDIDATE_NAMES[selected_index]
        metric_images = {"original": tensor_to_metric_rgb(input_tensor),
                         **{key: tensor_to_metric_rgb(outputs["candidates"][key]) for key in CANDIDATE_KEYS},
                         "selected": tensor_to_metric_rgb(outputs["selected_image"])}
        metrics = {key: {**quality_metrics(image), "psnr": None, "ssim": None,
                         "delta_psnr": None, "delta_ssim": None,
                         "preservation": 1.0 if key == "original" else
                         (preservation[selected_name] if key == "selected" else preservation[key.title()])}
                   for key, image in metric_images.items()}
        if reference_tensor is not None:
            reference_array = tensor_to_metric_rgb(reference_tensor)
            for key, image in metric_images.items():
                psnr, ssim, _, _ = calculate_metrics(image, reference_array)
                metrics[key]["psnr"], metrics[key]["ssim"] = float(psnr), float(ssim)
            for key in (*CANDIDATE_KEYS, "selected"):
                metrics[key]["delta_psnr"] = metrics[key]["psnr"] - metrics["original"]["psnr"]
                metrics[key]["delta_ssim"] = metrics[key]["ssim"] - metrics["original"]["ssim"]
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"V7 inference failed: {error}") from error

    selector_rows = [[name, float(utilities[i]), float(weights[i])] for i, name in enumerate(CANDIDATE_NAMES)]
    preservation_rows = [[name, preservation[name]] for name in CANDIDATE_NAMES]
    if reference_tensor is None:
        reference_status = "**Reference metrics:** PSNR/SSIM unavailable because no reference image was supplied."
    else:
        reference_status = "**Reference metrics:** all outputs use the same uploaded reference image."

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = RESULTS_ROOT / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        for name, image in candidate_images.items():
            image.save(run_dir / f"{name}.png")
        selected_image.save(run_dir / "selected.png")
        metadata = {
            "model": "Paper 1 - V7", "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": CHECKPOINT_SHA256, "best_epoch": BEST_EPOCH, "mode": "inference only",
            "input_filename": Path(getattr(input_image, "filename", "upload.png")).name,
            "input_saved": False, "input_original_size": list(input_image.size), "model_input_size": list(INPUT_SIZE),
            "preprocessing": "EXIF transpose, RGB conversion, resize to 256x256 with Lanczos, float32 scaling to [0,1]",
            "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
            "selected_candidate": selected_name,
            "utilities": dict(zip(CANDIDATE_NAMES, map(float, utilities))),
            "selection_weights": dict(zip(CANDIDATE_NAMES, map(float, weights))),
            "preservation_scores": preservation, "metrics": metrics, "reference_status": reference_status,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as error:
        raise gr.Error(f"Inference succeeded, but saving V7 outputs failed: {error}") from error

    status = (f"**V7 selected: {selected_name}** · Device: {DEVICE_LABEL} · "
              f"Checkpoint: `paper1_v7_best_epoch_019.pth` · {reference_status}")
    return (original_display, candidate_images["conservative"], candidate_images["balanced"],
            candidate_images["aggressive"], selected_image, selector_rows, preservation_rows,
            quality_table(metrics), status)


CSS = """
:root { --ink: #163238; --muted: #5a7378; --sea: #0d716b; --line: #d5e2df; }
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; background: #f7faf9; color: var(--ink); }
#hero { padding: 24px 28px 20px; border-bottom: 1px solid var(--line); background: linear-gradient(110deg, #f7faf9, #e5f1ed 58%, #d8ebe6); }
#hero h1 { margin: 0; color: var(--ink); font-size: 30px; line-height: 1.18; }
#hero p { margin: 8px 0 0; color: var(--muted); font-size: 15px; }
#selected-output { border: 2px solid var(--sea); padding: 8px; background: white; }
#run-button { background: var(--sea); color: white; border: 0; }
footer { display: none !important; }
"""

with gr.Blocks(title="Paper 1 - V7 Adaptive Underwater Image Enhancement") as demo:
    gr.HTML("<div id='hero'><h1>Paper 1 - Information-Preserving Adaptive Underwater Image Enhancement</h1><p>V7 Custom Image Inference Demo</p></div>")
    gr.Markdown(f"**Model:** Paper 1 - V7　 **Checkpoint:** `paper1_v7_best_epoch_019.pth`　 **Best epoch:** {BEST_EPOCH}　 **Mode:** Inference only　 **Device:** {DEVICE_LABEL}")
    with gr.Row():
        input_image = gr.Image(label="Upload underwater image", type="pil", sources=["upload"], format="png")
        reference_image = gr.Image(label="Optional reference image", type="pil", sources=["upload"], format="png")
    run_button = gr.Button("Run V7", variant="primary", elem_id="run-button")
    status = gr.Markdown("Inference only. No uploaded image has been processed yet.")
    gr.Markdown("### Original")
    original_output = gr.Image(label="Original (model input view)", type="pil", interactive=False)
    gr.Markdown("### Candidate Enhancements")
    with gr.Row():
        conservative_output = gr.Image(label="Conservative", type="pil", interactive=False)
        balanced_output = gr.Image(label="Balanced", type="pil", interactive=False)
        aggressive_output = gr.Image(label="Aggressive", type="pil", interactive=False)
    gr.Markdown("### Selected Enhancement")
    selected_output = gr.Image(label="Selected by actual V7 selector", type="pil", interactive=False, elem_id="selected-output")
    gr.Markdown("#### V7 selector outputs")
    selector_table = gr.Dataframe(headers=["Candidate", "Utility", "Selection weight"], datatype=["str", "number", "number"], label="Actual V7 selector utilities and probabilities", interactive=False)
    gr.Markdown("#### Information Preservation")
    preservation_table = gr.Dataframe(headers=["Candidate", "Preservation score"], datatype=["str", "number"], label="V7 fixed preservation scores", interactive=False)
    gr.Markdown("#### Image Quality Metrics\nUIQM/UCIQE are no-reference quality measures. PSNR/SSIM require an optional aligned reference image.")
    reference_status = gr.Markdown()
    metrics_output = gr.Markdown(label="V7 image metrics")
    run_button.click(fn=run_v7, inputs=[input_image, reference_image], outputs=[original_output,
        conservative_output, balanced_output, aggressive_output, selected_output, selector_table,
        preservation_table, metrics_output, status])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7862,
                                                   inbrowser=False, css=CSS)
