from __future__ import annotations

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
from models.paper1_model_v4 import Paper1ModelV4


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "v4" / "paper1_v4_best.pth"
RESULTS_ROOT = PROJECT_ROOT / "results" / "demo" / "v4"
INPUT_SIZE = (256, 256)
CANDIDATE_NAMES = ("Conservative", "Balanced", "Aggressive")
CANDIDATE_KEYS = tuple(name.lower() for name in CANDIDATE_NAMES)


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"V4 checkpoint not found: {CHECKPOINT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != 10:
        raise RuntimeError(f"Expected V4 best checkpoint from epoch 10, found epoch {epoch}.")
    model = Paper1ModelV4().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    nonfinite = [
        name
        for name, tensor in model.state_dict().items()
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item()
    ]
    if nonfinite:
        raise RuntimeError(f"V4 checkpoint has non-finite model tensors: {nonfinite[:5]}")
    model.eval()
    return model, device, epoch


MODEL, DEVICE, BEST_EPOCH = load_model()
DEVICE_LABEL = (
    f"CUDA · {torch.cuda.get_device_name(0)}"
    if DEVICE.type == "cuda"
    else "CPU"
)


def prepare_image(image: Image.Image, label: str) -> tuple[Image.Image, torch.Tensor]:
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
    array = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def tensor_to_metric_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu()[0].permute(1, 2, 0).numpy()
    if not np.isfinite(array).all():
        raise gr.Error("A model image contains non-finite values; quality metrics are unavailable.")
    return np.clip(array, 0.0, 1.0)


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    values = {"uiqm": float(uiqm(image)), "uciqe": float(uciqe(image))}
    if not np.isfinite(list(values.values())).all():
        raise gr.Error("The existing UIQM/UCIQE implementation returned a non-finite result.")
    return values


def cell(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def quality_table(metrics: dict[str, dict[str, float | None]]) -> str:
    rows = []
    for name in ("original", *CANDIDATE_KEYS, "selected"):
        item = metrics[name]
        display_name = "Original" if name == "original" else "Selected" if name == "selected" else name.title()
        delta_keys = ("delta_psnr", "delta_ssim")
        delta_psnr = "—" if name == "original" else cell(item[delta_keys[0]], signed=True)
        delta_ssim = "—" if name == "original" else cell(item[delta_keys[1]], signed=True)
        rows.append(
            f"| {display_name} | {cell(item['psnr'])} | {delta_psnr} | {cell(item['ssim'])} | {delta_ssim} | "
            f"{item['uiqm']:.4f} | {item['uciqe']:.4f} | {item['preservation']:.6f} |"
        )
    return "\n".join([
        "| Output | PSNR | Δ PSNR | SSIM | Δ SSIM | UIQM | UCIQE | Preservation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
    ])


def run_v4(input_image: Image.Image | None, reference_image: Image.Image | None):
    if input_image is None:
        raise gr.Error("Upload an underwater image before running V4.")
    original_display, input_tensor = prepare_image(input_image, "Input image")
    reference_tensor = None
    if reference_image is not None:
        reference_display, reference_tensor = prepare_image(reference_image, "Reference image")
        if reference_display.size != original_display.size:
            raise gr.Error("Reference image could not be resized to the model input shape.")

    try:
        with torch.inference_mode():
            outputs = MODEL(input_tensor)
            if not torch.isfinite(outputs["selected_image"]).all().item():
                raise RuntimeError("non-finite selected image")
            selected_index = int(outputs["selected_index"][0].item())
            if selected_index not in range(len(CANDIDATE_NAMES)):
                raise RuntimeError(f"selector returned invalid candidate index {selected_index}")
            candidates = {
                name: image_from_tensor(outputs["candidates"][name])
                for name in CANDIDATE_KEYS
            }
            selected = image_from_tensor(outputs["selected_image"])
            metric_images = {
                "original": tensor_to_metric_rgb(input_tensor),
                **{name: tensor_to_metric_rgb(outputs["candidates"][name]) for name in CANDIDATE_KEYS},
                "selected": tensor_to_metric_rgb(outputs["selected_image"]),
            }
            preservation = {
                name: float(outputs["preservation_scores"][key][0].item())
                for name, key in zip(CANDIDATE_NAMES, CANDIDATE_KEYS)
            }
            preservation_by_key = dict(zip(CANDIDATE_KEYS, preservation.values()))
            selected_key = CANDIDATE_KEYS[selected_index]
            utilities = outputs["utilities"][0].detach().float().cpu().tolist()
            weights = outputs["selection_weights"][0].detach().float().cpu().tolist()
            if not np.isfinite(list(preservation.values()) + utilities + weights).all():
                raise RuntimeError("non-finite preservation or selector values")
            metrics = {
                name: {
                    **quality_metrics(image),
                    "preservation": (
                        1.0
                        if name == "original"
                        else preservation_by_key[selected_key if name == "selected" else name]
                    ),
                    "psnr": None,
                    "ssim": None,
                    "delta_psnr": None,
                    "delta_ssim": None,
                }
                for name, image in metric_images.items()
            }
            if reference_tensor is not None:
                reference_array = tensor_to_metric_rgb(reference_tensor)
                for name, image in metric_images.items():
                    psnr, ssim, _, _ = calculate_metrics(image, reference_array)
                    metrics[name]["psnr"] = float(psnr)
                    metrics[name]["ssim"] = float(ssim)
                for name in CANDIDATE_KEYS + ("selected",):
                    metrics[name]["delta_psnr"] = metrics[name]["psnr"] - metrics["original"]["psnr"]
                    metrics[name]["delta_ssim"] = metrics[name]["ssim"] - metrics["original"]["ssim"]
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"V4 inference failed: {error}") from error

    selected_name = CANDIDATE_NAMES[selected_index]
    selector_rows = [
        [name, float(utilities[index]), float(weights[index])]
        for index, name in enumerate(CANDIDATE_NAMES)
    ]
    preservation_rows = [[name, preservation[name]] for name in CANDIDATE_NAMES]
    metrics_markdown = quality_table(metrics)

    if reference_tensor is None:
        reference_status = "**Reference metrics:** PSNR/SSIM unavailable — no ground-truth image supplied."
    else:
        reference_status = "**Reference metrics:** all five outputs use the same uploaded reference."

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = RESULTS_ROOT / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        for name, image in candidates.items():
            image.save(run_dir / f"{name}.png")
        selected.save(run_dir / "selected.png")
        metadata = {
            "model": "Paper 1 — V4",
            "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "best_epoch": BEST_EPOCH,
            "mode": "inference only",
            "input_filename": Path(getattr(input_image, "filename", "upload.png")).name,
            "input_saved": False,
            "input_original_size": list(input_image.size),
            "model_input_size": list(INPUT_SIZE),
            "preprocessing": "EXIF transpose, RGB conversion, resize to 256x256 with Lanczos, float32 ToTensor-equivalent scaling to [0,1]",
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
            "selected_candidate": selected_name,
            "preservation_scores": preservation,
            "utilities": {name: float(value) for name, value in zip(CANDIDATE_NAMES, utilities)},
            "selection_weights": {name: float(value) for name, value in zip(CANDIDATE_NAMES, weights)},
            "metrics": metrics,
            "reference_status": reference_status,
        }
        with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    except Exception as error:
        raise gr.Error(f"Inference succeeded, but saving V4 outputs failed: {error}") from error

    status = (
        f"**V4 Selected Candidate: {selected_name}** · Device: {DEVICE_LABEL} · "
        f"Checkpoint: `paper1_v4_best.pth` (epoch {BEST_EPOCH}) · {reference_status}"
    )
    return (
        original_display,
        candidates["conservative"],
        candidates["balanced"],
        candidates["aggressive"],
        selected,
        selector_rows,
        preservation_rows,
        metrics_markdown,
        status,
    )


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

with gr.Blocks(title="Paper 1 — Information-Preserving Adaptive Underwater Image Enhancement") as demo:
    gr.HTML(
        """<div id="hero"><h1>Paper 1 — Information-Preserving Adaptive Underwater Image Enhancement</h1>
        <p>V4 Custom Image Inference Demo</p></div>"""
    )
    gr.Markdown(
        f"**Model:** Paper 1 — V4　 **Checkpoint:** `paper1_v4_best.pth`　 **Best epoch:** {BEST_EPOCH}　 **Mode:** Inference only　 **Device:** {DEVICE_LABEL}"
    )
    with gr.Row():
        input_image = gr.Image(label="Upload underwater image", type="pil", sources=["upload"], format="png")
        reference_image = gr.Image(label="Optional reference image", type="pil", sources=["upload"], format="png")
    run_button = gr.Button("Run V4", variant="primary", elem_id="run-button")
    status = gr.Markdown("Inference only. No uploaded image has been processed yet.")

    gr.Markdown("### Original")
    original_output = gr.Image(label="Original (model input view)", type="pil", interactive=False)
    gr.Markdown("### Candidate Enhancements")
    with gr.Row():
        conservative_output = gr.Image(label="Conservative", type="pil", interactive=False)
        balanced_output = gr.Image(label="Balanced", type="pil", interactive=False)
        aggressive_output = gr.Image(label="Aggressive", type="pil", interactive=False)
    gr.Markdown("### Selected Enhancement")
    selected_output = gr.Image(label="Selected by V4 selector", type="pil", interactive=False, elem_id="selected-output")

    gr.Markdown("#### A. Candidate / Selector")
    selector_table = gr.Dataframe(
        headers=["Candidate", "Utility", "Selection weight"],
        datatype=["str", "number", "number"],
        label="Actual V4 selector outputs",
        interactive=False,
    )
    gr.Markdown("#### B. Information Preservation")
    preservation_table = gr.Dataframe(
        headers=["Candidate", "Preservation score"],
        datatype=["str", "number"],
        label="V4 fixed preservation scores",
        interactive=False,
    )
    gr.Markdown("#### C. Image Quality Metrics")
    gr.Markdown("UIQM/UCIQE are no-reference quality measures, not ground-truth accuracy. Preservation remains a separate model score.")
    reference_status = gr.Markdown()
    metrics_output = gr.Markdown(label="V4 image metrics")

    run_button.click(
        fn=run_v4,
        inputs=[input_image, reference_image],
        outputs=[
            original_output,
            conservative_output,
            balanced_output,
            aggressive_output,
            selected_output,
            selector_table,
            preservation_table,
            metrics_output,
            status,
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        css=CSS,
    )
