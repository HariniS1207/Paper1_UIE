from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from evaluate_test import calculate_metrics, uciqe, uiqm
from models.paper1_model_v7 import Paper1ModelV7


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "v7" / "paper1_v7_best_epoch_019.pth"
EXPECTED_SHA256 = "A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E"
RESULTS_ROOT = PROJECT_ROOT / "results" / "demo_v7"
FIGURE_DIR = PROJECT_ROOT / "tools" / "figures"
FIGURE_WORKING_DIR = FIGURE_DIR / "working"
FIGURE_PNG = FIGURE_DIR / "fig5_v7_qualitative_selection.png"
FIGURE_PDF = FIGURE_DIR / "fig5_v7_qualitative_selection.pdf"
FIGURE_METADATA = FIGURE_DIR / "fig5_v7_qualitative_selection_metadata.json"
INPUT_SIZE = (256, 256)
CANDIDATE_NAMES = ("Conservative", "Balanced", "Aggressive")
CANDIDATE_KEYS = tuple(name.lower() for name in CANDIDATE_NAMES)
FIGURE_ROWS = ("Anemone + clownfish", "Reef + fish", "Open-water turtle")


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
    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as source:
                image = source.copy()
        except Exception as error:
            raise gr.Error(f"{label}: could not decode the uploaded image ({error}).") from error
    if not isinstance(image, Image.Image):
        raise gr.Error(f"{label}: upload a valid JPG or PNG image.")
    try:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 1 or image.height < 1:
            raise ValueError("image has an empty dimension")
        original_size = list(image.size)
        prepared = image.resize(INPUT_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        return prepared, tensor, original_size
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
    original_display, input_tensor, input_original_size = prepare_image(input_image, "Input image")
    reference_tensor = None
    if reference_image is not None:
        _, reference_tensor, _ = prepare_image(reference_image, "Reference image")

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
        original_display.save(run_dir / "input.png")
        for name, image in candidate_images.items():
            image.save(run_dir / f"{name}.png")
        selected_image.save(run_dir / "selected.png")
        metadata = {
            "model": "Paper 1 - V7", "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": CHECKPOINT_SHA256, "best_epoch": BEST_EPOCH, "mode": "inference only",
            "input_filename": Path(input_image).name if isinstance(input_image, (str, Path)) else
                              Path(getattr(input_image, "filename", "upload.png")).name,
            "input_saved": True, "input_source_file": "input.png",
            "input_original_size": input_original_size, "model_input_size": list(INPUT_SIZE),
            "candidate_source_files": {name: f"{name.lower()}.png" for name in CANDIDATE_NAMES},
            "selected_source_file": "selected.png",
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
            quality_table(metrics), status, str(run_dir))


def _font(size):
    for font_path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _validated_figure_record(run_dir: str, row_label: str):
    directory = Path(run_dir).resolve()
    if not directory.is_relative_to(RESULTS_ROOT.resolve()):
        raise gr.Error("The selected inference result is outside the V7 results directory.")
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise gr.Error("The selected inference has no saved metadata.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_sources = metadata.get("candidate_source_files") or {}
    source_names = {"input": metadata.get("input_source_file"),
                    "conservative": candidate_sources.get("Conservative"),
                    "balanced": candidate_sources.get("Balanced"),
                    "aggressive": candidate_sources.get("Aggressive"),
                    "selected": metadata.get("selected_source_file")}
    stable_dir = FIGURE_WORKING_DIR / f"{directory.name}_{uuid.uuid4().hex[:8]}"
    stable_dir.mkdir(parents=True, exist_ok=False)
    stable_sources = {}
    images = {}
    for key, filename in source_names.items():
        if not isinstance(filename, str) or not filename:
            raise gr.Error(f"Cannot load {key}: inference metadata has no saved source reference.")
        source_path = (directory / filename).resolve()
        if not source_path.is_relative_to(directory) or not source_path.is_file():
            raise gr.Error(f"Cannot load {key}: saved source '{filename}' is missing or outside its inference folder.")
        stable_path = stable_dir / source_path.name
        try:
            shutil.copyfile(source_path, stable_path)
            with Image.open(stable_path) as source:
                source.verify()
            with Image.open(stable_path) as source:
                image = source.convert("RGB").copy()
        except Exception as error:
            raise gr.Error(f"Cannot load {key} image '{source_path}': {type(error).__name__}: {error}") from error
        stable_sources[key] = str(stable_path)
        source_names[key] = filename
        images[key] = image
    if any(image.size != INPUT_SIZE for image in images.values()):
        raise gr.Error("All five saved images must have the same 256×256 dimensions.")
    candidate = metadata.get("selected_candidate")
    if candidate not in CANDIDATE_NAMES:
        raise gr.Error("The selected inference has an invalid V7 candidate name.")
    if ImageChops.difference(images["selected"], images[candidate.lower()]).getbbox() is not None:
        raise gr.Error("Selected output does not exactly match V7's selected candidate.")
    probability = metadata.get("selection_weights", {}).get(candidate)
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise gr.Error("The selected inference has no valid actual V7 selection probability.")
    if metadata.get("checkpoint") != str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)):
        raise gr.Error("The selected result was not produced with the frozen V7 checkpoint.")
    return {"row_label": row_label, "run_id": directory.name,
            "source_filename": metadata.get("input_filename", "unknown"),
            "source_files": source_names, "assets": stable_sources, "selected_candidate": candidate,
            "selection_probability": float(probability), "checkpoint": metadata["checkpoint"],
            "image_dimensions": list(images["input"].size)}


def _load_figure_record_images(record):
    images = {}
    for key in ("input", "conservative", "balanced", "aggressive", "selected"):
        path = Path(record["assets"][key])
        try:
            with Image.open(path) as source:
                images[key] = source.convert("RGB").copy()
        except Exception as error:
            raise gr.Error(f"Cannot render Figure 5 {key} image from '{path}': {type(error).__name__}: {error}") from error
    return images


def _build_figure(records):
    if len(records) != 3 or set(records) != set(FIGURE_ROWS):
        raise gr.Error("Select exactly one inference for each of the three Figure 5 rows.")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    margin, label_width, image_width = 22, 190, INPUT_SIZE[0]
    cell_pitch = image_width + 2
    header_h, row_label_h, image_h, annotation_h, row_gap = 38, 25, INPUT_SIZE[1], 34, 12
    width = margin * 2 + label_width + cell_pitch * 5
    height = margin + header_h + 3 * (row_label_h + image_h + annotation_h) + 2 * row_gap + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font, label_font, annotation_font = _font(18), _font(16), _font(15)
    headers = ("Input", "Conservative", "Balanced", "Aggressive", "Selected")
    x0 = margin + label_width
    y = margin
    for column, header in enumerate(headers):
        box = draw.textbbox((0, 0), header, font=header_font)
        x = x0 + column * cell_pitch + (image_width - (box[2] - box[0])) // 2
        draw.text((x, y + 4), header, fill=(30, 39, 45), font=header_font)
    y += header_h
    metadata_rows = []
    for row_label in FIGURE_ROWS:
        record = records[row_label]
        images = _load_figure_record_images(record)
        y_row = y
        draw.text((margin, y_row + 3), row_label, fill=(30, 39, 45), font=label_font)
        y_row += row_label_h
        ordered = ("input", "conservative", "balanced", "aggressive", "selected")
        for column, key in enumerate(ordered):
            slot_x = x0 + column * cell_pitch
            x = slot_x + 1
            image = images[key]
            canvas.paste(image, (x, y_row))
            draw.rectangle((slot_x, y_row - 1, slot_x + image_width + 1, y_row + image_h),
                           outline=(50, 75, 80) if key == "selected" else (190, 198, 202), width=1)
        y_note = y_row + image_h + 5
        note = f"Selected: {record['selected_candidate']} ({record['selection_probability'] * 100:.2f}%)"
        draw.text((x0, y_note), note, fill=(30, 39, 45), font=annotation_font)
        metadata_rows.append({"row_label": row_label, "run_id": record["run_id"],
                              "source_filename": record["source_filename"],
                              "source_files": record["source_files"],
                              "selected_candidate": record["selected_candidate"],
                              "selection_probability": record["selection_probability"],
                              "checkpoint": record["checkpoint"]})
        y = y_row + image_h + annotation_h + row_gap

    return canvas, metadata_rows


def select_for_figure(run_dir, row_label, selections):
    if not run_dir:
        raise gr.Error("Run V7 for an inference before selecting it for Figure 5.")
    selections = dict(selections or {})
    if row_label not in FIGURE_ROWS:
        raise gr.Error("Choose one of the three Figure 5 row labels.")
    if row_label in selections:
        raise gr.Error(f"A case is already assigned to '{row_label}'. Choose another row or clear the selection.")
    if len(selections) >= 3:
        raise gr.Error("Figure 5 already has exactly three selected cases. Clear selections to replace one.")
    selections[row_label] = _validated_figure_record(run_dir, row_label)
    status = f"Selected {len(selections)} of 3 cases. " + ", ".join(selections)
    selected_images = _load_figure_record_images(selections[row_label])
    panel_images = [selected_images[key] for key in ("input", "conservative", "balanced", "aggressive", "selected")]
    if len(selections) == 3:
        preview, _ = _build_figure(selections)
        status += "\n\nAll 15 images loaded. Review the Figure 5 preview, then export."
        return (selections, *panel_images, preview, status, None, None)
    return (selections, *panel_images, None, status, None, None)


def export_figure(selections):
    if len(selections or {}) != 3 or set(selections) != set(FIGURE_ROWS):
        raise gr.Error("Select and preview all three Figure 5 cases before exporting.")
    preview, metadata_rows = _build_figure(selections)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    preview.save(FIGURE_PNG, format="PNG", dpi=(300, 300), optimize=False)
    preview.save(FIGURE_PDF, format="PDF", resolution=300.0)
    FIGURE_METADATA.write_text(json.dumps({"figure": "Figure 5", "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
                                          "rows": metadata_rows, "columns": ["Input", "Conservative", "Balanced", "Aggressive", "Selected"],
                                          "image_dimensions_px": list(INPUT_SIZE),
                                          "verification": {"exactly_three_cases": True,
                                                           "all_images_same_inference": True,
                                                           "selected_matches_v7_candidate_pixel_for_pixel": True,
                                                           "probability_from_saved_v7_selection_weights": True,
                                                           "input_and_candidate_pixels_unresampled": True}}, indent=2), encoding="utf-8")
    return preview, f"Figure exported: `{FIGURE_PNG}` · `{FIGURE_PDF}` · `{FIGURE_METADATA}`", str(FIGURE_PNG), str(FIGURE_PDF)


def clear_figure_selection():
    return {}, None, None, None, None, None, None, "No Figure 5 cases selected.", None, None


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
        input_image = gr.Image(label="Upload underwater image", type="filepath", sources=["upload"])
        reference_image = gr.Image(label="Optional reference image", type="filepath", sources=["upload"])
    run_button = gr.Button("Run V7", variant="primary", elem_id="run-button")
    latest_result = gr.State(None)
    figure_selections = gr.State({})
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
    gr.Markdown("### Figure 5 Builder\nRun each inference, choose its row label, then select it. The builder exports after all three rows are filled.")
    with gr.Row():
        figure_row_label = gr.Dropdown(choices=list(FIGURE_ROWS), value=FIGURE_ROWS[0], label="Figure 5 row")
        select_figure_button = gr.Button("Select for Figure 5")
        clear_figure_button = gr.Button("Clear Figure 5 selections")
    figure_status = gr.Markdown("No Figure 5 cases selected.")
    gr.Markdown("#### Selected inference image sources")
    with gr.Row():
        figure_input_image = gr.Image(label="Input", type="pil", interactive=False)
        figure_conservative_image = gr.Image(label="Conservative", type="pil", interactive=False)
        figure_balanced_image = gr.Image(label="Balanced", type="pil", interactive=False)
        figure_aggressive_image = gr.Image(label="Aggressive", type="pil", interactive=False)
        figure_selected_image = gr.Image(label="Selected", type="pil", interactive=False)
    figure_preview = gr.Image(label="Figure 5 Preview", type="pil", interactive=False)
    export_figure_button = gr.Button("Export Figure 5", variant="primary")
    with gr.Row():
        figure_png_file = gr.File(label="High-resolution PNG")
        figure_pdf_file = gr.File(label="PDF")
    run_button.click(fn=run_v7, inputs=[input_image, reference_image], outputs=[original_output,
        conservative_output, balanced_output, aggressive_output, selected_output, selector_table,
        preservation_table, metrics_output, status, latest_result])
    select_figure_button.click(fn=select_for_figure, inputs=[latest_result, figure_row_label, figure_selections],
                               outputs=[figure_selections, figure_input_image, figure_conservative_image,
                                        figure_balanced_image, figure_aggressive_image, figure_selected_image,
                                        figure_preview, figure_status, figure_png_file, figure_pdf_file])
    export_figure_button.click(fn=export_figure, inputs=[figure_selections],
                               outputs=[figure_preview, figure_status, figure_png_file, figure_pdf_file])
    clear_figure_button.click(fn=clear_figure_selection, inputs=[],
                              outputs=[figure_selections, figure_input_image, figure_conservative_image,
                                       figure_balanced_image, figure_aggressive_image, figure_selected_image,
                                       figure_preview, figure_status, figure_png_file, figure_pdf_file])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7862,
                                                   inbrowser=False, css=CSS)
