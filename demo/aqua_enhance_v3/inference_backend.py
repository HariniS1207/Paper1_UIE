"""Loopback-only PyTorch inference service for AquaEnhance V3."""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((APP_DIR / "app_config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
from models.paper1_model_v7 import Paper1ModelV7

CANDIDATES = ("conservative", "balanced", "aggressive")
MAX_UPLOAD_BYTES = int(CONFIG["max_upload_bytes"])
MODEL = None
DEVICE = None
CHECKPOINT_INFO = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_model():
    relative = Path(CONFIG["checkpoint"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Checkpoint config must be a project-relative path without parent traversal.")
    checkpoint_path = (PROJECT_ROOT / relative).resolve()
    if not checkpoint_path.is_relative_to((PROJECT_ROOT / "checkpoints").resolve()):
        raise RuntimeError("Configured checkpoint must remain inside the project checkpoints directory.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Configured checkpoint was not found: {relative.as_posix()}")
    actual_sha = file_sha256(checkpoint_path)
    if actual_sha != CONFIG["expected_sha256"]:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: expected {CONFIG['expected_sha256']}, found {actual_sha}.")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(state.get("epoch", -1))
    if epoch != int(CONFIG["expected_epoch"]):
        raise RuntimeError(f"Expected checkpoint epoch {CONFIG['expected_epoch']}, found {epoch}.")
    model = Paper1ModelV7()
    model.load_state_dict(state["model_state_dict"], strict=True)
    nonfinite = [name for name, tensor in model.state_dict().items()
                 if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item()]
    if nonfinite:
        raise RuntimeError(f"Checkpoint has non-finite model tensors: {nonfinite[:5]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device, {"path": relative.as_posix(), "epoch": epoch, "sha256": actual_sha,
                           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}


def encode_png(tensor: torch.Tensor, height: int, width: int) -> str:
    image = tensor.detach().float().cpu()
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise RuntimeError(f"Expected one RGB output tensor, got shape {tuple(image.shape)}.")
    image = image[:, :, :height, :width]
    array = image[0].permute(1, 2, 0).numpy()
    if array.shape != (height, width, 3) or not np.isfinite(array).all():
        raise RuntimeError("Model output has invalid dimensions or non-finite values.")
    if array.min() < -1e-5 or array.max() > 1.00001:
        raise RuntimeError("Model output fell outside the display range [0,1].")
    buffer = BytesIO()
    Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def enhance_bytes(content: bytes) -> dict:
    if not content:
        raise ValueError("The uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image file is larger than the {MAX_UPLOAD_BYTES // 1_000_000} MB upload limit.")
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Use a JPG, PNG, or WebP image.")
            opened.load()
            oriented = ImageOps.exif_transpose(opened).convert("RGB")
            original = oriented.copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"Could not read the image. Check that it is a valid JPG, PNG, or WebP: {error}") from error
    width, height = original.size
    if width < 1 or height < 1:
        raise ValueError("The image has an empty dimension.")
    if max(width, height) > int(CONFIG["max_image_side"]):
        raise ValueError(f"Image side exceeds the {CONFIG['max_image_side']} pixel limit. Crop the image and try again.")
    if width * height > int(CONFIG["max_image_pixels"]):
        raise ValueError(f"Image has {width * height:,} pixels; the local inference limit is {int(CONFIG['max_image_pixels']):,}. Resize it yourself and upload again.")

    array = np.asarray(original, dtype=np.float32) / 255.0  # project preprocessing: RGB, ToTensor range [0,1]
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    # The consequence-check decoder uses four stride-2 layers. Right/bottom edge padding
    # to a multiple of 16 preserves aspect ratio and lets arbitrary input sizes pass through.
    pad_h = (-height) % 16
    pad_w = (-width) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    try:
        with torch.inference_mode():
            outputs = MODEL(tensor)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError("The image exceeded available accelerator memory. Try a smaller image; AquaEnhance does not resize it automatically.") from error
    selected = outputs.get("selected_image")
    if selected is None:
        raise RuntimeError("The model did not return its selected enhancement.")
    selected_index = int(outputs["selected_index"][0].item())
    if selected_index not in range(len(CANDIDATES)):
        raise RuntimeError(f"Selector returned an invalid candidate index: {selected_index}.")
    weights = outputs["selection_weights"][0].detach().float().cpu().numpy()
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-5):
        raise RuntimeError("Selector returned invalid probabilities.")
    candidates = outputs.get("candidates", {})
    if set(candidates) != set(CANDIDATES):
        raise RuntimeError("Model candidate outputs did not match the three documented candidates.")
    return {
        "original": encode_png(torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0), height, width),
        "selected": encode_png(selected, height, width),
        "candidates": {name: encode_png(candidates[name], height, width) for name in CANDIDATES},
        "selected_candidate": CANDIDATES[selected_index].title(),
        "selection_weights": {name.title(): float(weights[i]) for i, name in enumerate(CANDIDATES)},
        "original_dimensions": [width, height], "padded_dimensions": [width + pad_w, height + pad_h],
        "checkpoint": CHECKPOINT_INFO,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AquaEnhanceLocal/1.0"

    def log_message(self, _format, *_args):
        # Do not log uploaded filenames or image data.
        return

    def respond(self, status: int, body: dict):
        raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if urlparse(self.path).path != "/health":
            self.respond(404, {"error": "Not found."})
            return
        self.respond(200, {"ready": MODEL is not None, "checkpoint": CHECKPOINT_INFO})

    def do_POST(self):
        if urlparse(self.path).path != "/enhance":
            self.respond(404, {"error": "Not found."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_UPLOAD_BYTES:
                raise ValueError(f"Upload size must be between 1 byte and {MAX_UPLOAD_BYTES // 1_000_000} MB.")
            result = enhance_bytes(self.rfile.read(size))
            self.respond(200, result)
        except ValueError as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            self.respond(500, {"error": f"Inference failed safely: {type(error).__name__}: {error}"})


def main():
    global MODEL, DEVICE, CHECKPOINT_INFO
    MODEL, DEVICE, CHECKPOINT_INFO = load_model()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="run one in-memory synthetic inference and exit")
    args = parser.parse_args()
    if args.self_check:
        pixels = np.zeros((31, 47, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(47, dtype=np.uint8)[None, :] * 5
        pixels[:, :, 1] = np.arange(31, dtype=np.uint8)[:, None] * 7
        source = Image.fromarray(pixels, mode="RGB")
        source_bytes = BytesIO()
        source.save(source_bytes, format="PNG")
        before_hash = hashlib.sha256(source_bytes.getvalue()).hexdigest()
        result = enhance_bytes(source_bytes.getvalue())
        decoded = {"original": result["original"], "selected": result["selected"], **result["candidates"]}
        dimensions = {}
        for name, encoded in decoded.items():
            with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                image.load()
                if image.mode != "RGB" or image.size != (47, 31):
                    raise RuntimeError(f"Synthetic {name} output has invalid mode/dimensions: {image.mode}/{image.size}")
                dimensions[name] = list(image.size)
        selected_key = result["selected_candidate"].lower()
        if decoded["selected"] != result["candidates"][selected_key]:
            raise RuntimeError("Selector output is not the labeled selected candidate")
        if hashlib.sha256(source_bytes.getvalue()).hexdigest() != before_hash:
            raise RuntimeError("Synthetic source image bytes were modified")
        print(json.dumps({"status": "passed", "checkpoint": CHECKPOINT_INFO, "input_unchanged": True,
                          "input_dimensions": result["original_dimensions"], "padded_dimensions": result["padded_dimensions"],
                          "output_rgb_dimensions": dimensions, "selected_candidate": result["selected_candidate"],
                          "selection_weights": result["selection_weights"], "files_written": False}, indent=2), flush=True)
        return
    host = CONFIG["backend_host"]
    if host != "127.0.0.1":
        raise RuntimeError("The inference service is restricted to 127.0.0.1.")
    server = ThreadingHTTPServer((host, int(CONFIG["backend_port"])), Handler)
    print(f"AquaEnhance V3 backend ready on http://{host}:{CONFIG['backend_port']} | {CHECKPOINT_INFO['device']} | epoch {CHECKPOINT_INFO['epoch']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
