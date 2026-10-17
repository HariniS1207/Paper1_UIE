"""Compute V7 quality metrics from prediction arrays using canonical project functions."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_canonical_metric_functions(project_root: Path):
    source = project_root / "src" / "evaluate_test.py"
    text = source.read_text(encoding="utf-8")
    parsed = ast.parse(text, filename=str(source))
    names = {"eme", "uicm", "uism", "uiconm", "uiqm", "uciqe", "calculate_metrics"}
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    found = {node.name for node in nodes}
    if found != names:
        raise RuntimeError(f"Canonical metric source functions missing: {sorted(names - found)}")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {
        "np": np,
        "rgb2hsv": rgb2hsv,
        "rgb2lab": rgb2lab,
        "peak_signal_noise_ratio": peak_signal_noise_ratio,
        "structural_similarity": structural_similarity,
    }
    exec(compile(module, str(source), "exec"), namespace)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest().upper()
    return {name: namespace[name] for name in names}, source_hash


def load_canonical_metrics(project_root: Path):
    """Backwards-compatible accessor for the project's combined metric function."""
    functions, source_hash = load_canonical_metric_functions(project_root)
    return functions["calculate_metrics"], source_hash


def image_tensor_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def evaluate_archive(archive_path: Path, filenames: list[str], reference_dir: Path, calculate_metrics):
    values = {key: [] for key in ("psnr", "ssim", "uiqm", "uciqe")}
    with np.load(archive_path, allow_pickle=False) as archive:
        expected = {f"prediction_{i:03d}" for i in range(len(filenames))}
        if set(archive.files) != expected:
            raise RuntimeError(f"Prediction archive sample mismatch: found {len(archive.files)}, expected {len(expected)}")
        for index, filename in enumerate(filenames):
            prediction = np.asarray(archive[f"prediction_{index:03d}"], dtype=np.float32)
            reference = image_tensor_array(reference_dir / filename)
            if prediction.shape != reference.shape:
                raise RuntimeError(f"Shape mismatch for {filename}: {prediction.shape} vs {reference.shape}")
            if not np.isfinite(prediction).all() or prediction.min() < -1e-6 or prediction.max() > 1.0 + 1e-6:
                raise RuntimeError(f"Invalid prediction range or non-finite values for {filename}")
            result = calculate_metrics(prediction, reference)
            for key, value in zip(values, result):
                if not np.isfinite(value):
                    raise FloatingPointError(f"Non-finite {key} for {filename}")
                values[key].append(float(value))
    return {
        "sample_count": len(filenames),
        "mean": {key: float(np.mean(items)) for key, items in values.items()},
        "std": {key: float(np.std(items, ddof=1)) for key, items in values.items()},
        "per_image": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    filenames = [line.strip() for line in args.split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(filenames) != 370:
        raise RuntimeError(f"Expected 370 validation entries, found {len(filenames)}")
    calculate_metrics, source_hash = load_canonical_metrics(root)
    baseline = evaluate_archive(args.baseline, filenames, args.reference_dir, calculate_metrics)
    updated = evaluate_archive(args.updated, filenames, args.reference_dir, calculate_metrics)
    deltas = {}
    for key in baseline["mean"]:
        per_image_delta = np.asarray(updated["per_image"][key]) - np.asarray(baseline["per_image"][key])
        deltas[key] = {
            "mean_delta": float(np.mean(per_image_delta)),
            "improved_sample_count": int(np.count_nonzero(per_image_delta > 0)),
            "worsened_sample_count": int(np.count_nonzero(per_image_delta < 0)),
            "unchanged_sample_count": int(np.count_nonzero(per_image_delta == 0)),
        }
    report = {
        "sample_count": len(filenames),
        "metric_source": "canonical function definitions extracted from src/evaluate_test.py; no metric formulas reimplemented",
        "metric_source_sha256": source_hash,
        "runtime_note": "This process uses the existing project .venv only for NumPy/Pillow/scikit-image metrics; it does not import PyTorch or load model checkpoints.",
        "input_convention": "RGB float32 arrays in [0,1]; PSNR/SSIM data_range=1.0; calculate_metrics clips inputs to [0,1] as in src/evaluate_test.py",
        "baseline": {k: v for k, v in baseline.items() if k != "per_image"},
        "updated": {k: v for k, v in updated.items() if k != "per_image"},
        "updated_minus_baseline": deltas,
        "higher_is_better": ["psnr", "ssim", "uiqm", "uciqe"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "sample_count", "baseline_mean", "baseline_std", "updated_mean", "updated_std",
              "updated_minus_baseline", "improved_samples", "worsened_samples", "unchanged_samples", "direction"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in baseline["mean"]:
            writer.writerow({
                "metric": key,
                "sample_count": len(filenames),
                "baseline_mean": baseline["mean"][key],
                "baseline_std": baseline["std"][key],
                "updated_mean": updated["mean"][key],
                "updated_std": updated["std"][key],
                "updated_minus_baseline": deltas[key]["mean_delta"],
                "improved_samples": deltas[key]["improved_sample_count"],
                "worsened_samples": deltas[key]["worsened_sample_count"],
                "unchanged_samples": deltas[key]["unchanged_sample_count"],
                "direction": "higher is better",
            })
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
