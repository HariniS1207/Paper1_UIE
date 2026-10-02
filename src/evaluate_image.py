from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from evaluate_test import calculate_metrics, uciqe, uiqm
from paper1_model import Paper1Model


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v2" / "paper1_v2_epoch_006.pth"


def load_rgb_image(path: Path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def tensor_from_array(array: np.ndarray, device: torch.device):
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return tensor


def save_image(path: Path, array: np.ndarray):
    clipped = np.clip(array, 0.0, 1.0)
    uint8 = np.round(clipped * 255.0).astype(np.uint8)
    Image.fromarray(np.asarray(uint8, dtype=np.uint8)).save(path)


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", "psnr", "ssim", "uiqm", "uciqe", "preservation", "consequence_error", "utility", "selected"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one underwater image with the current Paper 1 model.")
    parser.add_argument("--input", type=Path, required=True, help="Input underwater image path.")
    parser.add_argument("--reference", type=Path, default=None, help="Optional paired reference image.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to results/inference/<image_name>.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use.")
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "inference" / args.input.stem
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = Paper1Model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    original_array = load_rgb_image(args.input)
    input_tensor = tensor_from_array(original_array, device)
    with torch.no_grad():
        outputs = model(input_tensor)

    candidate_names = ["conservative", "balanced", "aggressive"]
    candidates = outputs["candidates"]
    selected_index = int(outputs["selected_index"][0].item())
    selection_scores = outputs["selection_weights"][0].detach().cpu().tolist()
    utilities = outputs["utilities"][0].detach().cpu().tolist()

    candidate_rows = []
    metrics_dict = {}

    for candidate_name in candidate_names:
        candidate_tensor = candidates[candidate_name][0].detach().cpu().numpy().transpose(1, 2, 0)
        candidate_rows.append({
            "candidate": candidate_name,
            "image": candidate_tensor,
            "preservation": float(outputs["preservation_scores"][candidate_name][0].item()),
            "consequence_error": float(outputs["consequence_error"][0].item()),
            "utility": float(utilities[candidate_names.index(candidate_name)]),
        })

    for name, record in zip(candidate_names, candidate_rows):
        candidate_array = record["image"]
        candidate_metric = {"uiqm": float(uiqm(candidate_array)), "uciqe": float(uciqe(candidate_array))}
        if args.reference is not None:
            reference_array = load_rgb_image(args.reference)
            psnr, ssim, uiqm_score, uciqe_score = calculate_metrics(candidate_array, reference_array)
            candidate_metric.update({"psnr": float(psnr), "ssim": float(ssim)})
        else:
            candidate_metric.update({"psnr": "N/A", "ssim": "N/A"})
        candidate_metric["preservation"] = record["preservation"]
        candidate_metric["consequence_error"] = record["consequence_error"]
        candidate_metric["utility"] = record["utility"]
        metrics_dict[name] = candidate_metric

    selected_array = outputs["selected_image"][0].detach().cpu().numpy().transpose(1, 2, 0)
    save_image(output_dir / "original.png", original_array)
    for candidate_name in candidate_names:
        save_image(output_dir / f"{candidate_name}.png", candidates[candidate_name][0].detach().cpu().numpy().transpose(1, 2, 0))
    save_image(output_dir / "selected.png", selected_array)

    if args.reference is not None:
        reference_array = load_rgb_image(args.reference)
        reference_metrics = calculate_metrics(selected_array, reference_array)
        selected_metrics = {
            "psnr": float(reference_metrics[0]),
            "ssim": float(reference_metrics[1]),
            "uiqm": float(uiqm(selected_array)),
            "uciqe": float(uciqe(selected_array)),
            "preservation": float(outputs["preservation_scores"][candidate_names[selected_index]][0].item()),
            "consequence_error": float(outputs["consequence_error"][0].item()),
            "utility": float(utilities[selected_index]),
            "selected": candidate_names[selected_index],
        }
    else:
        selected_metrics = {
            "psnr": "N/A",
            "ssim": "N/A",
            "uiqm": float(uiqm(selected_array)),
            "uciqe": float(uciqe(selected_array)),
            "preservation": float(outputs["preservation_scores"][candidate_names[selected_index]][0].item()),
            "consequence_error": float(outputs["consequence_error"][0].item()),
            "utility": float(utilities[selected_index]),
            "selected": candidate_names[selected_index],
        }

    rows = []
    for candidate_name in candidate_names + ["selected"]:
        if candidate_name == "selected":
            candidate_metric = selected_metrics
            name = "selected"
        else:
            candidate_metric = metrics_dict[candidate_name]
            name = candidate_name
        rows.append({
            "candidate": name,
            "psnr": candidate_metric.get("psnr", "N/A"),
            "ssim": candidate_metric.get("ssim", "N/A"),
            "uiqm": candidate_metric.get("uiqm", "N/A"),
            "uciqe": candidate_metric.get("uciqe", "N/A"),
            "preservation": candidate_metric.get("preservation", "N/A"),
            "consequence_error": candidate_metric.get("consequence_error", "N/A"),
            "utility": candidate_metric.get("utility", "N/A"),
            "selected": "yes" if name == candidate_names[selected_index] else "no",
        })
    save_csv(output_dir / "candidate_comparison.csv", rows)

    metrics_payload = {
        "input": str(args.input),
        "reference": str(args.reference) if args.reference is not None else None,
        "checkpoint": str(args.checkpoint),
        "selected_candidate": candidate_names[selected_index],
        "selection_weights": {name: float(score) for name, score in zip(candidate_names, selection_scores)},
        "utilities": {name: float(util) for name, util in zip(candidate_names, utilities)},
        "candidate_metrics": metrics_dict,
        "selected_metrics": selected_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, default=str)

    print("=" * 92)
    print("PAPER 1 — UNDERWATER IMAGE ENHANCEMENT")
    print("=" * 92)
    print(f"Input: {args.input.name}")
    print("Candidate Evaluation")
    print("-" * 92)
    print(f"{'Candidate':<15}{'PSNR':>12}{'SSIM':>12}{'UIQM':>12}{'UCIQE':>12}")
    for row in rows[:-1]:
        print(f"{row['candidate']:<15}{str(row['psnr']):>12}{str(row['ssim']):>12}{str(row['uiqm']):>12}{str(row['uciqe']):>12}")
    print(f"{rows[-1]['candidate']:<15}{str(rows[-1]['psnr']):>12}{str(rows[-1]['ssim']):>12}{str(rows[-1]['uiqm']):>12}{str(rows[-1]['uciqe']):>12}")
    print("\nPreservation")
    print("-" * 92)
    for name in candidate_names + ["selected"]:
        metric = metrics_dict.get(name, selected_metrics)
        print(f"{name:<15}{metric.get('preservation', 'N/A')}")
    print("\nConsequence")
    for name in candidate_names + ["selected"]:
        metric = metrics_dict.get(name, selected_metrics)
        print(f"{name:<15}{metric.get('consequence_error', 'N/A')}")
    print("\nSelection")
    print(f"Selected candidate: {candidate_names[selected_index]}")
    print("Selection weights:")
    for name, score in zip(candidate_names, selection_scores):
        print(f"  {name}: {score:.6f}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
