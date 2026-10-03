from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from euvp_dataset import EUVPPairedDataset
from evaluate_test import calculate_metrics
from paper1_model import Paper1Model
from transforms import PairedTransform


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
)
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "test.txt"
BEST_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v3" / "paper1_v3_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v3_final"
CANDIDATES = ("conservative", "balanced", "aggressive")
OUTPUTS = ("original", *CANDIDATES, "selected")


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def to_uint8(image):
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_comparison_panel(path, images):
    width, height = images[0][1].shape[1], images[0][1].shape[0]
    label_height = 28
    panel = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, array) in enumerate(images):
        x = index * width
        panel.paste(Image.fromarray(to_uint8(array)), (x, label_height))
        draw.text((x + 6, 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def main():
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Test manifest is missing: {SPLIT_PATH}")
    if not BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Selected best checkpoint is missing: {BEST_CHECKPOINT}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing final evaluation output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    best_epoch = int(checkpoint["epoch"])
    model = Paper1Model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = EUVPPairedDataset(
        input_dir=DATASET_ROOT / "trainA",
        reference_dir=DATASET_ROOT / "trainB",
        split_file=SPLIT_PATH,
        transform=PairedTransform(training=False),
    )
    if len(dataset) < 5:
        raise RuntimeError(f"Expected at least five test images; found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    selected_panel_indices = {
        round(index * (len(dataset) - 1) / 4) for index in range(5)
    }

    metric_values = {
        name: {metric: [] for metric in ("psnr", "ssim", "uiqm", "uciqe")}
        for name in OUTPUTS
    }
    preservation_values = {name: [] for name in (*CANDIDATES, "selected")}
    selection_counts = {name: 0 for name in CANDIDATES}
    original_selected_improvements = {"psnr": 0, "ssim": 0}
    selected_matches_psnr_best = 0
    selected_psnr_gaps = []
    selected_ssim_gaps = []
    per_image = []
    sample_panels = []
    dataset_offset = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Final V3 test evaluation"):
            original = batch["underwater"].to(device)
            reference = batch["reference"].to(device)
            outputs = model(original)
            candidate_images = outputs["candidates"]
            selected_indices = outputs["selected_index"].detach().cpu().tolist()

            for local_index, filename in enumerate(batch["filename"]):
                global_index = dataset_offset + local_index
                reference_image = reference[local_index].detach().cpu().numpy().transpose(1, 2, 0)
                original_image = original[local_index].detach().cpu().numpy().transpose(1, 2, 0)
                images = {"original": original_image}
                candidate_scores = {}
                candidate_l1 = {}

                for candidate_name in CANDIDATES:
                    image = candidate_images[candidate_name][local_index].detach().cpu().numpy().transpose(1, 2, 0)
                    images[candidate_name] = image
                    psnr, ssim, uiqm, uciqe = calculate_metrics(image, reference_image)
                    candidate_scores[candidate_name] = {
                        "psnr": float(psnr),
                        "ssim": float(ssim),
                        "uiqm": float(uiqm),
                        "uciqe": float(uciqe),
                        "preservation": float(outputs["preservation_scores"][candidate_name][local_index].item()),
                    }
                    candidate_l1[candidate_name] = float(np.mean(np.abs(np.clip(image, 0.0, 1.0) - reference_image)))
                    for metric_name in ("psnr", "ssim", "uiqm", "uciqe"):
                        metric_values[candidate_name][metric_name].append(candidate_scores[candidate_name][metric_name])
                    preservation_values[candidate_name].append(candidate_scores[candidate_name]["preservation"])

                selected_name = CANDIDATES[selected_indices[local_index]]
                selected_image = images[selected_name]
                images["selected"] = selected_image
                selected_metrics = dict(candidate_scores[selected_name])
                for metric_name in ("psnr", "ssim", "uiqm", "uciqe"):
                    metric_values["selected"][metric_name].append(selected_metrics[metric_name])

                preservation_values["selected"].append(selected_metrics["preservation"])
                selection_counts[selected_name] += 1

                original_metrics = calculate_metrics(original_image, reference_image)
                original_metric_dict = dict(zip(("psnr", "ssim", "uiqm", "uciqe"), map(float, original_metrics)))
                for metric_name in original_metric_dict:
                    metric_values["original"][metric_name].append(original_metric_dict[metric_name])
                    if selected_metrics[metric_name] > original_metric_dict[metric_name] and metric_name in original_selected_improvements:
                        original_selected_improvements[metric_name] += 1

                reference_best_name = max(CANDIDATES, key=lambda name: candidate_scores[name]["psnr"])
                selected_matches_psnr_best += int(selected_name == reference_best_name)
                selected_psnr_gaps.append(candidate_scores[reference_best_name]["psnr"] - selected_metrics["psnr"])
                selected_ssim_gaps.append(candidate_scores[reference_best_name]["ssim"] - selected_metrics["ssim"])

                per_image.append({
                    "filename": filename,
                    "selected_candidate": selected_name,
                    "reference_best_candidate_by_psnr": reference_best_name,
                    "selected_matches_reference_best_by_psnr": selected_name == reference_best_name,
                    "reference_best_candidate_by_l1": min(CANDIDATES, key=lambda name: candidate_l1[name]),
                    "original": original_metric_dict,
                    "candidates": candidate_scores,
                    "selected": selected_metrics,
                    "selected_minus_original": {
                        key: selected_metrics[key] - original_metric_dict[key]
                        for key in ("psnr", "ssim", "uiqm", "uciqe")
                    },
                })

                if global_index in selected_panel_indices:
                    sample_panels.append((filename, images.copy()))
            dataset_offset += original.shape[0]

    total = len(dataset)
    summary = {
        "checkpoint": str(BEST_CHECKPOINT),
        "best_epoch": best_epoch,
        "split": "test.txt",
        "test_samples": total,
        "metric_implementation": "src/evaluate_test.py calculate_metrics; skimage PSNR/SSIM plus existing UIQM/UCIQE; RGB [0,1]",
        "aggregate_metrics": {
            name: {metric: summarize(values) for metric, values in metric_values[name].items()}
            for name in OUTPUTS
        },
        "candidate_selection_distribution": {
            name: {"count": count, "fraction": count / total, "percent": 100.0 * count / total}
            for name, count in selection_counts.items()
        },
        "selected_vs_reference_best_by_psnr": {
            "matches": selected_matches_psnr_best,
            "fraction": selected_matches_psnr_best / total,
            "mean_psnr_gap_from_best": float(np.mean(selected_psnr_gaps)),
            "mean_ssim_gap_from_psnr_best": float(np.mean(selected_ssim_gaps)),
        },
        "original_vs_selected": {
            metric: {
                "improved_count": count,
                "improved_percent": 100.0 * count / total,
                "mean_delta": summary_placeholder,
            }
            for metric, count in original_selected_improvements.items()
            for summary_placeholder in [float(np.mean([
                row["selected_minus_original"][metric] for row in per_image
            ]))]
        },
        "preservation_statistics": {
            name: summarize(values) for name, values in preservation_values.items()
        },
        "qualitative_panel_indices": sorted(selected_panel_indices),
        "test_used_for_training_or_checkpoint_selection": False,
        "per_image": per_image,
    }

    with (OUTPUT_DIR / "final_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    for index, (filename, images) in enumerate(sample_panels, start=1):
        panel_images = [(name.title(), images[name]) for name in OUTPUTS]
        save_comparison_panel(OUTPUT_DIR / "panels" / f"{index:02d}_{Path(filename).stem}.png", panel_images)

    print(json.dumps({key: value for key, value in summary.items() if key != "per_image"}, indent=2))
    print("Test evaluation complete; test data was not used for training or checkpoint selection.")


if __name__ == "__main__":
    main()
