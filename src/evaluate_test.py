from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from euvp_dataset import EUVPPairedDataset
from paper1_model import Paper1Model
from transforms import PairedTransform


# ============================================================
# Configuration
# ============================================================

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
)

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "v2"
    / "paper1_v2_epoch_006.pth"
)

NAMES = ["Original", "Conservative", "Balanced", "Aggressive", "Selected"]
CANDIDATE_NAMES = ["conservative", "balanced", "aggressive"]


# ============================================================
# Utility metrics
# ============================================================

def eme(channel, block_size=8):
    h, w = channel.shape
    total = 0.0
    count = 0

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = channel[y : min(y + block_size, h), x : min(x + block_size, w)]
            if block.size == 0:
                continue
            max_val = float(np.max(block))
            min_val = float(np.min(block))
            if min_val <= 0:
                min_val = 1e-6
            if max_val <= 0:
                continue
            total += np.log(max_val / min_val)
            count += 1

    if count == 0:
        return 0.0
    return (2.0 / count) * total


def uicm(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    RG = R - G
    YB = 0.5 * (R + G) - B

    rg_mean = float(np.mean(RG))
    rg_std = float(np.std(RG))
    yb_mean = float(np.mean(YB))
    yb_std = float(np.std(YB))

    return -0.0268 * rg_mean + 0.1586 * rg_std - 0.0786 * yb_mean + 0.0112 * yb_std


def uism(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return 0.299 * eme(R) + 0.587 * eme(G) + 0.114 * eme(B)


def uiconm(image):
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return eme(gray)


def uiqm(image):
    c1 = 0.0282
    c2 = 0.2953
    c3 = 0.6761
    color = uicm(image)
    sharpness = uism(image)
    contrast = uiconm(image)
    return c1 * color + c2 * sharpness + c3 * contrast


def uciqe(image):
    lab = rgb2lab(image)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)
    contrast_l = (np.percentile(L, 99) - np.percentile(L, 1)) / 100.0
    hsv = rgb2hsv(image)
    saturation = hsv[:, :, 1]
    mean_s = np.mean(saturation)
    return 0.4680 * sigma_c + 0.2745 * contrast_l + 0.2576 * mean_s


def calculate_metrics(prediction, reference):
    prediction = np.clip(prediction, 0.0, 1.0)
    reference = np.clip(reference, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(reference, prediction, data_range=1.0)
    ssim = structural_similarity(reference, prediction, channel_axis=2, data_range=1.0)
    return psnr, ssim, uiqm(prediction), uciqe(prediction)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_image_array(path: Path, image_rgb: np.ndarray):
    image_rgb = np.clip(image_rgb, 0.0, 1.0)
    uint8 = np.round(image_rgb * 255.0).astype(np.uint8)
    Image.fromarray(uint8).save(path)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_metric_list(values):
    if not values:
        return {"mean": None, "std": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_split(checkpoint_path: Path, split_name: str, dataset_root: Path, batch_size: int = 8, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EUVPPairedDataset(
        input_dir=dataset_root / "trainA",
        reference_dir=dataset_root / "trainB",
        split_file=PROJECT_ROOT / "data" / "splits" / f"{split_name}.txt",
        transform=PairedTransform(training=False),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = Paper1Model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    metrics = {name: {"psnr": [], "ssim": [], "uiqm": [], "uciqe": []} for name in NAMES}
    selection_counts = {0: 0, 1: 0, 2: 0}
    improvement_counts = {"Conservative": {"psnr": 0, "ssim": 0}, "Balanced": {"psnr": 0, "ssim": 0}, "Aggressive": {"psnr": 0, "ssim": 0}, "Selected": {"psnr": 0, "ssim": 0}}
    per_image_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name} split"):
            underwater = batch["underwater"].to(device, non_blocking=True)
            reference = batch["reference"].to(device, non_blocking=True)
            outputs = model(underwater)
            candidates = outputs["candidates"]
            selected = outputs["selected_image"]

            for idx in outputs["selected_index"].detach().cpu().numpy():
                selection_counts[int(idx)] += 1

            original_np = underwater.detach().cpu().numpy().transpose(0, 2, 3, 1)
            reference_np = reference.detach().cpu().numpy().transpose(0, 2, 3, 1)
            candidate_arrays = {
                name: candidates[name].detach().cpu().numpy().transpose(0, 2, 3, 1) for name in CANDIDATE_NAMES
            }
            selected_np = selected.detach().cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(original_np.shape[0]):
                reference_img = np.clip(reference_np[i], 0.0, 1.0)
                images = {
                    "Original": original_np[i],
                    "Conservative": candidate_arrays["conservative"][i],
                    "Balanced": candidate_arrays["balanced"][i],
                    "Aggressive": candidate_arrays["aggressive"][i],
                    "Selected": selected_np[i],
                }
                image_results = {}

                for name, image in images.items():
                    psnr, ssim, uiqm_score, uciqe_score = calculate_metrics(image, reference_img)
                    image_results[name] = (psnr, ssim, uiqm_score, uciqe_score)
                    metrics[name]["psnr"].append(psnr)
                    metrics[name]["ssim"].append(ssim)
                    metrics[name]["uiqm"].append(uiqm_score)
                    metrics[name]["uciqe"].append(uciqe_score)

                original_psnr = image_results["Original"][0]
                original_ssim = image_results["Original"][1]
                for name in ["Conservative", "Balanced", "Aggressive", "Selected"]:
                    if image_results[name][0] > original_psnr:
                        improvement_counts[name]["psnr"] += 1
                    if image_results[name][1] > original_ssim:
                        improvement_counts[name]["ssim"] += 1

                filename = batch["filename"][i]
                per_image_rows.append(
                    {
                        "filename": filename,
                        "original_psnr": float(image_results["Original"][0]),
                        "original_ssim": float(image_results["Original"][1]),
                        "original_uiqm": float(image_results["Original"][2]),
                        "original_uciqe": float(image_results["Original"][3]),
                        "conservative_psnr": float(image_results["Conservative"][0]),
                        "conservative_ssim": float(image_results["Conservative"][1]),
                        "conservative_uiqm": float(image_results["Conservative"][2]),
                        "conservative_uciqe": float(image_results["Conservative"][3]),
                        "balanced_psnr": float(image_results["Balanced"][0]),
                        "balanced_ssim": float(image_results["Balanced"][1]),
                        "balanced_uiqm": float(image_results["Balanced"][2]),
                        "balanced_uciqe": float(image_results["Balanced"][3]),
                        "aggressive_psnr": float(image_results["Aggressive"][0]),
                        "aggressive_ssim": float(image_results["Aggressive"][1]),
                        "aggressive_uiqm": float(image_results["Aggressive"][2]),
                        "aggressive_uciqe": float(image_results["Aggressive"][3]),
                        "selected_psnr": float(image_results["Selected"][0]),
                        "selected_ssim": float(image_results["Selected"][1]),
                        "selected_uiqm": float(image_results["Selected"][2]),
                        "selected_uciqe": float(image_results["Selected"][3]),
                    }
                )

    aggregate = {}
    for name in NAMES:
        aggregate[name] = {
            "psnr": summarize_metric_list(metrics[name]["psnr"]),
            "ssim": summarize_metric_list(metrics[name]["ssim"]),
            "uiqm": summarize_metric_list(metrics[name]["uiqm"]),
            "uciqe": summarize_metric_list(metrics[name]["uciqe"]),
        }

    total = sum(selection_counts.values())
    selection_distribution = {
        "conservative": {"count": selection_counts[0], "fraction": 0.0 if total == 0 else selection_counts[0] / total},
        "balanced": {"count": selection_counts[1], "fraction": 0.0 if total == 0 else selection_counts[1] / total},
        "aggressive": {"count": selection_counts[2], "fraction": 0.0 if total == 0 else selection_counts[2] / total},
    }

    improvement_summary = {
        name: {
            "psnr_improved": improvement_counts[name]["psnr"],
            "ssim_improved": improvement_counts[name]["ssim"],
            "psnr_fraction": improvement_counts[name]["psnr"] / len(dataset),
            "ssim_fraction": improvement_counts[name]["ssim"] / len(dataset),
        }
        for name in ["Conservative", "Balanced", "Aggressive", "Selected"]
    }

    return {
        "dataset_size": len(dataset),
        "aggregate": aggregate,
        "selection_distribution": selection_distribution,
        "improvement": improvement_summary,
        "per_image": per_image_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the current Paper 1 model on a validation or test split.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint to load.")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Split to evaluate.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "current_model_evaluation", help="Folder to store evaluation metrics and samples.")
    parser.add_argument("--max-samples", type=int, default=8, help="Number of sample outputs to save.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate_split(args.checkpoint, args.split, args.dataset_root, batch_size=args.batch_size, device=device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "aggregate_metrics.json", results["aggregate"])
    save_json(output_dir / "selection_distribution.json", results["selection_distribution"])
    save_json(output_dir / "improvement_summary.json", results["improvement"])

    fieldnames = [
        "filename",
        "original_psnr", "original_ssim", "original_uiqm", "original_uciqe",
        "conservative_psnr", "conservative_ssim", "conservative_uiqm", "conservative_uciqe",
        "balanced_psnr", "balanced_ssim", "balanced_uiqm", "balanced_uciqe",
        "aggressive_psnr", "aggressive_ssim", "aggressive_uiqm", "aggressive_uciqe",
        "selected_psnr", "selected_ssim", "selected_uiqm", "selected_uciqe",
    ]
    save_csv(output_dir / "per_image_results.csv", results["per_image"], fieldnames)

    aggregate_rows = []
    for name in NAMES:
        row = {"candidate": name}
        for metric_name, metric_stats in results["aggregate"][name].items():
            for stat_key, stat_value in metric_stats.items():
                row[f"{metric_name}_{stat_key}"] = stat_value
        aggregate_rows.append(row)
    save_csv(output_dir / "candidate_metrics.csv", aggregate_rows, list(aggregate_rows[0].keys()) if aggregate_rows else ["candidate"])

    config = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "dataset_root": str(args.dataset_root),
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "image_range": "[0,1]",
        "metric_implementation": {
            "psnr": "skimage.metrics.peak_signal_noise_ratio(data_range=1.0)",
            "ssim": "skimage.metrics.structural_similarity(channel_axis=2, data_range=1.0)",
            "uiqm": "custom UIQM implementation in src/evaluate_test.py",
            "uciqe": "custom UCIQE implementation in src/evaluate_test.py",
        },
    }
    save_json(output_dir / "configuration.json", config)

    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    model = Paper1Model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"], strict=True)
    model.eval()

    sample_dataset = EUVPPairedDataset(
        input_dir=args.dataset_root / "trainA",
        reference_dir=args.dataset_root / "trainB",
        split_file=PROJECT_ROOT / "data" / "splits" / f"{args.split}.txt",
        transform=PairedTransform(training=False),
    )
    sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=0)
    for index, batch in enumerate(sample_loader):
        if index >= args.max_samples:
            break
        underwater = batch["underwater"].to(device)
        reference = batch["reference"].to(device)
        outputs = model(underwater)
        selected = outputs["selected_image"][0].detach().cpu().numpy().transpose(1, 2, 0)
        conservative = outputs["candidates"]["conservative"][0].detach().cpu().numpy().transpose(1, 2, 0)
        balanced = outputs["candidates"]["balanced"][0].detach().cpu().numpy().transpose(1, 2, 0)
        aggressive = outputs["candidates"]["aggressive"][0].detach().cpu().numpy().transpose(1, 2, 0)
        original = underwater[0].detach().cpu().numpy().transpose(1, 2, 0)
        ref_np = reference[0].detach().cpu().numpy().transpose(1, 2, 0)

        save_image_array(sample_dir / f"{index:02d}_original.png", original)
        save_image_array(sample_dir / f"{index:02d}_conservative.png", conservative)
        save_image_array(sample_dir / f"{index:02d}_balanced.png", balanced)
        save_image_array(sample_dir / f"{index:02d}_aggressive.png", aggressive)
        save_image_array(sample_dir / f"{index:02d}_selected.png", selected)
        save_image_array(sample_dir / f"{index:02d}_reference.png", ref_np)

    print()
    print("=" * 76)
    print("PAPER 1 — CURRENT MODEL EVALUATION")
    print("=" * 76)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Samples: {results['dataset_size']}")

    print(f"{'Output':<15}{'PSNR':>12}{'SSIM':>12}{'UIQM':>12}{'UCIQE':>12}")
    print("-" * 76)
    for name in NAMES:
        stats = results["aggregate"][name]
        print(f"{name:<15}{stats['psnr']['mean']:>12.4f}{stats['ssim']['mean']:>12.4f}{stats['uiqm']['mean']:>12.4f}{stats['uciqe']['mean']:>12.4f}")

    print(f"\nSelection distribution: {results['selection_distribution']}")
    print(f"Saved evaluation bundle to: {output_dir}")


if __name__ == "__main__":
    main()
