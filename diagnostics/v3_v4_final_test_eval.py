from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from data.euvp_dataset import EUVPPairedDataset
from data.transforms import PairedTransform
from evaluate_test import calculate_metrics
from models.paper1_model import Paper1Model
from models.paper1_model_v4 import Paper1ModelV4


DATA_ROOT = PROJECT_ROOT / "data" / "EUVP" / "EUVP-Dataset" / "EUVP" / "Paired" / "underwater_imagenet"
TEST_SPLIT = PROJECT_ROOT / "data" / "splits" / "test.txt"
V3_PATH = PROJECT_ROOT / "checkpoints" / "v3" / "paper1_v3_best.pth"
V4_PATH = PROJECT_ROOT / "checkpoints" / "v4" / "paper1_v4_best.pth"
V2_PATH = PROJECT_ROOT / "checkpoints" / "v2" / "paper1_v2_epoch_006.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_test"
REPORT_JSON = OUTPUT_DIR / "v3_vs_v4_test_report.json"
REPORT_MD = OUTPUT_DIR / "v3_vs_v4_test_report.md"
CANDIDATES = ("conservative", "balanced", "aggressive")
METHODS = ("original", "v3_selected", "v4_conservative", "v4_balanced", "v4_aggressive", "v4_selected")
METRICS = ("psnr", "ssim", "uiqm", "uciqe", "preservation")
TIE_TOLERANCE = 1e-6
EXPECTED_HASHES = {
    "v3": "80163DBA423CCDE672318A3DD4F93B68982ADE6A0CD3ED737FA8D326948D0555",
    "v2": "BBC8027FF23DCD477CD2E4BD0C95D7A42222AAAAE2E6A6197AF3D198149C53D1",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def checkpoint_hashes():
    return {
        "v3": sha256(V3_PATH),
        "v4": sha256(V4_PATH),
        "v2": sha256(V2_PATH),
    }


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def distribution(values):
    return summarize(values)


def percentage(mask):
    values = np.asarray(mask, dtype=bool)
    return {"count": int(values.sum()), "percent": float(values.mean() * 100.0)}


def evaluate_rgb(prediction, reference):
    prediction = np.clip(prediction, 0.0, 1.0)
    reference = np.clip(reference, 0.0, 1.0)
    psnr, ssim, uiqm, uciqe = calculate_metrics(prediction, reference)
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "uiqm": float(uiqm),
        "uciqe": float(uciqe),
    }


def rgb_tensor_to_array(tensor):
    array = tensor.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    if not np.isfinite(array).all():
        raise FloatingPointError("Non-finite model image encountered during final evaluation.")
    return np.clip(array, 0.0, 1.0)


def render_markdown(report):
    lines = [
        "# Paper 1 V3 vs V4 — Final Held-Out Test Evaluation",
        "",
        "## 1. Evaluation Configuration",
        "",
        f"- Split: `{report['configuration']['split']}` ({report['configuration']['test_image_count']} images)",
        f"- Preprocessing: {report['configuration']['preprocessing']}",
        f"- Metric implementation: `{report['configuration']['metric_source']}`; RGB HWC float `[0,1]`.",
        f"- V3 checkpoint: `{report['configuration']['v3_checkpoint']}` (epoch {report['configuration']['v3_epoch']})",
        f"- V4 checkpoint: `{report['configuration']['v4_checkpoint']}` (epoch {report['configuration']['v4_epoch']})",
        f"- V2 SHA recorded for safety only: `{report['configuration']['v2_sha256_before']}`",
        "- No training, tuning, or epoch selection occurred during this evaluation.",
        "",
        "## 2. Main Metrics",
        "",
        "| Method | PSNR | SSIM | UIQM | UCIQE | Preservation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "original": "Original",
        "v3_selected": "V3 Selected",
        "v4_conservative": "V4 Conservative",
        "v4_balanced": "V4 Balanced",
        "v4_aggressive": "V4 Aggressive",
        "v4_selected": "V4 Selected",
    }
    for method in METHODS:
        stats = report["metric_summary"][method]
        lines.append(
            f"| {labels[method]} | {stats['psnr']['mean']:.4f} | {stats['ssim']['mean']:.6f} | "
            f"{stats['uiqm']['mean']:.4f} | {stats['uciqe']['mean']:.4f} | {stats['preservation']['mean']:.6f} |"
        )
        lines.extend([
            "",
            "Original preservation is 1.0 by identity (input compared with itself).",
        ])

    lines.extend(["", "## 3. Delta From Original", "", "| Method | Δ PSNR | Δ SSIM | Δ UIQM | Δ UCIQE | Δ Preservation |", "|---|---:|---:|---:|---:|---:|"])
    for method in ("v3_selected", "v4_conservative", "v4_balanced", "v4_aggressive", "v4_selected"):
        delta = report["mean_delta_from_original"][method]
        lines.append(
            f"| {labels[method]} | {delta['psnr']:+.4f} | {delta['ssim']:+.6f} | {delta['uiqm']:+.4f} | "
            f"{delta['uciqe']:+.4f} | {delta['preservation']:+.6f} |"
        )

    lines.extend(["", "## 4. V4 Candidate Comparison", "", "| Candidate | PSNR | SSIM | UIQM | UCIQE | Preservation | Reference L1 | Consequence L1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for candidate in CANDIDATES:
        item = report["v4_candidate_summary"][candidate]
        lines.append(
            f"| {candidate.title()} | {item['psnr']['mean']:.4f} | {item['ssim']['mean']:.6f} | {item['uiqm']['mean']:.4f} | "
            f"{item['uciqe']['mean']:.4f} | {item['preservation']['mean']:.6f} | {item['reference_l1']['mean']:.6f} | {item['consequence_loss']['mean']:.6f} |"
        )

    lines.extend(["", "## 5. V4 Selection Distribution", ""])
    for candidate in CANDIDATES:
        item = report["v4_selection_distribution"][candidate]
        lines.append(f"- {candidate.title()}: {item['count']} ({item['percent']:.2f}%)")
    lines.extend([
        "",
        f"Reference-L1 target counts: {report['reference_l1_target_distribution']}",
        f"Selector/reference-L1 agreement: {report['selector_reference_agreement']['count']}/{report['configuration']['test_image_count']} ({report['selector_reference_agreement']['percent']:.2f}%).",
        f"Consequence-best agreement: {report['consequence_best_agreement']['count']}/{report['configuration']['test_image_count']} ({report['consequence_best_agreement']['percent']:.2f}%).",
        "",
        "## 6. V3 vs V4 Selected",
        "",
        "| Metric | Mean V4−V3 | Median V4−V3 | Std paired Δ | V4 > V3 | V4 < V3 | Tied (±1e-6) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for metric in METRICS:
        item = report["v4_vs_v3_selected"][metric]
        lines.append(
            f"| {metric.upper()} | {item['mean_difference']:+.6f} | {item['median_difference']:+.6f} | {item['std_difference']:.6f} | "
            f"{item['v4_greater']['percent']:.2f}% | {item['v4_less']['percent']:.2f}% | {item['tied']['percent']:.2f}% |"
        )

    lines.extend(["", "## 7. Per-Image Improvements vs Original", "", "| Model | Metric | Improved | Percentage | Mean paired Δ | Std paired Δ |", "|---|---|---:|---:|---:|---:|"])
    for model_name, method in (("V3", "v3_selected"), ("V4", "v4_selected")):
        for metric in METRICS:
            item = report["improvement_vs_original"][method][metric]
            lines.append(
                f"| {model_name} Selected | {metric.upper()} | {item['count']} | {item['percent']:.2f}% | "
                f"{item['mean_delta']:+.6f} | {item['std_delta']:.6f} |"
            )

    lines.extend([
        "",
        "## 8. Statistical Summary",
        "",
        "| Method | Metric | Mean | Median | Std | Min | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        for metric in METRICS:
            stats = report["metric_summary"][method][metric]
            lines.append(
                f"| {labels[method]} | {metric.upper()} | {stats['mean']:.6f} | {stats['median']:.6f} | "
                f"{stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |"
            )

    lines.extend(["", "## 9. Quality Trade-Offs", ""])
    for metric, comparison in report["v4_selected_tradeoffs"]["v4_selected_vs_original"].items():
        lines.append(f"- V4 Selected vs Original, {metric.upper()}: {comparison['improved']} improved, {comparison['declined']} declined, {comparison['tied']} tied.")
    lines.extend([
        "- UIQM/UCIQE are no-reference underwater quality measures, not ground-truth reconstruction accuracy.",
        "- Preservation is a separate model score and is not established by changes in UIQM/UCIQE.",
        "",
        "## 10. Extreme Cases",
    ])
    for key, record in report["v4_selected_extremes"].items():
        lines.append(f"- {key}: `{record['filename']}`; Original {record['original']:.6f}, V4 Selected {record['selected']:.6f}, Δ {record['delta']:+.6f}.")

    lines.extend([
        "",
        "## 11. Scale and Diversity Diagnostics",
        "",
        f"V4 learned gains: {report['v4_learned_gains']}",
        f"Mean residual L2: {report['v4_mean_residual_l2']}",
        f"Mean absolute residual: {report['v4_mean_absolute_residual']}",
        f"Residual cosine similarity: {report['v4_residual_cosine']}",
        "",
        "## 12. Configuration, Hashes, and Safety",
        "",
        f"- V3 SHA before/after: `{report['configuration']['v3_sha256_before']}` / `{report['configuration']['v3_sha256_after']}`",
        f"- V4 SHA before/after: `{report['configuration']['v4_sha256_before']}` / `{report['configuration']['v4_sha256_after']}`",
        f"- V2 SHA before/after: `{report['configuration']['v2_sha256_before']}` / `{report['configuration']['v2_sha256_after']}`",
        "- Only `test.txt` was evaluated. No training, test tuning, or checkpoint changes occurred.",
        f"- Evaluator: `{report['configuration']['evaluator_script']}`; metric functions from `{report['configuration']['metric_source']}`.",
        "",
        "## 13. Scientific Interpretation and Limitations",
        "",
        report["scientific_interpretation"],
        "",
        "## 14. Exact Next Step",
        "",
        report["recommended_next_step"],
        "",
    ])
    return "\n".join(lines)


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing final report directory: {OUTPUT_DIR}")
    for path in (V2_PATH, V3_PATH, V4_PATH, TEST_SPLIT):
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes_before = checkpoint_hashes()
    for name, expected in EXPECTED_HASHES.items():
        if hashes_before[name] != expected:
            raise RuntimeError(f"Frozen {name.upper()} checkpoint hash mismatch: {hashes_before[name]}")

    v3_checkpoint = torch.load(V3_PATH, map_location="cpu", weights_only=False)
    v4_checkpoint = torch.load(V4_PATH, map_location="cpu", weights_only=False)
    if int(v3_checkpoint["epoch"]) != 8 or int(v4_checkpoint["epoch"]) != 10:
        raise RuntimeError("Unexpected V3/V4 best checkpoint epoch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3_model = Paper1Model().to(device)
    v3_model.load_state_dict(v3_checkpoint["model_state_dict"], strict=True)
    v3_model.eval()
    v4_model = Paper1ModelV4().to(device)
    v4_model.load_state_dict(v4_checkpoint["model_state_dict"], strict=True)
    v4_model.eval()

    dataset = EUVPPairedDataset(
        DATA_ROOT / "trainA",
        DATA_ROOT / "trainB",
        TEST_SPLIT,
        transform=PairedTransform(training=False),
    )
    if len(dataset) != 370:
        raise RuntimeError(f"Expected 370 held-out test images; found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    metric_values = {method: {metric: [] for metric in METRICS} for method in METHODS}
    candidate_values = {
        name: {metric: [] for metric in (*METRICS, "reference_l1", "consequence_loss", "utility", "probability")}
        for name in CANDIDATES
    }
    improvement_flags = {
        method: {metric: [] for metric in METRICS}
        for method in ("v3_selected", "v4_selected")
    }
    per_image = []
    v4_selection_counts = {name: 0 for name in CANDIDATES}
    reference_target_counts = {name: 0 for name in CANDIDATES}
    consequence_target_counts = {name: 0 for name in CANDIDATES}
    selector_agreement = 0
    consequence_agreement = 0
    aggressive_residual_l2 = []
    residual_abs = {name: [] for name in CANDIDATES}
    residual_l2 = {name: [] for name in CANDIDATES}
    pairwise_cosine = {
        ("conservative", "balanced"): [],
        ("balanced", "aggressive"): [],
        ("conservative", "aggressive"): [],
    }
    extrema = {
        "max_psnr_improvement": None,
        "max_psnr_degradation": None,
        "max_ssim_improvement": None,
        "max_ssim_degradation": None,
    }

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Final V3/V4 test evaluation"):
            original = batch["underwater"].to(device)
            reference = batch["reference"].to(device)
            v3_output = v3_model(original)
            v4_output = v4_model(original)

            original_arrays = rgb_tensor_to_array(original)
            reference_arrays = rgb_tensor_to_array(reference)
            v4_candidate_arrays = {
                name: rgb_tensor_to_array(v4_output["candidates"][name])
                for name in CANDIDATES
            }
            v3_selected_arrays = rgb_tensor_to_array(v3_output["selected_image"])
            v4_selected_arrays = rgb_tensor_to_array(v4_output["selected_image"])

            for index, filename in enumerate(batch["filename"]):
                original_metrics = evaluate_rgb(original_arrays[index], reference_arrays[index])
                original_metrics["preservation"] = 1.0
                v3_selected_name = CANDIDATES[int(v3_output["selected_index"][index].item())]
                v3_metrics = evaluate_rgb(v3_selected_arrays[index], reference_arrays[index])
                v3_metrics["preservation"] = float(v3_output["preservation_scores"][v3_selected_name][index].item())

                v4_metrics = {}
                l1s = {}
                consequences = {}
                utilities = {}
                probabilities = {}
                residual_arrays = {}
                for name_index, name in enumerate(CANDIDATES):
                    prediction = v4_candidate_arrays[name][index]
                    candidate_metrics = evaluate_rgb(prediction, reference_arrays[index])
                    candidate_metrics["preservation"] = float(v4_output["preservation_scores"][name][index].item())
                    v4_metrics[name] = candidate_metrics
                    l1s[name] = float(np.mean(np.abs(np.clip(prediction, 0.0, 1.0) - reference_arrays[index])))
                    consequence_output = v4_model.consequence_check(original[index:index + 1], v4_output["candidates"][name][index:index + 1])
                    consequences[name] = float(F.l1_loss(consequence_output["reconstructed"], original[index:index + 1]).item())
                    utilities[name] = float(v4_output["utilities"][index, name_index].item())
                    probabilities[name] = float(v4_output["selection_weights"][index, name_index].item())
                    residual_arrays[name] = v4_output["candidates"][name][index] - original[index]
                    residual_l2[name].append(float(torch.linalg.vector_norm(residual_arrays[name].float()).item()))
                    residual_abs[name].append(float(residual_arrays[name].abs().mean().item()))
                    for metric in METRICS:
                        candidate_values[name][metric].append(candidate_metrics[metric])
                    candidate_values[name]["reference_l1"].append(l1s[name])
                    candidate_values[name]["consequence_loss"].append(consequences[name])
                    candidate_values[name]["utility"].append(utilities[name])
                    candidate_values[name]["probability"].append(probabilities[name])

                selected_name = CANDIDATES[int(v4_output["selected_index"][index].item())]
                v4_selected_metrics = dict(v4_metrics[selected_name])
                target_name = min(CANDIDATES, key=lambda name: l1s[name])
                consequence_best_name = min(CANDIDATES, key=lambda name: consequences[name])
                v4_selection_counts[selected_name] += 1
                reference_target_counts[target_name] += 1
                consequence_target_counts[consequence_best_name] += 1
                selector_agreement += int(selected_name == target_name)
                consequence_agreement += int(selected_name == consequence_best_name)
                aggressive_residual_l2.append(residual_l2["aggressive"][-1])

                for left, right in pairwise_cosine:
                    pairwise_cosine[(left, right)].append(float(F.cosine_similarity(
                        residual_arrays[left].flatten().float(),
                        residual_arrays[right].flatten().float(),
                        dim=0,
                        eps=1e-8,
                    ).item()))

                v4_outputs = {
                    "v4_conservative": v4_metrics["conservative"],
                    "v4_balanced": v4_metrics["balanced"],
                    "v4_aggressive": v4_metrics["aggressive"],
                    "v4_selected": v4_selected_metrics,
                }
                metric_map = {
                    "original": original_metrics,
                    "v3_selected": v3_metrics,
                    **v4_outputs,
                }
                for method, metrics in metric_map.items():
                    for metric in METRICS:
                        metric_values[method][metric].append(metrics[metric])
                for model_name, metrics in (("v3_selected", v3_metrics), ("v4_selected", v4_selected_metrics)):
                    for metric in METRICS:
                        improvement_flags[model_name][metric].append(metrics[metric] > original_metrics[metric])

                record = {
                    "filename": filename,
                    "original": original_metrics,
                    "v3_selected": {**v3_metrics, "selected_candidate": v3_selected_name},
                    "v4_candidates": v4_metrics,
                    "v4_selected": {**v4_selected_metrics, "selected_candidate": selected_name},
                    "v4_reference_l1": l1s,
                    "v4_consequence_loss": consequences,
                    "v4_utilities": utilities,
                    "v4_probabilities": probabilities,
                    "v4_reference_l1_best_candidate": target_name,
                    "v4_consequence_best_candidate": consequence_best_name,
                }
                per_image.append(record)

                for metric in ("psnr", "ssim"):
                    delta = v4_selected_metrics[metric] - original_metrics[metric]
                    if metric == "psnr":
                        keys = ("max_psnr_improvement", "max_psnr_degradation")
                    else:
                        keys = ("max_ssim_improvement", "max_ssim_degradation")
                    positive_key, negative_key = keys
                    key = positive_key if delta >= 0 else negative_key
                    existing = extrema[key]
                    if existing is None or (delta > existing["delta"] if delta >= 0 else delta < existing["delta"]):
                        extrema[key] = {
                            "filename": filename,
                            "original": original_metrics[metric],
                            "selected": v4_selected_metrics[metric],
                            "delta": float(delta),
                        }

    hashes_after = checkpoint_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError(f"Checkpoint hash changed during final evaluation: before={hashes_before}, after={hashes_after}")

    n = len(dataset)
    metric_summary = {
        method: {metric: summarize(values) for metric, values in methods.items()}
        for method, methods in metric_values.items()
    }
    mean_delta_from_original = {
        method: {
            metric: float(np.mean(np.asarray(metric_values[method][metric]) - np.asarray(metric_values["original"][metric])))
            for metric in METRICS
        }
        for method in METHODS if method != "original"
    }
    improvement_summary = {}
    for method, metrics in improvement_flags.items():
        improvement_summary[method] = {}
        for metric, flags in metrics.items():
            deltas = np.asarray(metric_values[method][metric]) - np.asarray(metric_values["original"][metric])
            improvement_summary[method][metric] = {
                "count": int(np.sum(flags)),
                "percent": float(np.mean(flags) * 100.0),
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
            }

    paired_v4_v3 = {}
    for metric in METRICS:
        differences = np.asarray(metric_values["v4_selected"][metric]) - np.asarray(metric_values["v3_selected"][metric])
        paired_v4_v3[metric] = {
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "std_difference": float(differences.std()),
            "v4_greater": percentage(differences > TIE_TOLERANCE),
            "v4_less": percentage(differences < -TIE_TOLERANCE),
            "tied": percentage(np.abs(differences) <= TIE_TOLERANCE),
        }

    tradeoffs = {
        "v4_selected_vs_original": {
            metric: {
                "improved": int(np.sum(np.asarray(metric_values["v4_selected"][metric]) > np.asarray(metric_values["original"][metric]))),
                "declined": int(np.sum(np.asarray(metric_values["v4_selected"][metric]) < np.asarray(metric_values["original"][metric]))),
                "tied": int(np.sum(np.asarray(metric_values["v4_selected"][metric]) == np.asarray(metric_values["original"][metric]))),
            }
            for metric in METRICS
        },
        "psnr_ssim_up_uiqm_down": int(sum(
            row["v4_selected"]["psnr"] > row["original"]["psnr"]
            and row["v4_selected"]["ssim"] > row["original"]["ssim"]
            and row["v4_selected"]["uiqm"] < row["original"]["uiqm"]
            for row in per_image
        )),
        "psnr_ssim_up_uciqe_down": int(sum(
            row["v4_selected"]["psnr"] > row["original"]["psnr"]
            and row["v4_selected"]["ssim"] > row["original"]["ssim"]
            and row["v4_selected"]["uciqe"] < row["original"]["uciqe"]
            for row in per_image
        )),
        "reference_metrics_up_preservation_down": int(sum(
            row["v4_selected"]["psnr"] > row["original"]["psnr"]
            and row["v4_selected"]["ssim"] > row["original"]["ssim"]
            and row["v4_selected"]["preservation"] < row["original"]["preservation"]
            for row in per_image
        )),
    }

    v4_selected_mean = metric_summary["v4_selected"]
    v3_selected_mean = metric_summary["v3_selected"]
    scientific_interpretation = (
        f"On {n} held-out test images, V4 Selected mean PSNR/SSIM were "
        f"{v4_selected_mean['psnr']['mean']:.4f}/{v4_selected_mean['ssim']['mean']:.6f}, versus "
        f"V3 Selected {v3_selected_mean['psnr']['mean']:.4f}/{v3_selected_mean['ssim']['mean']:.6f}. "
        f"V4 selected Aggressive on {v4_selection_counts['aggressive']} images and matched the reference-L1-best candidate "
        f"on {selector_agreement}/{n}. These are measurements, not a winner claim; UIQM/UCIQE are no-reference metrics, "
        "and preservation is a separate model score."
    )
    recommended_next_step = (
        "Freeze these test results. Review the metric trade-offs and validation-selected epoch, then plan a controlled ablation "
        "on the same train/validation protocol. Do not tune V4 using this test set."
    )

    report = {
        "configuration": {
            "evaluator_script": "diagnostics/v3_v4_final_test_eval.py",
            "metric_source": "src/evaluate_test.py:calculate_metrics, uiqm, uciqe",
            "preprocessing": "EUVPPairedDataset + PairedTransform(training=False): RGB conversion, ToTensor-equivalent [0,1], no random augmentation",
            "image_range": "RGB float [0,1], HWC for metric helper",
            "split": "data/splits/test.txt only",
            "test_image_count": n,
            "v3_checkpoint": "checkpoints/v3/paper1_v3_best.pth",
            "v3_epoch": int(v3_checkpoint["epoch"]),
            "v4_checkpoint": "checkpoints/v4/paper1_v4_best.pth",
            "v4_epoch": int(v4_checkpoint["epoch"]),
            "v2_checkpoint_hashed_not_loaded": "checkpoints/v2/paper1_v2_epoch_006.pth",
            "v3_sha256_before": hashes_before["v3"],
            "v3_sha256_after": hashes_after["v3"],
            "v4_sha256_before": hashes_before["v4"],
            "v4_sha256_after": hashes_after["v4"],
            "v2_sha256_before": hashes_before["v2"],
            "v2_sha256_after": hashes_after["v2"],
            "test_only_dataset_split_evaluated": True,
            "tie_tolerance": TIE_TOLERANCE,
            "tie_tolerance_units": "metric units",
            "no_training_or_tuning": True,
        },
        "metric_summary": metric_summary,
        "mean_delta_from_original": mean_delta_from_original,
        "v4_candidate_summary": {
            name: {**metric_summary[f"v4_{name}"],
                  "reference_l1": summarize(candidate_values[name]["reference_l1"]),
                  "consequence_loss": summarize(candidate_values[name]["consequence_loss"])}
            for name in CANDIDATES
        },
        "v4_selection_distribution": {
            name: {"count": count, "percent": 100.0 * count / n}
            for name, count in v4_selection_counts.items()
        },
        "reference_l1_target_distribution": {
            name: {"count": count, "percent": 100.0 * count / n}
            for name, count in reference_target_counts.items()
        },
        "consequence_best_distribution": {
            name: {"count": count, "percent": 100.0 * count / n}
            for name, count in consequence_target_counts.items()
        },
        "selector_reference_agreement": {"count": selector_agreement, "percent": 100.0 * selector_agreement / n},
        "consequence_best_agreement": {"count": consequence_agreement, "percent": 100.0 * consequence_agreement / n},
        "v4_vs_v3_selected": paired_v4_v3,
        "improvement_vs_original": improvement_summary,
        "v4_selected_tradeoffs": tradeoffs,
        "v4_selected_extremes": extrema,
        "v4_learned_gains": {name: float(value.detach().item()) for name, value in v4_model.candidate_generator.candidate_gains().items()},
        "v4_mean_residual_l2": {name: distribution(values) for name, values in residual_l2.items()},
        "v4_mean_absolute_residual": {name: distribution(values) for name, values in residual_abs.items()},
        "v4_residual_cosine": {
            f"{left}_vs_{right}": distribution(values)
            for (left, right), values in pairwise_cosine.items()
        },
        "scientific_interpretation": scientific_interpretation,
        "limitations": [
            "V3 and V4 are evaluated on the same held-out test images with the same preprocessing and metric functions.",
            "V2 historical checkpoint is not strictly compatible with the current architecture and was not loaded.",
            "UIQM/UCIQE are no-reference image quality metrics, not ground-truth accuracy.",
            "Preservation is a model-defined fixed score and is not itself a downstream task metric.",
            "Per-image best-candidate labels use reference L1 only as an analysis target and do not control selection.",
        ],
        "recommended_next_step": recommended_next_step,
        "per_image": per_image,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    with REPORT_JSON.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, indent=2))
    print("REPORT_JSON", REPORT_JSON)
    print("REPORT_MARKDOWN", REPORT_MD)
    print("test.txt accessed: YES; training/validation splits accessed: NO")


if __name__ == "__main__":
    main()
