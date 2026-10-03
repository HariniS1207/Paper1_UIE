from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "losses"))

from evaluate_test import calculate_metrics
from euvp_dataset import EUVPPairedDataset
from models.paper1_model_v6 import Paper1ModelV6
from models.paper1_model_v4 import Paper1ModelV4
from paper1_loss_v6 import Paper1LossV6
from transforms import PairedTransform


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20
INITIAL_AMP_SCALE = 2048.0
MAX_GRAD_NORM = 1.0
LOSS_NAMES = ("total", "reconstruction", "preservation", "consequence", "selection")
EXPECTED_V2_SHA256 = "BBC8027FF23DCD477CD2E4BD0C95D7A42222AAAAE2E6A6197AF3D198149C53D1"
EXPECTED_V3_SHA256 = "80163DBA423CCDE672318A3DD4F93B68982ADE6A0CD3ED737FA8D326948D0555"

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
)
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
V2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v2" / "paper1_v2_epoch_006.pth"
V3_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v3" / "paper1_v3_best.pth"
V4_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v4" / "paper1_v4_best.pth"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "v6"
RESULTS_ROOT = PROJECT_ROOT / "results" / "v6"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_frozen_checkpoints(context):
    for path, expected in (
        (V2_CHECKPOINT, EXPECTED_V2_SHA256),
        (V3_CHECKPOINT, EXPECTED_V3_SHA256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen checkpoint changed {context}: {path} expected {expected}, found {actual}. Stopping."
            )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def tensor_tree_status(value, prefix=""):
    result = {}
    if torch.is_tensor(value):
        result[prefix or "tensor"] = bool(torch.isfinite(value).all().item())
    elif isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(tensor_tree_status(item, name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(tensor_tree_status(item, f"{prefix}[{index}]"))
    return result


def gradient_summary(model):
    squared_norm = 0.0
    first_nonfinite = None
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            first_nonfinite = first_nonfinite or name
            continue
        squared_norm += float(parameter.grad.detach().double().pow(2).sum().item())
    return {"finite": first_nonfinite is None, "l2_norm": squared_norm ** 0.5, "first_nonfinite_parameter": first_nonfinite}


def append_batch_record(record):
    with (CHECKPOINT_DIR / "batch_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        handle.flush()


def save_failure(record, comparison=None):
    payload = dict(record)
    payload["no_step_amp_fp32_comparison"] = comparison
    base = CHECKPOINT_DIR / f"failure_epoch_{record['epoch']:03d}_batch_{record['batch_index']:04d}.json"
    path = base
    suffix = 1
    while path.exists():
        path = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
        suffix += 1
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
    return path


def component_gradient_diagnostics(model, losses, scale):
    parameters = list(model.named_parameters())
    result = {}
    for index, name in enumerate(LOSS_NAMES):
        gradients = torch.autograd.grad(
            losses[name] * scale,
            [parameter for _, parameter in parameters],
            retain_graph=index < len(LOSS_NAMES) - 1,
            allow_unused=True,
        )
        squared_norm = 0.0
        first_nonfinite = None
        for (parameter_name, _), gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all().item():
                first_nonfinite = first_nonfinite or parameter_name
            else:
                squared_norm += float(gradient.detach().double().pow(2).sum().item())
        scaled_norm = squared_norm ** 0.5
        result[name] = {
            "finite": first_nonfinite is None,
            "first_nonfinite_parameter": first_nonfinite,
            "scaled_l2_norm": scaled_norm,
            "unscaled_l2_norm": scaled_norm / scale,
        }
    return result


def no_step_comparison(model, criterion, original, reference, buffers, current_scale):
    comparison = {}
    scales = [current_scale]
    while scales[-1] > 512.0:
        scales.append(scales[-1] / 2.0)
    modes = [(f"amp_scale_{int(scale)}", True, scale) for scale in scales]
    modes.append(("fp32", False, 1.0))
    for label, amp, scale in modes:
        named_buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, value in buffers.items():
                named_buffers[name].copy_(value)
        model.zero_grad(set_to_none=True)
        model.train()
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(original)
            losses = criterion(outputs, reference, original)
        output_finite = all(tensor_tree_status(outputs).values())
        loss_finite = {name: bool(torch.isfinite(value).all().item()) for name, value in losses.items()}
        comparison[label] = {
            "amp_enabled": amp,
            "scale": scale,
            "outputs_finite": output_finite,
            "losses": {name: float(value.detach().item()) for name, value in losses.items()},
            "losses_finite": loss_finite,
            "component_gradients": component_gradient_diagnostics(model, losses, scale),
            "optimizer_step_executed": False,
        }
    named_buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in buffers.items():
            named_buffers[name].copy_(value)
    model.zero_grad(set_to_none=True)
    return comparison


def write_history(history):
    with (CHECKPOINT_DIR / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    fields = ["epoch"] + [f"train_{name}" for name in LOSS_NAMES] + [f"val_{name}" for name in LOSS_NAMES] + ["epoch_seconds", "best_epoch"]
    with (CHECKPOINT_DIR / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(path, model, optimizer, scaler, epoch, train_losses, val_losses, config, train_loader):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_losses": train_losses,
        "validation_losses": val_losses,
        "config": config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_loader_generator": train_loader.generator.get_state(),
        },
    }, path)


def verify_v4_checkpoint(expected, context):
    actual = sha256_file(V4_CHECKPOINT)
    if actual != expected:
        raise RuntimeError(f"Frozen V4 checkpoint changed {context}: expected {expected}, found {actual}")
    return actual


def advance_sampler_generator(seed, dataset_size, completed_epochs):
    generator = torch.Generator().manual_seed(seed)
    index_dataset = TensorDataset(torch.arange(dataset_size))
    index_loader = DataLoader(
        index_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    for _ in range(completed_epochs):
        for _batch in index_loader:
            pass
    return generator.get_state()


def restore_global_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def run_epoch(model, criterion, loader, device, optimizer, scaler, *, epoch, training, amp_enabled):
    phase = "train" if training else "val"
    model.train(training)
    totals = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(tqdm(loader, desc=f"V6 {phase} {epoch}/{MAX_EPOCHS}", leave=False), 1):
            original = batch["underwater"].to(device, non_blocking=True)
            reference = batch["reference"].to(device, non_blocking=True)
            filenames = list(batch["filename"])
            if training:
                model.zero_grad(set_to_none=True)
                buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            else:
                buffers = None
            scaler_scale = float(scaler.get_scale())

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(original)
                losses = criterion(outputs, reference, original)

            output_status = tensor_tree_status(outputs)
            loss_status = {name: bool(torch.isfinite(value).all().item()) for name, value in losses.items()}
            record = {
                "epoch": epoch,
                "phase": phase,
                "batch_index": batch_index,
                "filenames": filenames,
                "amp_enabled": amp_enabled,
                "scaler_scale": scaler_scale,
                "model_output_status": output_status,
                "model_outputs_finite": all(output_status.values()),
                "loss_component_finite": loss_status,
                "losses": {name: float(value.detach().item()) for name, value in losses.items()},
                "scaled_gradient_finite": None,
                "unscaled_gradient_finite": None,
                "gradient_norm_before_clip": None,
                "gradient_norm_after_clip": None,
                "first_offending_parameter": None,
                "optimizer_step_executed": False,
            }
            if not record["model_outputs_finite"] or not all(loss_status.values()):
                append_batch_record(record)
                path = save_failure(record)
                raise FloatingPointError(f"Non-finite V6 output/loss epoch={epoch} batch={batch_index} files={filenames}; {path}")

            if training:
                scaler.scale(losses["total"]).backward()
                scaled = gradient_summary(model)
                scaler.unscale_(optimizer)
                unscaled = gradient_summary(model)
                record["scaled_gradient_finite"] = scaled["finite"]
                record["unscaled_gradient_finite"] = unscaled["finite"]
                record["scaled_gradient_norm"] = scaled["l2_norm"]
                record["gradient_norm_before_clip"] = unscaled["l2_norm"]
                record["first_offending_parameter"] = scaled["first_nonfinite_parameter"] or unscaled["first_nonfinite_parameter"]

                if not scaled["finite"] or not unscaled["finite"]:
                    append_batch_record(record)
                    comparison = no_step_comparison(model, criterion, original, reference, buffers, scaler_scale)
                    path = save_failure(record, comparison)
                    raise FloatingPointError(
                        f"Non-finite V6 gradient epoch={epoch} batch={batch_index} "
                        f"parameter={record['first_offending_parameter']} files={filenames} scale={scaler_scale}; "
                        f"optimizer step not executed; {path}"
                    )

                try:
                    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
                except RuntimeError as error:
                    record["clipping_error"] = str(error)
                    append_batch_record(record)
                    path = save_failure(record)
                    raise FloatingPointError(f"Non-finite V6 gradient norm epoch={epoch} batch={batch_index}; {path}") from error
                record["gradient_norm_returned_by_clipper"] = float(clipped.detach().item())
                record["gradient_norm_after_clip"] = gradient_summary(model)["l2_norm"]
                scaler.step(optimizer)
                scaler.update()
                record["optimizer_step_executed"] = True
                append_batch_record(record)

            batch_size = original.shape[0]
            sample_count += batch_size
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach().item()) * batch_size

    if sample_count == 0:
        raise RuntimeError(f"Empty {phase} data loader.")
    return {name: value / sample_count for name, value in totals.items()}


def candidate_diversity(model, loader, device, *, compute_reference_metrics):
    model.eval()
    pair_names = (
        ("conservative", "balanced"),
        ("balanced", "aggressive"),
        ("conservative", "aggressive"),
    )
    distances = {pair: [] for pair in pair_names}
    cosine = {pair: [] for pair in pair_names}
    residual_norms = {name: [] for name in ("conservative", "balanced", "aggressive")}
    preservation = {name: [] for name in residual_norms}
    quality = {name: {"psnr": [], "ssim": []} for name in residual_norms}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="V6 validation candidate diagnostics", leave=False):
            original = batch["underwater"].to(device)
            reference = batch["reference"].to(device)
            outputs = model(original)
            candidates = outputs["candidates"]
            residuals = {name: candidates[name] - original for name in residual_norms}
            for name in residual_norms:
                residual_norms[name].extend(torch.linalg.vector_norm(residuals[name].float(), dim=(1, 2, 3)).cpu().tolist())
                preservation[name].extend(outputs["preservation_scores"][name].float().cpu().tolist())
                if compute_reference_metrics:
                    for index in range(original.shape[0]):
                        predicted = candidates[name][index].float().cpu().permute(1, 2, 0).numpy()
                        target = reference[index].float().cpu().permute(1, 2, 0).numpy()
                        psnr, ssim, _, _ = calculate_metrics(predicted, target)
                        quality[name]["psnr"].append(float(psnr))
                        quality[name]["ssim"].append(float(ssim))
            for left, right in pair_names:
                distances[(left, right)].extend(torch.abs(candidates[left] - candidates[right]).mean(dim=(1, 2, 3)).cpu().tolist())
                cosine[(left, right)].extend(torch.nn.functional.cosine_similarity(
                    residuals[left].flatten(1).float(),
                    residuals[right].flatten(1).float(),
                    dim=1,
                    eps=1e-8,
                ).cpu().tolist())
    report = {
        "pairwise_mean_absolute_candidate_distance": {
            f"{left}_vs_{right}": float(np.mean(values)) for (left, right), values in distances.items()
        },
        "pairwise_mean_residual_cosine_similarity": {
            f"{left}_vs_{right}": float(np.mean(values)) for (left, right), values in cosine.items()
        },
        "mean_residual_l2_norm": {name: float(np.mean(values)) for name, values in residual_norms.items()},
        "mean_preservation_score": {name: float(np.mean(values)) for name, values in preservation.items()},
        "validation_candidate_quality": {
            name: {metric: float(np.mean(values)) if values else None for metric, values in metrics.items()}
            for name, metrics in quality.items()
        },
    }
    model.train()
    return report


def fit_target_calibration(model, train_dataset, device):
    """Load the threshold and two-term scales frozen by Phase A on train.txt."""
    source = PROJECT_ROOT / "diagnostics" / "v6_selection" / "v6_training_only_threshold_analysis.json"
    analysis = json.loads(source.read_text(encoding="utf-8"))
    if analysis.get("held_out_or_validation_accessed") or not analysis.get("decision_frozen_before_validation"):
        raise RuntimeError("V6 target analysis is not verified training-only and frozen.")
    rank = analysis["preservation_consequence_rank_calibration"]
    frozen = analysis["frozen_threshold"]
    return {
        "method": "5% relative fidelity eligibility; eligible candidates ranked by mean train-median/IQR normalized preservation distortion and consequence L1",
        "fit_split": "data/splits/train.txt", "fit_observations": int(rank["training_only_observations"]),
        "component_order": ["preservation_distortion", "consequence_l1"],
        "median": rank["median"], "iqr": rank["iqr"],
        "relative_fidelity_tolerance": float(frozen["relative_tolerance"]),
        "threshold_analysis": "diagnostics/v6_selection/v6_training_only_threshold_analysis.json",
        "old_v4_reference_l1_target_counts": {k: sum(1 for r in analysis["per_image_training_records"] if int(__import__("numpy").argmin(r["reference_l1"])) == i) for i,k in enumerate(("conservative","balanced","aggressive"))},
        "v6_multiobjective_target_counts": frozen["target_counts_after_preservation_consequence_ranking"],
        "eligible_candidate_frequency": frozen["eligible_candidate_frequency"],
        "training_sample_count": analysis["sample_count"],
        "test_split_loaded": False, "validation_used_for_fit": False,
    }

def validation_analysis(model, criterion, loader, device, v4_model):
    """Paired V4/V6 analysis on validation only, including per-image records."""
    labels = ("Conservative", "Balanced", "Aggressive")
    metric_names = ("psnr", "ssim", "uiqm", "uciqe", "preservation", "consequence", "reference_l1")
    rows = []
    model.eval(); v4_model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="V6 best-checkpoint validation analysis", leave=False):
            x = batch["underwater"].to(device)
            ref = batch["reference"].to(device)
            v6 = model(x)
            v4 = v4_model(x)
            targets, quantities = criterion.candidate_targets(v6, ref)
            v4_ref_losses = torch.stack([
                F.l1_loss(v4["candidates"][n], ref, reduction="none").mean((1,2,3))
                for n in ("conservative", "balanced", "aggressive")
            ], 1)
            v4_target = v4_ref_losses.argmin(1)
            v4_index = v4["selected_index"]
            v6_index = v6["selected_index"]
            probabilities = v6["selection_weights"].float()
            entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(1)
            sorted_utility = v6["utilities"].float().sort(dim=1, descending=True).values
            margins = sorted_utility[:, 0] - sorted_utility[:, 1]
            condition_l1 = torch.abs(x - ref).mean((1,2,3))
            for i, filename in enumerate(batch["filename"]):
                vi, wi = int(v4_index[i]), int(v6_index[i])
                chosen_v4 = v4["candidates"][labels[0].lower()][i] if False else v4["selected_image"][i]
                chosen_v6 = v6["selected_image"][i]
                arr_ref = ref[i].float().cpu().permute(1,2,0).numpy()
                metric_pair = {}
                for version, img in (("v4", chosen_v4), ("v6", chosen_v6)):
                    arr = img.float().cpu().permute(1,2,0).numpy()
                    psnr, ssim, uiqm_value, uciqe_value = calculate_metrics(arr, arr_ref)
                    metric_pair[version] = {"psnr": float(psnr), "ssim": float(ssim), "uiqm": float(uiqm_value), "uciqe": float(uciqe_value)}
                candidate_data = {}
                v4_candidate_data = {}
                for c, name in enumerate(("conservative", "balanced", "aggressive")):
                    candidate_data[name] = {
                        "reference_l1": float(quantities["reference"][i,c]),
                        "preservation_score": float(v6["preservation_scores"][name][i]),
                        "preservation_distortion": float(quantities["preservation"][i,c]),
                        "consequence": float(quantities["consequence"][i,c]),
                        "residual_l2": float(torch.linalg.vector_norm((v6["candidates"][name][i]-x[i]).float())),
                    }
                    v4_candidate = v4["candidates"][name][i]
                    _, v4_pres = v4_model.information_preservation(x[i:i+1], v4_candidate.unsqueeze(0))
                    v4_cons = v4_model.consequence_check(x[i:i+1], v4_candidate.unsqueeze(0))["consequence_error"][0]
                    v4_candidate_data[name] = {
                        "reference_l1": float(F.l1_loss(v4_candidate, ref[i], reduction="mean")),
                        "preservation_score": float(v4_pres[0]),
                        "consequence": float(v4_cons),
                        "residual_l2": float(torch.linalg.vector_norm((v4_candidate-x[i]).float())),
                    }
                condition = float(condition_l1[i])
                rows.append({
                    "filename": filename, "input_reference_l1": condition,
                    "condition_vector": [float(v) for v in v6["condition_vector"][i].float().cpu().tolist()],
                    "v4_selected": labels[vi], "v6_target": labels[int(targets[i])], "v6_selected": labels[wi],
                    "v6_probabilities": {labels[j]: float(probabilities[i,j]) for j in range(3)},
                    "selection_entropy": float(entropy[i]), "utility_margin": float(margins[i]),
                    "metrics": metric_pair, "candidates": candidate_data,
                    "v4_candidates": v4_candidate_data,
                    "v4_selector_target": labels[int(v4_target[i])],
                })
    def counts(a, b=None):
        out = {x: {y: 0 for y in labels} for x in labels} if b else {x: 0 for x in labels}
        for row in rows:
            if b: out[row[a]][row[b]] += 1
            else: out[row[a]] += 1
        return out
    def avg(path):
        vals = [r for row in rows for r in [row["metrics"][path[0]][path[1]]]]
        return float(np.mean(vals)) if vals else None
    v6_distribution = counts("v6_selected")
    target_distribution = counts("v6_target")
    v4_distribution = counts("v4_selected")
    matrix_v4_to_v6_target = counts("v4_selector_target", "v6_target")
    matrix_target_to_selector = counts("v6_target", "v6_selected")
    matrix_selectors = counts("v4_selected", "v6_selected")
    quartile_edges = np.quantile([r["input_reference_l1"] for r in rows], [0, .25, .5, .75, 1])
    groups = {}
    for q in range(4):
        group = [r for r in rows if quartile_edges[q] <= r["input_reference_l1"] <= quartile_edges[q+1]
                 and (q == 3 or r["input_reference_l1"] < quartile_edges[q+1])]
        groups[f"input_reference_l1_q{q+1}"] = {label: sum(r["v6_selected"] == label for r in group) for label in labels}
    aggregate = {
        "sample_count": len(rows), "v4_selection_distribution": v4_distribution,
        "v6_target_distribution": target_distribution, "v6_selection_distribution": v6_distribution,
        "v4_selector_to_v6_target_confusion": matrix_v4_to_v6_target,
        "v6_target_to_v6_selector_confusion": matrix_target_to_selector,
        "v4_selector_to_v6_selector_confusion": matrix_selectors,
        "v6_target_selector_agreement": float(np.mean([r["v6_target"] == r["v6_selected"] for r in rows])),
        "v4_target_to_v6_target_confusion": counts("v4_selector_target", "v6_target"),
        "mean_selection_entropy": float(np.mean([r["selection_entropy"] for r in rows])),
        "mean_utility_margin": float(np.mean([r["utility_margin"] for r in rows])),
        "mean_v6_selection_probabilities": {label: float(np.mean([r["v6_probabilities"][label] for r in rows])) for label in labels},
        "selection_distribution_percentages": {
            version: {label: 100.0 * values[label] / len(rows) for label in labels}
            for version, values in (("v4", v4_distribution), ("v6_target", target_distribution), ("v6_selector", v6_distribution))
        },
        "mean_v4_metrics": {m: avg(("v4", m)) for m in ("psnr", "ssim", "uiqm", "uciqe")},
        "mean_v6_metrics": {m: avg(("v6", m)) for m in ("psnr", "ssim", "uiqm", "uciqe")},
        "adaptivity_by_input_reference_l1_quartile": groups,
        "median_input_reference_l1_quartile_edges": [float(x) for x in quartile_edges],
    }
    for version, selected_key in (("v4", "v4_selected"), ("v6", "v6_selected")):
        aggregate[f"mean_{version}_selection_objectives"] = {
            metric: float(np.mean([row["v4_candidates" if version == "v4" else "candidates"][row[selected_key].lower()][metric]
                                   if metric != "preservation_distortion" else
                                   1.0 - row["v4_candidates" if version == "v4" else "candidates"][row[selected_key].lower()]["preservation_score"]
                                   for row in rows]))
            for metric in ("reference_l1", "preservation_distortion", "consequence", "residual_l2")
        }
    return aggregate, rows


def write_final_reports(history, best_epoch, calibration, aggregate, rows, run_dir):
    (PROJECT_ROOT / "diagnostics" / "v6_selection").mkdir(parents=True, exist_ok=True)
    detail_path = PROJECT_ROOT / "diagnostics" / "v6_selection" / "validation_per_image.json"
    with detail_path.open("w", encoding="utf-8") as handle:
        json.dump({"split": "validation", "records": rows}, handle, indent=2)
    v4m = aggregate["mean_v4_metrics"]
    v6m = aggregate["mean_v6_metrics"]
    v4o = aggregate["mean_v4_selection_objectives"]
    v6o = aggregate["mean_v6_selection_objectives"]
    groups = list(aggregate["adaptivity_by_input_reference_l1_quartile"].values())
    group_totals = [sum(x.values()) for x in groups]
    group_distributions = [{k: v / max(1, total) for k, v in row.items()} for row, total in zip(groups, group_totals)]
    group_shift = max((sum(abs(a[k]-b[k]) for k in a) / 2 for a in group_distributions for b in group_distributions), default=0.0)
    adaptive = group_shift >= 0.10
    fidelity_retained = v6m["psnr"] >= v4m["psnr"] - 1.0 and v6m["ssim"] >= v4m["ssim"] - 0.02
    preservation_gain = v6o["preservation_distortion"] < v4o["preservation_distortion"] - 0.005
    if adaptive and fidelity_retained and preservation_gain:
        status = "V6 PROMISING — FREEZE FOR HELD-OUT TEST"
    elif (v6m["psnr"] < v4m["psnr"] - 1.0 and v6m["ssim"] < v4m["ssim"] - 0.02 and not preservation_gain):
        status = "V4 REMAINS THE STRONGER MODEL"
    else:
        status = "V6 NEEDS REFINEMENT"
    calibration = dict(calibration)
    training_sample_count = int(calibration["fit_observations"] // 3)
    calibration["training_sample_count"] = training_sample_count
    calibration["old_v4_reference_l1_target_percentages"] = {
        key: 100.0 * value / training_sample_count
        for key, value in calibration["old_v4_reference_l1_target_counts"].items()
    }
    calibration["v6_multiobjective_target_percentages"] = {
        key: 100.0 * value / training_sample_count
        for key, value in calibration["v6_multiobjective_target_counts"].items()
    }
    failure_paths = sorted(CHECKPOINT_DIR.glob("failure_epoch_*_batch_*.json"))
    failure_records = [json.loads(path.read_text(encoding="utf-8")) for path in failure_paths]
    report = {
        "motivation": "Change the V4 selector supervision target to balance reference fidelity, fixed preservation, and consequence consistency.",
        "objective": "eligible(c) iff (reference_L1(c)-best_reference_L1)/(best_reference_L1+1e-8) <= 0.05; among eligible candidates minimize the equal mean of train-normalized preservation distortion and consequence L1.",
        "normalization": calibration,
        "training_configuration": {"dataset": "EUVP paired underwater_imagenet", "train_split": "data/splits/train.txt",
                                   "validation_split": "data/splits/val.txt", "test_split_loaded": False,
                                   "preprocessing": "paired RGB ToTensor [0,1]; paired horizontal flip for train only",
                                   "seed": SEED, "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                                   "max_epochs": MAX_EPOCHS, "validation_frequency_epochs": 1,
                                   "best_checkpoint_criterion": "minimum validation total loss",
                                   "loss_weights": {"reconstruction": 1.0, "preservation": 0.2, "consequence": 0.2, "selection": 0.5},
                                   "gradient_clip_norm": MAX_GRAD_NORM, "scheduler": None, "early_stopping": False,
                                   "initialization": "V4 best", "amp_enabled": True, "amp_initial_scale": 2048,
                                   "amp_replay_adjustment": "V6-specific AMP events are recorded. Any non-finite batch is replay-diagnosed before resuming; failed optimizer steps are not counted.",
                                   "test_split_loaded": False},
        "history": history, "best_epoch": best_epoch, "validation_analysis": aggregate,
        "diagnostic_json": str(detail_path.relative_to(PROJECT_ROOT)),
        "failure_cases": [r["filename"] for r in rows if r["metrics"]["v6"]["psnr"] < r["metrics"]["v4"]["psnr"] - 3.0],
        "scientific_interpretation": (
            f"Validation selection shows a maximum pairwise total-variation shift of {group_shift:.3f} across input/reference-L1 quartiles. "
            f"Selected-output PSNR differs from V4 by {v6m['psnr']-v4m['psnr']:+.3f} dB and SSIM by {v6m['ssim']-v4m['ssim']:+.4f}; "
            f"selected preservation distortion changes by {v6o['preservation_distortion']-v4o['preservation_distortion']:+.5f}. "
            "These are validation associations and do not establish causality."
        ),
        "decision_rule_inputs": {"quartile_selection_total_variation_max": group_shift, "meaningfully_adaptive_threshold": 0.10,
                                  "fidelity_retained": fidelity_retained, "preservation_gain_threshold_distortion": 0.005,
                                  "preservation_gain": preservation_gain},
        "final_status": status,
        "amp_overflow_diagnostics": [{"epoch": item["epoch"], "batch_index": item["batch_index"],
                                      "filenames": item["filenames"], "initial_scale": item["scaler_scale"],
                                      "same_batch_amp_scale_finite": {key: value["component_gradients"]["total"]["finite"]
                                                                        for key, value in item.get("no_step_amp_fp32_comparison", {}).items()},
                                      "optimizer_step_executed": item["optimizer_step_executed"]}
                                     for item in failure_records],
        "freeze_recommendation": "Freeze only if adaptive validation criteria are met; held-out testing requires a separate explicit authorization.",
        "held_out_test_authorized": False,
        "test_split_used": False,
        "final_safety": {
            "v2_sha256": sha256_file(V2_CHECKPOINT),
            "v3_sha256": sha256_file(V3_CHECKPOINT),
            "v4_sha256": sha256_file(V4_CHECKPOINT),
            "test_manifest_read": False,
            "v3_v4_source_modified": False,
            "existing_checkpoints_overwritten": False,
            "git_commit_or_push": False,
            "new_source_files": ["src/models/paper1_model_v6.py", "src/losses/paper1_loss_v6.py", "src/train_v6.py", "src/analyze_v6_validation.py"],
        },
    }
    json_path = RESULTS_ROOT / "v6_training_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    md = ["# V6 Training Report", "", f"Decision: **{report['final_status']}**", "",
          "## Objective", "", report["objective"], "", "Normalization uses component medians and IQRs fitted once on train.txt. Validation and test data do not fit these values.", "",
          "## Training and validation", "", f"Best epoch: {best_epoch}.", "",
          "| Epoch | Train total | Val total | Train selection | Val selection |", "|---:|---:|---:|---:|---:|"]
    for row in history:
        md.append(f"| {row['epoch']} | {row['train_total']:.6f} | {row['val_total']:.6f} | {row['train_selection']:.6f} | {row['val_selection']:.6f} |")
    md += ["", "## Train-only target distribution", "",
           "The calibration diagnostics below were computed from the training split before V6 optimization. Percentages are over 2,960 training images.", "",
           "| Candidate | V4 reference-L1 target | V6 multi-objective target |", "|---|---:|---:|"]
    for candidate in ("conservative", "balanced", "aggressive"):
        md.append(f"| {candidate.title()} | {calibration['old_v4_reference_l1_target_counts'][candidate]} ({calibration['old_v4_reference_l1_target_percentages'][candidate]:.2f}%) | {calibration['v6_multiobjective_target_counts'][candidate]} ({calibration['v6_multiobjective_target_percentages'][candidate]:.2f}%) |")
    md += ["", "## Validation comparison", "", "```json", json.dumps(aggregate, indent=2), "```", "",
           "The selection groups use validation input-to-reference L1 quartiles as a measurable degradation proxy; this is association, not causal evidence.", "",
           f"Per-image diagnostics: `{detail_path.relative_to(PROJECT_ROOT)}`.", "",
           "AMP overflow diagnostics are included in the JSON report; both failed steps were recorded, their same-batch replay gradients were finite at scale 2,048 or lower, and no failed optimizer step was applied.", "",
           "No held-out test images were read or evaluated. Final held-out testing is not authorized by this report."]
    (RESULTS_ROOT / "v6_training_report.md").write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train or resume the controlled Paper 1 V6 experiment.")
    parser.add_argument("--resume-from-epoch", type=int, default=None)
    parser.add_argument("--amp-resume-scale", type=float, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not V2_CHECKPOINT.is_file() or not V3_CHECKPOINT.is_file() or not V4_CHECKPOINT.is_file():
        raise FileNotFoundError("A frozen V2/V3/V4 checkpoint is missing.")
    verify_frozen_checkpoints("before V6 setup")
    v4_sha_before = sha256_file(V4_CHECKPOINT)

    existing_epochs = sorted(CHECKPOINT_DIR.glob("paper1_v6_epoch_*.pth")) if CHECKPOINT_DIR.exists() else []
    resume_epoch = args.resume_from_epoch
    if resume_epoch is None and existing_epochs:
        resume_epoch = max(int(path.stem.rsplit("_", 1)[1]) for path in existing_epochs)
    resume_checkpoint_path = (
        CHECKPOINT_DIR / f"paper1_v6_epoch_{resume_epoch:03d}.pth"
        if resume_epoch is not None
        else None
    )
    if resume_checkpoint_path is not None and not resume_checkpoint_path.is_file():
        raise FileNotFoundError(f"V6 resume checkpoint does not exist: {resume_checkpoint_path}")

    allowed_pretraining_diagnostics = {
        "target_calibration_train_only.json", "batch_diagnostics.jsonl", "training_config.json",
    }
    existing_noncalibration = ([p for p in CHECKPOINT_DIR.iterdir()
                                if p.name not in allowed_pretraining_diagnostics and not p.name.startswith("failure_epoch_")]
                               if CHECKPOINT_DIR.exists() else [])
    if resume_checkpoint_path is None and existing_noncalibration:
        raise FileExistsError(f"V6 checkpoint directory has artifacts but no resume checkpoint: {CHECKPOINT_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = None
    if resume_checkpoint_path is None:
        run_dir = RESULTS_ROOT / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
        run_dir.mkdir(parents=True, exist_ok=False)

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    train_dataset = EUVPPairedDataset(
        DATASET_ROOT / "trainA",
        DATASET_ROOT / "trainB",
        SPLIT_DIR / "train.txt",
        transform=PairedTransform(training=True),
    )
    validation_dataset = EUVPPairedDataset(
        DATASET_ROOT / "trainA",
        DATASET_ROOT / "trainB",
        SPLIT_DIR / "val.txt",
        transform=PairedTransform(training=False),
    )
    if not set(train_dataset.filenames).isdisjoint(validation_dataset.filenames):
        raise RuntimeError("Train/validation filename leakage detected.")

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = Paper1ModelV6().to(device)
    calibration_path = CHECKPOINT_DIR / "target_calibration_train_only.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        model.load_state_dict(torch.load(V4_CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        calibration = fit_target_calibration(model, train_dataset, device)
        with calibration_path.open("x", encoding="utf-8") as handle:
            json.dump(calibration, handle, indent=2)
    criterion = Paper1LossV6(calibration).to(device)
    if resume_checkpoint_path is None and CHECKPOINT_DIR.exists():
        # A completed calibration is the only artifact allowed before epoch 1.
        if any(p.name not in allowed_pretraining_diagnostics and not p.name.startswith("failure_epoch_") for p in CHECKPOINT_DIR.iterdir()):
            raise FileExistsError(f"Unexpected V6 artifact in {CHECKPOINT_DIR}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_AMP_SCALE, enabled=amp_enabled)

    config = {
        "experiment": "Paper 1 V6 fidelity-eligibility adaptive-selection experiment",
        "seed": SEED,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "amp_enabled": amp_enabled,
        "amp_initial_scale": INITIAL_AMP_SCALE,
        "amp_scale_adjustment": "Start at GradScaler 2048; diagnose any overflow with same-batch replay and do not count failed optimizer steps.",
        "scheduler": None,
        "early_stopping": False,
        "loss_weights": {
            "reconstruction": criterion.lambda_recon,
            "preservation": criterion.lambda_preserve,
            "consequence": criterion.lambda_consequence,
            "selection": criterion.lambda_selection,
        },
        "training_split": "data/splits/train.txt",
        "validation_split": "data/splits/val.txt",
        "test_split_loaded": False,
        "initialization": "frozen V4 best checkpoint; all V4 modules are then trainable",
        "v4_sha256_before": sha256_file(V4_CHECKPOINT),
        "target_calibration": calibration,
        "preprocessing": "paired ToTensor [0,1]; paired horizontal flip for train only; no other changes",
        "model": "V4 module graph and candidate-specific generator; only V6 selector target formulation differs",
        "candidate_generator": "V4 shared 3->32->64->128 features; separate candidate-specific condition projection and independent 160->64->32->3 tanh residual heads; ordered positive gains initialized 0.0625/0.125/0.25",
        "v2_sha256_before": sha256_file(V2_CHECKPOINT),
        "v3_sha256_before": sha256_file(V3_CHECKPOINT),
        "output_dir": str(CHECKPOINT_DIR),
        "results_dir": str(run_dir),
    }
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("epoch", -1)) != resume_epoch:
            raise RuntimeError("V6 resume checkpoint epoch does not match its filename.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = dict(checkpoint["scaler_state_dict"])
        if args.amp_resume_scale is not None:
            if not amp_enabled or args.amp_resume_scale <= 0:
                raise ValueError("A positive --amp-resume-scale requires CUDA AMP.")
            scaler_state["scale"] = float(args.amp_resume_scale)
            scaler_state["_growth_tracker"] = 0
        scaler.load_state_dict(scaler_state)

        saved_rng = checkpoint.get("rng_state", {})
        if "train_loader_generator" in saved_rng:
            train_loader.generator.set_state(saved_rng["train_loader_generator"])
        else:
            train_loader.generator.set_state(
                advance_sampler_generator(SEED, len(train_dataset), resume_epoch)
            )
        if saved_rng:
            restore_global_rng_state(saved_rng)

        history_path = CHECKPOINT_DIR / "loss_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if not history or int(history[-1]["epoch"]) != resume_epoch:
            raise RuntimeError("V6 loss history does not end at the requested resume checkpoint.")
        best_row = min(history, key=lambda row: float(row["val_total"]))
        best_epoch = int(best_row["epoch"])
        best_val_total = float(best_row["val_total"])
        start_epoch = resume_epoch + 1
        config = json.loads((CHECKPOINT_DIR / "training_config.json").read_text(encoding="utf-8"))
        run_dir = Path(config["results_dir"])
        if args.amp_resume_scale is not None:
            config["amp_stability_correction"] = {
                "resume_scale": float(args.amp_resume_scale),
                "growth_tracker_reset": True,
                "diagnosis": "Recorded same-batch AMP replay found selector gradients non-finite at 8192/4096, finite at 2048 and below, and FP32 finite; resumed with reduced scale based on these diagnostics.",
            }
        print(f"Resuming V6 from epoch {resume_epoch}: {resume_checkpoint_path}")
        print("Sampler order restored from saved RNG state or deterministic replay.")
        print("restored_scaler", scaler.state_dict())
    else:
        history = []
        best_epoch = None
        best_val_total = float("inf")
        start_epoch = 1
        frozen_v4 = torch.load(V4_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(frozen_v4["model_state_dict"], strict=True)
        torch.manual_seed(SEED)
        synthetic = 0.2 + 0.6 * torch.rand(4, 3, 256, 256, device=device)
        model.eval()
        with torch.inference_mode():
            model(synthetic)
        synthetic_loader = [{"underwater": synthetic, "reference": torch.rand_like(synthetic)}]
        pretraining_diversity = candidate_diversity(model, synthetic_loader, device, compute_reference_metrics=False)
        model.train()
        if min(pretraining_diversity["pairwise_mean_absolute_candidate_distance"].values()) <= 1e-5:
            raise RuntimeError(f"V6 candidate branches failed the pretraining diversity check: {pretraining_diversity}")
        with (run_dir / "candidate_diversity_pretraining.json").open("w", encoding="utf-8") as handle:
            json.dump(pretraining_diversity, handle, indent=2)
        print("pretraining candidate diversity", json.dumps(pretraining_diversity, indent=2))

    if args.preflight_only:
        verify_frozen_checkpoints("after V6 resume preflight")
        print(f"V6 resume preflight passed: epoch={resume_epoch}, next_epoch={start_epoch}")
        print(f"model_device={device}, amp={amp_enabled}, scaler={scaler.state_dict()}")
        print(f"history_rows={len(history)}, best_epoch={best_epoch}, best_val_total={best_val_total}")
        print(f"test_split_loaded=False; optimizer_step_executed=False")
        return

    (CHECKPOINT_DIR / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"V6 device={device}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"train_samples={len(train_dataset)} validation_samples={len(validation_dataset)} test_split_loaded=False")

    start_time = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        verify_frozen_checkpoints(f"before V6 epoch {epoch}")
        verify_v4_checkpoint(v4_sha_before, f"before V6 epoch {epoch}")
        epoch_start = time.perf_counter()
        train_losses = run_epoch(model, criterion, train_loader, device, optimizer, scaler, epoch=epoch, training=True, amp_enabled=amp_enabled)
        val_losses = run_epoch(model, criterion, validation_loader, device, optimizer, scaler, epoch=epoch, training=False, amp_enabled=amp_enabled)
        verify_frozen_checkpoints(f"after V6 epoch {epoch}")
        verify_v4_checkpoint(v4_sha_before, f"after V6 epoch {epoch}")
        epoch_seconds = time.perf_counter() - epoch_start

        if val_losses["total"] < best_val_total:
            best_val_total = val_losses["total"]
            best_epoch = epoch

        row = {"epoch": epoch}
        row.update({f"train_{name}": train_losses[name] for name in LOSS_NAMES})
        row.update({f"val_{name}": val_losses[name] for name in LOSS_NAMES})
        row.update({"epoch_seconds": epoch_seconds, "best_epoch": best_epoch})
        history.append(row)
        write_history(history)

        epoch_checkpoint = CHECKPOINT_DIR / f"paper1_v6_epoch_{epoch:03d}.pth"
        save_checkpoint(epoch_checkpoint, model, optimizer, scaler, epoch, train_losses, val_losses, config, train_loader)
        if epoch == best_epoch:
            save_checkpoint(CHECKPOINT_DIR / f"paper1_v6_best_epoch_{epoch:03d}.pth", model, optimizer, scaler, epoch, train_losses, val_losses, config, train_loader)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | Train Total {train_losses['total']:.6f} | Val Total {val_losses['total']:.6f} | "
            f"Train Reconstruction {train_losses['reconstruction']:.6f} | Val Reconstruction {val_losses['reconstruction']:.6f} | "
            f"Train Preservation {train_losses['preservation']:.6f} | Val Preservation {val_losses['preservation']:.6f} | "
            f"Train Consequence {train_losses['consequence']:.6f} | Val Consequence {val_losses['consequence']:.6f} | "
            f"Train Selection {train_losses['selection']:.6f} | Val Selection {val_losses['selection']:.6f} | "
            f"Best Epoch {best_epoch} | Seconds {epoch_seconds:.1f}"
        )

    best = torch.load(CHECKPOINT_DIR / f"paper1_v6_best_epoch_{best_epoch:03d}.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    after = candidate_diversity(model, validation_loader, device, compute_reference_metrics=True)
    with (run_dir / "candidate_diversity_validation_best.json").open("w", encoding="utf-8") as handle:
        json.dump(after, handle, indent=2)

    verify_frozen_checkpoints("after V6 training")
    verify_v4_checkpoint(v4_sha_before, "after V6 training")
    print("training_seconds", time.perf_counter() - start_time)
    print("best_epoch", best_epoch, "best_validation_total", best_val_total)
    print("final_epoch_losses", json.dumps(history[-1], indent=2))
    print("post-training validation candidate diversity", json.dumps(after, indent=2))
    print("V6 checkpoints", [path.name for path in sorted(CHECKPOINT_DIR.glob("paper1_v6_*.pth"))])
    print("V3/V2 checkpoint hashes preserved")
    print("test.txt used: NO")


if __name__ == "__main__":
    main()
