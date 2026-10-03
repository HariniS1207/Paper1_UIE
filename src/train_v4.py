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
from models.paper1_model_v4 import Paper1ModelV4
from paper1_loss import Paper1Loss
from transforms import PairedTransform


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20
INITIAL_AMP_SCALE = 8192.0
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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "v4"
RESULTS_ROOT = PROJECT_ROOT / "results" / "v4"


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
        for batch_index, batch in enumerate(tqdm(loader, desc=f"V4 {phase} {epoch}/{MAX_EPOCHS}", leave=False), 1):
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
                raise FloatingPointError(f"Non-finite V4 output/loss epoch={epoch} batch={batch_index} files={filenames}; {path}")

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
                        f"Non-finite V4 gradient epoch={epoch} batch={batch_index} "
                        f"parameter={record['first_offending_parameter']} files={filenames} scale={scaler_scale}; "
                        f"optimizer step not executed; {path}"
                    )

                try:
                    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
                except RuntimeError as error:
                    record["clipping_error"] = str(error)
                    append_batch_record(record)
                    path = save_failure(record)
                    raise FloatingPointError(f"Non-finite V4 gradient norm epoch={epoch} batch={batch_index}; {path}") from error
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
        for batch in tqdm(loader, desc="V4 validation candidate diagnostics", leave=False):
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


def main():
    parser = argparse.ArgumentParser(description="Train or resume the controlled Paper 1 V4 experiment.")
    parser.add_argument("--resume-from-epoch", type=int, default=None)
    parser.add_argument("--amp-resume-scale", type=float, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not V2_CHECKPOINT.is_file() or not V3_CHECKPOINT.is_file():
        raise FileNotFoundError("A frozen V2/V3 checkpoint is missing.")
    verify_frozen_checkpoints("before V4 setup")

    existing_epochs = sorted(CHECKPOINT_DIR.glob("paper1_v4_epoch_*.pth")) if CHECKPOINT_DIR.exists() else []
    resume_epoch = args.resume_from_epoch
    if resume_epoch is None and existing_epochs:
        resume_epoch = max(int(path.stem.rsplit("_", 1)[1]) for path in existing_epochs)
    if resume_epoch is None and CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
        raise FileExistsError(f"V4 checkpoint directory has artifacts but no resume checkpoint: {CHECKPOINT_DIR}")
    resume_checkpoint_path = (
        CHECKPOINT_DIR / f"paper1_v4_epoch_{resume_epoch:03d}.pth"
        if resume_epoch is not None
        else None
    )
    if resume_checkpoint_path is not None and not resume_checkpoint_path.is_file():
        raise FileNotFoundError(f"V4 resume checkpoint does not exist: {resume_checkpoint_path}")

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

    model = Paper1ModelV4().to(device)
    criterion = Paper1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_AMP_SCALE, enabled=amp_enabled)

    config = {
        "experiment": "Paper 1 V4 candidate-specific generator controlled experiment",
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
        "preprocessing": "paired ToTensor [0,1]; paired horizontal flip for train only; no other changes",
        "model": "V3 condition encoder + V4 candidate-specific generator + V3 fixed preservation + V3 selector + V3 consequence check",
        "candidate_generator": "shared 3->32->64->128 features; separate candidate-specific condition projection and independent 160->64->32->3 convolutional tanh residual head for each candidate; positive learnable ordered gains initialized 0.0625/0.125/0.25",
        "v2_sha256_before": sha256_file(V2_CHECKPOINT),
        "v3_sha256_before": sha256_file(V3_CHECKPOINT),
        "output_dir": str(CHECKPOINT_DIR),
        "results_dir": str(run_dir),
    }
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("epoch", -1)) != resume_epoch:
            raise RuntimeError("V4 resume checkpoint epoch does not match its filename.")
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
            raise RuntimeError("V4 loss history does not end at the requested resume checkpoint.")
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
                "diagnosis": "V4 selection-loss backward overflowed in AMP on a recorded batch while FP32 gradients were finite; AMP resume scale was reduced based on same-batch no-step diagnostics.",
            }
        print(f"Resuming V4 from epoch {resume_epoch}: {resume_checkpoint_path}")
        print("Sampler order restored from saved RNG state or deterministic replay.")
        print("restored_scaler", scaler.state_dict())
    else:
        history = []
        best_epoch = None
        best_val_total = float("inf")
        start_epoch = 1
        torch.manual_seed(SEED)
        synthetic = 0.2 + 0.6 * torch.rand(4, 3, 256, 256, device=device)
        model.eval()
        with torch.inference_mode():
            model(synthetic)
        synthetic_loader = [{"underwater": synthetic, "reference": torch.rand_like(synthetic)}]
        pretraining_diversity = candidate_diversity(model, synthetic_loader, device, compute_reference_metrics=False)
        model.train()
        if min(pretraining_diversity["pairwise_mean_absolute_candidate_distance"].values()) <= 1e-5:
            raise RuntimeError(f"V4 candidate branches failed the pretraining diversity check: {pretraining_diversity}")
        with (run_dir / "candidate_diversity_pretraining.json").open("w", encoding="utf-8") as handle:
            json.dump(pretraining_diversity, handle, indent=2)
        print("pretraining candidate diversity", json.dumps(pretraining_diversity, indent=2))

    if args.preflight_only:
        verify_frozen_checkpoints("after V4 resume preflight")
        print(f"V4 resume preflight passed: epoch={resume_epoch}, next_epoch={start_epoch}")
        print(f"model_device={device}, amp={amp_enabled}, scaler={scaler.state_dict()}")
        print(f"history_rows={len(history)}, best_epoch={best_epoch}, best_val_total={best_val_total}")
        print(f"test_split_loaded=False; optimizer_step_executed=False")
        return

    (CHECKPOINT_DIR / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"V4 device={device}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"train_samples={len(train_dataset)} validation_samples={len(validation_dataset)} test_split_loaded=False")

    start_time = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        verify_frozen_checkpoints(f"before V4 epoch {epoch}")
        epoch_start = time.perf_counter()
        train_losses = run_epoch(model, criterion, train_loader, device, optimizer, scaler, epoch=epoch, training=True, amp_enabled=amp_enabled)
        val_losses = run_epoch(model, criterion, validation_loader, device, optimizer, scaler, epoch=epoch, training=False, amp_enabled=amp_enabled)
        verify_frozen_checkpoints(f"after V4 epoch {epoch}")
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

        epoch_checkpoint = CHECKPOINT_DIR / f"paper1_v4_epoch_{epoch:03d}.pth"
        save_checkpoint(epoch_checkpoint, model, optimizer, scaler, epoch, train_losses, val_losses, config, train_loader)
        if epoch == best_epoch:
            save_checkpoint(CHECKPOINT_DIR / "paper1_v4_best.pth", model, optimizer, scaler, epoch, train_losses, val_losses, config, train_loader)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | Train Total {train_losses['total']:.6f} | Val Total {val_losses['total']:.6f} | "
            f"Train Reconstruction {train_losses['reconstruction']:.6f} | Val Reconstruction {val_losses['reconstruction']:.6f} | "
            f"Train Preservation {train_losses['preservation']:.6f} | Val Preservation {val_losses['preservation']:.6f} | "
            f"Train Consequence {train_losses['consequence']:.6f} | Val Consequence {val_losses['consequence']:.6f} | "
            f"Train Selection {train_losses['selection']:.6f} | Val Selection {val_losses['selection']:.6f} | "
            f"Best Epoch {best_epoch} | Seconds {epoch_seconds:.1f}"
        )

    best = torch.load(CHECKPOINT_DIR / "paper1_v4_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    after = candidate_diversity(model, validation_loader, device, compute_reference_metrics=True)
    with (run_dir / "candidate_diversity_validation_best.json").open("w", encoding="utf-8") as handle:
        json.dump(after, handle, indent=2)

    verify_frozen_checkpoints("after V4 training")
    print("training_seconds", time.perf_counter() - start_time)
    print("best_epoch", best_epoch, "best_validation_total", best_val_total)
    print("final_epoch_losses", json.dumps(history[-1], indent=2))
    print("post-training validation candidate diversity", json.dumps(after, indent=2))
    print("V4 checkpoints", [path.name for path in sorted(CHECKPOINT_DIR.glob("paper1_v4_*.pth"))])
    print("V3/V2 checkpoint hashes preserved")
    print("test.txt used: NO")


if __name__ == "__main__":
    main()
