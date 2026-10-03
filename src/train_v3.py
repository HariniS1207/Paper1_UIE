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
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "losses"))

from euvp_dataset import EUVPPairedDataset
from paper1_loss import Paper1Loss
from paper1_model import Paper1Model
from transforms import PairedTransform


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 8
MAX_GRAD_NORM = 1.0
EXPECTED_V2_SHA256 = "BBC8027FF23DCD477CD2E4BD0C95D7A42222AAAAE2E6A6197AF3D198149C53D1"

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
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "v3"
LOSS_NAMES = ("total", "reconstruction", "preservation", "consequence", "selection")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_v2_checkpoint(expected_hash: str, context: str) -> str:
    current_hash = sha256_file(V2_CHECKPOINT)
    if current_hash != expected_hash:
        raise RuntimeError(
            f"Frozen V2 checkpoint SHA-256 changed {context}: "
            f"expected {expected_hash}, found {current_hash}. Stopping."
        )
    return current_hash


def check_finite_losses(losses, epoch: int, phase: str) -> None:
    for name in LOSS_NAMES:
        value = losses[name]
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(
                f"NaN/Inf detected at epoch {epoch}, phase={phase}, component={name}. Stopping."
            )


def check_finite_gradients(model: torch.nn.Module, epoch: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(
                f"NaN/Inf detected at epoch {epoch}, phase=train, component=gradient:{name}. Stopping."
            )


def tensor_tree_status(value, prefix=""):
    status = {}
    if torch.is_tensor(value):
        status[prefix or "tensor"] = bool(torch.isfinite(value).all().item())
    elif isinstance(value, dict):
        for name, item in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            status.update(tensor_tree_status(item, path))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else str(index)
            status.update(tensor_tree_status(item, path))
    return status


def gradient_summary(named_parameters):
    total_squared_norm = 0.0
    candidate_squared_norm = 0.0
    first_nonfinite = None
    gradients_present = 0
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        gradients_present += 1
        if not torch.isfinite(gradient).all().item():
            if first_nonfinite is None:
                first_nonfinite = name
            continue
        squared_norm = float(gradient.detach().double().pow(2).sum().item())
        total_squared_norm += squared_norm
        if name.startswith("candidate_generator."):
            candidate_squared_norm += squared_norm
    return {
        "finite": first_nonfinite is None,
        "first_nonfinite_parameter": first_nonfinite,
        "gradients_present": gradients_present,
        "l2_norm": total_squared_norm ** 0.5,
        "candidate_generator_l2_norm": candidate_squared_norm ** 0.5,
    }


def append_batch_record(record) -> None:
    path = OUTPUT_DIR / "batch_diagnostics.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        handle.flush()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def save_failure_record(record, comparison=None) -> Path:
    payload = dict(record)
    payload["no_step_amp_fp32_comparison"] = comparison
    base_path = OUTPUT_DIR / f"failure_epoch_{record['epoch']:03d}_batch_{record['batch_index']:04d}.json"
    path = base_path
    suffix = 1
    while path.exists():
        path = base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")
        suffix += 1
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
    return path


def restore_buffers(model, snapshot) -> None:
    current_buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in snapshot.items():
            current_buffers[name].copy_(value)


def component_gradient_diagnostics(model, losses, scale):
    parameters = list(model.named_parameters())
    outputs = {}
    names = list(LOSS_NAMES)
    for index, loss_name in enumerate(names):
        gradients = torch.autograd.grad(
            losses[loss_name] * scale,
            [parameter for _, parameter in parameters],
            retain_graph=index < len(names) - 1,
            allow_unused=True,
        )
        total_squared_norm = 0.0
        candidate_squared_norm = 0.0
        first_nonfinite = None
        for (parameter_name, _), gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all().item():
                if first_nonfinite is None:
                    first_nonfinite = parameter_name
                continue
            squared_norm = float(gradient.detach().double().pow(2).sum().item())
            total_squared_norm += squared_norm
            if parameter_name.startswith("candidate_generator."):
                candidate_squared_norm += squared_norm

        scaled_norm = total_squared_norm ** 0.5
        unscaled_norm = scaled_norm / scale
        outputs[loss_name] = {
            "scaled_gradient_finite": first_nonfinite is None,
            "scaled_gradient_first_nonfinite_parameter": first_nonfinite,
            "scaled_gradient_l2_norm": scaled_norm,
            "scaled_candidate_generator_l2_norm": candidate_squared_norm ** 0.5,
            "unscaled_gradient_finite": first_nonfinite is None,
            "unscaled_gradient_l2_norm": unscaled_norm,
            "unscaled_candidate_generator_l2_norm": candidate_squared_norm ** 0.5 / scale,
            "first_nonfinite_parameter": first_nonfinite,
        }
    return outputs


def diagnose_failed_batch(model, criterion, original, reference, buffer_snapshot, scaler_scale):
    comparison = {}
    amp_scales = [scaler_scale]
    while amp_scales[-1] > 2048.0:
        amp_scales.append(amp_scales[-1] / 2.0)
    diagnostics = [(True, f"amp_scale_{int(scale)}", scale) for scale in amp_scales]
    diagnostics.append((False, "fp32", 1.0))

    for amp_enabled, label, scale in diagnostics:
        restore_buffers(model, buffer_snapshot)
        model.zero_grad(set_to_none=True)
        model.train(True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(original)
            losses = criterion(outputs, reference, original)
        output_status = tensor_tree_status(outputs)
        loss_status = {
            name: bool(torch.isfinite(value).all().item())
            for name, value in losses.items()
        }
        gradients = component_gradient_diagnostics(model, losses, scale)
        comparison[label] = {
            "amp_enabled": amp_enabled,
            "scaler_scale": scale,
            "model_outputs_finite": all(output_status.values()),
            "model_output_status": output_status,
            "losses": {name: float(value.detach().item()) for name, value in losses.items()},
            "loss_finite": loss_status,
            "gradient_diagnostics_by_loss": gradients,
            "optimizer_step_executed": False,
        }
    restore_buffers(model, buffer_snapshot)
    model.zero_grad(set_to_none=True)
    return comparison


def reconstruct_rng_through_epoch(train_loader, validation_loader, completed_epochs):
    for _ in range(completed_epochs):
        for _batch in train_loader:
            pass
        for _batch in validation_loader:
            pass


def capture_rng_state(train_loader):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": train_loader.generator.get_state(),
    }


def restore_rng_state(state, train_loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_loader.generator.set_state(state["train_loader_generator"])


def make_loader(dataset, *, training: bool, generator=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def run_epoch(model, criterion, loader, device, scaler, *, epoch: int, training: bool, amp_enabled: bool):
    phase = "train" if training else "val"
    model.train(training)
    totals = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(
            tqdm(loader, desc=f"V3 {phase} {epoch}/{EPOCHS}", leave=False),
            start=1,
        ):
            original = batch["underwater"].to(device, non_blocking=True)
            reference = batch["reference"].to(device, non_blocking=True)
            filenames = list(batch["filename"])

            if training:
                model.zero_grad(set_to_none=True)
                buffer_snapshot = {
                    name: value.detach().clone()
                    for name, value in model.named_buffers()
                }
                scaler_scale = float(scaler.get_scale())
            else:
                buffer_snapshot = None
                scaler_scale = float(scaler.get_scale())

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(original)
                losses = criterion(outputs, reference, original)

            output_status = tensor_tree_status(outputs)
            loss_status = {
                name: bool(torch.isfinite(value).all().item())
                for name, value in losses.items()
            }
            record = {
                "epoch": epoch,
                "phase": phase,
                "batch_index": batch_index,
                "filenames": filenames,
                "grad_scaler_scale": scaler_scale,
                "amp_enabled": amp_enabled,
                "model_outputs_finite": all(output_status.values()),
                "model_output_status": output_status,
                "loss_component_finite": loss_status,
                "losses": {name: float(value.detach().item()) for name, value in losses.items()},
                "total_loss_finite": loss_status.get("total", False),
                "scaled_gradient_finite": None,
                "unscaled_gradient_finite": None,
                "gradient_norm_before_clip": None,
                "gradient_norm_after_clip": None,
                "first_offending_parameter": None,
                "optimizer_step_executed": False,
            }

            if not record["model_outputs_finite"] or not all(loss_status.values()):
                append_batch_record(record)
                diagnostic_path = save_failure_record(record)
                raise FloatingPointError(
                    f"Non-finite model output or loss at epoch {epoch}, batch {batch_index}; "
                    f"files={filenames}; diagnostic={diagnostic_path}"
                )

            if training:
                scaler.scale(losses["total"]).backward()
                scaled_gradient_status = gradient_summary(model.named_parameters())
                scaler.unscale_(optimizer)
                unscaled_gradient_status = gradient_summary(model.named_parameters())
                record["scaled_gradient_finite"] = scaled_gradient_status["finite"]
                record["scaled_gradient_norm"] = scaled_gradient_status["l2_norm"]
                record["unscaled_gradient_finite"] = unscaled_gradient_status["finite"]
                record["gradient_norm_before_clip"] = unscaled_gradient_status["l2_norm"]
                record["first_offending_parameter"] = (
                    scaled_gradient_status["first_nonfinite_parameter"]
                    or unscaled_gradient_status["first_nonfinite_parameter"]
                )

                if not scaled_gradient_status["finite"] or not unscaled_gradient_status["finite"]:
                    append_batch_record(record)
                    comparison = diagnose_failed_batch(
                        model,
                        criterion,
                        original,
                        reference,
                        buffer_snapshot,
                        scaler_scale,
                    )
                    diagnostic_path = save_failure_record(record, comparison)
                    raise FloatingPointError(
                        f"Non-finite gradient at epoch {epoch}, batch {batch_index}, "
                        f"parameter={record['first_offending_parameter']}, files={filenames}, "
                        f"scaler_scale={scaler_scale}; diagnostic={diagnostic_path}. "
                        "Optimizer step was not executed."
                    )

                try:
                    clipped_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=MAX_GRAD_NORM,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as error:
                    record["gradient_clipping_error"] = str(error)
                    append_batch_record(record)
                    diagnostic_path = save_failure_record(record)
                    raise FloatingPointError(
                        f"Non-finite gradient norm during clipping at epoch {epoch}, "
                        f"batch {batch_index}, files={filenames}; diagnostic={diagnostic_path}"
                    ) from error

                record["gradient_norm_returned_by_clipper"] = float(clipped_norm.detach().item())
                record["gradient_norm_after_clip"] = gradient_summary(model.named_parameters())["l2_norm"]
                scaler.step(optimizer)
                scaler.update()
                record["optimizer_step_executed"] = True

            if training:
                append_batch_record(record)

            batch_size = original.shape[0]
            sample_count += batch_size
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach().item()) * batch_size

    if sample_count == 0:
        raise RuntimeError(f"The {phase} loader is empty.")
    return {name: value / sample_count for name, value in totals.items()}


def write_history(history) -> None:
    json_path = OUTPUT_DIR / "loss_history.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    csv_path = OUTPUT_DIR / "loss_history.csv"
    fields = ["epoch"] + [f"train_{name}" for name in LOSS_NAMES] + [f"val_{name}" for name in LOSS_NAMES] + ["best_epoch"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    train_losses,
    val_losses,
    config,
    train_loader,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_losses": train_losses,
            "validation_losses": val_losses,
            "config": config,
            "rng_state": capture_rng_state(train_loader),
        },
        path,
    )


def main() -> None:
    global optimizer

    parser = argparse.ArgumentParser(description="Train or resume the controlled Paper 1 V3 experiment.")
    parser.add_argument("--resume-from-epoch", type=int, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument("--amp-resume-scale", type=float, default=None)
    args = parser.parse_args()

    if not V2_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Frozen V2 checkpoint is missing: {V2_CHECKPOINT}")
    v2_sha_before = verify_v2_checkpoint(EXPECTED_V2_SHA256, "before V3 setup")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_files = sorted(OUTPUT_DIR.glob("paper1_v3_epoch_*.pth"))
    resume_epoch = args.resume_from_epoch
    if resume_epoch is None and checkpoint_files:
        resume_epoch = max(
            int(path.stem.rsplit("_", 1)[1]) for path in checkpoint_files
        )
    if resume_epoch is None and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"V3 output directory contains run artifacts but no epoch checkpoint: {OUTPUT_DIR}"
        )
    resume_path = (
        OUTPUT_DIR / f"paper1_v3_epoch_{resume_epoch:03d}.pth"
        if resume_epoch is not None
        else None
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_path}")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()

    train_dataset = EUVPPairedDataset(
        input_dir=DATASET_ROOT / "trainA",
        reference_dir=DATASET_ROOT / "trainB",
        split_file=SPLIT_DIR / "train.txt",
        transform=PairedTransform(training=True),
    )
    validation_dataset = EUVPPairedDataset(
        input_dir=DATASET_ROOT / "trainA",
        reference_dir=DATASET_ROOT / "trainB",
        split_file=SPLIT_DIR / "val.txt",
        transform=PairedTransform(training=False),
    )
    train_names = train_dataset.filenames
    validation_names = validation_dataset.filenames
    if len(train_names) != len(set(train_names)) or len(validation_names) != len(set(validation_names)):
        raise RuntimeError("Duplicate filenames found in train/validation split manifests.")
    if not set(train_names).isdisjoint(validation_names):
        raise RuntimeError("Train/validation split leakage detected. Stopping.")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)
    train_loader = make_loader(train_dataset, training=True, generator=loader_generator)
    validation_loader = make_loader(validation_dataset, training=False)

    model = Paper1Model().to(device)
    criterion = Paper1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("epoch") != resume_epoch:
            raise RuntimeError(
                f"Resume checkpoint epoch mismatch: path says {resume_epoch}, "
                f"checkpoint says {checkpoint.get('epoch')}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = dict(checkpoint["scaler_state_dict"])
        if args.amp_resume_scale is not None:
            if not amp_enabled:
                raise ValueError("--amp-resume-scale requires CUDA AMP to be enabled.")
            if args.amp_resume_scale <= 0:
                raise ValueError("--amp-resume-scale must be positive.")
            scaler_state["scale"] = float(args.amp_resume_scale)
            scaler_state["_growth_tracker"] = 0
        scaler.load_state_dict(scaler_state)

        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"], train_loader)
            rng_recovery = "checkpoint RNG state"
        else:
            reconstruct_rng_through_epoch(train_loader, validation_loader, resume_epoch)
            rng_recovery = f"deterministic replay of train/validation RNG through epoch {resume_epoch}"

        history_path = OUTPUT_DIR / "loss_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if not history or int(history[-1]["epoch"]) != resume_epoch:
            raise RuntimeError(
                f"Loss history does not end at the requested resume epoch {resume_epoch}; "
                "refusing to continue with inconsistent history."
            )
        best_row = min(history, key=lambda row: float(row["val_total"]))
        best_epoch = int(best_row["epoch"])
        best_validation_total = float(best_row["val_total"])
        start_epoch = resume_epoch + 1
        config_path = OUTPUT_DIR / "training_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.amp_resume_scale is not None:
            config["amp_stability_correction"] = {
                "resume_scale": float(args.amp_resume_scale),
                "growth_tracker_reset": True,
                "evidence": "Same epoch-5 failure batch was finite for all loss-component gradients at AMP scales <= 16384 and non-finite at 32768/65536; FP32 was finite.",
            }
        print(f"Resuming from epoch {resume_epoch}: {resume_path}")
        print(f"RNG recovery: {rng_recovery}")
        if args.amp_resume_scale is not None:
            print(
                f"AMP stability correction: scale={args.amp_resume_scale:g}, "
                "growth tracker reset to 0"
            )
    else:
        history = []
        best_epoch = None
        best_validation_total = float("inf")
        start_epoch = 1
        config = {
            "experiment": "Paper 1 V3 fixed-preservation controlled run",
            "initialization": "random from scratch; no checkpoint loaded",
            "seed": SEED,
            "deterministic_algorithms": True,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "scheduler": None,
            "early_stopping": False,
            "amp_enabled": amp_enabled,
            "gradient_clip_norm": MAX_GRAD_NORM,
            "validation_frequency_epochs": 1,
            "checkpoint_frequency_epochs": 1,
            "best_checkpoint_criterion": "minimum validation total loss",
            "loss_weights": {
                "reconstruction": criterion.lambda_recon,
                "preservation": criterion.lambda_preserve,
                "consequence": criterion.lambda_consequence,
                "selection": criterion.lambda_selection,
            },
            "dataset_root": str(DATASET_ROOT),
            "training_split": "data/splits/train.txt",
            "validation_split": "data/splits/val.txt",
            "test_split_used": False,
            "preprocessing": "paired RGB ToTensor [0,1]; train-only paired horizontal flip",
            "output_dir": str(OUTPUT_DIR),
            "frozen_v2_checkpoint": str(V2_CHECKPOINT),
            "frozen_v2_sha256_before": v2_sha_before,
        }

    config["batch_diagnostic_logging"] = {
        "file": "batch_diagnostics.jsonl",
        "failure_records": "failure_epoch_*_batch_*.json",
        "fields": [
            "epoch", "batch_index", "filenames", "scaler_scale",
            "model_output_finiteness", "loss_finiteness",
            "scaled_gradient_finiteness", "unscaled_gradient_finiteness",
            "gradient_norm", "first_offending_parameter", "optimizer_step_executed",
        ],
        "no_step_amp_fp32_diagnostic_on_gradient_failure": True,
    }
    with (OUTPUT_DIR / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"V3 device: {device}")
    print(f"CUDA: available={torch.cuda.is_available()}, version={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}; validation samples: {len(validation_dataset)}")
    print(f"Batch size: {BATCH_SIZE}; epochs: {EPOCHS}; Adam lr: {LEARNING_RATE}; AMP: {amp_enabled}")
    print("Test split: not loaded or used")

    final_epoch = min(args.stop_after_epoch or EPOCHS, EPOCHS)
    if final_epoch < start_epoch:
        raise ValueError(f"stop-after epoch {final_epoch} precedes start epoch {start_epoch}")

    for epoch in range(start_epoch, final_epoch + 1):
        verify_v2_checkpoint(EXPECTED_V2_SHA256, f"before epoch {epoch}")
        train_losses = run_epoch(
            model,
            criterion,
            train_loader,
            device,
            scaler,
            epoch=epoch,
            training=True,
            amp_enabled=amp_enabled,
        )
        validation_losses = run_epoch(
            model,
            criterion,
            validation_loader,
            device,
            scaler,
            epoch=epoch,
            training=False,
            amp_enabled=amp_enabled,
        )
        verify_v2_checkpoint(EXPECTED_V2_SHA256, f"after epoch {epoch}")

        if validation_losses["total"] < best_validation_total:
            best_validation_total = validation_losses["total"]
            best_epoch = epoch

        row = {"epoch": epoch}
        row.update({f"train_{name}": train_losses[name] for name in LOSS_NAMES})
        row.update({f"val_{name}": validation_losses[name] for name in LOSS_NAMES})
        row["best_epoch"] = best_epoch
        history.append(row)
        write_history(history)

        epoch_path = OUTPUT_DIR / f"paper1_v3_epoch_{epoch:03d}.pth"
        save_checkpoint(
            epoch_path,
            model,
            optimizer,
            scaler,
            epoch,
            train_losses,
            validation_losses,
            config,
            train_loader,
        )
        if epoch == best_epoch:
            save_checkpoint(
                OUTPUT_DIR / "paper1_v3_best.pth",
                model,
                optimizer,
                scaler,
                epoch,
                train_losses,
                validation_losses,
                config,
                train_loader,
            )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Total {train_losses['total']:.6f} | Val Total {validation_losses['total']:.6f} | "
            f"Train Reconstruction {train_losses['reconstruction']:.6f} | Val Reconstruction {validation_losses['reconstruction']:.6f} | "
            f"Train Preservation {train_losses['preservation']:.6f} | Val Preservation {validation_losses['preservation']:.6f} | "
            f"Train Consequence {train_losses['consequence']:.6f} | Val Consequence {validation_losses['consequence']:.6f} | "
            f"Train Selection {train_losses['selection']:.6f} | Val Selection {validation_losses['selection']:.6f} | "
            f"Best Epoch {best_epoch}"
        )

        verify_v2_checkpoint(EXPECTED_V2_SHA256, f"after saving epoch {epoch}")
        if epoch == args.stop_after_epoch:
            print(f"Stopped after requested epoch {epoch}; no test data was used.")
            return

    v2_sha_after = verify_v2_checkpoint(EXPECTED_V2_SHA256, "after V3 training")
    print("\nTraining/validation losses by epoch:")
    print(json.dumps(history, indent=2))
    print(f"Best epoch: {best_epoch}; best validation total: {best_validation_total:.6f}")
    print(f"Final epoch: {EPOCHS}; final train total: {history[-1]['train_total']:.6f}; final validation total: {history[-1]['val_total']:.6f}")
    print("Checkpoints:")
    for checkpoint_path in sorted(OUTPUT_DIR.glob("paper1_v3_*.pth")):
        print(f"  {checkpoint_path}")
    print(f"V2 checkpoint SHA-256 before: {v2_sha_before}")
    print(f"V2 checkpoint SHA-256 after:  {v2_sha_after}")
    print("Test split used: NO")


if __name__ == "__main__":
    main()