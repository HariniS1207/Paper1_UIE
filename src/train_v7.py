from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src" / "data", ROOT / "src" / "models", ROOT / "src" / "losses"):
    import sys
    sys.path.insert(0, str(path))

from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from models.paper1_model_v7 import Paper1ModelV7
from losses.paper1_loss_v7 import Paper1LossV7

SEED, BATCH_SIZE, LR, MAX_EPOCHS = 42, 8, 1e-4, 20
NAMES = ("Conservative", "Balanced", "Aggressive")
LOWER = tuple(n.lower() for n in NAMES)
DATASET = ROOT / "data" / "EUVP" / "EUVP-Dataset" / "EUVP" / "Paired" / "underwater_imagenet"
SPLITS = ROOT / "data" / "splits"
V4_CKPT = ROOT / "checkpoints" / "v4" / "paper1_v4_best.pth"
CKPT_DIR = ROOT / "checkpoints" / "v7"
RESULTS_DIR = ROOT / "results" / "v7"
ANALYSIS_PATH = ROOT / "diagnostics" / "v7_selection" / "v7_candidate_space_analysis.json"
HASH_MANIFEST = ROOT / "diagnostics" / "v7_selection" / "frozen_checkpoint_hashes_before.json"
LOSS_KEYS = ("total", "reconstruction", "preservation", "consequence", "selection")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha_state(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def verify_frozen_manifest():
    before = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
    for version, files in before.items():
        directory = ROOT / "checkpoints" / version
        actual = {p.name: sha256(p) for p in sorted(directory.glob("*.pth"))}
        if actual != files:
            raise RuntimeError(f"Frozen {version} checkpoint inventory/hash changed during V7: {version}")


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def rng_state(loader_generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng(state, loader_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader_generator"])


def targets_for(outputs, reference):
    quantities = Paper1LossV7.candidate_objectives(outputs, reference)
    targets, ranks = Paper1LossV7.relative_rank_targets(quantities)
    return targets, quantities, ranks


def run_epoch(model, criterion, loader, device, optimizer, scaler, *, epoch, training, amp_enabled, amp_events):
    model.train(training)
    # Frozen condition features and candidate weights must remain fixed.
    model.condition_encoder.eval()
    model.candidate_generator.eval()
    model.information_preservation.eval()
    model.consequence_check.eval()
    phase = "train" if training else "val"
    sums = {k: 0.0 for k in LOSS_KEYS}
    target_counts = torch.zeros(3, dtype=torch.long)
    selector_counts = torch.zeros(3, dtype=torch.long)
    n_seen = 0
    for batch_index, batch in enumerate(tqdm(loader, desc=f"V7 {phase} {epoch}/{MAX_EPOCHS}", leave=False), 1):
        x = batch["underwater"].to(device, non_blocking=True)
        ref = batch["reference"].to(device, non_blocking=True)
        if training:
            attempt = 0
            while True:
                attempt += 1
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(x)
                    losses = criterion(outputs, ref, x)
                finite_loss = all(bool(torch.isfinite(loss.detach()).all()) for loss in losses.values())
                if not finite_loss:
                    finite_grad = False
                else:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    finite_grad = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                      for p in model.selection_network.parameters())
                if finite_loss and finite_grad:
                    torch.nn.utils.clip_grad_norm_(model.selection_network.parameters(), 1.0, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    break
                old_scale = float(scaler.get_scale())
                event = {"epoch": epoch, "batch_index": batch_index, "replay": attempt,
                         "scale_before": old_scale, "loss_finite": finite_loss,
                         "gradient_finite": finite_grad, "optimizer_step_applied": False,
                         "filenames": list(batch["filename"])}
                amp_events.append(event)
                optimizer.zero_grad(set_to_none=True)
                if attempt >= 6:
                    raise FloatingPointError(f"Non-finite V7 selector gradients after deterministic replay: {event}")
                scaler.update(new_scale=max(old_scale / 2.0, 1.0))
        else:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(x)
                losses = criterion(outputs, ref, x)
        targets, _, _ = targets_for(outputs, ref)
        target_counts += torch.bincount(targets.detach().cpu(), minlength=3)
        selector_counts += torch.bincount(outputs["selected_index"].detach().cpu(), minlength=3)
        bs = x.size(0)
        for key in LOSS_KEYS:
            sums[key] += float(losses[key].detach().float().item()) * bs
        n_seen += bs
    if not n_seen:
        raise RuntimeError(f"Empty {phase} loader")
    return ({k: v / n_seen for k, v in sums.items()}, target_counts.tolist(), selector_counts.tolist())


def save_checkpoint(path, model, optimizer, scaler, epoch, train_loss, val_loss, history,
                    config, loader_generator, best_epoch, best_val):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_loss, "validation_loss": val_loss, "history": history,
                "config": config, "rng_state": rng_state(loader_generator),
                "best_epoch": best_epoch, "best_validation_total": best_val}, path)


def main():
    parser = argparse.ArgumentParser(description="Train selector-only Paper 1 V7; train/validation only.")
    parser.add_argument("--resume", action="store_true", help="resume the last complete V7 epoch")
    args = parser.parse_args()
    if not ANALYSIS_PATH.is_file() or not HASH_MANIFEST.is_file():
        raise FileNotFoundError("V7 candidate-space evidence/hash manifest is missing")
    verify_frozen_manifest()
    if not V4_CKPT.is_file():
        raise FileNotFoundError(V4_CKPT)
    if not args.resume:
        if CKPT_DIR.exists() and any(CKPT_DIR.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing V7 checkpoint artifacts: {CKPT_DIR}")
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = RESULTS_DIR / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        epochs = sorted(CKPT_DIR.glob("paper1_v7_epoch_*.pth"))
        if not epochs:
            raise FileNotFoundError("No completed V7 epoch exists to resume")
        run_dir = Path(json.loads((CKPT_DIR / "training_config.json").read_text())["results_dir"])

    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    train_ds = EUVPPairedDataset(DATASET / "trainA", DATASET / "trainB", SPLITS / "train.txt",
                                 transform=PairedTransform(training=True))
    val_ds = EUVPPairedDataset(DATASET / "trainA", DATASET / "trainB", SPLITS / "val.txt",
                               transform=PairedTransform(training=False))
    if set(train_ds.filenames) & set(val_ds.filenames):
        raise RuntimeError("Train/validation split overlap")
    loader_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available(), generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    model = Paper1ModelV7().to(device)
    v4_state = torch.load(V4_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    # Transfer the exact frozen V4 condition/generator/preservation/consequence
    # modules. V7's replacement selector is initialized independently.
    frozen_state = {k: v for k, v in v4_state.items() if not k.startswith("selection_network.")}
    missing, unexpected = model.load_state_dict(frozen_state, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith("selection_network.")}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(f"V4-to-V7 initialization mismatch: missing={missing}, unexpected={unexpected}")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.selection_network.parameters():
        p.requires_grad_(True)
    model.condition_encoder.eval()
    frozen_hash = {name: sha_state(getattr(model, name)) for name in
                   ("condition_encoder", "candidate_generator", "information_preservation", "consequence_check")}
    criterion = Paper1LossV7().to(device)
    optimizer = torch.optim.Adam(model.selection_network.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048.0, enabled=amp_enabled)
    amp_events, history, best_epoch, best_val, start_epoch = [], [], None, float("inf"), 1
    config_path = CKPT_DIR / "training_config.json"
    if args.resume:
        epoch_paths = sorted(CKPT_DIR.glob("paper1_v7_epoch_*.pth"))
        last = epoch_paths[-1]
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["epoch"] != int(last.stem.rsplit("_", 1)[1]):
            raise RuntimeError("Latest V7 epoch filename/content mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        history = state["history"]
        amp_events = state.get("amp_events", [])
        best_epoch, best_val = state["best_epoch"], state["best_validation_total"]
        start_epoch = state["epoch"] + 1
        restore_rng(state["rng_state"], loader_gen)
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
        train_only_targets = analysis["training_only_rank_target_audit"]
        config = {
            "experiment": "Paper 1 V7 selector-only, condition-aware compatibility experiment",
            "initialization": "V4 best for all frozen modules; new selector initialized with seed 42",
            "frozen_modules": list(frozen_hash), "frozen_module_state_sha256_before": frozen_hash,
            "candidate_generator": "unchanged V4 candidate-specific residual branches; frozen for V7",
            "selector": "shared candidate encoder + condition projection + elementwise candidate-condition interaction; candidate preservation score and consequence error as utility inputs",
            "target": "For each image rank its three candidates separately on reference L1, preservation distortion, and consequence L1; ties receive average rank; average the three normalized ranks (0 best, 1 worst); argmin is the target. No class quotas or train-global scales.",
            "target_training_only_audit": train_only_targets,
            "training_target_counts_from_frozen_v4_train_data": train_only_targets["target_counts"],
            "seed": SEED, "batch_size": BATCH_SIZE, "optimizer": "Adam",
            "learning_rate": LR, "max_epochs": MAX_EPOCHS, "best_checkpoint_criterion": "minimum validation total loss",
            "amp_enabled": amp_enabled, "amp_initial_scale": 2048.0, "scheduler": None,
            "early_stopping": False, "train_split": "data/splits/train.txt",
            "validation_split": "data/splits/val.txt", "test_split_loaded": False,
            "preprocessing": "paired RGB ToTensor [0,1]; paired random horizontal flip train only",
            "loss_weights": {"reconstruction": 1.0, "preservation": 0.2, "consequence": 0.2, "selection": 0.5},
            "selector_tradeoff_losses": "Expected reference L1, preservation distortion, and consequence errors under selector probabilities; makes all three existing terms directly train the selector.",
            "results_dir": str(run_dir), "v4_checkpoint_sha256": sha256(V4_CKPT),
            "frozen_checkpoint_hashes_before": json.loads(HASH_MANIFEST.read_text(encoding="utf-8")),
            "repository_head": "dad17a1c0f9f76b75f6fd685f593537d592550dd",
        }
        with config_path.open("x", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    if start_epoch > MAX_EPOCHS:
        raise RuntimeError("V7 checkpoint already contains all configured epochs")
    print(f"V7 selector-only training: train={len(train_ds)} val={len(val_ds)} device={device} amp={amp_enabled}; test split loaded: NO")
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        verify_frozen_manifest()
        start = time.perf_counter()
        train_loss, train_targets, train_selected = run_epoch(
            model, criterion, train_loader, device, optimizer, scaler,
            epoch=epoch, training=True, amp_enabled=amp_enabled, amp_events=amp_events)
        val_loss, val_targets, val_selected = run_epoch(
            model, criterion, val_loader, device, optimizer, scaler,
            epoch=epoch, training=False, amp_enabled=amp_enabled, amp_events=amp_events)
        for name, digest in frozen_hash.items():
            if sha_state(getattr(model, name)) != digest:
                raise RuntimeError(f"Frozen V7 module changed: {name}")
        verify_frozen_manifest()
        if val_loss["total"] < best_val:
            best_val, best_epoch = val_loss["total"], epoch
        record = {"epoch": epoch, "train": train_loss, "validation": val_loss,
                  "train_target_counts": dict(zip(NAMES, train_targets)),
                  "train_selector_counts": dict(zip(NAMES, train_selected)),
                  "validation_target_counts": dict(zip(NAMES, val_targets)),
                  "validation_selector_counts": dict(zip(NAMES, val_selected)),
                  "seconds": time.perf_counter() - start, "best_epoch": best_epoch}
        history.append(record)
        payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scaler_state_dict": scaler.state_dict(), "history": history, "config": config,
                   "rng_state": rng_state(loader_gen), "epoch": epoch, "best_epoch": best_epoch,
                   "best_validation_total": best_val, "amp_events": amp_events}
        torch.save(payload, CKPT_DIR / f"paper1_v7_epoch_{epoch:03d}.pth")
        if epoch == best_epoch:
            torch.save(payload, CKPT_DIR / f"paper1_v7_best_epoch_{epoch:03d}.pth")
        (CKPT_DIR / "loss_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (CKPT_DIR / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["epoch", "train_total", "validation_total", "train_reconstruction", "validation_reconstruction",
                      "train_preservation", "validation_preservation", "train_consequence", "validation_consequence",
                      "train_selection", "validation_selection", "best_epoch", "seconds"]
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for row in history:
                flat = {"epoch": row["epoch"], "best_epoch": row["best_epoch"], "seconds": row["seconds"]}
                for phase in ("train", "validation"):
                    for key, value in row[phase].items(): flat[f"{phase}_{key}"] = value
                writer.writerow(flat)
        print(f"Epoch {epoch}/{MAX_EPOCHS} train={train_loss['total']:.5f} val={val_loss['total']:.5f} "
              f"val_sel={val_loss['selection']:.5f} best={best_epoch} val_targets={val_targets} "
              f"val_selected={val_selected} elapsed={record['seconds']:.1f}s", flush=True)
    config["amp_events"] = amp_events
    config["frozen_module_state_sha256_after"] = {name: sha_state(getattr(model, name)) for name in frozen_hash}
    if config["frozen_module_state_sha256_after"] != frozen_hash:
        raise RuntimeError("Frozen modules changed during the V7 run")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    verify_frozen_manifest()
    best_path = CKPT_DIR / f"paper1_v7_best_epoch_{best_epoch:03d}.pth"
    print(f"V7 training complete. best_epoch={best_epoch}, best_val_total={best_val:.6f}, best_sha256={sha256(best_path)}")


if __name__ == "__main__":
    main()
