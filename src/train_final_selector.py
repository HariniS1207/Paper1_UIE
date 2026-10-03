"""One final train/validation-only selector compatibility experiment."""

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
import sys
for p in (ROOT / "src", ROOT / "src" / "data", ROOT / "src" / "models", ROOT / "src" / "losses"):
    sys.path.insert(0, str(p))
from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from models.paper1_model_final_selector import Paper1ModelFinalSelector
from losses.paper1_loss_final_selector import Paper1LossFinalSelector

SEED, BATCH, LR, EPOCHS = 42, 8, 1e-4, 20
NAMES = ("Conservative", "Balanced", "Aggressive")
LOWER = tuple(n.lower() for n in NAMES)
DATA = ROOT / "data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet"
SPLITS = ROOT / "data/splits"
V4 = ROOT / "checkpoints/v4/paper1_v4_best.pth"
CKPT = ROOT / "checkpoints/final_selector"
RESULTS = ROOT / "results/final_selector"
DIAG = ROOT / "diagnostics/final_selector"
MANIFEST = ROOT / "diagnostics/v7_selection/frozen_checkpoint_hashes_before.json"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def state_digest(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest().upper()


def verify_old_checkpoints():
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for version, files in expected.items():
        actual = {p.name: digest(p) for p in sorted((ROOT / "checkpoints" / version).glob("*.pth"))}
        if actual != files:
            raise RuntimeError(f"Existing {version} checkpoint inventory changed")


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def run_epoch(model, criterion, loader, optimizer, scaler, device, training, epoch, amp, amp_events):
    model.train(training)
    for module in (model.condition_encoder, model.candidate_generator, model.information_preservation, model.consequence_check):
        module.eval()
    sums = {k: 0.0 for k in ("total", "reconstruction", "preservation", "consequence", "selection")}
    target_mass = torch.zeros(3, dtype=torch.float64)
    targets_hard = torch.zeros(3, dtype=torch.long)
    selected = torch.zeros(3, dtype=torch.long)
    n = 0
    for bi, batch in enumerate(tqdm(loader, desc=f"Final selector {'train' if training else 'val'} {epoch}/{EPOCHS}", leave=False), 1):
        x, ref = batch["underwater"].to(device), batch["reference"].to(device)
        if training:
            attempt = 0
            while True:
                attempt += 1; optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    out = model(x); losses = criterion(out, ref, x)
                finite_loss = all(torch.isfinite(v.detach()).all().item() for v in losses.values())
                finite_grad = False
                if finite_loss:
                    scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
                    finite_grad = all(p.grad is None or torch.isfinite(p.grad).all().item()
                                      for p in model.selection_network.parameters())
                if finite_loss and finite_grad:
                    torch.nn.utils.clip_grad_norm_(model.selection_network.parameters(), 1.0, error_if_nonfinite=True)
                    scaler.step(optimizer); scaler.update(); break
                old = float(scaler.get_scale())
                event = {"epoch": epoch, "batch_index": bi, "replay": attempt,
                         "scale_before": old, "loss_finite": bool(finite_loss),
                         "gradient_finite": bool(finite_grad), "optimizer_step_applied": False,
                         "filenames": list(batch["filename"])}
                amp_events.append(event); optimizer.zero_grad(set_to_none=True)
                if attempt >= 6: raise FloatingPointError(f"Final selector FP16/FP32 replay exhausted: {event}")
                # Replay this exact batch after lowering scale; final fallback is FP32.
                if attempt >= 3: amp = False
                scaler.update(new_scale=max(old / 2.0, 1.0))
        else:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model(x); losses = criterion(out, ref, x)
        with torch.no_grad():
            objectives = criterion.candidate_objectives(out, ref)
            soft, _ = criterion.soft_targets(objectives)
            hard = soft.argmax(dim=1)
        target_mass += soft.double().sum(dim=0).cpu()
        targets_hard += torch.bincount(hard.cpu(), minlength=3)
        selected += torch.bincount(out["selected_index"].detach().cpu(), minlength=3)
        for k, value in losses.items(): sums[k] += float(value.detach().float()) * x.shape[0]
        n += x.shape[0]
    return ({k: v / n for k, v in sums.items()}, target_mass.tolist(),
            targets_hard.tolist(), selected.tolist())


def main():
    for path in (CKPT, RESULTS, DIAG):
        path.mkdir(parents=True, exist_ok=True)
    if any(CKPT.iterdir()) or any(DIAG.iterdir()) or any(p.is_file() for p in RESULTS.iterdir()):
        raise FileExistsError("Refusing to overwrite existing final-selector artifacts")
    if not MANIFEST.is_file() or not V4.is_file(): raise FileNotFoundError("V4 checkpoint/hash manifest missing")
    verify_old_checkpoints(); seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = torch.cuda.is_available()
    train = EUVPPairedDataset(DATA / "trainA", DATA / "trainB", SPLITS / "train.txt", transform=PairedTransform(training=True))
    val = EUVPPairedDataset(DATA / "trainA", DATA / "trainB", SPLITS / "val.txt", transform=PairedTransform(training=False))
    if set(train.filenames) & set(val.filenames): raise RuntimeError("Train/validation overlap")
    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=amp, generator=gen)
    val_loader = DataLoader(val, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=amp)
    model = Paper1ModelFinalSelector().to(device)
    v4_state = torch.load(V4, map_location="cpu", weights_only=False)["model_state_dict"]
    frozen = {k: v for k, v in v4_state.items() if not k.startswith("selection_network.")}
    missing, unexpected = model.load_state_dict(frozen, strict=False)
    if unexpected or set(missing) != {k for k in model.state_dict() if k.startswith("selection_network.")}:
        raise RuntimeError(f"V4 module initialization mismatch: {missing}, {unexpected}")
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.selection_network.parameters(): p.requires_grad_(True)
    frozen_names = ("condition_encoder", "candidate_generator", "information_preservation", "consequence_check")
    frozen_hashes = {n: state_digest(getattr(model, n)) for n in frozen_names}
    criterion = Paper1LossFinalSelector().to(device)
    optimizer = torch.optim.Adam(model.selection_network.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048.0, enabled=amp)
    config = {
        "experiment": "Final condition-by-candidate bilinear compatibility selector",
        "selector": "Per-policy learned bilinear f(condition)^T W_k g(candidate), plus learned candidate-specific preservation/consequence scalar terms and bias",
        "initialization": "Exact V4 condition encoder, candidate generator, preservation and consequence modules; new selector initialized with seed 42",
        "frozen_modules": list(frozen_names), "frozen_module_state_sha256_before": frozen_hashes,
        "target": "Per-image equal-weight average normalized ranks over fidelity L1, preservation distortion, consequence error; softmax(-mean_rank/0.25), no quotas or resampling",
        "seed": SEED, "batch_size": BATCH, "optimizer": "Adam", "learning_rate": LR, "max_epochs": EPOCHS,
        "best_checkpoint_criterion": "minimum validation total loss", "amp_enabled": amp,
        "amp_initial_scale": 2048.0, "train_split": "data/splits/train.txt",
        "validation_split": "data/splits/val.txt", "test_split_loaded": False,
        "preprocessing": "paired RGB ToTensor [0,1]; paired random horizontal flip train only",
        "loss_weights": {"reconstruction": 1.0, "preservation": 0.2, "consequence": 0.2, "selection": 0.5},
        "v4_checkpoint_sha256": digest(V4), "frozen_checkpoint_hashes_before": json.loads(MANIFEST.read_text()),
        "repository_head_before": "dad17a1c0f9f76b75f6fd685f593537d592550dd",
    }
    run_dir = RESULTS / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(exist_ok=False); config["results_dir"] = str(run_dir.relative_to(ROOT))
    history, amp_events, best_val, best_epoch = [], [], float("inf"), None
    print(f"Final selector train={len(train)} validation={len(val)} device={device} amp={amp}; test split loaded: NO", flush=True)
    for epoch in range(1, EPOCHS + 1):
        verify_old_checkpoints(); start = time.perf_counter()
        tr = run_epoch(model, criterion, train_loader, optimizer, scaler, device, True, epoch, amp, amp_events)
        va = run_epoch(model, criterion, val_loader, optimizer, scaler, device, False, epoch, amp, amp_events)
        for name, before in frozen_hashes.items():
            if state_digest(getattr(model, name)) != before: raise RuntimeError(f"Frozen module changed: {name}")
        verify_old_checkpoints()
        if va[0]["total"] < best_val: best_val, best_epoch = va[0]["total"], epoch
        record = {"epoch": epoch, "train": tr[0], "validation": va[0],
                  "train_target_soft_mass": dict(zip(NAMES, tr[1])), "train_target_hard_counts": dict(zip(NAMES, tr[2])),
                  "train_selected_counts": dict(zip(NAMES, tr[3])), "validation_target_soft_mass": dict(zip(NAMES, va[1])),
                  "validation_target_hard_counts": dict(zip(NAMES, va[2])), "validation_selected_counts": dict(zip(NAMES, va[3])),
                  "seconds": time.perf_counter() - start, "best_epoch": best_epoch}
        history.append(record)
        payload = {"epoch": epoch, "best_epoch": best_epoch, "best_validation_total": best_val,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scaler_state_dict": scaler.state_dict(), "history": history, "amp_events": amp_events, "config": config}
        torch.save(payload, CKPT / f"final_selector_epoch_{epoch:03d}.pth")
        if epoch == best_epoch: torch.save(payload, CKPT / f"final_selector_best_epoch_{epoch:03d}.pth")
        (CKPT / "loss_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}/{EPOCHS} train={tr[0]['total']:.5f} val={va[0]['total']:.5f} best={best_epoch} selected={va[3]} elapsed={record['seconds']:.1f}s", flush=True)
    config["amp_events"] = amp_events
    config["frozen_module_state_sha256_after"] = {n: state_digest(getattr(model, n)) for n in frozen_names}
    config["checkpoint_sha256"] = digest(CKPT / f"final_selector_best_epoch_{best_epoch:03d}.pth")
    config["previous_checkpoints_unchanged"] = True
    (CKPT / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RESULTS / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete best_epoch={best_epoch} best_val={best_val:.6f} sha256={config['checkpoint_sha256']}", flush=True)


if __name__ == "__main__": main()
