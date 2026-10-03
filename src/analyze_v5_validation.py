"""Validation-only post-training V4/V5 analysis; never loads a test split."""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "losses"))

from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from models.paper1_model_v4 import Paper1ModelV4
from models.paper1_model_v5 import Paper1ModelV5
from paper1_loss_v5 import Paper1LossV5
from train_v5 import BATCH_SIZE, DATASET_ROOT, SPLIT_DIR, candidate_diversity, validation_analysis, write_final_reports


def main():
    ckpt_dir = ROOT / "checkpoints" / "v5"
    results = ROOT / "results" / "v5"
    config = json.loads((ckpt_dir / "training_config.json").read_text(encoding="utf-8"))
    history = json.loads((ckpt_dir / "loss_history.json").read_text(encoding="utf-8"))
    best_epoch = min(history, key=lambda r: float(r["val_total"]))["epoch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration = json.loads((ckpt_dir / "target_calibration_train_only.json").read_text(encoding="utf-8"))
    val = EUVPPairedDataset(DATASET_ROOT / "trainA", DATASET_ROOT / "trainB", SPLIT_DIR / "val.txt", transform=PairedTransform(training=False))
    loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    v5 = Paper1ModelV5().to(device)
    checkpoint = torch.load(ckpt_dir / f"paper1_v5_best_epoch_{best_epoch:03d}.pth", map_location=device, weights_only=False)
    v5.load_state_dict(checkpoint["model_state_dict"], strict=True)
    v4 = Paper1ModelV4().to(device)
    v4_checkpoint = torch.load(ROOT / "checkpoints" / "v4" / "paper1_v4_best.pth", map_location=device, weights_only=False)
    v4.load_state_dict(v4_checkpoint["model_state_dict"], strict=True)
    criterion = Paper1LossV5(calibration).to(device)
    aggregate, rows = validation_analysis(v5, criterion, loader, device, v4)
    aggregate["v5_candidate_diversity"] = candidate_diversity(v5, loader, device, compute_reference_metrics=True)
    aggregate["v4_candidate_diversity"] = candidate_diversity(v4, loader, device, compute_reference_metrics=True)
    run_dir = Path(config["results_dir"])
    write_final_reports(history, best_epoch, calibration, aggregate, rows, run_dir)
    print(json.dumps({"best_epoch": best_epoch, "validation_analysis": aggregate}, indent=2))


if __name__ == "__main__":
    main()
