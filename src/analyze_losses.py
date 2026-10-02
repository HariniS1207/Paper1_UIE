from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "losses"))

from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from paper1_model import Paper1Model
from paper1_loss import Paper1Loss


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "v2"
    / "paper1_v2_epoch_006.pth"
)

BATCH_SIZE = 8


# ============================================================
# Dataset
# ============================================================

dataset = EUVPPairedDataset(
    input_dir=DATASET_ROOT / "trainA",
    reference_dir=DATASET_ROOT / "trainB",
    split_file=SPLIT_DIR / "val.txt",
    transform=PairedTransform(training=False),
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# Model
# ============================================================

model = Paper1Model().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

criterion = Paper1Loss().to(DEVICE)


# ============================================================
# Accumulators
# ============================================================

loss_names = [
    "total",
    "reconstruction",
    "preservation",
    "consequence",
    "selection"
]

loss_sums = {
    name: 0.0
    for name in loss_names
}

num_batches = 0


# ============================================================
# Evaluation
# ============================================================

with torch.no_grad():

    for batch in tqdm(
        loader,
        desc="Analyzing validation losses"
    ):

        underwater = batch["underwater"].to(
            DEVICE,
            non_blocking=True
        )

        reference = batch["reference"].to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available()
        ):

            outputs = model(
                underwater
            )

            losses = criterion(
                outputs,
                reference,
                underwater
            )

        for name in loss_names:
            loss_sums[name] += losses[name].item()

        num_batches += 1


# ============================================================
# Results
# ============================================================

print()
print("=" * 60)
print("PAPER 1 V2 — VALIDATION LOSS ANALYSIS")
print("=" * 60)

print(
    f"Checkpoint : {CHECKPOINT.name}"
)

print(
    f"Validation images: {len(dataset)}"
)

print()

for name in loss_names:

    average = (
        loss_sums[name]
        / num_batches
    )

    print(
        f"{name.capitalize():<15}: "
        f"{average:.6f}"
    )

print("=" * 60)