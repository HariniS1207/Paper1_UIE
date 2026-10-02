from pathlib import Path

import torch
from torch.utils.data import DataLoader

from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
)


dataset = EUVPPairedDataset(
    input_dir=DATASET_ROOT / "trainA",
    reference_dir=DATASET_ROOT / "trainB",
    split_file=PROJECT_ROOT / "data" / "splits" / "train.txt",
    transform=PairedTransform(training=True),
)


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


underwater, reference = None, None

batch = next(iter(loader))

underwater = batch["underwater"].to(device)
reference = batch["reference"].to(device)


print(f"Underwater batch shape: {underwater.shape}")
print(f"Reference batch shape: {reference.shape}")

print(f"Underwater device: {underwater.device}")
print(f"Reference device: {reference.device}")

print("\nDataLoader + GPU test: PASSED")