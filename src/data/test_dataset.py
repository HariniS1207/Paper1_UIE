from pathlib import Path

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


print(f"Dataset size: {len(dataset)}")

sample = dataset[0]

print(f"Filename: {sample['filename']}")
print(f"Underwater type: {type(sample['underwater'])}")
print(f"Reference type: {type(sample['reference'])}")
print(f"Underwater shape: {sample['underwater'].shape}")
print(f"Reference shape: {sample['reference'].shape}")
print("\nDataset test: PASSED")