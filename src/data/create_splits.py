from pathlib import Path
import random


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_A = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
    / "trainA"
)

TRAIN_B = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
    / "trainB"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

SEED = 42


# -----------------------------
# Verify dataset
# -----------------------------
input_files = sorted(
    file.name for file in TRAIN_A.glob("*.jpg")
)

reference_files = sorted(
    file.name for file in TRAIN_B.glob("*.jpg")
)

print(f"Input images  : {len(input_files)}")
print(f"Reference images: {len(reference_files)}")


if len(input_files) != len(reference_files):
    raise RuntimeError("Input/reference image counts do not match.")

if set(input_files) != set(reference_files):
    raise RuntimeError("Input/reference filenames do not match.")

print("Pair verification: PASSED")


# -----------------------------
# Shuffle reproducibly
# -----------------------------
random.seed(SEED)

filenames = input_files.copy()
random.shuffle(filenames)


# -----------------------------
# 80 / 10 / 10 split
# -----------------------------
total = len(filenames)

train_end = int(0.80 * total)
val_end = train_end + int(0.10 * total)

train_files = filenames[:train_end]
val_files = filenames[train_end:val_end]
test_files = filenames[val_end:]


# -----------------------------
# Save split manifests
# -----------------------------
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files,
}

for split_name, files in splits.items():
    output_file = SPLIT_DIR / f"{split_name}.txt"

    with output_file.open("w", encoding="utf-8") as f:
        for filename in files:
            f.write(filename + "\n")

    print(f"{split_name:5s}: {len(files)} pairs -> {output_file}")


print("\nSplit creation completed.")
print(f"Random seed: {SEED}")