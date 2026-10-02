import torch
from torch.utils.data import DataLoader

from data.euvp_dataset import EUVPPairedDataset
from data.transforms import PairedTransform
from models.paper1_model import Paper1Model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIR = "data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet/trainA"
REFERENCE_DIR = "data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet/trainB"

VAL_LIST = "data/splits/val.txt"
CHECKPOINT = "checkpoints/v2/paper1_v2_epoch_006.pth"

BATCH_SIZE = 8


dataset = EUVPPairedDataset(
    input_dir=INPUT_DIR,
    reference_dir=REFERENCE_DIR,
    split_file=VAL_LIST,
    transform=PairedTransform(training=False)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


model = Paper1Model().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()


counts = torch.zeros(3, dtype=torch.long)
selected_counts = torch.zeros(3, dtype=torch.long)

total_images = 0


candidate_names = [
    "conservative",
    "balanced",
    "aggressive"
]


with torch.no_grad():

    for batch in loader:

        underwater = batch["underwater"].to(
            DEVICE,
            non_blocking=True
        )

        reference = batch["reference"].to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(underwater)

        candidates = outputs["candidates"]

        candidate_losses = []

        for name in candidate_names:

            loss = torch.mean(
                torch.abs(
                    candidates[name] - reference
                ),
                dim=(1, 2, 3)
            )

            candidate_losses.append(loss)

        candidate_losses = torch.stack(
            candidate_losses,
            dim=1
        )

        # Candidate with lowest reference L1
        target_indices = torch.argmin(
            candidate_losses,
            dim=1
        )

        for i in range(3):
            counts[i] += (
                target_indices == i
            ).sum().cpu()

        # Candidate actually selected by model
        selected_indices = outputs["selected_index"]

        for i in range(3):
            selected_counts[i] += (
                selected_indices == i
            ).sum().cpu()

        total_images += underwater.size(0)


print()
print("=" * 60)
print("PAPER 1 V2 — VALIDATION SELECTION ANALYSIS")
print("=" * 60)

print(f"Validation images: {total_images}")

print()
print("REFERENCE-L1 BEST CANDIDATE")
print("-" * 60)

for i, name in enumerate(candidate_names):

    percentage = (
        100.0 *
        counts[i].item() /
        total_images
    )

    print(
        f"{name:12s}: "
        f"{counts[i].item():3d} "
        f"({percentage:6.2f}%)"
    )


print()
print("MODEL SELECTED CANDIDATE")
print("-" * 60)

for i, name in enumerate(candidate_names):

    percentage = (
        100.0 *
        selected_counts[i].item() /
        total_images
    )

    print(
        f"{name:12s}: "
        f"{selected_counts[i].item():3d} "
        f"({percentage:6.2f}%)"
    )

print("=" * 60)