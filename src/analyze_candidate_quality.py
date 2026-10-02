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

candidate_names = [
    "conservative",
    "balanced",
    "aggressive"
]


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

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


stats = {
    name: {
        "l1": 0.0,
        "preservation": 0.0,
        "consequence": 0.0,
        "count": 0
    }
    for name in candidate_names
}


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

        # --------------------------------------------------
        # Evaluate each candidate independently
        # --------------------------------------------------

        for name in candidate_names:

            candidate = candidates[name]

            # 1. Reference L1
            l1 = torch.mean(
                torch.abs(candidate - reference),
                dim=(1, 2, 3)
            )

            # 2. Information preservation
            preservation_map, preservation_score = (
                model.information_preservation(
                    underwater,
                    candidate
                )
            )

            # 3. Consequence reconstruction
            consequence_output = model.consequence_check(
            underwater,
            candidate
            )

            consequence_error = consequence_output["consequence_error"]

            stats[name]["l1"] += l1.sum().item()
            stats[name]["preservation"] += (
                preservation_score.sum().item()
            )
            stats[name]["consequence"] += (
                consequence_error.sum().item()
            )

            stats[name]["count"] += underwater.size(0)


print()
print("=" * 75)
print("PAPER 1 V2 — CANDIDATE QUALITY ANALYSIS")
print("=" * 75)

print(
    f"{'Candidate':<15}"
    f"{'Reference L1':>15}"
    f"{'Preservation':>15}"
    f"{'Consequence':>15}"
)

print("-" * 75)

for name in candidate_names:

    count = stats[name]["count"]

    avg_l1 = stats[name]["l1"] / count
    avg_preservation = stats[name]["preservation"] / count
    avg_consequence = stats[name]["consequence"] / count

    print(
        f"{name:<15}"
        f"{avg_l1:>15.6f}"
        f"{avg_preservation:>15.6f}"
        f"{avg_consequence:>15.6f}"
    )

print("=" * 75)