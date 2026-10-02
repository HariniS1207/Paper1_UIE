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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
V2_CHECKPOINT_DIR = CHECKPOINT_DIR / "v2"

VAL_BATCH_SIZE = 8

CHECKPOINTS = [
    V2_CHECKPOINT_DIR / "paper1_v2_epoch_004.pth",
    V2_CHECKPOINT_DIR / "paper1_v2_epoch_005.pth",
    V2_CHECKPOINT_DIR / "paper1_v2_epoch_006.pth",
    V2_CHECKPOINT_DIR / "paper1_v2_epoch_007.pth",
    V2_CHECKPOINT_DIR / "paper1_v2_epoch_008.pth",
]


val_dataset = EUVPPairedDataset(
    input_dir=DATASET_ROOT / "trainA",
    reference_dir=DATASET_ROOT / "trainB",
    split_file=SPLIT_DIR / "val.txt",
    transform=PairedTransform(training=False),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


model = Paper1Model().to(DEVICE)
criterion = Paper1Loss().to(DEVICE)


print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("Validation samples:", len(val_dataset))
print()


for checkpoint_path in CHECKPOINTS:

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in tqdm(
            val_loader,
            desc=checkpoint_path.name
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

                outputs = model(underwater)

                losses = criterion(
                    outputs,
                    reference,
                    underwater
                )

            total_loss += losses["total"].item()

    avg_loss = total_loss / len(val_loader)

    print(
        f"{checkpoint_path.name} "
        f"Validation Loss: {avg_loss:.6f}"
    )

print("\nValidation completed.")