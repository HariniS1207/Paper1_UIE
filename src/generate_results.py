from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from paper1_model import Paper1Model


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
    / "paper1_v2_epoch_008.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "visuals"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


dataset = EUVPPairedDataset(
    input_dir=DATASET_ROOT / "trainA",
    reference_dir=DATASET_ROOT / "trainB",
    split_file=SPLIT_DIR / "test.txt",
    transform=PairedTransform(training=False),
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)


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


with torch.no_grad():

    batch = next(iter(loader))

    underwater = batch["underwater"].to(DEVICE)

    outputs = model(underwater)

    selected = outputs["selected_image"]

    conservative = outputs["candidates"]["conservative"]
    balanced = outputs["candidates"]["balanced"]
    aggressive = outputs["candidates"]["aggressive"]

    for i in range(4):

        filename = batch["filename"][i]

        save_image(
            underwater[i],
            OUTPUT_DIR / f"{i}_underwater.png"
        )

        save_image(
            conservative[i],
            OUTPUT_DIR / f"{i}_conservative.png"
        )

        save_image(
            balanced[i],
            OUTPUT_DIR / f"{i}_balanced.png"
        )

        save_image(
            aggressive[i],
            OUTPUT_DIR / f"{i}_aggressive.png"
        )

        save_image(
            selected[i],
            OUTPUT_DIR / f"{i}_selected.png"
        )

        save_image(
            batch["reference"][i],
            OUTPUT_DIR / f"{i}_reference.png"
        )

        print(
            f"{filename} -> "
            f"selected index = "
            f"{outputs['selected_index'][i].item()}"
        )

print("\nVisual results saved to:")
print(OUTPUT_DIR)