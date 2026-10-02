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


# ==================================================
# CONFIG
# ==================================================

BATCH_SIZE = 8
START_EPOCH = 3
END_EPOCH = 8
LEARNING_RATE = 1e-4

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
CHECKPOINT_DIR.mkdir(exist_ok=True)

RESUME_CHECKPOINT = (
    CHECKPOINT_DIR / "paper1_epoch_003.pth"
)


# ==================================================
# DATASET
# ==================================================

train_dataset = EUVPPairedDataset(
    input_dir=DATASET_ROOT / "trainA",
    reference_dir=DATASET_ROOT / "trainB",
    split_file=SPLIT_DIR / "train.txt",
    transform=PairedTransform(training=True),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ==================================================
# MODEL
# ==================================================

model = Paper1Model().to(DEVICE)

criterion = Paper1Loss().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ==================================================
# RESUME FROM EPOCH 3
# ==================================================

if RESUME_CHECKPOINT.exists():

    print(
        f"Loading checkpoint: {RESUME_CHECKPOINT}"
    )

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    print(
        f"Resumed from epoch {checkpoint['epoch']}"
    )


# ==================================================
# TRAIN
# ==================================================

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("Training samples:", len(train_dataset))
print("Batch size:", BATCH_SIZE)
print(
    f"Epoch range: {START_EPOCH + 1} → {END_EPOCH}"
)
print("Mixed precision: ENABLED")
print()


for epoch in range(START_EPOCH + 1, END_EPOCH + 1):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{END_EPOCH}"
    )

    for batch in progress_bar:

        underwater = batch["underwater"].to(
            DEVICE,
            non_blocking=True
        )

        reference = batch["reference"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
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

            total_loss = losses["total"]

        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            total_loss.item()
        )

        progress_bar.set_postfix(
            loss=f"{total_loss.item():.4f}"
        )

    epoch_loss = (
        running_loss
        / len(train_loader)
    )

    print(
        f"\nEpoch [{epoch}/{END_EPOCH}] "
        f"Loss: {epoch_loss:.6f}"
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"paper1_epoch_{epoch:03d}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "loss": epoch_loss,
        },
        checkpoint_path
    )

    print(
        f"Checkpoint saved: {checkpoint_path}\n"
    )

print("Training completed.")