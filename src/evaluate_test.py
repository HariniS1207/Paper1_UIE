from pathlib import Path
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, rgb2hsv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from paper1_model import Paper1Model


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
# UIQM
# ============================================================

def eme(channel, block_size=8):

    h, w = channel.shape

    total = 0.0
    count = 0

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):

            block = channel[
                y:min(y + block_size, h),
                x:min(x + block_size, w)
            ]

            if block.size == 0:
                continue

            max_val = np.max(block)
            min_val = np.min(block)

            if min_val <= 0:
                min_val = 1e-6

            if max_val <= 0:
                continue

            total += np.log(max_val / min_val)
            count += 1

    if count == 0:
        return 0.0

    return (2.0 / count) * total


def uicm(image):

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    RG = R - G
    YB = 0.5 * (R + G) - B

    rg_mean = np.mean(RG)
    rg_std = np.std(RG)

    yb_mean = np.mean(YB)
    yb_std = np.std(YB)

    return (
        -0.0268 * rg_mean
        + 0.1586 * rg_std
        - 0.0786 * yb_mean
        + 0.0112 * yb_std
    )


def uism(image):

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    return (
        0.299 * eme(R)
        + 0.587 * eme(G)
        + 0.114 * eme(B)
    )


def uiconm(image):

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    return eme(gray)


def uiqm(image):

    c1 = 0.0282
    c2 = 0.2953
    c3 = 0.6761

    color = uicm(image)
    sharpness = uism(image)
    contrast = uiconm(image)

    return (
        c1 * color
        + c2 * sharpness
        + c3 * contrast
    )


# ============================================================
# UCIQE
# ============================================================

def uciqe(image):

    lab = rgb2lab(image)

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    chroma = np.sqrt(
        a ** 2 + b ** 2
    )

    sigma_c = np.std(chroma)

    contrast_l = (
        np.percentile(L, 99)
        - np.percentile(L, 1)
    ) / 100.0

    hsv = rgb2hsv(image)

    saturation = hsv[:, :, 1]

    mean_s = np.mean(saturation)

    return (
        0.4680 * sigma_c
        + 0.2745 * contrast_l
        + 0.2576 * mean_s
    )


# ============================================================
# Metric helper
# ============================================================

def calculate_metrics(
    prediction,
    reference
):

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )

    reference = np.clip(
        reference,
        0.0,
        1.0
    )

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=2,
        data_range=1.0
    )

    uiqm_score = uiqm(prediction)
    uciqe_score = uciqe(prediction)

    return (
        psnr,
        ssim,
        uiqm_score,
        uciqe_score
    )


# ============================================================
# Dataset
# ============================================================

dataset = EUVPPairedDataset(
    input_dir=DATASET_ROOT / "trainA",
    reference_dir=DATASET_ROOT / "trainB",
    split_file=SPLIT_DIR / "test.txt",
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
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Storage
# ============================================================

names = [
    "Original",
    "Conservative",
    "Balanced",
    "Aggressive",
    "Selected"
]

metrics = {
    name: {
        "psnr": [],
        "ssim": [],
        "uiqm": [],
        "uciqe": []
    }
    for name in names
}

selection_counts = {
    0: 0,
    1: 0,
    2: 0
}

# Number of images where enhancement improves
# over original baseline.
improvement_counts = {
    "Conservative": {"psnr": 0, "ssim": 0},
    "Balanced": {"psnr": 0, "ssim": 0},
    "Aggressive": {"psnr": 0, "ssim": 0},
    "Selected": {"psnr": 0, "ssim": 0}
}


# ============================================================
# Evaluation
# ============================================================

with torch.no_grad():

    for batch in tqdm(
        loader,
        desc="Evaluating test set"
    ):

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

        selected = outputs["selected_image"]

        # ----------------------------------------------------
        # Selection statistics
        # ----------------------------------------------------

        indices = (
            outputs["selected_index"]
            .detach()
            .cpu()
            .numpy()
        )

        for idx in indices:
            selection_counts[int(idx)] += 1

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        original_np = (
            underwater
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
        )

        reference_np = (
            reference
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
        )

        conservative_np = (
            candidates["conservative"]
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
        )

        balanced_np = (
            candidates["balanced"]
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
        )

        aggressive_np = (
            candidates["aggressive"]
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
        )

        selected_np = (
            selected
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
        )

        # ----------------------------------------------------
        # Per-image evaluation
        # ----------------------------------------------------

        for i in range(
            original_np.shape[0]
        ):

            reference_img = np.clip(
                reference_np[i],
                0.0,
                1.0
            )

            images = {
                "Original": original_np[i],
                "Conservative": conservative_np[i],
                "Balanced": balanced_np[i],
                "Aggressive": aggressive_np[i],
                "Selected": selected_np[i]
            }

            image_results = {}

            for name, image in images.items():

                result = calculate_metrics(
                    image,
                    reference_img
                )

                image_results[name] = result

                metrics[name]["psnr"].append(
                    result[0]
                )

                metrics[name]["ssim"].append(
                    result[1]
                )

                metrics[name]["uiqm"].append(
                    result[2]
                )

                metrics[name]["uciqe"].append(
                    result[3]
                )

            # ------------------------------------------------
            # Improvement over original
            # ------------------------------------------------

            original_psnr = image_results[
                "Original"
            ][0]

            original_ssim = image_results[
                "Original"
            ][1]

            for name in [
                "Conservative",
                "Balanced",
                "Aggressive",
                "Selected"
            ]:

                if image_results[name][0] > original_psnr:
                    improvement_counts[name]["psnr"] += 1

                if image_results[name][1] > original_ssim:
                    improvement_counts[name]["ssim"] += 1


# ============================================================
# Final Results
# ============================================================

print()
print("=" * 75)
print("PAPER 1 V2 — CANDIDATE-WISE TEST ANALYSIS")
print("=" * 75)

print(
    f"Checkpoint : {CHECKPOINT.name}"
)

print(
    f"Test images: {len(dataset)}"
)

print()

print(
    f"{'Output':<15}"
    f"{'PSNR':>12}"
    f"{'SSIM':>12}"
    f"{'UIQM':>12}"
    f"{'UCIQE':>12}"
)

print("-" * 75)

for name in names:

    mean_psnr = np.mean(
        metrics[name]["psnr"]
    )

    mean_ssim = np.mean(
        metrics[name]["ssim"]
    )

    mean_uiqm = np.mean(
        metrics[name]["uiqm"]
    )

    mean_uciqe = np.mean(
        metrics[name]["uciqe"]
    )

    print(
        f"{name:<15}"
        f"{mean_psnr:>12.4f}"
        f"{mean_ssim:>12.4f}"
        f"{mean_uiqm:>12.4f}"
        f"{mean_uciqe:>12.4f}"
    )


# ============================================================
# Improvement Statistics
# ============================================================

print()
print("=" * 75)
print("IMPROVEMENT OVER ORIGINAL UNDERWATER INPUT")
print("=" * 75)

print(
    f"{'Output':<15}"
    f"{'PSNR Improved':>20}"
    f"{'SSIM Improved':>20}"
)

print("-" * 75)

for name in [
    "Conservative",
    "Balanced",
    "Aggressive",
    "Selected"
]:

    psnr_count = improvement_counts[
        name
    ]["psnr"]

    ssim_count = improvement_counts[
        name
    ]["ssim"]

    print(
        f"{name:<15}"
        f"{psnr_count:>10}/{len(dataset):<9}"
        f"{100 * psnr_count / len(dataset):>7.2f}%"
        f"{ssim_count:>10}/{len(dataset):<9}"
        f"{100 * ssim_count / len(dataset):>7.2f}%"
    )


# ============================================================
# Selection Distribution
# ============================================================

print()
print("=" * 75)
print("SELECTION DISTRIBUTION")
print("=" * 75)

total = sum(
    selection_counts.values()
)

print(
    f"Conservative (0): "
    f"{selection_counts[0]} "
    f"({100 * selection_counts[0] / total:.2f}%)"
)

print(
    f"Balanced (1): "
    f"{selection_counts[1]} "
    f"({100 * selection_counts[1] / total:.2f}%)"
)

print(
    f"Aggressive (2): "
    f"{selection_counts[2]} "
    f"({100 * selection_counts[2] / total:.2f}%)"
)

print("=" * 75)