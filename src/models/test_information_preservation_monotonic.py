import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from models.information_preservation import InformationPreservationModule


def make_reference_image(size=(256, 256), device=None):
    h, w = size
    y = torch.linspace(0.0, 1.0, steps=h, device=device).view(-1, 1)
    x = torch.linspace(0.0, 1.0, steps=w, device=device).view(1, -1)
    base = 0.45 + 0.25 * torch.sin(2.5 * torch.pi * x) + 0.2 * torch.cos(2.1 * torch.pi * y)
    color_r = 0.2 + 0.5 * (base + 0.15 * torch.sin(4.0 * torch.pi * x))
    color_g = 0.4 + 0.35 * base
    color_b = 0.6 + 0.25 * (1.0 - base)
    image = torch.stack([color_r, color_g, color_b], dim=0).unsqueeze(0)
    image = torch.clamp(image, 0.0, 1.0)
    image = image.repeat(4, 1, 1, 1)
    return image


def apply_blur(x, kernel_size=9):
    pad = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype) / (kernel_size ** 2)
    return F.conv2d(x, kernel.expand(3, 1, kernel_size, kernel_size), padding=pad, groups=3)


def apply_noise(x, sigma):
    return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)


def apply_color_cast(x, shift):
    out = x.clone()
    out[:, 0] = torch.clamp(out[:, 0] + shift, 0.0, 1.0)
    out[:, 1] = torch.clamp(out[:, 1] - 0.10, 0.0, 1.0)
    return out


def run_case(device):
    original = make_reference_image(device=device)
    mild = apply_blur(original, kernel_size=5)
    mild = apply_noise(mild, 0.02)
    strong = apply_blur(original, kernel_size=11)
    strong = apply_noise(strong, 0.06)
    severe = torch.clamp(0.5 * original + 0.5 * torch.rand_like(original), 0.0, 1.0)
    with torch.no_grad():
        _, id_score = InformationPreservationModule().to(device)(original, original)
        _, mild_score = InformationPreservationModule().to(device)(original, mild)
        _, strong_score = InformationPreservationModule().to(device)(original, strong)
        _, severe_score = InformationPreservationModule().to(device)(original, severe)

    values = {
        "identity": id_score.mean().item(),
        "mild": mild_score.mean().item(),
        "strong": strong_score.mean().item(),
        "severe": severe_score.mean().item(),
    }
    print("Preservation monotonicity test on", device)
    for k, v in values.items():
        print(f"{k}: {v:.6f}")

    assert values["identity"] > values["mild"] > values["strong"] > values["severe"]
    return values


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_case(device)

    if torch.cuda.is_available():
        run_case(torch.device("cuda"))

    print("Information preservation monotonicity test: PASSED")


if __name__ == "__main__":
    main()
