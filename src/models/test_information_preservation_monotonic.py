import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from models.information_preservation import InformationPreservationModule


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
)


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


def apply_noise(x, sigma, seed=42):
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return torch.clamp(x + sigma * noise, 0.0, 1.0)


def apply_luma_preserving_color_cast(x, shift=0.05):
    out = x.clone()
    out[:, 0] = out[:, 0] + shift
    out[:, 1] = out[:, 1] - (0.299 / 0.587) * shift
    return out


def apply_edge_destruction(x):
    low_resolution = F.interpolate(x, size=(16, 16), mode="area")
    return F.interpolate(low_resolution, size=x.shape[-2:], mode="bilinear", align_corners=False)


def make_cases(original):
    contrast = torch.clamp((original - 0.5) * 1.25 + 0.5, 0.0, 1.0)
    color_cast = apply_luma_preserving_color_cast(original)
    blur_kernel_size = 9
    blurred = apply_blur(original, kernel_size=blur_kernel_size)
    noisy = apply_noise(original, sigma=0.04, seed=42)
    edge_destroyed = apply_edge_destruction(original)
    combined = apply_noise(
        torch.clamp(
            apply_blur(color_cast, kernel_size=blur_kernel_size)
            + 0.06,
            0.0,
            1.0,
        ),
        sigma=0.025,
        seed=84,
    )
    return {
        "identity": original,
        "brightness_plus": torch.clamp(original + 0.08, 0.0, 1.0),
        "brightness_minus": torch.clamp(original - 0.08, 0.0, 1.0),
        "contrast": contrast,
        "color_cast": color_cast,
        "blur": blurred,
        "noise": noisy,
        "edge_destruction": edge_destroyed,
        "combined_distortion": combined,
    }


def report_case(module, original, candidate, name):
    diagnostics = module.compute_distortions(original, candidate)
    values = {key: float(value.mean().item()) for key, value in diagnostics.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise AssertionError(f"Non-finite preservation result for {name}: {values}")
    print(
        f"{name}: score={values['preservation_score']:.6f}, "
        f"raw_luma={values['raw_luma_distortion']:.6f}, "
        f"structure={values['standardized_structure_distortion']:.6f}, "
        f"chroma={values['chroma_distortion']:.6f}, "
        f"sobel={values['sobel_distortion']:.6f}"
    )
    return values


def test_synthetic_cases(device):
    torch.manual_seed(42)
    original = make_reference_image(device=device)
    module = InformationPreservationModule().to(device).eval()

    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    assert all(not parameter.requires_grad for parameter in module.parameters())

    results = {
        name: report_case(module, original, candidate, name)
        for name, candidate in make_cases(original).items()
    }

    identity = results["identity"]
    assert abs(identity["preservation_score"] - 1.0) < 1e-6
    assert all(identity[key] < 1e-7 for key in identity if key != "preservation_score")
    assert all(
        result["preservation_score"] < identity["preservation_score"]
        for name, result in results.items()
        if name != "identity"
    )

    for name in ("brightness_plus", "brightness_minus"):
        assert results[name]["raw_luma_distortion"] > 0.0

    assert results["contrast"]["raw_luma_distortion"] > 0.0
    assert results["color_cast"]["chroma_distortion"] > 0.0
    assert results["blur"]["standardized_structure_distortion"] > 0.0
    assert results["noise"]["sobel_distortion"] > 0.0
    assert results["edge_destruction"]["sobel_distortion"] > 0.0
    assert results["combined_distortion"]["raw_luma_distortion"] > 0.0
    assert results["combined_distortion"]["standardized_structure_distortion"] > 0.0
    assert results["combined_distortion"]["chroma_distortion"] > 0.0
    assert results["combined_distortion"]["sobel_distortion"] > 0.0

    enhanced = make_cases(original)["combined_distortion"].detach().clone().requires_grad_(True)
    objective = module.compute_distortions(original, enhanced)["total_distortion"].mean()
    objective.backward()
    gradient = enhanced.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    print(
        f"gradient_check: finite=True, nonzero=True, "
        f"L1={gradient.abs().sum().item():.6f}"
    )
    print("synthetic_checks: identity_maximal=True, all_finite=True, non_trainable=True")
    return results


def test_real_euvp_image(device):
    split_path = PROJECT_ROOT / "data" / "splits" / "train.txt"
    with split_path.open("r", encoding="utf-8") as handle:
        filename = next(line.strip() for line in handle if line.strip())
    image_path = DATASET_ROOT / "trainA" / filename
    with Image.open(image_path) as image:
        original = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)

    module = InformationPreservationModule().to(device).eval()
    print(f"real_euvp_image: {image_path.relative_to(PROJECT_ROOT)}")
    identity = report_case(module, original, original, "real_identity")
    brightness = report_case(
        module,
        original,
        torch.clamp(original + 0.05, 0.0, 1.0),
        "real_brightness_plus",
    )
    assert abs(identity["preservation_score"] - 1.0) < 1e-6
    assert brightness["raw_luma_distortion"] > 0.0
    assert brightness["preservation_score"] < identity["preservation_score"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Controlled fixed-preservation tests on", device)
    test_synthetic_cases(device)
    test_real_euvp_image(device)
    print("Fixed-preservation controlled tests: PASSED")


if __name__ == "__main__":
    main()
