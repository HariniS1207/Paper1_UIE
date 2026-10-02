import torch

from src.models.information_preservation import (
    InformationPreservationModule
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    module = InformationPreservationModule().to(device)
    module.eval()

    # Synthetic original image
    original = torch.rand(
        4, 3, 256, 256,
        device=device
    )

    # Different distortion levels
    conservative = torch.clamp(
        original * 0.95,
        0.0,
        1.0
    )

    balanced = torch.clamp(
        original * 0.75,
        0.0,
        1.0
    )

    aggressive = torch.clamp(
        original * 0.40,
        0.0,
        1.0
    )

    with torch.no_grad():

        _, p_conservative = module(
            original,
            conservative
        )

        _, p_balanced = module(
            original,
            balanced
        )

        _, p_aggressive = module(
            original,
            aggressive
        )

    print("\nInformation Preservation V3")
    print("--------------------------------")

    print(
        "Conservative:",
        p_conservative.mean().item()
    )

    print(
        "Balanced:",
        p_balanced.mean().item()
    )

    print(
        "Aggressive:",
        p_aggressive.mean().item()
    )


if __name__ == "__main__":
    main()