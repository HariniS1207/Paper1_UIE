import torch
from selection_network import UtilitySelectionNetwork


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UtilitySelectionNetwork().to(device)

condition_vector = torch.rand(
    8, 128,
    device=device
)

candidates = {
    "conservative": torch.rand(
        8, 3, 256, 256,
        device=device
    ),
    "balanced": torch.rand(
        8, 3, 256, 256,
        device=device
    ),
    "aggressive": torch.rand(
        8, 3, 256, 256,
        device=device
    )
}

preservation_scores = {
    "conservative": torch.rand(8, device=device),
    "balanced": torch.rand(8, device=device),
    "aggressive": torch.rand(8, device=device)
}

utilities, selection_weights, selected_index = model(
    candidates,
    condition_vector,
    preservation_scores
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Utilities:", utilities.shape)
print("Selection weights:", selection_weights.shape)
print("Selected index:", selected_index.shape)
print("Selected candidates:", selected_index)

assert utilities.shape == (8, 3)
assert selection_weights.shape == (8, 3)
assert selected_index.shape == (8,)

assert torch.allclose(
    selection_weights.sum(dim=1),
    torch.ones(8, device=device),
    atol=1e-6
)

print("\nUtility Selection Network test: PASSED")
