import torch
from consequence_check import ConsequenceCheckModule


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ConsequenceCheckModule().to(device)

original = torch.rand(
    8, 3, 256, 256,
    device=device
)

enhanced = torch.rand(
    8, 3, 256, 256,
    device=device
)

outputs = model(
    original,
    enhanced
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Original:", original.shape)
print("Enhanced:", enhanced.shape)
print("Reconstructed:", outputs["reconstructed"].shape)
print("Consequence error:", outputs["consequence_error"].shape)
print("Consequence score:", outputs["consequence_score"].shape)

assert outputs["reconstructed"].shape == (
    8, 3, 256, 256
)

assert outputs["consequence_error"].shape == (8,)

assert outputs["consequence_score"].shape == (8,)

print("\nConsequence Check Module test: PASSED")
