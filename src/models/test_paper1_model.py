import torch
from paper1_model import Paper1Model


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Paper1Model().to(device)

x = torch.rand(
    8, 3, 256, 256,
    device=device
)

outputs = model(x)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Input:", x.shape)
print("Condition map:", outputs["condition_map"].shape)
print("Condition vector:", outputs["condition_vector"].shape)

for name, image in outputs["candidates"].items():
    print(f"{name}: {image.shape}")

for name, score in outputs["preservation_scores"].items():
    print(
        f"{name} preservation score:",
        score.shape
    )

print("Utilities:", outputs["utilities"].shape)
print(
    "Selection weights:",
    outputs["selection_weights"].shape
)
print(
    "Selected index:",
    outputs["selected_index"].shape
)
print(
    "Selected image:",
    outputs["selected_image"].shape
)

assert outputs["selected_image"].shape == (
    8, 3, 256, 256
)

assert outputs["utilities"].shape == (8, 3)

assert outputs["selection_weights"].shape == (8, 3)

assert outputs["selected_index"].shape == (8,)

print("\nFull Paper 1 model integration test: PASSED")
print(
    "Reconstructed:",
    outputs["reconstructed"].shape
)

print(
    "Consequence error:",
    outputs["consequence_error"].shape
)

print(
    "Consequence score:",
    outputs["consequence_score"].shape
)

assert outputs["reconstructed"].shape == (
    8, 3, 256, 256
)

assert outputs["consequence_error"].shape == (8,)

assert outputs["consequence_score"].shape == (8,)