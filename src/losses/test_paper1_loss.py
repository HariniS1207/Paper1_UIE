import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "src" / "models"

sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper1_model import Paper1Model
from paper1_loss import Paper1Loss


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Paper1Model().to(device)
criterion = Paper1Loss().to(device)

original = torch.rand(
    2, 3, 256, 256,
    device=device
)

reference = torch.rand(
    2, 3, 256, 256,
    device=device
)

outputs = model(original)

losses = criterion(
    outputs,
    reference,
    original
)

print("Device:", device)

for name, value in losses.items():
    print(f"{name} loss: {value.item():.6f}")

assert torch.isfinite(losses["total"])
assert torch.isfinite(losses["reconstruction"])
assert torch.isfinite(losses["preservation"])
assert torch.isfinite(losses["consequence"])
assert torch.isfinite(losses["selection"])

losses["total"].backward()

print("\nPaper 1 Loss test: PASSED")
print("Backward pass: PASSED")