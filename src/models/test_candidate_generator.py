import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from candidate_generator import EnhancementCandidateGenerator


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = EnhancementCandidateGenerator().to(device)

x = torch.rand(
    8, 3, 256, 256,
    device=device
)

condition = torch.rand(
    8, 128,
    device=device
)

outputs = model(
    x,
    condition
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

for name, output in outputs.items():
    print(
        f"{name}: {output.shape}"
    )

assert outputs["conservative"].shape == (
    8, 3, 256, 256
)

assert outputs["balanced"].shape == (
    8, 3, 256, 256
)

assert outputs["aggressive"].shape == (
    8, 3, 256, 256
)

# Verify all outputs stay in valid image range
for output in outputs.values():
    assert output.min() >= 0.0
    assert output.max() <= 1.0

print(
    "\nControlled Enhancement Candidate "
    "Generator test: PASSED"
)