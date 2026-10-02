import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from information_preservation import InformationPreservationModule


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = InformationPreservationModule().to(device)

original = torch.rand(
    8, 3, 256, 256,
    device=device
)

enhanced = torch.rand(
    8, 3, 256, 256,
    device=device
)

preservation_map, preservation_score = model(
    original,
    enhanced
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Original:", original.shape)
print("Enhanced:", enhanced.shape)
print("Preservation map:", preservation_map.shape)
print("Preservation score:", preservation_score.shape)

assert preservation_map.shape == (8, 1, 256, 256)
assert preservation_score.shape == (8,)

print("\nInformation Preservation Module test: PASSED")