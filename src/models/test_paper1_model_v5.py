"""Synthetic V5 model/objective smoke check; this never loads dataset splits."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "losses"))

from models.paper1_model_v5 import Paper1ModelV5
from paper1_loss_v5 import Paper1LossV5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Paper1ModelV5().to(device)
    criterion = Paper1LossV5({"median": [0.1, 0.1, 0.1], "iqr": [0.1, 0.1, 0.1]}).to(device)
    x = torch.rand(1, 3, 256, 256, device=device)
    reference = torch.rand_like(x)
    outputs = model(x)
    losses = criterion(outputs, reference, x)
    target, components = criterion.candidate_targets(outputs, reference)
    assert outputs["candidates"]["conservative"].shape == (1, 3, 256, 256)
    assert target.shape == (1,) and int(target.item()) in (0, 1, 2)
    assert all(torch.isfinite(v).all() for v in losses.values())
    assert all(torch.isfinite(v).all() for v in components.values())
    losses["total"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("V5 synthetic forward/backward passed on", device)


if __name__ == "__main__":
    main()
