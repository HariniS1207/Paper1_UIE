import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "losses"))

from models.paper1_model_v4 import Paper1ModelV4
from paper1_loss import Paper1Loss



def diversity_report(original, candidates):
    residuals = {
        name: image - original
        for name, image in candidates.items()
    }
    names = ("conservative", "balanced", "aggressive")
    report = {}
    for index, left in enumerate(names):
        report[left] = {
            "residual_norm": float(torch.linalg.vector_norm(residuals[left].float(), dim=(1, 2, 3)).mean().item())
        }
        for right in names[index + 1:]:
            distance = torch.abs(candidates[left] - candidates[right]).mean().item()
            cosine = torch.nn.functional.cosine_similarity(
                residuals[left].flatten(1).float(),
                residuals[right].flatten(1).float(),
                dim=1,
                eps=1e-8,
            ).mean().item()
            report[f"{left}_vs_{right}"] = {
                "mean_absolute_candidate_distance": float(distance),
                "residual_cosine_similarity": float(cosine),
            }
    return report


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Paper1ModelV4().to(device)
    model.train()

    original = 0.2 + 0.6 * torch.rand(4, 3, 256, 256, device=device)
    reference = torch.rand_like(original)
    outputs = model(original)
    candidates = outputs["candidates"]

    for name in ("conservative", "balanced", "aggressive"):
        image = candidates[name]
        assert image.shape == (4, 3, 256, 256), (name, image.shape)
        assert torch.isfinite(image).all(), name
        assert image.min().item() >= 0.0 and image.max().item() <= 1.0, name

    diversity = diversity_report(original, candidates)
    for key, values in diversity.items():
        if "mean_absolute_candidate_distance" in values:
            assert values["mean_absolute_candidate_distance"] > 1e-5, (key, values)
            assert values["residual_cosine_similarity"] < 0.98, (key, values)

    criterion = Paper1Loss().to(device)
    losses = criterion(outputs, reference, original)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total"].backward()

    for name in ("conservative", "balanced", "aggressive"):
        branch_parameters = [
            parameter
            for parameter_name, parameter in model.candidate_generator.named_parameters()
            if f"{name}" in parameter_name
        ]
        assert branch_parameters
        assert any(parameter.grad is not None for parameter in branch_parameters)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in branch_parameters
        )

    assert torch.isfinite(outputs["selected_image"]).all()
    assert outputs["selected_index"].min().item() >= 0
    assert outputs["selected_index"].max().item() < 3

    print("device", device)
    print("candidate_gains", {name: float(value.detach().item()) for name, value in model.candidate_generator.candidate_gains().items()})
    print("candidate_diversity", diversity)
    print("losses", {name: float(value.detach().item()) for name, value in losses.items()})
    print("candidate_outputs_shape", tuple(candidates["balanced"].shape))
    print("finite_outputs=True; range=[0,1]; branch_gradients=finite_nonzero; selector_valid=True")
    print("V4 candidate generator pretraining tests: PASSED")


if __name__ == "__main__":
    main()
