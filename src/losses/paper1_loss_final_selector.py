"""Per-image soft Borda targets for the final selector-only experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Paper1LossFinalSelector(nn.Module):
    names = ("conservative", "balanced", "aggressive")

    def __init__(self, target_temperature=0.25, lambda_recon=1.0,
                 lambda_preserve=0.2, lambda_consequence=0.2, lambda_selection=0.5):
        super().__init__()
        self.target_temperature = float(target_temperature)
        self.lambda_recon, self.lambda_preserve = lambda_recon, lambda_preserve
        self.lambda_consequence, self.lambda_selection = lambda_consequence, lambda_selection

    @classmethod
    def candidate_objectives(cls, outputs, reference):
        candidates = outputs["candidates"]
        fidelity = torch.stack([
            F.l1_loss(candidates[n].float(), reference.float(), reduction="none").mean((1, 2, 3))
            for n in cls.names
        ], dim=1)
        preservation = torch.stack([1.0 - outputs["preservation_scores"][n].float() for n in cls.names], dim=1)
        consequence = torch.stack([outputs["candidate_consequence_errors"][n].float() for n in cls.names], dim=1)
        return torch.stack((fidelity, preservation, consequence), dim=2)

    @staticmethod
    def normalized_ranks(objectives):
        # Average ranks handle ties; normalize each image/objective independently.
        values = objectives.detach().float()
        less_better = values.unsqueeze(2) > values.unsqueeze(1)
        equal = values.unsqueeze(2) == values.unsqueeze(1)
        ranks = less_better.sum(dim=2).float() + 0.5 * (equal.sum(dim=2).float() - 1.0)
        return ranks / 2.0

    def soft_targets(self, objectives):
        ranks = self.normalized_ranks(objectives)
        mean_rank = ranks.mean(dim=2)
        targets = torch.softmax(-mean_rank / self.target_temperature, dim=1)
        return targets, {"normalized_ranks": ranks, "mean_rank": mean_rank}

    def forward(self, outputs, reference, original):
        weights = outputs["selection_weights"].float()
        objectives = self.candidate_objectives(outputs, reference)
        target, _ = self.soft_targets(objectives)
        axes = (weights.unsqueeze(-1) * objectives).sum(dim=1).mean(dim=0)
        selection = -(target * F.log_softmax(outputs["utilities"].float(), dim=1)).sum(dim=1).mean()
        total = (self.lambda_recon * axes[0] + self.lambda_preserve * axes[1]
                 + self.lambda_consequence * axes[2] + self.lambda_selection * selection)
        return {"total": total, "reconstruction": axes[0], "preservation": axes[1],
                "consequence": axes[2], "selection": selection}
