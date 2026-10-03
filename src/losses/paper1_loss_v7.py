"""V7 per-image candidate-relative rank targets and selector-aware tradeoffs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Paper1LossV7(nn.Module):
    candidate_names = ("conservative", "balanced", "aggressive")

    def __init__(self, lambda_recon=1.0, lambda_preserve=0.2,
                 lambda_consequence=0.2, lambda_selection=0.5):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_preserve = lambda_preserve
        self.lambda_consequence = lambda_consequence
        self.lambda_selection = lambda_selection

    @staticmethod
    def candidate_objectives(outputs, reference):
        names = Paper1LossV7.candidate_names
        candidates = outputs["candidates"]
        fidelity = torch.stack([
            F.l1_loss(candidates[n].float(), reference.float(), reduction="none").mean((1, 2, 3))
            for n in names
        ], dim=1)
        preservation = torch.stack([
            1.0 - outputs["preservation_scores"][n].float() for n in names
        ], dim=1)
        consequence = torch.stack([
            outputs["candidate_consequence_errors"][n].float() for n in names
        ], dim=1)
        return torch.stack((fidelity, preservation, consequence), dim=2)

    @staticmethod
    def relative_rank_targets(objectives):
        """Equal-weight per-image Borda rank over fidelity/preservation/consequence.

        For every image and objective, the three candidates receive normalized
        average ranks in [0, 1] (ties share their average rank). The lowest mean
        rank is the training target. No class quotas or train-global metric scale
        are used.
        """
        batch, candidates, dimensions = objectives.shape
        if candidates != 3 or dimensions != 3:
            raise ValueError("V7 requires three candidates and three objective axes")
        values = objectives.detach().float()
        less_better = values.unsqueeze(2) > values.unsqueeze(1)
        equal = values.unsqueeze(2) == values.unsqueeze(1)
        ranks = less_better.sum(dim=2).float() + 0.5 * (equal.sum(dim=2).float() - 1.0)
        normalized_ranks = ranks / 2.0
        mean_rank = normalized_ranks.mean(dim=2)
        target = mean_rank.argmin(dim=1)
        return target, {"mean_rank": mean_rank, "normalized_ranks": normalized_ranks}

    def candidate_targets(self, outputs, reference):
        objectives = self.candidate_objectives(outputs, reference)
        target, ranks = self.relative_rank_targets(objectives)
        return target, {"objectives": objectives, **ranks}

    def forward(self, outputs, reference, original):
        weights = outputs["selection_weights"].float()
        objectives = self.candidate_objectives(outputs, reference)
        target, _ = self.relative_rank_targets(objectives)
        # Expected terms give selector parameters a direct gradient for all three
        # research objectives; this avoids the V4-V6 hard-argmax consequence path.
        reconstruction = (weights * objectives[:, :, 0]).sum(1).mean()
        preservation = (weights * objectives[:, :, 1]).sum(1).mean()
        consequence = (weights * objectives[:, :, 2]).sum(1).mean()
        selection = F.cross_entropy(outputs["utilities"].float(), target)
        total = (self.lambda_recon * reconstruction
                 + self.lambda_preserve * preservation
                 + self.lambda_consequence * consequence
                 + self.lambda_selection * selection)
        return {"total": total, "reconstruction": reconstruction,
                "preservation": preservation, "consequence": consequence,
                "selection": selection}
