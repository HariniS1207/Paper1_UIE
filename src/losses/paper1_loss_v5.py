"""V5 loss: V4 losses with a train-calibrated multi-objective selector target."""

import torch
import torch.nn.functional as F

from paper1_loss import Paper1Loss


class Paper1LossV5(Paper1Loss):
    names = ("conservative", "balanced", "aggressive")

    def __init__(self, calibration):
        super().__init__()
        self.register_buffer("median", torch.as_tensor(calibration["median"], dtype=torch.float32))
        self.register_buffer("iqr", torch.as_tensor(calibration["iqr"], dtype=torch.float32))

    def candidate_targets(self, outputs, reference):
        candidates = outputs["candidates"]
        l_ref = torch.stack([
            F.l1_loss(candidates[n], reference, reduction="none").mean((1, 2, 3))
            for n in self.names
        ], dim=1)
        l_pres = torch.stack([
            1.0 - outputs["preservation_scores"][n] for n in self.names
        ], dim=1)
        l_cons = torch.stack([
            outputs["candidate_consequence_errors"][n] for n in self.names
        ], dim=1)
        raw = torch.stack((l_ref, l_pres, l_cons), dim=2)
        normalized = (raw - self.median.view(1, 1, 3)) / self.iqr.view(1, 1, 3)
        score = normalized.mean(dim=2)
        return score.argmin(dim=1), {"reference": l_ref, "preservation": l_pres, "consequence": l_cons, "score": score}

    def forward(self, outputs, reference, original):
        candidates = outputs["candidates"]
        weights = outputs["selection_weights"]
        l_ref = torch.stack([
            F.l1_loss(candidates[n], reference, reduction="none").mean((1, 2, 3))
            for n in self.names
        ], dim=1)
        reconstruction = (weights * l_ref).sum(1).mean()
        preservation = torch.stack([1.0 - outputs["preservation_scores"][n] for n in self.names], 1).mean()
        consequence = F.l1_loss(outputs["reconstructed"], original)
        targets, _ = self.candidate_targets(outputs, reference)
        selection = F.cross_entropy(outputs["utilities"], targets)
        total = (self.lambda_recon * reconstruction + self.lambda_preserve * preservation
                 + self.lambda_consequence * consequence + self.lambda_selection * selection)
        return {"total": total, "reconstruction": reconstruction, "preservation": preservation,
                "consequence": consequence, "selection": selection}
