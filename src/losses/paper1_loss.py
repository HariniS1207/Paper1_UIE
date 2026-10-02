import torch
import torch.nn as nn
import torch.nn.functional as F


class Paper1Loss(nn.Module):
    def __init__(
        self,
        lambda_recon=1.0,
        lambda_preserve=0.2,
        lambda_consequence=0.2,
        lambda_selection=0.5,
    ):
        super().__init__()

        self.lambda_recon = lambda_recon
        self.lambda_preserve = lambda_preserve
        self.lambda_consequence = lambda_consequence
        self.lambda_selection = lambda_selection

    def forward(self, outputs, reference, original):

        candidates = outputs["candidates"]
        selection_weights = outputs["selection_weights"]
        utilities = outputs["utilities"]

        # -------------------------------------------------
        # 1. Per-image reconstruction losses
        # -------------------------------------------------

        per_image_candidate_losses = torch.stack(
            [
                F.l1_loss(
                    candidates["conservative"],
                    reference,
                    reduction="none"
                ).mean(dim=(1, 2, 3)),

                F.l1_loss(
                    candidates["balanced"],
                    reference,
                    reduction="none"
                ).mean(dim=(1, 2, 3)),

                F.l1_loss(
                    candidates["aggressive"],
                    reference,
                    reduction="none"
                ).mean(dim=(1, 2, 3)),
            ],
            dim=1
        )

        # -------------------------------------------------
        # 2. Soft candidate reconstruction loss
        # -------------------------------------------------

        reconstruction_loss = (
            selection_weights
            * per_image_candidate_losses
        ).sum(dim=1).mean()

        # -------------------------------------------------
        # 3. Information preservation loss
        # -------------------------------------------------

        preservation_scores = outputs[
            "preservation_scores"
        ]

        preservation_loss = torch.stack(
            [
                1.0 - preservation_scores["conservative"],
                1.0 - preservation_scores["balanced"],
                1.0 - preservation_scores["aggressive"],
            ],
            dim=1
        ).mean()

        # -------------------------------------------------
        # 4. Consequence consistency loss
        # -------------------------------------------------

        consequence_loss = F.l1_loss(
            outputs["reconstructed"],
            original
        )

        # -------------------------------------------------
        # 5. Per-image selection target
        # -------------------------------------------------

        target_indices = torch.argmin(
            per_image_candidate_losses,
            dim=1
        )

        selection_loss = F.cross_entropy(
            utilities,
            target_indices
        )

        # -------------------------------------------------
        # 6. Total loss
        # -------------------------------------------------

        total_loss = (
            self.lambda_recon * reconstruction_loss
            + self.lambda_preserve * preservation_loss
            + self.lambda_consequence * consequence_loss
            + self.lambda_selection * selection_loss
        )

        return {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "preservation": preservation_loss,
            "consequence": consequence_loss,
            "selection": selection_loss,
        }