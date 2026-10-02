import torch
import torch.nn as nn


class EnhancementCandidateGenerator(nn.Module):
    """
    Generates one shared enhancement residual and creates
    controlled Conservative / Balanced / Aggressive responses.
    """

    def __init__(self, condition_dim=128):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ReLU(inplace=True)
        )

        self.residual_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, condition_vector=None):

        features = self.feature_extractor(x)

        if condition_vector is not None:

            condition = self.condition_projection(
                condition_vector
            )

            condition = condition.unsqueeze(-1).unsqueeze(-1)

            condition = condition.expand(
                -1,
                -1,
                x.shape[2],
                x.shape[3]
            )

            features = torch.cat(
                [features, condition],
                dim=1
            )

        else:
            # Fallback for standalone testing
            zeros = torch.zeros(
                features.size(0),
                128,
                features.size(2),
                features.size(3),
                device=features.device,
                dtype=features.dtype
            )

            features = torch.cat(
                [features, zeros],
                dim=1
            )

        # Shared learned enhancement direction
        residual = self.residual_head(features)

        # Limit maximum correction
        residual = 0.25 * residual

        # Controlled enhancement strengths
        conservative = torch.clamp(
            x + 0.25 * residual,
            0.0,
            1.0
        )

        balanced = torch.clamp(
            x + 0.50 * residual,
            0.0,
            1.0
        )

        aggressive = torch.clamp(
            x + 1.00 * residual,
            0.0,
            1.0
        )

        return {
            "conservative": conservative,
            "balanced": balanced,
            "aggressive": aggressive
        }