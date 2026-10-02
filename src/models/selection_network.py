import torch
import torch.nn as nn


class UtilitySelectionNetwork(nn.Module):
    def __init__(self, condition_dim=128):
        super().__init__()

        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(inplace=True)
        )

        self.utility_head = nn.Sequential(
            nn.Linear(64 + 64 + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        candidates,
        condition_vector,
        preservation_scores
    ):
        condition_features = self.condition_projection(
            condition_vector
        )

        utilities = []

        candidate_names = [
            "conservative",
            "balanced",
            "aggressive"
        ]

        for name in candidate_names:
            candidate = candidates[name]

            features = self.candidate_encoder(candidate)
            features = features.flatten(1)

            preservation = preservation_scores[name].unsqueeze(1)

            combined = torch.cat(
                [
                    features,
                    condition_features,
                    preservation
                ],
                dim=1
            )

            utility = self.utility_head(combined)

            utilities.append(utility)

        utilities = torch.cat(utilities, dim=1)

        selection_weights = torch.softmax(
            utilities,
            dim=1
        )

        selected_index = torch.argmax(
            selection_weights,
            dim=1
        )

        return utilities, selection_weights, selected_index