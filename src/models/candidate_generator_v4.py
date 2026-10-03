import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateSpecificEnhancementGenerator(nn.Module):
    """Shared condition-aware features with independently learned candidate heads."""

    candidate_names = ("conservative", "balanced", "aggressive")

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
        self.condition_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(condition_dim, 32),
                nn.ReLU(inplace=True),
            )
            for name in self.candidate_names
        })
        self.candidate_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(160, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Tanh(),
            )
            for name in self.candidate_names
        })

        initial_gains = {
            "conservative": 0.0625,
            "balanced_increment": 0.0625,
            "aggressive_increment": 0.125,
        }
        self.gain_logits = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(math.log(math.expm1(value))))
            for name, value in initial_gains.items()
        })

    def candidate_gains(self):
        conservative = F.softplus(self.gain_logits["conservative"])
        balanced = conservative + F.softplus(self.gain_logits["balanced_increment"])
        aggressive = balanced + F.softplus(self.gain_logits["aggressive_increment"])
        return {
            "conservative": conservative,
            "balanced": balanced,
            "aggressive": aggressive,
        }

    def forward(self, x, condition_vector=None):
        shared_features = self.feature_extractor(x)
        if condition_vector is None:
            condition_vector = shared_features.new_zeros((x.shape[0], 128))

        gains = self.candidate_gains()
        candidates = {}
        for name in self.candidate_names:
            condition = self.condition_projections[name](condition_vector)
            condition = condition.unsqueeze(-1).unsqueeze(-1).expand(
                -1,
                -1,
                x.shape[2],
                x.shape[3],
            )
            branch_features = torch.cat((shared_features, condition), dim=1)
            branch_residual = self.candidate_heads[name](branch_features)
            candidates[name] = torch.clamp(x + gains[name] * branch_residual, 0.0, 1.0)
        return candidates
