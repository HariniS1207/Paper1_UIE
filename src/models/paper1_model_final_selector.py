"""Final isolated selector experiment: V4 candidates, explicit bilinear compatibility."""

import torch
import torch.nn as nn

from models.candidate_generator_v4 import CandidateSpecificEnhancementGenerator
from models.condition_encoder import ConditionEncoder
from models.consequence_check import ConsequenceCheckModule
from models.information_preservation import InformationPreservationModule


class BilinearCompatibilitySelector(nn.Module):
    names = ("conservative", "balanced", "aggressive")

    def __init__(self, condition_dim=128, embedding_dim=64):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(64, embedding_dim), nn.LayerNorm(embedding_dim), nn.Tanh()
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.Tanh()
        )
        # Explicit bilinear condition-candidate compatibility. A separate matrix
        # per policy lets the condition value candidate suitability differently.
        self.compatibility = nn.Parameter(torch.empty(3, embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.compatibility)
        self.metric_weights = nn.Parameter(torch.zeros(3, 2))
        self.candidate_bias = nn.Parameter(torch.zeros(3))

    def forward(self, candidates, condition_vector, preservation_scores, consequence_errors):
        z = self.condition_projection(condition_vector)
        scores = []
        for k, name in enumerate(self.names):
            c = self.candidate_encoder(candidates[name]).flatten(1)
            c = self.candidate_projection(c)
            wkz = torch.matmul(z, self.compatibility[k])
            compatibility = (wkz * c).sum(dim=1) / (c.shape[1] ** 0.5)
            metrics = torch.stack((preservation_scores[name], consequence_errors[name]), dim=1).float()
            metric_term = (metrics * self.metric_weights[k]).sum(dim=1)
            scores.append(compatibility + metric_term + self.candidate_bias[k])
        utilities = torch.stack(scores, dim=1)
        probabilities = torch.softmax(utilities.float(), dim=1)
        return utilities, probabilities, probabilities.argmax(dim=1)


class Paper1ModelFinalSelector(nn.Module):
    names = BilinearCompatibilitySelector.names

    def __init__(self, condition_dim=128):
        super().__init__()
        self.condition_encoder = ConditionEncoder(condition_dim=condition_dim)
        self.candidate_generator = CandidateSpecificEnhancementGenerator()
        self.information_preservation = InformationPreservationModule()
        self.selection_network = BilinearCompatibilitySelector(condition_dim=condition_dim)
        self.consequence_check = ConsequenceCheckModule()

    def forward(self, x):
        condition_map, condition_vector = self.condition_encoder(x)
        candidates = self.candidate_generator(x, condition_vector)
        preservation_scores, candidate_consequence_errors = {}, {}
        with torch.no_grad():
            for name in self.names:
                _, preservation_scores[name] = self.information_preservation(x, candidates[name])
                candidate_consequence_errors[name] = self.consequence_check(x, candidates[name])["consequence_error"]
        utilities, weights, selected_index = self.selection_network(
            candidates, condition_vector, preservation_scores, candidate_consequence_errors
        )
        candidate_stack = torch.stack([candidates[n] for n in self.names], dim=1)
        selected_image = candidate_stack[torch.arange(x.shape[0], device=x.device), selected_index]
        with torch.no_grad():
            selected_consequence = self.consequence_check(x, selected_image)
        return {
            "condition_map": condition_map, "condition_vector": condition_vector,
            "candidates": candidates, "preservation_scores": preservation_scores,
            "candidate_consequence_errors": candidate_consequence_errors,
            "utilities": utilities, "selection_weights": weights, "selected_index": selected_index,
            "selected_image": selected_image, "reconstructed": selected_consequence["reconstructed"],
            "consequence_error": selected_consequence["consequence_error"],
            "consequence_score": selected_consequence["consequence_score"],
        }
