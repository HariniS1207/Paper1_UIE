"""V7 freezes the V4 candidate space and learns condition-aware utilities."""

import torch
import torch.nn as nn

from models.candidate_generator_v4 import CandidateSpecificEnhancementGenerator
from models.condition_encoder import ConditionEncoder
from models.consequence_check import ConsequenceCheckModule
from models.information_preservation import InformationPreservationModule


class ConditionAwareUtilitySelectionNetwork(nn.Module):
    """Shared candidate scorer with explicit condition/candidate interactions."""

    candidate_names = ("conservative", "balanced", "aggressive")

    def __init__(self, condition_dim=128):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.candidate_projection = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.condition_projection = nn.Sequential(nn.Linear(condition_dim, 64), nn.ReLU(inplace=True))
        # Concatenated candidate and condition features plus their elementwise
        # interaction let the same candidate be valued differently by condition.
        self.utility_head = nn.Sequential(
            nn.Linear(64 * 3 + 2, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, candidates, condition_vector, preservation_scores, consequence_errors):
        condition = self.condition_projection(condition_vector)
        utilities = []
        for name in self.candidate_names:
            candidate_features = self.candidate_encoder(candidates[name]).flatten(1)
            candidate_features = self.candidate_projection(candidate_features)
            interaction = candidate_features * condition
            metrics = torch.stack(
                (preservation_scores[name], consequence_errors[name]), dim=1
            ).to(candidate_features.dtype)
            features = torch.cat((candidate_features, condition, interaction, metrics), dim=1)
            utilities.append(self.utility_head(features))
        utilities = torch.cat(utilities, dim=1)
        probabilities = torch.softmax(utilities.float(), dim=1)
        selected_index = probabilities.argmax(dim=1)
        return utilities, probabilities, selected_index


class Paper1ModelV7(nn.Module):
    """V4 candidate branches, frozen outside selector-only V7 training."""

    candidate_names = ("conservative", "balanced", "aggressive")

    def __init__(self, condition_dim=128):
        super().__init__()
        self.condition_encoder = ConditionEncoder(condition_dim=condition_dim)
        self.candidate_generator = CandidateSpecificEnhancementGenerator()
        self.information_preservation = InformationPreservationModule()
        self.selection_network = ConditionAwareUtilitySelectionNetwork(condition_dim=condition_dim)
        self.consequence_check = ConsequenceCheckModule()

    def forward(self, x):
        condition_map, condition_vector = self.condition_encoder(x)
        candidates = self.candidate_generator(x, condition_vector)
        preservation_scores = {}
        candidate_consequence_errors = {}
        # These fixed/frozen measurements condition the learned selector. Their
        # definitions remain the existing V4/V5/V6 implementations.
        with torch.no_grad():
            for name in self.candidate_names:
                _, preservation_scores[name] = self.information_preservation(x, candidates[name])
                candidate_consequence_errors[name] = self.consequence_check(
                    x, candidates[name]
                )["consequence_error"]
        utilities, selection_weights, selected_index = self.selection_network(
            candidates, condition_vector, preservation_scores, candidate_consequence_errors
        )
        candidate_stack = torch.stack([candidates[n] for n in self.candidate_names], dim=1)
        selected_image = candidate_stack[torch.arange(x.shape[0], device=x.device), selected_index]
        with torch.no_grad():
            selected_consequence = self.consequence_check(x, selected_image)
        return {
            "condition_map": condition_map,
            "condition_vector": condition_vector,
            "candidates": candidates,
            "preservation_scores": preservation_scores,
            "candidate_consequence_errors": candidate_consequence_errors,
            "utilities": utilities,
            "selection_weights": selection_weights,
            "selected_index": selected_index,
            "selected_image": selected_image,
            "reconstructed": selected_consequence["reconstructed"],
            "consequence_error": selected_consequence["consequence_error"],
            "consequence_score": selected_consequence["consequence_score"],
        }
