"""V5 retains the V4 architecture and exposes candidate consequence losses."""

import torch
import torch.nn as nn

from models.candidate_generator_v4 import CandidateSpecificEnhancementGenerator
from models.condition_encoder import ConditionEncoder
from models.consequence_check import ConsequenceCheckModule
from models.information_preservation import InformationPreservationModule
from models.selection_network import UtilitySelectionNetwork


class Paper1ModelV5(nn.Module):
    """Exact V4 modules; the extra per-candidate consequence calls expose target inputs."""

    def __init__(self, condition_dim=128):
        super().__init__()
        self.condition_encoder = ConditionEncoder(condition_dim=condition_dim)
        self.candidate_generator = CandidateSpecificEnhancementGenerator()
        self.information_preservation = InformationPreservationModule()
        self.selection_network = UtilitySelectionNetwork(condition_dim=condition_dim)
        self.consequence_check = ConsequenceCheckModule()

    def forward(self, x):
        condition_map, condition_vector = self.condition_encoder(x)
        candidates = self.candidate_generator(x, condition_vector)
        names = ("conservative", "balanced", "aggressive")
        preservation_scores = {}
        for name in names:
            _, preservation_scores[name] = self.information_preservation(x, candidates[name])
        utilities, selection_weights, selected_index = self.selection_network(
            candidates, condition_vector, preservation_scores
        )
        candidate_stack = torch.stack([candidates[name] for name in names], dim=1)
        batch_indices = torch.arange(x.size(0), device=x.device)
        selected_image = candidate_stack[batch_indices, selected_index]
        selected_consequence = self.consequence_check(x, selected_image)
        # Target-only candidate consequence values need no gradient. Keeping their
        # three reconstruction graphs alive would multiply peak training memory.
        with torch.no_grad():
            candidate_consequence_errors = {
                name: self.consequence_check(x, candidates[name])["consequence_error"]
                for name in names
            }
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
