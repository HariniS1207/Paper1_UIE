import torch
import torch.nn as nn

from models.candidate_generator_v4 import CandidateSpecificEnhancementGenerator
from models.condition_encoder import ConditionEncoder
from models.consequence_check import ConsequenceCheckModule
from models.information_preservation import InformationPreservationModule
from models.selection_network import UtilitySelectionNetwork


class Paper1ModelV4(nn.Module):
    """Paper 1 pipeline with only the candidate generator replaced for V4."""

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

        preservation_scores = {}
        for name, candidate in candidates.items():
            _, score = self.information_preservation(x, candidate)
            preservation_scores[name] = score

        utilities, selection_weights, selected_index = self.selection_network(
            candidates,
            condition_vector,
            preservation_scores,
        )

        candidate_stack = torch.stack(
            [
                candidates["conservative"],
                candidates["balanced"],
                candidates["aggressive"],
            ],
            dim=1,
        )
        batch_indices = torch.arange(x.size(0), device=x.device)
        selected_image = candidate_stack[batch_indices, selected_index]
        consequence = self.consequence_check(x, selected_image)

        return {
            "condition_map": condition_map,
            "condition_vector": condition_vector,
            "candidates": candidates,
            "preservation_scores": preservation_scores,
            "utilities": utilities,
            "selection_weights": selection_weights,
            "selected_index": selected_index,
            "selected_image": selected_image,
            "reconstructed": consequence["reconstructed"],
            "consequence_error": consequence["consequence_error"],
            "consequence_score": consequence["consequence_score"],
        }
