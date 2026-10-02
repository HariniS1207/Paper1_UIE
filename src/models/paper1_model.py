import torch
import torch.nn as nn

from models.condition_encoder import ConditionEncoder
from models.candidate_generator import EnhancementCandidateGenerator
from models.information_preservation import InformationPreservationModule
from models.selection_network import UtilitySelectionNetwork
from models.consequence_check import ConsequenceCheckModule


class Paper1Model(nn.Module):
    def __init__(self, condition_dim=128):
        super().__init__()

        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim
        )

        self.candidate_generator = EnhancementCandidateGenerator()

        self.information_preservation = (
            InformationPreservationModule()
        )

        self.selection_network = UtilitySelectionNetwork(
            condition_dim=condition_dim
        )

        self.consequence_check = ConsequenceCheckModule()

    def forward(self, x):
        # 1. Estimate image condition
        condition_map, condition_vector = (
            self.condition_encoder(x)
        )

        # 2. Generate enhancement candidates
        candidates = self.candidate_generator(x, condition_vector)

        # 3. Estimate information preservation
        preservation_scores = {}

        for name, candidate in candidates.items():
            _, score = self.information_preservation(
                x,
                candidate
            )
            preservation_scores[name] = score

        # 4. Select enhancement candidate
        utilities, selection_weights, selected_index = (
            self.selection_network(
                candidates,
                condition_vector,
                preservation_scores
            )
        )

        # 5. Stack candidates
        candidate_stack = torch.stack(
            [
                candidates["conservative"],
                candidates["balanced"],
                candidates["aggressive"]
            ],
            dim=1
        )

        batch_indices = torch.arange(
            x.size(0),
            device=x.device
        )

        selected_image = candidate_stack[
            batch_indices,
            selected_index
        ]

        # 6. Check consequence of selected enhancement
        consequence = self.consequence_check(
            x,
            selected_image
        )

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