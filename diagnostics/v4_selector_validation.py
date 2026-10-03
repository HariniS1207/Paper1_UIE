from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from models.paper1_model_v4 import Paper1ModelV4
from data.euvp_dataset import EUVPPairedDataset
from data.transforms import PairedTransform


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v4" / "paper1_v4_best.pth"
V2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v2" / "paper1_v2_epoch_006.pth"
V3_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v3" / "paper1_v3_best.pth"
EXPECTED_V2_SHA256 = "BBC8027FF23DCD477CD2E4BD0C95D7A42222AAAAE2E6A6197AF3D198149C53D1"
EXPECTED_V3_SHA256 = "80163DBA423CCDE672318A3DD4F93B68982ADE6A0CD3ED737FA8D326948D0555"
EXPECTED_V4_EPOCH = 10
DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "EUVP"
    / "EUVP-Dataset"
    / "EUVP"
    / "Paired"
    / "underwater_imagenet"
)
VAL_SPLIT = PROJECT_ROOT / "data" / "splits" / "val.txt"
OUTPUT_PATH = PROJECT_ROOT / "diagnostics" / "v4_selector_validation" / "v4_validation_diagnosis.json"
CANDIDATES = ("conservative", "balanced", "aggressive")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_frozen_hashes():
    hashes = {
        "v2": sha256(V2_CHECKPOINT),
        "v3": sha256(V3_CHECKPOINT),
    }
    if hashes["v2"] != EXPECTED_V2_SHA256 or hashes["v3"] != EXPECTED_V3_SHA256:
        raise RuntimeError(f"Frozen checkpoint hash mismatch: {hashes}")
    return hashes


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def percentage(mask):
    values = np.asarray(mask, dtype=bool)
    return {"count": int(values.sum()), "percent": float(values.mean() * 100.0)}


def summarize_group(records, title):
    n = len(records)
    if n == 0:
        return {"count": 0}
    selected_counts = {name: sum(row["selected"] == name for row in records) for name in CANDIDATES}
    target_agreement = sum(row["selected"] == row["reference_l1_target"] for row in records)
    return {
        "count": n,
        "selector_choice_distribution": {
            name: {"count": count, "percent": 100.0 * count / n}
            for name, count in selected_counts.items()
        },
        "selector_reference_agreement": {"count": target_agreement, "percent": 100.0 * target_agreement / n},
        "mean_aggressive_probability": float(np.mean([row["probabilities"]["aggressive"] for row in records])),
        "mean_reference_l1": {
            name: float(np.mean([row["reference_l1"][name] for row in records]))
            for name in CANDIDATES
        },
        "mean_consequence_loss": {
            name: float(np.mean([row["consequence_loss"][name] for row in records]))
            for name in CANDIDATES
        },
        "mean_preservation_score": {
            name: float(np.mean([row["preservation_score"][name] for row in records]))
            for name in CANDIDATES
        },
    }


def main():
    if not CHECKPOINT.is_file() or not VAL_SPLIT.is_file():
        raise FileNotFoundError("V4 best checkpoint or val.txt is missing.")
    frozen_hashes_before = verify_frozen_hashes()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != EXPECTED_V4_EPOCH:
        raise RuntimeError(f"Expected V4 epoch {EXPECTED_V4_EPOCH}; got {checkpoint.get('epoch')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Paper1ModelV4().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = EUVPPairedDataset(
        DATA_ROOT / "trainA",
        DATA_ROOT / "trainB",
        VAL_SPLIT,
        transform=PairedTransform(training=False),
    )
    if len(dataset) != 370:
        raise RuntimeError(f"Expected 370 validation images; found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    l1_by_candidate = {name: [] for name in CANDIDATES}
    preservation_by_candidate = {name: [] for name in CANDIDATES}
    preservation_loss_by_candidate = {name: [] for name in CANDIDATES}
    consequence_by_candidate = {name: [] for name in CANDIDATES}
    utility_by_candidate = {name: [] for name in CANDIDATES}
    probability_by_candidate = {name: [] for name in CANDIDATES}
    residual_norm_by_candidate = {name: [] for name in CANDIDATES}
    residual_abs_by_candidate = {name: [] for name in CANDIDATES}
    diversity_cosines = {
        ("conservative", "balanced"): [],
        ("balanced", "aggressive"): [],
        ("conservative", "aggressive"): [],
    }
    confusion = {target: {selected: 0 for selected in CANDIDATES} for target in CANDIDATES}
    records = []
    selection_margins = []
    probability_margins = []
    entropies = []
    selected_counts = {name: 0 for name in CANDIDATES}
    aggressive_best = []
    aggressive_selected = []
    aggressive_lowest_consequence = []
    aggressive_lowest_preservation_loss = []
    aggressive_highest_utility = []
    aggressive_largest_residual = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="V4 selector validation diagnosis"):
            original = batch["underwater"].to(device)
            reference = batch["reference"].to(device)
            output = model(original)
            candidates = output["candidates"]
            stacked_l1 = torch.stack([
                F.l1_loss(candidates[name], reference, reduction="none").mean(dim=(1, 2, 3))
                for name in CANDIDATES
            ], dim=1)
            target_indices = torch.argmin(stacked_l1, dim=1)
            target_names = [CANDIDATES[index] for index in target_indices.cpu().tolist()]
            selected_names = [CANDIDATES[index] for index in output["selected_index"].cpu().tolist()]
            probabilities = output["selection_weights"].float()
            utilities = output["utilities"].float()
            sorted_utilities = torch.sort(utilities, dim=1, descending=True).values
            sorted_probabilities = torch.sort(probabilities, dim=1, descending=True).values
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)

            residuals = {name: candidates[name] - original for name in CANDIDATES}
            residual_norms = {
                name: torch.linalg.vector_norm(residuals[name].float(), dim=(1, 2, 3))
                for name in CANDIDATES
            }
            residual_abss = {
                name: residuals[name].abs().mean(dim=(1, 2, 3))
                for name in CANDIDATES
            }
            consequence_losses = {}
            for name in CANDIDATES:
                consequence_output = model.consequence_check(original, candidates[name])
                consequence_losses[name] = F.l1_loss(consequence_output["reconstructed"], original, reduction="none").mean(dim=(1, 2, 3))

            candidate_preservation = {
                name: output["preservation_scores"][name].float()
                for name in CANDIDATES
            }
            candidate_preservation_loss = {name: 1.0 - candidate_preservation[name] for name in CANDIDATES}

            for batch_index in range(original.shape[0]):
                ref_l1 = {name: float(stacked_l1[batch_index, index].item()) for index, name in enumerate(CANDIDATES)}
                preserve = {name: float(candidate_preservation[name][batch_index].item()) for name in CANDIDATES}
                preserve_loss = {name: float(candidate_preservation_loss[name][batch_index].item()) for name in CANDIDATES}
                consequence = {name: float(consequence_losses[name][batch_index].item()) for name in CANDIDATES}
                utility = {name: float(utilities[batch_index, index].item()) for index, name in enumerate(CANDIDATES)}
                probability = {name: float(probabilities[batch_index, index].item()) for index, name in enumerate(CANDIDATES)}
                residual_norm = {name: float(residual_norms[name][batch_index].item()) for name in CANDIDATES}
                residual_abs = {name: float(residual_abss[name][batch_index].item()) for name in CANDIDATES}
                target = target_names[batch_index]
                selected = selected_names[batch_index]
                confusion[target][selected] += 1
                selected_counts[selected] += 1
                is_aggressive_best = target == "aggressive"
                aggressive_best.append(is_aggressive_best)
                aggressive_selected.append(selected == "aggressive")
                aggressive_lowest_consequence.append(consequence["aggressive"] == min(consequence.values()))
                aggressive_lowest_preservation_loss.append(preserve_loss["aggressive"] == min(preserve_loss.values()))
                aggressive_highest_utility.append(utility["aggressive"] == max(utility.values()))
                aggressive_largest_residual.append(residual_norm["aggressive"] == max(residual_norm.values()))
                selection_margins.append(float((sorted_utilities[batch_index, 0] - sorted_utilities[batch_index, 1]).item()))
                probability_margins.append(float((sorted_probabilities[batch_index, 0] - sorted_probabilities[batch_index, 1]).item()))
                entropies.append(float(entropy[batch_index].item()))
                records.append({
                    "filename": batch["filename"][batch_index],
                    "reference_l1_target": target,
                    "selected": selected,
                    "reference_l1": ref_l1,
                    "preservation_score": preserve,
                    "preservation_loss": preserve_loss,
                    "consequence_loss": consequence,
                    "utilities": utility,
                    "probabilities": probability,
                    "residual_norm": residual_norm,
                    "mean_absolute_residual": residual_abs,
                })

            for name in CANDIDATES:
                l1_by_candidate[name].extend(stacked_l1[:, CANDIDATES.index(name)].cpu().tolist())
                preservation_by_candidate[name].extend(candidate_preservation[name].cpu().tolist())
                preservation_loss_by_candidate[name].extend(candidate_preservation_loss[name].cpu().tolist())
                consequence_by_candidate[name].extend(consequence_losses[name].cpu().tolist())
                utility_by_candidate[name].extend(utilities[:, CANDIDATES.index(name)].cpu().tolist())
                probability_by_candidate[name].extend(probabilities[:, CANDIDATES.index(name)].cpu().tolist())
                residual_norm_by_candidate[name].extend(residual_norms[name].cpu().tolist())
                residual_abs_by_candidate[name].extend(residual_abss[name].cpu().tolist())
            for pair in diversity_cosines:
                left, right = pair
                values = F.cosine_similarity(
                    residuals[left].flatten(1).float(),
                    residuals[right].flatten(1).float(),
                    dim=1,
                    eps=1e-8,
                )
                diversity_cosines[pair].extend(values.cpu().tolist())

    target_counts = {
        name: int(sum(record["reference_l1_target"] == name for record in records))
        for name in CANDIDATES
    }
    n = len(records)
    gains = {name: float(value.detach().item()) for name, value in model.candidate_generator.candidate_gains().items()}
    raw_gain_logits = {name: float(value.detach().item()) for name, value in model.candidate_generator.gain_logits.items()}
    groups = {
        "aggressive_is_reference_best": summarize_group([row for row in records if row["reference_l1_target"] == "aggressive"], "aggressive_best"),
        "aggressive_is_not_reference_best": summarize_group([row for row in records if row["reference_l1_target"] != "aggressive"], "not_aggressive_best"),
    }
    report = {
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_split": str(VAL_SPLIT.relative_to(PROJECT_ROOT)),
        "test_split_accessed": False,
        "validation_image_count": n,
        "device": str(device),
        "checkpoint_hashes_before": frozen_hashes_before,
        "checkpoint_hashes_after": verify_frozen_hashes(),
        "candidate_reference_l1": {name: distribution(values) for name, values in l1_by_candidate.items()},
        "candidate_preservation_score": {name: distribution(values) for name, values in preservation_by_candidate.items()},
        "candidate_preservation_loss": {name: distribution(values) for name, values in preservation_loss_by_candidate.items()},
        "candidate_consequence_loss": {name: distribution(values) for name, values in consequence_by_candidate.items()},
        "candidate_selector_utility": {name: distribution(values) for name, values in utility_by_candidate.items()},
        "candidate_selector_probability": {name: distribution(values) for name, values in probability_by_candidate.items()},
        "reference_l1_target_distribution": {
            name: {"count": target_counts[name], "percent": 100.0 * target_counts[name] / n}
            for name in CANDIDATES
        },
        "selector_choice_distribution": {
            name: {"count": selected_counts[name], "percent": 100.0 * selected_counts[name] / n}
            for name in CANDIDATES
        },
        "selector_reference_agreement": {
            "count": sum(record["selected"] == record["reference_l1_target"] for record in records),
            "percent": 100.0 * sum(record["selected"] == record["reference_l1_target"] for record in records) / n,
        },
        "confusion_matrix_reference_target_rows_selector_choice_columns": confusion,
        "selection_confidence": {
            "top_minus_second_utility": distribution(selection_margins),
            "top_minus_second_probability": distribution(probability_margins),
            "entropy_nats": distribution(entropies),
        },
        "gains": gains,
        "raw_gain_logits": raw_gain_logits,
        "gain_ordering_conservative_lt_balanced_lt_aggressive": gains["conservative"] < gains["balanced"] < gains["aggressive"],
        "residual_scale": {
            "mean_l2_norm": {name: distribution(values) for name, values in residual_norm_by_candidate.items()},
            "mean_absolute_residual": {name: distribution(values) for name, values in residual_abs_by_candidate.items()},
            "pairwise_residual_cosine_similarity": {
                f"{left}_vs_{right}": distribution(values)
                for (left, right), values in diversity_cosines.items()
            },
        },
        "aggressive_relationships": {
            "aggressive_reference_l1_best": percentage(aggressive_best),
            "aggressive_selected": percentage(aggressive_selected),
            "aggressive_lowest_consequence_loss": percentage(aggressive_lowest_consequence),
            "aggressive_lowest_preservation_loss": percentage(aggressive_lowest_preservation_loss),
            "aggressive_highest_utility": percentage(aggressive_highest_utility),
            "aggressive_largest_residual_norm": percentage(aggressive_largest_residual),
        },
        "conditional_analysis": groups,
        "records": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    print(f"Full per-image diagnostic saved: {OUTPUT_PATH}")
    print("test.txt accessed: NO")


if __name__ == "__main__":
    main()
