"""Validation-only analysis for the frozen V7 selector experiment."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "data", ROOT / "src" / "models", ROOT / "src" / "losses"):
    sys.path.insert(0, str(p))

from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from models.paper1_model import Paper1Model as Paper1ModelV3
from models.paper1_model_v4 import Paper1ModelV4
from models.paper1_model_v7 import Paper1ModelV7
from losses.paper1_loss_v7 import Paper1LossV7
from evaluate_test import calculate_metrics
from train_v7 import DATASET, SPLITS, BATCH_SIZE, CKPT_DIR, RESULTS_DIR, NAMES, sha256, verify_frozen_manifest


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return state


def mean_dict(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def metric_dict(image, reference):
    a = image.detach().float().cpu().permute(1, 2, 0).numpy()
    b = reference.detach().float().cpu().permute(1, 2, 0).numpy()
    psnr, ssim, uiqm, uciqe = calculate_metrics(a, b)
    return {"psnr": float(psnr), "ssim": float(ssim), "uiqm": float(uiqm), "uciqe": float(uciqe)}


def main():
    verify_frozen_manifest()
    ckptdir = ROOT / "checkpoints" / "v7"
    history = json.loads((ckptdir / "loss_history.json").read_text(encoding="utf-8"))
    best = min(history, key=lambda r: r["validation"]["total"])
    best_epoch = int(best["epoch"])
    checkpoint_path = ckptdir / f"paper1_v7_best_epoch_{best_epoch:03d}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EUVPPairedDataset(DATASET / "trainA", DATASET / "trainB", SPLITS / "val.txt",
                                transform=PairedTransform(training=False))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    v7 = Paper1ModelV7().to(device)
    state = load_checkpoint(v7, checkpoint_path, device)
    v4 = Paper1ModelV4().to(device)
    load_checkpoint(v4, ROOT / "checkpoints" / "v4" / "paper1_v4_best.pth", device)
    v3 = Paper1ModelV3().to(device)
    load_checkpoint(v3, ROOT / "checkpoints" / "v3" / "paper1_v3_best.pth", device)
    records = []
    pair_names = ((0, 1), (1, 2), (0, 2))
    for batch in tqdm(loader, desc="V7 validation analysis", leave=False):
        x, ref = batch["underwater"].to(device), batch["reference"].to(device)
        with torch.inference_mode():
            o7, o4, o3 = v7(x), v4(x), v3(x)
            targets, q = Paper1LossV7().candidate_targets(o7, ref)
        outputs = {"V7": o7, "V4": o4, "V3": o3}
        for i, filename in enumerate(batch["filename"]):
            m = {version: metric_dict(out["selected_image"][i], ref[i]) for version, out in outputs.items()}
            original_metrics = metric_dict(x[i], ref[i])
            candidate = {}
            residuals = []
            for j, name in enumerate(NAMES):
                low = name.lower()
                residual = (o7["candidates"][low][i] - x[i]).float()
                residuals.append(residual.flatten())
                cm = metric_dict(o7["candidates"][low][i], ref[i])
                candidate[low] = {
                    "reference_l1": float(q["objectives"][i, j, 0]),
                    "preservation_distortion": float(q["objectives"][i, j, 1]),
                    "preservation_score": 1.0 - float(q["objectives"][i, j, 1]),
                    "consequence": float(q["objectives"][i, j, 2]),
                    "residual_l2": float(torch.linalg.vector_norm(residual)),
                    "psnr": cm["psnr"],
                    "ssim": cm["ssim"],
                }
            distances, cosines = {}, {}
            for a, b in pair_names:
                key = f"{NAMES[a]}_vs_{NAMES[b]}"
                ra, rb = residuals[a], residuals[b]
                distances[key] = float((o7["candidates"][NAMES[a].lower()][i] - o7["candidates"][NAMES[b].lower()][i]).abs().mean())
                cosines[key] = float(torch.nn.functional.cosine_similarity(ra, rb, dim=0, eps=1e-8))
            selected_name = NAMES[int(o7["selected_index"][i])]
            selected_low = selected_name.lower()
            v3_name = NAMES[int(o3["selected_index"][i])]
            entropy = -(o7["selection_weights"][i].float().clamp_min(1e-12)
                        * o7["selection_weights"][i].float().clamp_min(1e-12).log()).sum()
            sorted_u = o7["utilities"][i].float().sort(descending=True).values
            pareto = []
            vals = q["objectives"][i].float().cpu().numpy()
            for ci, cname in enumerate(NAMES):
                dominated = any(oi != ci and np.all(vals[oi] <= vals[ci]) and np.any(vals[oi] < vals[ci])
                                for oi in range(3))
                if not dominated:
                    pareto.append(cname)
            records.append({
                "filename": filename,
                "severity_input_reference_l1": float((x[i] - ref[i]).abs().mean()),
                "condition_vector": o7["condition_vector"][i].float().cpu().tolist(),
                "target": NAMES[int(targets[i])], "selected": selected_name,
                "v4_selected": NAMES[int(o4["selected_index"][i])], "v3_selected": v3_name,
                "probabilities": {NAMES[k]: float(o7["selection_weights"][i, k]) for k in range(3)},
                "entropy_nats": float(entropy), "utility_margin": float(sorted_u[0] - sorted_u[1]),
                "candidate_metrics": candidate, "pareto_optimal": pareto,
                "pairwise_candidate_distance_mean_abs": distances,
                "pairwise_residual_cosine": cosines,
                "selected_metrics": m,
                "original_metrics": original_metrics,
                "selected_preservation_score": float(o7["preservation_scores"][selected_low][i]),
                "selected_consequence": float(o7["candidate_consequence_errors"][selected_low][i]),
                "v3_selected_preservation_score": float(o3["preservation_scores"][v3_name.lower()][i]),
                "v3_selected_consequence": float(o3["consequence_error"][i]),
            })

    # Use already-declared validation severity quartiles so diagnostic grouping is comparable to V6.
    edges = [0.047843124717473984, 0.09782713651657104, 0.11644341424107552,
             0.13583612069487572, 0.302097350358963]
    groups = []
    for qi in range(4):
        subset = [r for r in records if r["severity_input_reference_l1"] >= edges[qi]
                  and (r["severity_input_reference_l1"] <= edges[qi + 1] if qi == 3
                       else r["severity_input_reference_l1"] < edges[qi + 1])]
        counts = {n: sum(r["selected"] == n for r in subset) for n in NAMES}
        target_counts = {n: sum(r["target"] == n for r in subset) for n in NAMES}
        groups.append({"quartile": f"Q{qi+1}", "n": len(subset),
                       "mean_severity": float(np.mean([r["severity_input_reference_l1"] for r in subset])),
                       "target_counts": target_counts, "selected_counts": counts,
                       "selected_percentages": {n: 100 * counts[n] / len(subset) for n in NAMES},
                       "target_percentages": {n: 100 * target_counts[n] / len(subset) for n in NAMES},
                       "candidate_quality": {n.lower(): {metric: float(np.mean([r["candidate_metrics"][n.lower()][metric] for r in subset]))
                                                                  for metric in ("reference_l1", "preservation_score", "consequence", "psnr", "ssim")}
                                              for n in NAMES},
                       "selected_quality": {version: mean_dict([r["selected_metrics"][version] for r in subset])
                                            for version in ("V7",)},
                       "selected_preservation_score": float(np.mean([r["selected_preservation_score"] for r in subset])),
                       "selected_consequence": float(np.mean([r["selected_consequence"] for r in subset]))})
    selector_counts = {n: sum(r["selected"] == n for r in records) for n in NAMES}
    target_counts = {n: sum(r["target"] == n for r in records) for n in NAMES}
    agreement = float(np.mean([r["target"] == r["selected"] for r in records]))
    prob_means = {n: float(np.mean([r["probabilities"][n] for r in records])) for n in NAMES}
    overall_metrics = {v: mean_dict([r["selected_metrics"][v] for r in records]) for v in ("V3", "V4", "V7")}
    overall_metrics["Original"] = mean_dict([r["original_metrics"] for r in records])
    preservation_consequence = {
        "V7": {"preservation_score": float(np.mean([r["selected_preservation_score"] for r in records])),
               "consequence_error": float(np.mean([r["selected_consequence"] for r in records]))},
        "V3": {"preservation_score": float(np.mean([r["v3_selected_preservation_score"] for r in records])),
               "consequence_error": float(np.mean([r["v3_selected_consequence"] for r in records]))},
    }
    diversity = {
        "pairwise_candidate_distance_mean_abs": {k: float(np.mean([r["pairwise_candidate_distance_mean_abs"][k] for r in records]))
                                                 for k in records[0]["pairwise_candidate_distance_mean_abs"]},
        "pairwise_residual_cosine_mean": {k: float(np.mean([r["pairwise_residual_cosine"][k] for r in records]))
                                          for k in records[0]["pairwise_residual_cosine"]},
        "mean_residual_l2": {n.lower(): float(np.mean([r["candidate_metrics"][n.lower()]["residual_l2"] for r in records])) for n in NAMES},
    }
    confusion = {t: {s: 0 for s in NAMES} for t in NAMES}
    for r in records: confusion[r["target"]][r["selected"]] += 1
    dist_groups = [{n: counts[n] / len(records) for n in NAMES} for counts in [selector_counts]]
    tv = max((0.5 * sum(abs(a[n] - b[n]) for n in NAMES)
              for a in [{n: g["selected_counts"][n] / g["n"] for n in NAMES} for g in groups]
              for b in [{n: g["selected_counts"][n] / g["n"] for n in NAMES} for g in groups]), default=0.0)

    # Read only validation summaries/history for prior models; never inspect test outputs.
    comparison = {}
    for ver in ("V3", "V4", "V5", "V6"):
        ckdir = ROOT / "checkpoints" / ver.lower()
        hist_path = ckdir / "loss_history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            h = min(hist, key=lambda r: float(r.get("val_total", r.get("validation", {}).get("total", float("inf")))))
            if "validation" in h:
                losses = h["validation"]
            else:
                losses = {k: h.get("val_" + k) for k in ("total", "reconstruction", "preservation", "consequence", "selection")}
            comparison[ver] = {"best_epoch": h["epoch"], "validation_losses": losses,
                               "selected_metrics": overall_metrics.get(ver)}
    for ver in ("V5", "V6"):
        report_path = ROOT / "results" / ver.lower() / f"{ver.lower()}_training_report.json"
        if report_path.exists():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            va = rep.get("validation_analysis", {})
            key = f"mean_{ver.lower()}_metrics"
            if key in va: comparison[ver]["selected_metrics"] = va[key]
            obj = va.get(f"mean_{ver.lower()}_selection_objectives", {})
            if obj: comparison[ver]["selected_preservation_distortion_and_consequence"] = {
                "preservation_distortion": obj.get("preservation_distortion"), "consequence_error": obj.get("consequence")}
            for tkey in (f"{ver.lower()}_target_distribution", f"{ver.lower()}_selection_distribution"):
                if tkey in va: comparison[ver][tkey] = va[tkey]
    comparison["V7"] = {"best_epoch": best_epoch, "validation_losses": best["validation"],
                        "selected_metrics": overall_metrics["V7"], **{"target_counts": target_counts,
                        "selected_counts": selector_counts, "target_selector_agreement": agreement}}
    result = {
        "split": "validation only", "sample_count": len(records),
        "best_epoch": best_epoch, "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path), "validation_total": best["validation"]["total"],
        "target_counts": target_counts, "selector_counts": selector_counts,
        "target_selector_agreement": agreement, "target_to_selected_confusion": confusion,
        "mean_selection_probabilities": prob_means,
        "mean_selection_entropy_nats": float(np.mean([r["entropy_nats"] for r in records])),
        "mean_utility_margin": float(np.mean([r["utility_margin"] for r in records])),
        "severity_quartile_edges": edges, "severity_groups": groups,
        "maximum_pairwise_selection_total_variation_across_quartiles": tv,
        "selected_metrics": overall_metrics, "selected_preservation_consequence": preservation_consequence,
        "candidate_diversity": diversity,
        "candidate_quality_overall": {n.lower(): {metric: float(np.mean([r["candidate_metrics"][n.lower()][metric] for r in records]))
                                                    for metric in ("reference_l1", "preservation_score", "consequence", "psnr", "ssim")}
                                       for n in NAMES},
        "comparison_v3_v4_v5_v6_v7_validation_only": comparison,
        "amp_events": state.get("amp_events", []),
        "frozen_module_hashes_unchanged": json.loads((ckptdir / "training_config.json").read_text())["frozen_module_state_sha256_after"],
        "frozen_checkpoint_hashes_before": json.loads((ROOT / "diagnostics" / "v7_selection" / "frozen_checkpoint_hashes_before.json").read_text()),
        "test_split_loaded": False,
    }
    diagnostics = ROOT / "diagnostics" / "v7_selection"
    output_paths = [diagnostics / "validation_per_image.json", diagnostics / "v7_selector_diagnosis.json",
                    RESULTS_DIR / "v7_training_report.json"]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Refusing to overwrite existing V7 validation/report artifacts")
    (diagnostics / "validation_per_image.json").write_text(json.dumps({"split": "validation", "records": records}, indent=2), encoding="utf-8")
    (diagnostics / "v7_selector_diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULTS_DIR / "v7_training_report.json").write_text(json.dumps({
        "training_config": json.loads((ckptdir / "training_config.json").read_text()),
        "history": history, "best_epoch": best_epoch, "best_validation": best["validation"],
        "validation_diagnosis": result, "test_set_accessed": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
