"""One frozen, official V7 test evaluation. No model/training changes."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src/data", ROOT / "src/models", ROOT / "src/losses"):
    sys.path.insert(0, str(path))
from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from models.paper1_model_v7 import Paper1ModelV7
from evaluate_test import calculate_metrics

CHECKPOINT = ROOT / "checkpoints/v7/paper1_v7_best_epoch_019.pth"
EXPECTED_CHECKPOINT_SHA256 = "A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E"
SPLIT = ROOT / "data/splits/test.txt"
DATASET = ROOT / "data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet"
V7_DIAG = ROOT / "diagnostics/v7_selection/v7_selector_diagnosis.json"
REPORT_MD = ROOT / "results/v7/v7_final_test_report.md"
REPORT_JSON = ROOT / "results/v7/v7_final_test_report.json"
PER_IMAGE = ROOT / "diagnostics/v7_selection/test_per_image.json"
COMPARISON_CSV = ROOT / "results/v7/v7_final_test_comparison.csv"
NAMES = ("Conservative", "Balanced", "Aggressive")
KEYS = tuple(n.lower() for n in NAMES)

FROZEN_BASELINES = {
    "Original": {"psnr": 16.8987, "ssim": 0.737308, "uiqm": 1.8555, "uciqe": 5.6085, "preservation": 1.0},
    "V3 Selected": {"psnr": 20.4374, "ssim": 0.783765, "uiqm": 1.8003, "uciqe": 5.3948, "preservation": 0.957900},
    "V4 Selected": {"psnr": 23.3363, "ssim": 0.817886, "uiqm": 1.7163, "uciqe": 5.8820, "preservation": 0.954935},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def checkpoint_inventory():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in sorted((ROOT / "checkpoints").rglob("*.pth"))}


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "median": float(np.median(a)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def metric_dict(image, reference):
    values = calculate_metrics(np.clip(image, 0.0, 1.0), np.clip(reference, 0.0, 1.0))
    return dict(zip(("psnr", "ssim", "uiqm", "uciqe"), map(float, values)))


def main():
    before_manifest_path = ROOT / "diagnostics/v7_selection/final_test_checkpoint_hashes_before.json"
    for path in (REPORT_MD, REPORT_JSON, PER_IMAGE, COMPARISON_CSV):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing final-test artifact: {path}")
    if not CHECKPOINT.is_file() or not SPLIT.is_file():
        raise FileNotFoundError("Frozen V7 checkpoint or official test split is missing")

    # Verify the frozen model before any test example is loaded.
    checkpoint_before = sha256(CHECKPOINT)
    if checkpoint_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Frozen V7 checkpoint hash mismatch: {checkpoint_before}")
    checkpoint_hashes_now = checkpoint_inventory()
    split_hash_now = sha256(SPLIT)
    if before_manifest_path.exists():
        frozen_before = json.loads(before_manifest_path.read_text(encoding="utf-8"))
        if (frozen_before["checkpoint_hashes"] != checkpoint_hashes_now or
                frozen_before["v7_sha256"] != checkpoint_before or
                frozen_before["test_manifest_sha256"] != split_hash_now):
            raise RuntimeError("Checkpoint or test manifest differs from the first pre-evaluation snapshot")
        checkpoint_hashes_before = frozen_before["checkpoint_hashes"]
        split_hash_before = frozen_before["test_manifest_sha256"]
    else:
        checkpoint_hashes_before = checkpoint_hashes_now
        split_hash_before = split_hash_now
        before_manifest_path.write_text(
            json.dumps({"checkpoint_hashes": checkpoint_hashes_before,
                        "v7_sha256": checkpoint_before, "test_manifest_sha256": split_hash_before}, indent=2),
            encoding="utf-8")

    # The severity procedure and boundaries are fixed from the earlier V7 validation analysis.
    val_diagnosis = json.loads(V7_DIAG.read_text(encoding="utf-8"))
    severity_edges = [float(v) for v in val_diagnosis["severity_quartile_edges"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 19:
        raise RuntimeError(f"Expected epoch 19 checkpoint, found {checkpoint.get('epoch')}")
    model = Paper1ModelV7().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = EUVPPairedDataset(DATASET / "trainA", DATASET / "trainB", SPLIT,
                                transform=PairedTransform(training=False))
    if len(dataset) != 370:
        raise RuntimeError(f"Expected 370 official test pairs, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    metric_keys = ("psnr", "ssim", "uiqm", "uciqe")
    metric_values = {version: {m: [] for m in metric_keys}
                     for version in ("Original", *NAMES, "V7 Selected")}
    preservation_values = {name: [] for name in ("Original", *NAMES, "V7 Selected")}
    consequence_values = {name: [] for name in (*NAMES, "V7 Selected")}
    per_image = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Frozen V7 official test evaluation"):
            x = batch["underwater"].to(device)
            ref = batch["reference"].to(device)
            out = model(x)
            if not torch.isfinite(out["selected_image"]).all().item():
                raise FloatingPointError("Non-finite V7 selected output")
            weights = out["selection_weights"].float()
            utilities = out["utilities"].float()
            if not torch.isfinite(weights).all().item() or not torch.isfinite(utilities).all().item():
                raise FloatingPointError("Non-finite V7 utility or selection probability")
            if (weights < 0).any().item() or not torch.allclose(weights.sum(1), torch.ones(x.size(0), device=device), atol=1e-5):
                raise RuntimeError("Invalid V7 selection probabilities")
            indices = out["selected_index"].detach().cpu().tolist()
            for i, filename in enumerate(batch["filename"]):
                original = x[i].detach().float().cpu().permute(1, 2, 0).numpy()
                reference = ref[i].detach().float().cpu().permute(1, 2, 0).numpy()
                candidates = {name: out["candidates"][name.lower()][i].detach().float().cpu().permute(1, 2, 0).numpy()
                              for name in NAMES}
                scores = {"Original": metric_dict(original, reference)}
                candidate_metrics = {}
                for name in NAMES:
                    candidate_metrics[name] = metric_dict(candidates[name], reference)
                    candidate_metrics[name]["preservation"] = float(out["preservation_scores"][name.lower()][i])
                    candidate_metrics[name]["consequence_error"] = float(out["candidate_consequence_errors"][name.lower()][i])
                    scores[name] = candidate_metrics[name]
                original_preservation = 1.0
                idx = int(indices[i])
                if idx not in range(3) or int(weights[i].argmax()) != idx:
                    raise RuntimeError("Selected candidate does not match highest V7 probability")
                selected_name = NAMES[idx]
                selected_metrics = dict(candidate_metrics[selected_name])
                scores["V7 Selected"] = selected_metrics
                for version, metrics in scores.items():
                    for key in metric_keys:
                        metric_values[version][key].append(metrics[key])
                preservation_values["Original"].append(original_preservation)
                for name in NAMES:
                    preservation_values[name].append(candidate_metrics[name]["preservation"])
                    consequence_values[name].append(candidate_metrics[name]["consequence_error"])
                preservation_values["V7 Selected"].append(selected_metrics["preservation"])
                consequence_values["V7 Selected"].append(selected_metrics["consequence_error"])
                p = weights[i].detach().cpu().tolist()
                u = utilities[i].detach().cpu().tolist()
                entropy = float(-(weights[i] * weights[i].clamp_min(1e-12).log()).sum())
                sorted_u = utilities[i].sort(descending=True).values
                severity = float((x[i] - ref[i]).abs().mean())
                per_image.append({
                    "filename": filename, "severity_input_reference_l1": severity,
                    "selected_candidate": selected_name,
                    "selection_probabilities": dict(zip(NAMES, map(float, p))),
                    "utilities": dict(zip(NAMES, map(float, u))),
                    "candidate_metrics": candidate_metrics,
                    "selected_metrics": selected_metrics,
                    "original_metrics": scores["Original"],
                    "selected_minus_original": {k: selected_metrics[k] - scores["Original"][k] for k in metric_keys},
                    "selected_preservation_change_from_original": selected_metrics["preservation"] - original_preservation,
                    "selection_entropy_nats": entropy,
                    "utility_margin": float(sorted_u[0] - sorted_u[1]),
                })

    total = len(per_image)
    selection_counts = {name: sum(r["selected_candidate"] == name for r in per_image) for name in NAMES}
    selection_distribution = {name: {"count": count, "percent": 100.0 * count / total,
                                     "fraction": count / total} for name, count in selection_counts.items()}
    mean_prob = {name: float(np.mean([r["selection_probabilities"][name] for r in per_image])) for name in NAMES}
    severity_groups = []
    for qi in range(4):
        lo, hi = severity_edges[qi], severity_edges[qi + 1]
        subset = [r for r in per_image if r["severity_input_reference_l1"] >= lo and
                  (r["severity_input_reference_l1"] <= hi if qi == 3 else r["severity_input_reference_l1"] < hi)]
        counts = {name: sum(r["selected_candidate"] == name for r in subset) for name in NAMES}
        severity_groups.append({
            "quartile": f"Q{qi+1}", "lower_edge_inclusive": lo, "upper_edge": hi,
            "n": len(subset), "mean_severity": float(np.mean([r["severity_input_reference_l1"] for r in subset])) if subset else None,
            "selection_counts": counts,
            "selection_percentages": {name: (100.0 * counts[name] / len(subset) if subset else None) for name in NAMES},
            "mean_entropy_nats": float(np.mean([r["selection_entropy_nats"] for r in subset])) if subset else None,
            "mean_max_probability": float(np.mean([max(r["selection_probabilities"].values()) for r in subset])) if subset else None,
            "mean_utility_margin": float(np.mean([r["utility_margin"] for r in subset])) if subset else None,
        })
    out_of_edges = [r for r in per_image if r["severity_input_reference_l1"] < severity_edges[0]
                    or r["severity_input_reference_l1"] > severity_edges[-1]]
    metric_summary = {version: {key: summarize(vals) for key, vals in data.items()}
                      for version, data in metric_values.items()}
    preservation_summary = {name: summarize(values) for name, values in preservation_values.items()}
    consequence_summary = {name: summarize(values) for name, values in consequence_values.items()}
    improvement = {}
    for key in metric_keys:
        deltas = [r["selected_minus_original"][key] for r in per_image]
        improved = sum(delta > 0 for delta in deltas)
        improvement[key] = {"improved_count": improved, "improved_percent": 100.0 * improved / total,
                            "mean_change": float(np.mean(deltas))}
    pres_delta = [r["selected_preservation_change_from_original"] for r in per_image]
    pres_improved = sum(v > 0 for v in pres_delta)
    improvement["preservation"] = {"improved_count": pres_improved,
                                   "improved_percent": 100.0 * pres_improved / total,
                                   "mean_change": float(np.mean(pres_delta))}
    v7_vs_v4 = {key: metric_summary["V7 Selected"][key]["mean"] - FROZEN_BASELINES["V4 Selected"][key]
                for key in metric_keys}
    v7_vs_v4["preservation"] = (preservation_summary["V7 Selected"]["mean"]
                                 - FROZEN_BASELINES["V4 Selected"]["preservation"])
    test_decision = (
        "A" if min(selection_counts.values()) / total >= 0.10 and len(severity_groups) == 4
        and len({max(g["selection_percentages"], key=g["selection_percentages"].get) for g in severity_groups}) >= 2
        else "B" if sum(v > 0 for v in selection_counts.values()) >= 2
        else "C"
    )
    checkpoint_after = sha256(CHECKPOINT)
    checkpoint_hashes_after = checkpoint_inventory()
    split_hash_after = sha256(SPLIT)
    checkpoint_integrity = (checkpoint_before == checkpoint_after and
                            checkpoint_hashes_before == checkpoint_hashes_after)
    if not checkpoint_integrity:
        raise RuntimeError("One or more checkpoint hashes changed during final test evaluation")
    if split_hash_before != split_hash_after:
        raise RuntimeError("Official test split manifest changed during evaluation")

    report = {
        "evaluation": "final frozen V7 held-out test evaluation", "test_set_accessed": True,
        "test_used_for_training_or_model_selection": False, "sample_count": total,
        "split": "data/splits/test.txt", "test_manifest_sha256_before_after": split_hash_before,
        "model": "Paper1ModelV7", "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "best_epoch": 19,
        "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after,
        "all_checkpoint_files_unchanged": checkpoint_integrity,
        "checkpoint_hashes_before": checkpoint_hashes_before, "checkpoint_hashes_after": checkpoint_hashes_after,
        "metric_implementation": "src/evaluate_test.py calculate_metrics; identical existing PSNR/SSIM/UIQM/UCIQE helpers; RGB [0,1]",
        "preprocessing": "official paired EUVPPairedDataset and PairedTransform(training=False); RGB ToTensor [0,1]; no resizing or random augmentation",
        "device": str(device), "aggregate_metrics": metric_summary,
        "preservation_statistics": preservation_summary, "consequence_error_statistics": consequence_summary,
        "selection_distribution": selection_distribution, "mean_selection_probabilities": mean_prob,
        "mean_selection_entropy_nats": float(np.mean([r["selection_entropy_nats"] for r in per_image])),
        "mean_max_selection_probability": float(np.mean([max(r["selection_probabilities"].values()) for r in per_image])),
        "mean_utility_margin": float(np.mean([r["utility_margin"] for r in per_image])),
        "severity_quartile_edges_from_v7_validation": severity_edges,
        "severity_quartile_groups": severity_groups,
        "severity_outside_established_edges_count": len(out_of_edges),
        "frozen_test_baselines": FROZEN_BASELINES,
        "v7_selected_minus_v4_selected": v7_vs_v4,
        "v7_selected_vs_original_improvements": improvement,
        "final_adaptive_selection_assessment": test_decision,
        "decision_text": {"A": "V7 demonstrates adaptive selection on the held-out test set with meaningful candidate variation.",
                          "B": "V7 shows some test-set candidate variation, but remains strongly biased toward one candidate.",
                          "C": "V7 does not demonstrate meaningful adaptive selection on the test set."}[test_decision],
        "test_image_examples_cherry_picked": False,
    }

    # Comparison CSV contains the frozen reference numbers unchanged plus V7 test means.
    csv_rows = []
    for name, vals in FROZEN_BASELINES.items(): csv_rows.append({"model": name, **vals})
    csv_rows.append({"model": "V7 Selected", **{m: metric_summary["V7 Selected"][m]["mean"] for m in metric_keys},
                     "preservation": preservation_summary["V7 Selected"]["mean"],
                     "consequence_error": consequence_summary["V7 Selected"]["mean"]})
    with COMPARISON_CSV.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in csv_rows for k in row)))
        writer.writeheader(); writer.writerows(csv_rows)
    PER_IMAGE.write_text(json.dumps({"split": "test", "records": per_image}, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("checkpoint_hashes_before", "checkpoint_hashes_after")}, indent=2))


def write_markdown(r):
    lines = ["# Final frozen V7 test evaluation", "", "**This is the single final held-out evaluation. V7 was not retrained, tuned, or modified.**", "",
        f"- Test images: {r['sample_count']}", f"- Checkpoint: `{r['checkpoint']}` (epoch 19)",
        f"- SHA-256 before/after: `{r['checkpoint_sha256_before']}`", "- Test data used for training/model selection: **No**.",
        "- Preprocessing: official paired test split with `PairedTransform(training=False)`, RGB tensor [0,1], no resize or random augmentation.", "",
        "## Aggregate metrics", "", "| Output | PSNR | SSIM | UIQM | UCIQE | Preservation | Consequence error |", "|---|---:|---:|---:|---:|---:|---:|"]
    rows = [("Original", r["aggregate_metrics"]["Original"], r["preservation_statistics"]["Original"].get("mean"), None)]
    for name in NAMES:
        rows.append((name, r["aggregate_metrics"][name], r["preservation_statistics"][name]["mean"], r["consequence_error_statistics"][name]["mean"]))
    rows.append(("V7 Selected", r["aggregate_metrics"]["V7 Selected"], r["preservation_statistics"]["V7 Selected"]["mean"], r["consequence_error_statistics"]["V7 Selected"]["mean"]))
    for name, metrics, preservation, consequence in rows:
        vals = [metrics[k]["mean"] for k in ("psnr", "ssim", "uiqm", "uciqe")]
        lines.append(f"| {name} | {vals[0]:.4f} | {vals[1]:.6f} | {vals[2]:.4f} | {vals[3]:.4f} | {preservation:.6f} | {consequence if consequence is not None else 'N/A'} |")
    lines += ["", "## Frozen baseline comparison", "", "V3 and V4 values below are the frozen figures supplied for this final comparison and were not changed.", "",
        "| Model | PSNR | SSIM | UIQM | UCIQE | Preservation |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in FROZEN_BASELINES.items(): lines.append(f"| {name} | {values['psnr']:.4f} | {values['ssim']:.6f} | {values['uiqm']:.4f} | {values['uciqe']:.4f} | {values['preservation']:.6f} |")
    v7 = r["aggregate_metrics"]["V7 Selected"]
    lines.append(f"| V7 Selected (test) | {v7['psnr']['mean']:.4f} | {v7['ssim']['mean']:.6f} | {v7['uiqm']['mean']:.4f} | {v7['uciqe']['mean']:.4f} | {r['preservation_statistics']['V7 Selected']['mean']:.6f} |")
    lines += ["", "### V7 Selected - V4 Selected", "", "| Metric | Difference |", "|---|---:|"]
    for key, value in r["v7_selected_minus_v4_selected"].items(): lines.append(f"| {key} | {value:+.6f} |")
    lines += ["", "## Selection analysis", "", "| Candidate | Count | Percent | Mean probability |", "|---|---:|---:|---:|"]
    for name in NAMES:
        dist = r["selection_distribution"][name]
        lines.append(f"| {name} | {dist['count']} | {dist['percent']:.2f}% | {r['mean_selection_probabilities'][name]:.6f} |")
    lines += ["", f"- Mean entropy: {r['mean_selection_entropy_nats']:.6f} nats.",
        f"- Mean maximum selection probability: {r['mean_max_selection_probability']:.6f}.",
        f"- Mean utility margin: {r['mean_utility_margin']:.6f}.", "",
        "### Fixed validation severity quartiles", "",
        "The boundaries below are reused unchanged from the V7 validation analysis; test images outside the established edge range are reported separately.", "",
        "| Quartile | n | Mean severity | C | B | A | Entropy | Max probability | Utility margin |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in r["severity_quartile_groups"]:
        p=group["selection_percentages"]
        lines.append(f"| {group['quartile']} | {group['n']} | {group['mean_severity'] if group['mean_severity'] is not None else float('nan'):.6f} | {p['Conservative'] if p['Conservative'] is not None else float('nan'):.2f}% | {p['Balanced'] if p['Balanced'] is not None else float('nan'):.2f}% | {p['Aggressive'] if p['Aggressive'] is not None else float('nan'):.2f}% | {group['mean_entropy_nats'] if group['mean_entropy_nats'] is not None else float('nan'):.5f} | {group['mean_max_probability'] if group['mean_max_probability'] is not None else float('nan'):.5f} | {group['mean_utility_margin'] if group['mean_utility_margin'] is not None else float('nan'):.5f} |")
    lines += ["", f"Images outside the fixed severity interval: {r['severity_outside_established_edges_count']}.", "",
        "## Selected versus Original improvement counts", "", "| Metric | Improved images | Percent | Mean change |", "|---|---:|---:|---:|"]
    for key, value in r["v7_selected_vs_original_improvements"].items(): lines.append(f"| {key} | {value['improved_count']} | {value['improved_percent']:.2f}% | {value['mean_change']:+.6f} |")
    lines += ["", "## Final assessment", "", f"**{r['final_adaptive_selection_assessment']}. {r['decision_text']}**", "",
        "The measurements are reported as observed; the test set was not used for cherry-picking, tuning, training, or checkpoint selection.",
        "", "Checkpoint integrity: all `.pth` files under `checkpoints/` match their before-evaluation hashes. The V7 checkpoint hash before and after evaluation is identical.",
        "", "**TEST SET ACCESSED: YES - FINAL FROZEN EVALUATION ONLY**", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
