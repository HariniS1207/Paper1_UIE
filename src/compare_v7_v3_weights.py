"""Compare the original V7 and selected V3 Paper1ModelV7 checkpoints.

The script is read-only with respect to checkpoints. It strictly loads both
state dictionaries and writes CSV, JSON, and Markdown summaries to the chosen
output directory (default: results/v7_v3_weight_comparison).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
V7_DEFAULT = ROOT / "checkpoints/v7/paper1_v7_best_epoch_019.pth"
V3_DEFAULT = ROOT / "checkpoints/v7_enhancement_only_v3/paper1_v7_enhancement_only_v3_best_epoch_009.pth"
OUTPUT_DEFAULT = ROOT / "results/v7_v3_weight_comparison"
EXPECTED_HASHES = {
    "V7": "A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E",
    "V3": "29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA",
}
TRAINABLE_MODULES = ("candidate_generator", "selection_network")
FROZEN_MODULES = ("condition_encoder", "information_preservation", "consequence_check")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_state(path: Path, label: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    actual = sha256(path)
    expected = EXPECTED_HASHES[label]
    if actual != expected:
        raise RuntimeError(f"{label} checkpoint hash mismatch: expected {expected}, found {actual}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{label} checkpoint has no non-empty model_state_dict")
    return state, {"path": path.relative_to(ROOT).as_posix(), "sha256": actual,
                   "epoch": checkpoint.get("epoch"), "size_bytes": path.stat().st_size}


def finite_tensor(value: torch.Tensor, label: str) -> None:
    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{label} contains NaN or infinity")


def compare_tensor(name: str, before: torch.Tensor, after: torch.Tensor,
                   parameter_names: set[str], buffer_names: set[str]) -> dict[str, Any]:
    if tuple(before.shape) != tuple(after.shape):
        raise RuntimeError(f"Shape mismatch at {name}: {tuple(before.shape)} vs {tuple(after.shape)}")
    finite_tensor(before, f"V7 {name}")
    finite_tensor(after, f"V3 {name}")
    if name in parameter_names:
        kind = "parameter"
    elif name in buffer_names:
        kind = "buffer"
    else:
        kind = "state_tensor"
    same_dtype = before.dtype == after.dtype
    exact = same_dtype and torch.equal(before, after)
    if before.is_complex() or after.is_complex():
        a, b = before.to(torch.complex128), after.to(torch.complex128)
    elif before.is_floating_point() or after.is_floating_point():
        a, b = before.to(torch.float64), after.to(torch.float64)
    else:
        # Tensor equality is exact for integer and boolean buffers. Float64 is
        # used only for the magnitude statistics after differences are counted.
        a, b = before, after
    changed_mask = a != b
    changed_elements = int(changed_mask.sum().item())
    total = before.numel()
    diff = (b.to(torch.complex128) - a.to(torch.complex128)) if (a.is_complex() or b.is_complex()) else (b.to(torch.float64) - a.to(torch.float64))
    abs_diff = diff.abs().to(torch.float64)
    sum_abs = float(abs_diff.sum().item())
    l2 = float(torch.linalg.vector_norm(diff).item())
    base_l2 = float(torch.linalg.vector_norm(a.to(torch.complex128) if a.is_complex() else a.to(torch.float64)).item())
    relative = l2 / base_l2 if base_l2 != 0 else None
    module = name.split(".", 1)[0]
    return {
        "name": name, "kind": kind, "module": module,
        "shape": list(before.shape), "dtype_v7": str(before.dtype), "dtype_v3": str(after.dtype),
        "exactly_identical": bool(exact), "changed_elements": changed_elements,
        "total_elements": total,
        "changed_element_percent": (100.0 * changed_elements / total) if total else 0.0,
        "mean_absolute_difference": (sum_abs / total) if total else 0.0,
        "maximum_absolute_difference": float(abs_diff.max().item()) if total else 0.0,
        "l2_difference": l2, "v7_l2_norm": base_l2, "relative_l2_difference": relative,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elements = sum(row["total_elements"] for row in rows)
    changed = sum(row["changed_elements"] for row in rows)
    v7_l2_sq = sum(row["v7_l2_norm"] ** 2 for row in rows)
    diff_l2_sq = sum(row["l2_difference"] ** 2 for row in rows)
    return {
        "tensor_count": len(rows),
        "exact_identical_tensor_count": sum(row["exactly_identical"] for row in rows),
        "changed_tensor_count": sum(not row["exactly_identical"] for row in rows),
        "total_elements": elements, "changed_elements": changed,
        "changed_element_percent": (100.0 * changed / elements) if elements else 0.0,
        "sum_absolute_difference": sum(row["mean_absolute_difference"] * row["total_elements"] for row in rows),
        "global_mean_absolute_difference": (sum(row["mean_absolute_difference"] * row["total_elements"] for row in rows) / elements) if elements else 0.0,
        "global_l2_difference": math.sqrt(diff_l2_sq),
        "global_v7_l2_norm": math.sqrt(v7_l2_sq),
        "global_relative_l2_difference": math.sqrt(diff_l2_sq) / math.sqrt(v7_l2_sq) if v7_l2_sq else None,
    }


def run(v7_path: Path, v3_path: Path, out_dir: Path) -> dict[str, Any]:
    sys_path = str(ROOT / "src")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from models.paper1_model_v7 import Paper1ModelV7

    v7_state, v7_info = load_state(v7_path, "V7")
    v3_state, v3_info = load_state(v3_path, "V3")
    v7_model, v3_model = Paper1ModelV7(), Paper1ModelV7()
    try:
        v7_model.load_state_dict(v7_state, strict=True)
        v3_model.load_state_dict(v3_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Strict model loading failed: {error}") from error
    expected_v7, expected_v3 = set(v7_model.state_dict()), set(v3_model.state_dict())
    if set(v7_state) != expected_v7 or set(v3_state) != expected_v3:
        raise RuntimeError("Checkpoint keys do not exactly match the Paper1ModelV7 state dictionary")
    if set(v7_state) != set(v3_state):
        raise RuntimeError("V7 and V3 state-dictionary keys differ")

    params = {name for name, _ in v7_model.named_parameters()}
    buffers = {name for name, _ in v7_model.named_buffers()}
    rows = [compare_tensor(name, v7_state[name], v3_state[name], params, buffers)
            for name in sorted(v7_state)]
    param_rows = [row for row in rows if row["kind"] == "parameter"]
    buffer_rows = [row for row in rows if row["kind"] == "buffer"]
    trainable_rows = [row for row in param_rows if row["module"] in TRAINABLE_MODULES]
    frozen_rows = [row for row in rows if row["module"] in FROZEN_MODULES]
    if any(not row["exactly_identical"] for row in frozen_rows):
        names = [row["name"] for row in frozen_rows if not row["exactly_identical"]]
        raise RuntimeError(f"Frozen V3 modules differ from V7; investigate before accepting comparison: {names[:20]}")
    unexpected_changes = [row["name"] for row in rows
                          if not row["exactly_identical"] and row["module"] not in TRAINABLE_MODULES]
    if unexpected_changes:
        raise RuntimeError(f"Changed tensors occur outside the documented V3 trainable scope: {unexpected_changes[:20]}")

    modules: dict[str, dict[str, Any]] = {}
    for module in sorted({row["module"] for row in rows}):
        module_rows = [row for row in rows if row["module"] == module]
        modules[module] = {"all_state": aggregate(module_rows),
                           "parameters": aggregate([r for r in module_rows if r["kind"] == "parameter"]),
                           "buffers": aggregate([r for r in module_rows if r["kind"] == "buffer"]),
                           "changed_names": [r["name"] for r in module_rows if not r["exactly_identical"]],
                           "changed_parameter_names": [r["name"] for r in module_rows if r["kind"] == "parameter" and not r["exactly_identical"]],
                           "changed_buffer_names": [r["name"] for r in module_rows if r["kind"] == "buffer" and not r["exactly_identical"]]}

    summary = {
        "checkpoints": {"V7": v7_info, "V3": v3_info},
        "strict_load": True,
        "model": "Paper1ModelV7",
        "training_scope_from_verified_v3_script": list(TRAINABLE_MODULES),
        "frozen_modules_from_verified_v3_script": list(FROZEN_MODULES),
        "all_state": aggregate(rows), "parameters": aggregate(param_rows),
        "buffers": aggregate(buffer_rows), "changed_trainable_parameters": aggregate(trainable_rows),
        "changed_trainable_parameter_elements": sum(r["changed_elements"] for r in trainable_rows),
        "derived_trainable_parameter_elements": sum(p.numel() for n, p in v7_model.named_parameters() if n.split(".", 1)[0] in TRAINABLE_MODULES),
        "frozen_modules_exactly_identical": all(r["exactly_identical"] for r in frozen_rows),
        "modules": modules,
        "aggregate_definitions": {
            "changed_tensor": "state tensor differs exactly, including dtype; exact equality compares every value",
            "changed_element": "number of element-wise unequal values; percentage denominator is all values in that category",
            "global_mean_absolute_difference": "sum of absolute element differences divided by total elements",
            "global_l2_difference": "square root of the sum of squared element differences across category",
            "global_relative_l2_difference": "global L2 difference divided by global V7 L2 norm; null when V7 norm is zero",
            "mean_absolute_difference": "per-tensor sum of absolute differences divided by tensor element count",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weight_comparison.json").write_text(json.dumps({"summary": summary, "tensors": rows}, indent=2), encoding="utf-8")
    fields = ["name", "kind", "module", "shape", "dtype_v7", "dtype_v3", "exactly_identical",
              "changed_elements", "total_elements", "changed_element_percent", "mean_absolute_difference",
              "maximum_absolute_difference", "l2_difference", "v7_l2_norm", "relative_l2_difference"]
    with (out_dir / "tensor_differences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "shape": json.dumps(row["shape"], separators=(",", ":"))})
    write_markdown(out_dir / "WEIGHT_COMPARISON_REPORT.md", summary)
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# V7–V3 checkpoint weight comparison", "",
             "Both checkpoint state dictionaries were loaded strictly into separate `Paper1ModelV7` instances. The script only reads checkpoints and verifies their expected SHA-256 values.", "",
             "## Checkpoints", "",
             "| Version | Path | Epoch | SHA-256 | Size |", "|---|---|---:|---|---:|"]
    for label in ("V7", "V3"):
        c = summary["checkpoints"][label]
        lines.append(f"| {label} | `{c['path']}` | {c['epoch']} | `{c['sha256']}` | {c['size_bytes']:,} bytes |")
    lines.extend(["", "## Summary", "",
                  "| Category | Tensors | Exactly identical | Changed tensors | Elements | Changed elements | Changed elements (%) | Global L2 difference | Relative L2 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for label in ("all_state", "parameters", "buffers", "changed_trainable_parameters"):
        a = summary[label]
        lines.append(f"| {label.replace('_', ' ').title()} | {a['tensor_count']:,} | {a['exact_identical_tensor_count']:,} | {a['changed_tensor_count']:,} | {a['total_elements']:,} | {a['changed_elements']:,} | {a['changed_element_percent']:.6f}% | {a['global_l2_difference']:.8g} | {a['global_relative_l2_difference'] if a['global_relative_l2_difference'] is not None else 'N/A (zero baseline norm)'} |")
    lines.extend(["", f"Changed trainable parameter elements: **{summary['changed_trainable_parameter_elements']:,}** of **{summary['derived_trainable_parameter_elements']:,}** trainable elements defined by the V3 script.",
                  f"Frozen modules exactly identical: **{summary['frozen_modules_exactly_identical']}**.", "", "## Changed tensors by model module", "",
                  "| Module | Parameter tensors changed | Changed parameter elements | Buffer tensors changed | Changed buffer elements |", "|---|---:|---:|---:|---:|"])
    for module, data in summary["modules"].items():
        p, b = data["parameters"], data["buffers"]
        lines.append(f"| `{module}` | {p['changed_tensor_count']} | {p['changed_elements']:,} | {b['changed_tensor_count']} | {b['changed_elements']:,} |")
    lines.extend(["", "### Changed parameter and buffer names", ""])
    for module, data in summary["modules"].items():
        if data["changed_parameter_names"] or data["changed_buffer_names"]:
            lines.append(f"- `{module}`")
            lines.extend(f"  - Parameter: `{name}`" for name in data["changed_parameter_names"])
            lines.extend(f"  - Buffer: `{name}`" for name in data["changed_buffer_names"])
    lines.extend(["", "## Interpretation", "",
                  "The V3 training script marks `candidate_generator` and `selection_network` trainable and freezes `condition_encoder`, `information_preservation`, and `consequence_check`. Frozen-module state tensors are required by this script to match V7 exactly; a discrepancy causes the comparison to stop. Weight differences verify final checkpoint changes but do not alone prove which tensors were trainable; that conclusion comes from the training script and pretraining audit.", "",
                  "Changed parameter names are enumerated in `tensor_differences.csv` and `weight_comparison.json`; raw tensor values are not included.", "",
                  "## Metric definitions", "",
                  "Per-tensor mean absolute difference is the absolute elementwise difference sum divided by tensor size. Per-tensor L2 is the Euclidean norm of the difference; relative L2 is that norm divided by the V7 tensor norm and is N/A for a zero V7 norm. Aggregate L2 combines squares across the category before taking a square root. Integer/bool buffers use exact value comparisons and are converted only for magnitude statistics.", "",
                  "Run from the project root with `python src/compare_v7_v3_weights.py`. Output files can be regenerated without changing either checkpoint.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7", type=Path, default=V7_DEFAULT)
    parser.add_argument("--v3", type=Path, default=V3_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    result = run(args.v7.resolve(), args.v3.resolve(), args.output_dir.resolve())
    print(json.dumps({"output_dir": args.output_dir.resolve().as_posix(), "summary": result["all_state"]}, indent=2))


if __name__ == "__main__":
    main()
