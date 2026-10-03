# V6 training-only fidelity threshold analysis

## Scope and frozen candidates

Computed only on `data/splits/train.txt` (2,960 paired images) with deterministic, no-flip preprocessing and the frozen V4 best checkpoint. No validation/test split was loaded. Reference L1, fixed preservation distortion, and consequence error match the existing V4/V5 definitions.

## Relative fidelity-gap distribution

Gap is `(candidate L1 - best candidate L1) / (best candidate L1 + 1e-8)`. Statistics are mean, median, population standard deviation and quantiles.

| Candidate | Mean | Median | Std | Q25 | Q75 | Q90 | Q95 | Q99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conservative | 1.48977 | 1.47689 | 0.63798 | 1.05913 | 1.90241 | 2.29718 | 2.58140 | 3.12821 | 4.08689 |
| Balanced | 1.38952 | 1.36609 | 0.61588 | 0.96942 | 1.79208 | 2.15619 | 2.44223 | 2.99776 | 3.68561 |
| Aggressive | 0.00194 | 0.00000 | 0.02866 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.01927 |

## Training-only threshold sweep

| Relative tolerance | Eligible candidate occurrences | Images with >1 eligible | Target counts after preservation/consequence rank |
|---:|---|---:|---|
| 1% | {'conservative': 11, 'balanced': 21, 'aggressive': 2931} | 3 | {'conservative': 9, 'balanced': 20, 'aggressive': 2931} |
| 2% | {'conservative': 13, 'balanced': 26, 'aggressive': 2934} | 11 | {'conservative': 9, 'balanced': 22, 'aggressive': 2929} |
| 3% | {'conservative': 16, 'balanced': 27, 'aggressive': 2935} | 16 | {'conservative': 10, 'balanced': 22, 'aggressive': 2928} |
| 5% | {'conservative': 23, 'balanced': 31, 'aggressive': 2938} | 25 | {'conservative': 12, 'balanced': 21, 'aggressive': 2927} |
| 7.5% | {'conservative': 30, 'balanced': 36, 'aggressive': 2939} | 33 | {'conservative': 17, 'balanced': 20, 'aggressive': 2923} |
| 10% | {'conservative': 33, 'balanced': 42, 'aggressive': 2942} | 41 | {'conservative': 18, 'balanced': 22, 'aggressive': 2920} |
| 15% | {'conservative': 42, 'balanced': 53, 'aggressive': 2946} | 54 | {'conservative': 21, 'balanced': 27, 'aggressive': 2912} |

## Frozen rule before V6 implementation

**Tolerance: 5% relative reference-L1 gap.** Eligibility means candidate reference L1 is at most 1.05 times that image's best candidate reference L1 (with `1e-8` denominator epsilon). This sets an explicit, understandable maximum fidelity sacrifice. The threshold was chosen for its direct interpretation, not candidate proportions, and was frozen before inspecting validation behavior. Training-only eligible frequencies: `{'conservative': 23, 'balanced': 31, 'aggressive': 2938}`; resulting target counts after the predeclared ranking: `{'conservative': 12, 'balanced': 21, 'aggressive': 2927}`.

Among eligible candidates, rank by the equal mean of training-median/IQR standardized fixed preservation distortion and consequence L1. Training-only medians `[0.00441405177116394, 0.11060812324285507]`, IQRs `[0.036563098430633545, 0.07987094204872847]`. These calibration values and all per-image candidate measurements are machine-readable in the JSON.
