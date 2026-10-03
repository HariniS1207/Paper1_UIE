# Final selector validation diagnosis

- Model: explicit per-candidate bilinear condition-candidate compatibility.
- Selected counts: {'Conservative': 2, 'Balanced': 2, 'Aggressive': 366}; target counts: {'Conservative': 48, 'Balanced': 59, 'Aggressive': 263}.
- Selected probabilities mean: {'Conservative': 0.17329061712968993, 'Balanced': 0.2512836593530468, 'Aggressive': 0.5754257243227314}.
- Target-selector agreement: 72.162%.
- Mean entropy: 0.9257; margin: 0.6854.

| Quartile | Severity | C | B | A | Entropy | Confidence | Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.0816 | 1.1% | 0.0% | 98.9% | 0.9331 | 0.5588 | 0.6053 |
| Q2 | 0.1070 | 1.1% | 1.1% | 97.8% | 0.9217 | 0.5569 | 0.5516 |
| Q3 | 0.1263 | 0.0% | 1.1% | 98.9% | 0.9275 | 0.5894 | 0.7708 |
| Q4 | 0.1650 | 0.0% | 0.0% | 100.0% | 0.9206 | 0.5974 | 0.8132 |


## Representative validation cases

- **Conservative:** `n01914609_12065.jpg`; severity L1=0.0805; C/B/A probabilities=0.483/0.049/0.468; candidate fidelity L1=0.1014, preservation=0.9932, consequence=0.1688.
- **Balanced:** `n01917289_652.jpg`; severity L1=0.1216; C/B/A probabilities=0.043/0.499/0.458; candidate fidelity L1=0.1387, preservation=0.9950, consequence=0.1611.
- **Aggressive:** `n01917289_440.jpg`; severity L1=0.2068; C/B/A probabilities=0.137/0.234/0.629; candidate fidelity L1=0.0868, preservation=0.9512, consequence=0.0409.

## Decision

**C - FAILURE to establish meaningful adaptive selection.** Aggressive was selected for 366/370 images; Conservative and Balanced were selected twice each. Selection remains Aggressive in 97.8% to 100% of each severity quartile. The soft probabilities have higher entropy than V7, but argmax decisions are collapsed. Stop model development and use V4 as the strongest enhancement baseline; retain the final-selector run and V7 as adaptive-selection investigations. No further model version. `TEST SET ACCESSED: NO`.


## Target-to-selector confusion matrix

Rows are soft-target argmax, columns are selected policy.

| Target / selected | C | B | A |
|---|---:|---:|---:|
| Conservative | 2 | 0 | 46 |
| Balanced | 0 | 2 | 57 |
| Aggressive | 0 | 0 | 263 |

## Candidate diversity

{
  "pairwise_candidate_distance_mean_abs": {
    "Conservative_vs_Balanced": 0.011103761438406199,
    "Balanced_vs_Aggressive": 0.10783049769296839,
    "Conservative_vs_Aggressive": 0.11362383519676891
  },
  "pairwise_residual_cosine_mean": {
    "Conservative_vs_Balanced": 0.7598270159995032,
    "Balanced_vs_Aggressive": 0.34405236883542023,
    "Conservative_vs_Aggressive": -0.0463193481297207
  },
  "mean_residual_l2": {
    "conservative": 5.525165676023509,
    "balanced": 9.35200071914776,
    "aggressive": 53.85591207968222
  }
}


Previous checkpoint integrity: all V2-V6 checkpoint hashes match the saved pre-experiment manifest (zero mismatches). V7 best checkpoint SHA-256 is unchanged at `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`.
