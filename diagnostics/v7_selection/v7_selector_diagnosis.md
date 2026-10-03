# V7 selector diagnosis

## Evidence and bottleneck

Candidate-space Pareto analysis (reference L1, preservation distortion, consequence error; strict dominance) shows alternatives are available. V6 Pareto-optimal rates: Conservative 81.35%, Balanced 34.32%, Aggressive 100%; each image has at least one non-dominated alternative among C/B. V4: 84.9%, 87.3%, 100%. V5: 45.9%, 100%, 53.5%. This identifies selector collapse as the primary bottleneck. Retain the V4 candidate generator and improve its selector.

## V7 validation selector

- Targets: `{'Conservative': 48, 'Balanced': 59, 'Aggressive': 263}`; selected: `{'Conservative': 42, 'Balanced': 56, 'Aggressive': 272}`.
- Agreement: 91.622%; entropy: 0.2251 nats; mean utility margin: 5.2123.
- Candidate mean absolute distances: `{'Conservative_vs_Balanced': 0.011103761438406199, 'Balanced_vs_Aggressive': 0.10783049769296839, 'Conservative_vs_Aggressive': 0.11362383519676891}`.

| Severity quartile | Mean severity | C | B | A |
|---|---:|---:|---:|---:|
| Q1 | 0.0816 | 11.8% | 24.7% | 63.4% |
| Q2 | 0.1070 | 19.6% | 26.1% | 54.3% |
| Q3 | 0.1263 | 7.6% | 4.3% | 88.0% |
| Q4 | 0.1650 | 6.5% | 5.4% | 88.2% |

Selection changes across severity groups and all branches are used. Candidate branches remain distinct. However, selection remains concentrated toward Aggressive overall, and validation fidelity is lower than V4/V6. Decision: **B**: improved trade-off, but insufficient evidence to claim a final genuinely adaptive model or freeze V7. No V8. Test set accessed: **NO**.
