# V7 Candidate-Space Forensic Analysis (Validation Artifacts)

Scope: frozen V4/V5/V6 validation outputs only (370 paired validation images). No held-out test manifest or images loaded. This diagnoses available candidates; it does not tune the V7 target. Pareto objectives minimize reference L1, fixed preservation distortion (1 - score), and consequence L1. A candidate is dominated only if another is no worse on all three and strictly better on at least one.

## Pareto-optimal candidates by severity quartile

Percentages are per-candidate occurrence among images in each group; multiple candidates can be Pareto-optimal for the same image. Severity quartile boundaries reuse the already-recorded V6 validation groups.

### V4

| Group | n | C Pareto % | B Pareto % | A Pareto % | Mean Pareto candidates/image | Images with C or B Pareto |
|---|---:|---:|---:|---:|---:|---:|
| all | 370 | 84.9 | 87.3 | 100.0 | 2.72 | 370 |
| Q1 | 93 | 76.3 | 84.9 | 100.0 | 2.61 | 93 |
| Q2 | 92 | 73.9 | 80.4 | 100.0 | 2.54 | 92 |
| Q3 | 92 | 95.7 | 90.2 | 100.0 | 2.86 | 92 |
| Q4 | 93 | 93.5 | 93.5 | 100.0 | 2.87 | 93 |
### V5

| Group | n | C Pareto % | B Pareto % | A Pareto % | Mean Pareto candidates/image | Images with C or B Pareto |
|---|---:|---:|---:|---:|---:|---:|
| all | 370 | 45.9 | 100.0 | 53.5 | 1.99 | 370 |
| Q1 | 93 | 44.1 | 100.0 | 34.4 | 1.78 | 93 |
| Q2 | 92 | 56.5 | 100.0 | 42.4 | 1.99 | 92 |
| Q3 | 92 | 45.7 | 100.0 | 64.1 | 2.10 | 92 |
| Q4 | 93 | 37.6 | 100.0 | 73.1 | 2.11 | 93 |
### V6

| Group | n | C Pareto % | B Pareto % | A Pareto % | Mean Pareto candidates/image | Images with C or B Pareto |
|---|---:|---:|---:|---:|---:|---:|
| all | 370 | 81.4 | 34.3 | 100.0 | 2.16 | 370 |
| Q1 | 93 | 60.2 | 53.8 | 100.0 | 2.14 | 93 |
| Q2 | 92 | 73.9 | 34.8 | 100.0 | 2.09 | 92 |
| Q3 | 92 | 94.6 | 20.7 | 100.0 | 2.15 | 92 |
| Q4 | 93 | 96.8 | 28.0 | 100.0 | 2.25 | 93 |

## Diagnosis

**CASE A ? selector bottleneck.** With the V6 candidate generator, Conservative is Pareto-optimal on 81.4% of validation images and Balanced on 34.3%; each appears in every severity quartile (Conservative 60.2?96.8%, Balanced 20.7?53.8%). Aggressive is also Pareto-optimal on all images, so there are genuine tradeoffs, not a single dominating candidate. V5 shows even more alternatives (Balanced 100%, Conservative 45.9%, Aggressive 53.5%). The V4 candidate branches are distinct, and non-aggressive branches remain non-dominated broadly. Keep candidate generation unchanged; make V7?s selector explicitly condition-aware and give it candidate-specific preservation/consequence inputs.

The prior V4/V5/V6 selectors each collapsed (to Aggressive/Balanced/Aggressive respectively), even though their candidate spaces contain alternatives. The V7 experiment therefore isolates the selector. The JSON includes per-image objectives and Pareto sets, severity quartile summaries, V6 condition vectors, and existing candidate distance/residual-diversity summaries.

Checkpoint hash inventory captured before V7: `frozen_checkpoint_hashes_before.json`.


## Frozen training-only V7 target audit

V7 uses an equal-weight candidate-relative Borda rank: for each training image, rank the three candidates independently on reference L1, preservation distortion, and consequence L1 (lower is better; ties share average rank); divide each by 2 and average. The target is the candidate with minimum average rank. There are no class quotas, resampling, or global metric scales. On the 2,960 V4-best training candidate records this yields counts C/B/A = 339/517/2,104 (11.5%/17.5%/71.1%). By increasing input-reference severity quartile, target counts are:

| Train group | n | Mean severity | Conservative | Balanced | Aggressive |
|---|---:|---:|---:|---:|---:|
| Q1 | 740 | 0.0815 | 113 | 254 | 373 |
| Q2 | 740 | 0.1058 | 109 | 177 | 454 |
| Q3 | 740 | 0.1259 | 71 | 67 | 602 |
| Q4 | 740 | 0.1688 | 46 | 19 | 675 |

The target is not class-balanced. Its shift toward Aggressive at higher measured degradation emerges from per-image candidate metrics in train.txt. Full per-image labels and Pareto sets are in the JSON. This target audit is training-only; validation was reserved for evaluation.
