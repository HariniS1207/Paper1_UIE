# V5 selector diagnosis

## Checkpoint and safety
V5 epoch 12 `73F62495B31F226A5BB71459A5F6587FBD68A9BE204BFD26FF1C46B0C0DEFE9B`; V4 `5C877EDD0094E0C3F35AD977B5EA7651FFA7E22FBF4C38C15006AECE19E7C8D7`; V3 `80163DBA423CCDE672318A3DD4F93B68982ADE6A0CD3ED737FA8D326948D0555`. 370 validation and 2,960 train images. Only train/validation lists were used; no held-out evaluation or optimizer step. Checkpoint hashes unchanged during diagnosis; no source/checkpoint was modified.

## Exact objective and target
`src/losses/paper1_loss_v5.py::Paper1LossV5.candidate_targets`: `S(c)=mean((L_k(c)-median_train[k])/IQR_train[k])`, equal weights, terms reference L1, preservation distortion `1-score`, and consequence L1. Training-only calibration: `data/splits/train.txt`, 8880 observations; median `[0.1054370328783989, 0.00441405177116394, 0.11057379096746445]`, IQR `[0.0668342262506485, 0.036563098430633545, 0.07983411848545074]`. The same calibration is applied on validation. JSON contains exact terms, z-normalized terms, combined score, and argmin for each image.

| Target source | Conservative | Balanced | Aggressive |
|---|---:|---:|---:|
| Initial training labels | 28 | 228 | 2704 |
| Epoch 12 train inference | 0 | 2958 | 2 |
| Validation | 0 | 370 | 0 |

## Validation quality
Values are mean / median / population std / min / max.

| Candidate | reference_l1 | preservation_distortion | consequence_l1 | combined_score |
|---|---|---|---|---|
| Conservative | 0.178506 / 0.177540 / 0.039263 / 0.066851 / 0.363197 | 0.031857 / 0.028945 / 0.008352 / 0.020932 / 0.062811 | 0.155638 / 0.159622 / 0.017684 / 0.081413 / 0.186460 | 0.802778 / 0.792190 / 0.202078 / 0.242186 / 1.585492 |
| Balanced | 0.066598 / 0.061964 / 0.024171 / 0.029791 / 0.222092 | 0.033660 / 0.033378 / 0.004486 / 0.023593 / 0.049709 | 0.041106 / 0.039390 / 0.013204 / 0.015037 / 0.082440 | -0.217133 / -0.237782 / 0.147142 / -0.495509 / 0.550739 |
| Aggressive | 0.185732 / 0.184608 / 0.039259 / 0.084250 / 0.357209 | 0.030744 / 0.029996 / 0.008324 / 0.011307 / 0.061133 | 0.166736 / 0.161858 / 0.039223 / 0.062289 / 0.274812 | 0.875009 / 0.828959 / 0.362233 / 0.096015 / 2.263124 |


V4 target rows by V5 target columns: `{"Conservative": {"Conservative": 0, "Balanced": 1, "Aggressive": 0}, "Balanced": {"Conservative": 0, "Balanced": 3, "Aggressive": 0}, "Aggressive": {"Conservative": 0, "Balanced": 366, "Aggressive": 0}}`. Component minima counts: `{"reference_l1": {"Conservative": 3, "Balanced": 367, "Aggressive": 0}, "preservation_distortion": {"Conservative": 136, "Balanced": 48, "Aggressive": 186}, "consequence_l1": {"Conservative": 0, "Balanced": 370, "Aggressive": 0}, "combined_score": {"Conservative": 0, "Balanced": 370, "Aggressive": 0}}`. Per-image transitions and decompositions are included in JSON.

## Selector and condition dependence
Target-to-selector confusion: `{"Conservative": {"Conservative": 0, "Balanced": 0, "Aggressive": 0}, "Balanced": {"Conservative": 0, "Balanced": 370, "Aggressive": 0}, "Aggressive": {"Conservative": 0, "Balanced": 0, "Aggressive": 0}}`; mean probability margin 0.99628, entropy 0.00793 nats. Both target and selector choose Balanced for all 370 validation images, so the observed collapse is already present in the target. Absolute utility means were not retained in the existing per-image validation artifact; probabilities/margins are reported rather than fabricating logits. The previous report's degradation quartiles each select Balanced. Condition vector is retained per image for downstream audit; no new clustering was used.

## Train/validation and normalization
Epoch 12 train target counts `{'Conservative': 0, 'Balanced': 2958, 'Aggressive': 2}` vs validation `{'Conservative': 0, 'Balanced': 370, 'Aggressive': 0}`; original train labels `{'conservative': 28, 'balanced': 228, 'aggressive': 2704}`. This indicates current train/validation target shift. Exact raw and normalized component distributions for both splits are in JSON. Normalization applies consistently, but was calibrated on V4-initialized candidate outputs before V5 updates, so stale calibration remains a possible contributor, not an implementation mismatch.

## Diversity and preservation/fidelity
Existing V5 and frozen V4 validation diversity summaries (pairwise cosine, residual norms, candidate distances, preservation, candidate quality) are included in JSON. Branches remain distinct, so candidate generation collapse is not supported. V5 selected mean preservation distortion is 0.033660 versus V4 0.045017 and consequence is 0.041106 versus 0.052315; reference L1 worsens 0.054925 to 0.066598. Selected PSNR decreases 1.713 dB and SSIM 0.0169. The preservation gain comes with a material fidelity cost.

## Root cause and decision
The V5 combined target ranks Balanced first on all validation images; the selector follows it with high confidence. This is a target collapse, not independent selector collapse. Candidate branches remain diverse. The normalization implementation is consistent, though its calibration predates V5 candidate changes. Preservation/consequence improve at material fidelity cost. Changing the generator is unsupported.

**Decision: A. TARGET FORMULATION NEEDS REFINEMENT.**

**Recommended single next modification:** impose a predeclared reference-L1 eligibility constraint during target construction, then rank eligible candidates using the existing preservation/consequence terms. Keep normalization, selector, and candidate generator fixed. No V6 was trained.
