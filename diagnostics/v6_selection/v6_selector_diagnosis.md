# V6 Selector Diagnosis (Validation Only)

## Result

**Failed adaptive-selection experiment.** The V6 selector chose Aggressive on all 370 validation images. Its target was 367 Aggressive, 1 Balanced, and 2 Conservative, so target-selector agreement of 99.19% is dominated by the Aggressive majority. Mean selection entropy was 0.0181 nats. The selection class is constant across validation severity groups, so V6 does not demonstrate condition-dependent branch choice.

V6 best checkpoint SHA-256: `6E4FFC2DFED363BB5748D90925F78D4D748A0805A7B46B77F81998432E205E0F`.

## Candidate quality on validation

| Candidate | Target | Selected | PSNR | SSIM | Preservation | Consequence L1 |
|---|---:|---:|---:|---:|---:|---:|
| Conservative | 2 | 0 | 16.402 | 0.7171 | 0.9952 | 0.1472 |
| Balanced | 1 | 0 | 16.005 | 0.7116 | 0.9944 | 0.1544 |
| Aggressive | 367 | 370 | 23.396 | 0.8140 | 0.9545 | 0.0531 |

## Severity groups

Severity is the validation input-reference mean absolute RGB error; quartiles are descriptive.

| Group | n | Mean severity | Target C/B/A | Selected C/B/A | PSNR | SSIM |
|---|---:|---:|---|---|---:|---:|
| Q1 | 93 | 0.0816 | 2/0/91 | 0/0/93 | 24.572 | 0.8132 |
| Q2 | 92 | 0.1070 | 0/1/91 | 0/0/92 | 23.841 | 0.8278 |
| Q3 | 92 | 0.1263 | 0/0/92 | 0/0/92 | 23.682 | 0.8183 |
| Q4 | 93 | 0.1650 | 0/0/93 | 0/0/93 | 21.498 | 0.7968 |

Aggressive selection probability vs severity Pearson r: 0.07519597834397013. Since branch selection is constant, probability correlation measures confidence variation only and cannot establish adaptive behavior.

## Validation-only comparison

| Model | Best epoch | Total | Reconstruction | Preservation | Consequence | Selection | Target / selection
|---|---:|---:|---:|---:|---:|---:|---|
| V3 | 8 | 0.41076 | 0.07890 | 0.04721 | 0.06035 | 0.62071 | not applicable (single output) |
| V4 | 10 | 0.09028 | 0.05506 | 0.01753 | 0.05231 | 0.04249 | 0/0/370 / 0/0/370 |
| V5 | 12 | 0.08238 | 0.06673 | 0.03214 | 0.04110 | 0.00200 | 0/370/0 / 0/370/0 |
| V6 | 1 | 0.08455 | 0.05503 | 0.01873 | 0.05313 | 0.03031 | 2/1/367 / 0/0/370 |

V3 has a single-output architecture; selection and candidate diversity are not applicable. V4 selected Aggressive for all validation images. V5 selected Balanced for all validation images. V6 targets were less collapsed, but its trained selector still selected Aggressive for every image. Candidate diversity and complete condition-group candidate metrics are recorded in the JSON diagnosis.

## Decision

Do not freeze V6 as an adaptive model and do not run the held-out test. The run is a negative result for the adaptive-selection claim; the next experiment should diagnose why the selector ignores its minority target classes before changing the target or adding losses.

Baseline V3/V4 hashes match the recorded known hashes; V5 hash is recorded. No test split or test images were read.

**TEST SET ACCESSED: NO**
