# Paper 1 Repository Audit and Experimental Status

## Experimental boundary

The terminology in this repository distinguishes four separate items:

- **V2** means the previously recorded baseline experiment and its historical validation results. Those results are recorded in `docs/v2_baseline_boundary.md`; the current repository cannot exactly reproduce them.
- **V2 checkpoint** means the frozen historical artifact `checkpoints/v2/paper1_v2_epoch_006.pth`. Its legacy trainable preservation branch is present in the checkpoint, but its exact original forward implementation is not recoverable from the current repository. Strict loading into the current model fails structurally.
- **Fixed-preservation experiment** means the new, non-trainable multi-scale preservation mechanism documented in `docs/preservation_method.md`. It is not V2, and its scores must not be mixed with the historical V2 results.
- **V3** means a future trained model using the new preservation mechanism. V3 has not been trained.

The checkpoint is preserved unchanged. Its presence does not establish exact V2 reproducibility.

## 1. Architecture overview

The project already implements the research pipeline described in the proposal:

- Condition encoder: `src/models/condition_encoder.py`
  - `ConditionEncoder`
  - `3 -> 32 -> 64 -> 128 -> 256` encoder with stride-2 conv blocks
  - adaptive average pooling and linear projection to a 128-d condition vector
  - returns `condition_map` and `condition_vector`

- Candidate generator: `src/models/candidate_generator.py`
  - `EnhancementCandidateGenerator`
  - shared residual generator with condition-aware fusion
  - produces `conservative`, `balanced`, and `aggressive` candidate outputs
  - residual is clamped to `[-1, 1]` before application, then each candidate is clamped into `[0, 1]`

- New fixed-preservation experiment: `src/models/information_preservation.py`
  - `InformationPreservationModule`
  - fixed, not trainable, multi-scale feature extractor based on grayscale and Sobel-gradient statistics
  - computes measurable distortion across fine/mid/coarse/gradient scales
  - converts distortion into bounded preservation `P in [0,1]`

- Selector: `src/models/selection_network.py`
  - `UtilitySelectionNetwork`
  - encodes each candidate and combines it with the condition vector and preservation score
  - outputs per-candidate utilities and softmax selection weights
  - selects the argmax candidate index

- Consequence check: `src/models/consequence_check.py`
  - `ConsequenceCheckModule`
  - encoder + decoder reconstructs the original underwater image from the chosen enhancement
  - returns reconstructed image, consequence error, and `exp(-error)` score

- End-to-end model: `src/models/paper1_model.py`
  - `Paper1Model` orchestrates the entire forward pass
  - returns candidate images, preservation scores, utilities, selection weights, selected index, selected image, reconstructed image, consequence error, and consequence score

## 2. Entry points and runtime flow

### Training entry point
- `src/train_v2.py`
- Works with the V2 checkpoint directory: `checkpoints/v2/`
- Batch size: 8
- Epochs: 8
- Learning rate: `1e-4`
- Uses AMP when CUDA is available

### Validation entry point
- `src/validate.py`
- Strict-loads each checkpoint and reports validation loss; incompatible architectures raise a load error

### Evaluation entry point
- `src/evaluate_test.py`
- Loads the selected checkpoint with `strict=True` and evaluates the current model on the paired validation/test split
- Prints aggregate PSNR / SSIM / UIQM / UCIQE tables
- Also reports selection distribution and improvement counts; it fails on the frozen V2 checkpoint because its architecture is incompatible

### Exact model entry point
- `src/models/paper1_model.py`
- `Paper1Model.forward(x)` is the end-to-end forward pass

### Exact loss implementation
- `src/losses/paper1_loss.py`
- The current loss is:
  - reconstruction: weighted soft candidate reconstruction using selection weights
  - preservation: mean of `1 - preservation_score`
  - consequence: `L1(reconstructed, original)`
  - selection: cross-entropy against argmin reference-L1 candidate

## 3. Dataset and splits

The dataset is defined in `src/data/euvp_dataset.py` and uses paired EUVP imagery:

- **Input directory**: `data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet/trainA`
- **Reference directory**: `data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet/trainB`
- **Split files**:
  - train: 2960 images
  - val: 370 images
  - test: 370 images

The transform in `src/data/transforms.py` converts images to tensors in `[0,1]` and applies identical horizontal flip to input and reference during training.

## 4. Tensor ranges and image normalization

- Image tensors from `ToTensor()` are in `[0,1]`
- Candidate outputs are explicitly clamped to `[0,1]`
- The reconstruction head uses a final `Sigmoid`, but the model still uses `[0,1]` image tensors consistently
- PSNR/SSIM are computed with `data_range=1.0` in `src/evaluate_test.py`
- UIQM/UCIQE are implemented on RGB data in `[0,1]` and operate on arrays clipped to `[0,1]` before computation

## 5. Metric implementations

### PSNR / SSIM
- `skimage.metrics.peak_signal_noise_ratio`
- `skimage.metrics.structural_similarity`
- both use `data_range=1.0`
- this is correct for the `[0,1]` image range used by the project

### UIQM / UCIQE
- implemented in `src/evaluate_test.py`
- custom formulas from underwater image quality literature
- both are computed on RGB arrays in `[0,1]`
- no normalization is performed to the extremely large published-paper scales; the code uses the local implementation directly
- this matters when comparing against externally published metrics with different scaling conventions

## 6. Checkpoint format

The checkpoint schema is:

- `epoch`
- `loss`
- `model_state_dict`
- `optimizer_state_dict`

Frozen historical V2 checkpoint artifact:

- `checkpoints/v2/paper1_v2_epoch_006.pth`

The checkpoint contains the legacy trainable preservation branch. Its exact original forward implementation is missing from the current repository, and strict loading into the current model fails structurally. It cannot currently support a claim of exact reproduction of the original V2 experiment. See `docs/v2_baseline_boundary.md`.

## 7. Current findings

### Confirmed findings
- Historical V2 validation results are recorded as **Previously recorded V2 baseline** in `docs/v2_baseline_boundary.md`; they are not claimed to be reproducible by the current repository.
- The fixed multi-scale preservation mechanism is a separate new experiment, not the original V2 preservation mechanism.
- The fixed-preservation monotonicity check verifies `identity > mild > strong > severe`.
- V3 training has not started.

### Known issues / gaps
- Test scripts are not robust to direct execution from the repo root unless the `src` directory is added to `sys.path`.
- Generated bundles exist under `results/v2_baseline/` and `results/inference/`. They predate strict-load cleanup and were produced through the earlier non-strict loading path, so they are not exact V2 reproduction artifacts.
- The current evaluation and single-image scripts use `strict=True`; the frozen V2 checkpoint fails because of the preservation-module architecture mismatch.
- The `PROJECT_AUDIT.md` and mathematical preservation doc were needed to record the actual architecture and research choices.

## 8. Summary

The historical V2 results, V2 checkpoint artifact, fixed-preservation experiment, and future V3 model are distinct. The original V2 forward implementation cannot be recovered from this repository, so exact V2 reproducibility is not established. The fixed-preservation mechanism is documented and its monotonicity test passes; it remains an experiment for a future model version and has not been used to claim new V2 results.
