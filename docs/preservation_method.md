# Fixed-Preservation Experiment

This document describes the **new fixed-preservation experiment**. It is not the preservation mechanism used by the historical V2 baseline. Its scores must not be mixed with the previously recorded V2 preservation scores. A future trained model using this mechanism is designated V3; V3 training has not started.

## Mathematical definition

Given an underwater input image $x$ and a candidate enhancement $e$, the preservation module measures how much information is retained after the candidate transformation.

The core formulation is:

$$
P(x, e) = 1 - D(F(x), F(e))
$$

where $F(\cdot)$ is a fixed multi-scale feature extractor and $D(\cdot, \cdot)$ is a bounded feature distortion measure.

The current implementation uses a fixed, frozen, non-trainable feature bank:

- fine-scale grayscale structure
- mid-scale grayscale structure after $2\times2$ pooling
- coarse-scale grayscale structure after $4\times4$ pooling
- gradient magnitude information derived from Sobel responses

The feature distortion is computed as a normalized absolute difference:

$$
D(f_x, f_e) = \frac{|f_x - f_e|}{|f_x| + \epsilon}
$$

followed by a bounded transform:

$$
\tilde{D} = \frac{D}{1 + D}
$$

which keeps the distortion stable and prevents runaway values.

The preservation score is then a weighted combination of the per-scale fidelity terms:

$$
P(x, e) = w_f P_f + w_m P_m + w_c P_c + w_g P_g
$$

with:

- $w_f = 0.30$
- $w_m = 0.25$
- $w_c = 0.20$
- $w_g = 0.25$

and each component is:

$$
P_s = 1 - \mathrm{mean}(\tilde{D}_s)
$$

## Rationale

This formulation avoids the common failure mode of a trainable scalar preservation head that can simply output values near 1 for almost every candidate. Instead, preservation is measured directly from the image content itself.

The module evaluates multiple scales of information:

- fine scale captures local structure and texture retention
- mid scale captures intermediate spatial relationships
- coarse scale captures global scene structure
- gradient scale captures edge and contour preservation

This creates a measurable and interpretable notion of what is preserved after enhancement.

## Implementation detail

The implementation is located at `src/models/information_preservation.py` and uses:

- grayscale conversion with standard luma weights: $0.299R + 0.587G + 0.114B$
- Sobel-based gradient magnitude extraction
- per-image normalization to reduce sensitivity to global illumination shifts
- mean distortion reduction across channel and spatial dimensions
- final clamp to `[0,1]`

The output includes:

- `preservation_map`: a spatial map in `[0,1]`
- `preservation_score`: a batch-level scalar score in `[0,1]`

## Normalization and stability

The distortion term is intentionally bounded and normalized to avoid a trainable head collapsing to near-1 values. The feature difference is normalized by the per-image intensity scale and then transformed through a monotonic saturation function.

This keeps the score numerically stable under minibatch training and GPU execution.

## Expected behavior

The score should satisfy the following ordering on realistic distortions:

- identity image $\rightarrow$ highest preservation
- mild blur/noise/contrast change $\rightarrow$ moderate reduction
- strong enhancement, severe smoothing, or strong color distortion $\rightarrow$ lower preservation

This behavior is validated by the monotonicity test in `src/models/test_information_preservation_monotonic.py`, which requires the ordering `identity > mild > strong > severe`.

## Limitations

This is a fixed, physics-agnostic preservation proxy rather than a full optical or degradation model. It is intentionally interpretable and stable, but it is not tied to a specific underwater radiative transfer model. It should therefore be viewed as a practical information-preservation signal rather than a full physical fidelity model.

It is best used as a research signal for candidate selection and candidate comparison, not as a calibrated physical ground truth metric.
