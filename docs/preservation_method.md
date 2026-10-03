# Fixed-Preservation Experiment

This document describes the **new fixed-preservation experiment**. It is not the preservation mechanism used by the historical V2 baseline. Its scores must not be mixed with the previously recorded V2 preservation scores. A future trained model using this mechanism is designated V3; V3 training has not started.

## Mathematical definition

Given an underwater input image $x$ and a candidate enhancement $e$, the preservation module measures how much information is retained after the candidate transformation.

The core formulation is:

$$
P(x, e) = 1 - D(F(x), F(e))
$$

where $F(\cdot)$ is a fixed multi-scale feature extractor and $D(\cdot, \cdot)$ is a bounded feature distortion measure.

The fixed feature bank contains raw luma, independently standardized luma, two opponent-chroma channels, and Sobel gradient magnitude. Raw luma and standardized luma/chroma are pooled at fine, $2\times2$, and $4\times4$ scales. Sobel magnitude is computed from raw luma and pooled by $2\times2$.

For RGB input in $[0,1]$:

$$
Y = 0.299R + 0.587G + 0.114B,\qquad
Z = \frac{Y-\mu(Y)}{\sigma(Y)+\epsilon}
$$

$$
C_{rg}=R-G,\qquad C_{by}=\frac{R+G}{2}-B
$$

At scale $s\in\{1,2,4\}$, let $P_s$ denote identity, $2\times2$ average pooling, or $4\times4$ average pooling. Define the per-scale distortions:

$$
D_I^s=\operatorname{mean}|P_s(Y_x)-P_s(Y_e)|
$$

$$
D_S^s=\operatorname{mean}\left[\rho\left(
\frac{|P_s(Z_x)-P_s(Z_e)|}
{\operatorname{mean}(|P_s(Z_x)|)+\epsilon}
\right)\right]
$$

$$
D_C^s=\operatorname{mean}_{c\in\{rg,by\}}
\left[\rho\left(\frac{|P_s(C_{x,c})-P_s(C_{e,c})|}{2}\right)\right],\qquad
\rho(t)=\frac{t}{1+t}
$$

The raw-luma difference needs no relative normalization because luma is in $[0,1]$. The opponent-channel difference is at most 2, so it is normalized by that fixed range before bounding. Standardized structure retains the existing per-image relative normalization and bounded difference. Let $D_G$ be the existing relative bounded difference between the pooled Sobel-magnitude maps.

Blend the three luma/chroma terms equally at each scale, then preserve the existing scale and gradient weights:

$$
D_s=\frac{D_I^s+D_S^s+D_C^s}{3},\qquad
P(x,e)=1-\left(0.30D_1+0.25D_2+0.20D_4+0.25D_G\right)
$$

The spatial map uses the same per-pixel blend at each scale, resized and combined with the existing $0.50/0.30/0.20$ fine/mid/coarse map weights. No trainable preservation parameters are introduced. The module remains differentiable almost everywhere.

## Rationale

This formulation avoids a trainable scalar preservation head that could self-report high scores. Raw luma adds sensitivity to brightness/intensity changes; standardized luma retains relative structure sensitivity; opponent chroma adds fixed color sensitivity; and Sobel magnitude measures edge/detail differences.

The module evaluates multiple scales of information:

- raw-luma scales capture intensity fidelity from local to coarse structure
- standardized-luma scales capture relative spatial structure and texture
- opponent-chroma scales capture red/green and yellow/blue color changes
- the gradient scale captures edge-strength and contour changes

This creates a measurable and interpretable notion of what is preserved after enhancement.

## Implementation detail

The implementation is located at `src/models/information_preservation.py` and uses:

- grayscale conversion with standard luma weights: $0.299R + 0.587G + 0.114B$
- raw luma pyramids with fixed-range absolute distortion
- independently standardized luma pyramids with relative bounded distortion
- fixed opponent channels $R-G$ and $(R+G)/2-B$ with a fixed range of 2
- Sobel-based gradient magnitude extraction
- per-image normalization only for the standardized-structure branch
- mean distortion reduction across channel and spatial dimensions
- final clamp to `[0,1]`

The output includes:

- `preservation_map`: a spatial map in `[0,1]`
- `preservation_score`: a batch-level scalar score in `[0,1]`

## Normalization and stability

Standardized-structure, chroma, and Sobel distortions use bounded transforms. Raw-luma distortion is bounded by the input range. The fixed branches contain no trainable parameters. Independent grayscale standardization intentionally makes the structure branch invariant to global affine intensity changes; the raw-luma branch ensures those changes still lower the overall preservation score.

This keeps the score numerically stable under minibatch training and GPU execution.

## Expected behavior

The score should satisfy the following ordering on realistic distortions:

- identity image $\rightarrow$ highest preservation
- mild blur/noise/contrast change $\rightarrow$ moderate reduction
- strong enhancement, severe smoothing, or strong color distortion $\rightarrow$ lower preservation

Controlled tests in `src/models/test_information_preservation_monotonic.py` report component and total distortions for identity, positive/negative brightness changes, contrast, a luma-preserving color cast, blur, noise, edge destruction, combined distortion, and a real EUVP training image.

## Limitations

This is a fixed, physics-agnostic preservation proxy rather than a full optical or degradation model. Raw-luma sensitivity also penalizes legitimate exposure correction, and the two opponent channels are not perceptually uniform. The existing zero-padded Sobel convolution can create a small boundary response to a uniform brightness offset; the raw-luma branch is the intended global-intensity signal. It is a practical information-preservation signal, not calibrated physical ground truth.

It is best used as a research signal for candidate selection and candidate comparison, not as a calibrated physical ground truth metric.
