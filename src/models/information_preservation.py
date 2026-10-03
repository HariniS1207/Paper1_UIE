import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedFeatureExtractor(nn.Module):
    """
    Frozen multi-scale feature extractor.

    Uses fixed image-processing operations rather than a learnable
    preservation head. This makes preservation measurable rather than
    self-reported by the network.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels for structural/edge information
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]]
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def grayscale(self, x):
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]

        return (
            0.299 * r +
            0.587 * g +
            0.114 * b
        )

    def chroma(self, x):
        red = x[:, 0:1]
        green = x[:, 1:2]
        blue = x[:, 2:3]
        red_green = red - green
        yellow_blue = 0.5 * (red + green) - blue
        return torch.cat((red_green, yellow_blue), dim=1)

    def gradient(self, x):
        gray = self.grayscale(x)

        gx = F.conv2d(
            gray,
            self.sobel_x,
            padding=1
        )

        gy = F.conv2d(
            gray,
            self.sobel_y,
            padding=1
        )

        magnitude = torch.sqrt(
            gx.pow(2) + gy.pow(2) + 1e-6
        )

        return magnitude

    def extract(self, x):
        """Return fixed raw-luma, structural, chroma, and edge pyramids."""
        luma = self.grayscale(x)
        standardized_luma = (
            luma - luma.mean(dim=(2, 3), keepdim=True)
        ) / (
            luma.std(dim=(2, 3), keepdim=True) + 1e-6
        )
        chroma = self.chroma(x)

        def pyramid(feature):
            return {
                "fine": feature,
                "mid": F.avg_pool2d(feature, kernel_size=2, stride=2),
                "coarse": F.avg_pool2d(feature, kernel_size=4, stride=4),
            }

        return {
            "raw_luma": pyramid(luma),
            "standardized_luma": pyramid(standardized_luma),
            "chroma": pyramid(chroma),
            "gradient": F.avg_pool2d(
                self.gradient(x),
                kernel_size=2,
                stride=2,
            ),
        }


class InformationPreservationModule(nn.Module):
    """
    Research-grade information preservation module.

    Preservation is computed from fixed multi-scale image features
    rather than a trainable preservation head.

    Output:
        preservation_map   : spatial preservation representation
        preservation_score : scalar score in [0, 1]
    """

    def __init__(
        self,
        fine_weight=0.30,
        mid_weight=0.25,
        coarse_weight=0.20,
        gradient_weight=0.25
    ):
        super().__init__()

        self.feature_extractor = FixedFeatureExtractor()

        self.fine_weight = fine_weight
        self.mid_weight = mid_weight
        self.coarse_weight = coarse_weight
        self.gradient_weight = gradient_weight

    def normalized_difference(self, original, enhanced):
        difference = torch.abs(original - enhanced)
        scale = (
            original.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
        )
        difference = difference / scale
        return difference / (1.0 + difference)

    @staticmethod
    def chroma_difference(original, enhanced):
        difference = torch.abs(original - enhanced) / 2.0
        return difference / (1.0 + difference)

    def _measure(self, original, enhanced):
        original_features = self.feature_extractor.extract(original)
        enhanced_features = self.feature_extractor.extract(enhanced)

        scale_weights = {
            "fine": self.fine_weight,
            "mid": self.mid_weight,
            "coarse": self.coarse_weight,
        }
        scale_distortions = {}
        spatial_distortions = {}
        component_distortions = {
            "raw_luma": 0.0,
            "standardized_structure": 0.0,
            "chroma": 0.0,
        }
        scale_weight_sum = sum(scale_weights.values())

        for scale, weight in scale_weights.items():
            raw_luma_difference = torch.abs(
                original_features["raw_luma"][scale]
                - enhanced_features["raw_luma"][scale]
            )
            structure_difference = self.normalized_difference(
                original_features["standardized_luma"][scale],
                enhanced_features["standardized_luma"][scale],
            )
            chroma_difference = self.chroma_difference(
                original_features["chroma"][scale],
                enhanced_features["chroma"][scale],
            )

            raw_luma_score = raw_luma_difference.mean(dim=(1, 2, 3))
            structure_score = structure_difference.mean(dim=(1, 2, 3))
            chroma_score = chroma_difference.mean(dim=(1, 2, 3))
            scale_distortions[scale] = (
                raw_luma_score + structure_score + chroma_score
            ) / 3.0
            component_distortions["raw_luma"] = (
                component_distortions["raw_luma"]
                + weight * raw_luma_score
            )
            component_distortions["standardized_structure"] = (
                component_distortions["standardized_structure"]
                + weight * structure_score
            )
            component_distortions["chroma"] = (
                component_distortions["chroma"] + weight * chroma_score
            )

            spatial_distortions[scale] = (
                raw_luma_difference
                + structure_difference
                + chroma_difference.mean(dim=1, keepdim=True)
            ) / 3.0

        gradient_difference = self.normalized_difference(
            original_features["gradient"],
            enhanced_features["gradient"],
        )
        sobel_score = gradient_difference.mean(dim=(1, 2, 3))

        raw_luma_score = component_distortions["raw_luma"] / scale_weight_sum
        structure_score = (
            component_distortions["standardized_structure"] / scale_weight_sum
        )
        chroma_score = component_distortions["chroma"] / scale_weight_sum
        total_distortion = (
            sum(
                weight * scale_distortions[scale]
                for scale, weight in scale_weights.items()
            )
            + self.gradient_weight * sobel_score
        )
        preservation_score = torch.clamp(1.0 - total_distortion, 0.0, 1.0)

        fine_difference = spatial_distortions["fine"]
        spatial_difference = (
            0.50 * fine_difference
            + 0.30
            * F.interpolate(
                spatial_distortions["mid"],
                size=fine_difference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + 0.20
            * F.interpolate(
                spatial_distortions["coarse"],
                size=fine_difference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        preservation_map = torch.clamp(1.0 - spatial_difference, 0.0, 1.0)

        diagnostics = {
            "preservation_score": preservation_score,
            "raw_luma_distortion": raw_luma_score,
            "standardized_structure_distortion": structure_score,
            "chroma_distortion": chroma_score,
            "sobel_distortion": sobel_score,
            "total_distortion": total_distortion,
        }
        return preservation_map, diagnostics

    def compute_distortions(self, original, enhanced):
        """Return per-image scalar preservation and component distortions."""
        _, diagnostics = self._measure(original, enhanced)
        return diagnostics

    def forward(self, original, enhanced):
        preservation_map, diagnostics = self._measure(original, enhanced)
        return preservation_map, diagnostics["preservation_score"]