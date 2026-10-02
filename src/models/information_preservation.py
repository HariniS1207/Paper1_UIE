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
        """
        Returns multi-scale fixed representations.
        """

        gray = self.grayscale(x)

        # Normalize per image to reduce sensitivity to global
        # illumination differences.
        gray = (
            gray - gray.mean(dim=(2, 3), keepdim=True)
        ) / (
            gray.std(dim=(2, 3), keepdim=True) + 1e-6
        )

        gradient = self.gradient(x)

        # Multi-scale representations
        scale_1 = gray

        scale_2 = F.avg_pool2d(
            gray,
            kernel_size=2,
            stride=2
        )

        scale_4 = F.avg_pool2d(
            gray,
            kernel_size=4,
            stride=4
        )

        gradient_2 = F.avg_pool2d(
            gradient,
            kernel_size=2,
            stride=2
        )

        return {
            "fine": scale_1,
            "mid": scale_2,
            "coarse": scale_4,
            "gradient": gradient_2
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
        """
        Computes normalized absolute feature difference.

        Returns values approximately in [0, 1] after normalization.
        """

        difference = torch.abs(original - enhanced)

        scale = (
            original.abs().mean(
                dim=(1, 2, 3),
                keepdim=True
            ) + 1e-6
        )

        difference = difference / scale

        # Robustly bound the distortion.
        difference = difference / (1.0 + difference)

        return difference

    def forward(self, original, enhanced):

        original_features = self.feature_extractor.extract(
            original
        )

        enhanced_features = self.feature_extractor.extract(
            enhanced
        )

        # ---------------------------------------------------------
        # Multi-scale feature distortion
        # ---------------------------------------------------------

        fine_difference = self.normalized_difference(
            original_features["fine"],
            enhanced_features["fine"]
        )

        mid_difference = self.normalized_difference(
            original_features["mid"],
            enhanced_features["mid"]
        )

        coarse_difference = self.normalized_difference(
            original_features["coarse"],
            enhanced_features["coarse"]
        )

        gradient_difference = self.normalized_difference(
            original_features["gradient"],
            enhanced_features["gradient"]
        )

        # ---------------------------------------------------------
        # Convert distortion -> preservation
        # ---------------------------------------------------------

        fine_score = 1.0 - fine_difference.mean(
            dim=(1, 2, 3)
        )

        mid_score = 1.0 - mid_difference.mean(
            dim=(1, 2, 3)
        )

        coarse_score = 1.0 - coarse_difference.mean(
            dim=(1, 2, 3)
        )

        gradient_score = 1.0 - gradient_difference.mean(
            dim=(1, 2, 3)
        )

        preservation_score = (
            self.fine_weight * fine_score +
            self.mid_weight * mid_score +
            self.coarse_weight * coarse_score +
            self.gradient_weight * gradient_score
        )

        # Numerical safety
        preservation_score = torch.clamp(
            preservation_score,
            0.0,
            1.0
        )

        # ---------------------------------------------------------
        # Spatial preservation map
        # ---------------------------------------------------------

        spatial_difference = (
            0.50 * fine_difference +
            0.30 * F.interpolate(
                mid_difference,
                size=fine_difference.shape[-2:],
                mode="bilinear",
                align_corners=False
            ) +
            0.20 * F.interpolate(
                coarse_difference,
                size=fine_difference.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        )

        preservation_map = 1.0 - spatial_difference

        preservation_map = torch.clamp(
            preservation_map,
            0.0,
            1.0
        )

        return preservation_map, preservation_score