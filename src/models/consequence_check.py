import torch
import torch.nn as nn


class ConsequenceCheckModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 3, 4, stride=2, padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, original, enhanced):
        enhanced_features = self.encoder(enhanced)

        reconstructed = self.reconstruction_head(
            enhanced_features
        )

        consequence_error = torch.mean(
            torch.abs(original - reconstructed),
            dim=(1, 2, 3)
        )

        consequence_score = torch.exp(
            -consequence_error
        )

        return {
            "reconstructed": reconstructed,
            "consequence_error": consequence_error,
            "consequence_score": consequence_score
        }