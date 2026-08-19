"""
DnCNN — residual-learning denoiser (Zhang et al., 2017).

Predicts the *noise residual* (input - clean) rather than the clean image
directly. This is deliberate: with speckle noise the "residual" the network
actually has to learn is x*n (multiplicative), which is still far easier to
fit than the raw image content, and the residual formulation trains faster
and more stably than direct clean-image regression (see original DnCNN
paper, Sec. 3).
"""
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_layers: int = 17, num_features: int = 64):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True)]

        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: degraded (noisy) image, range roughly [0,1] (may exceed slightly).
        Returns the *denoised* image, computed as input minus predicted residual."""
        residual = self.body(x)
        return x - residual
