"""
ESPCN — efficient sub-pixel-convolution super-resolution (Shi et al., 2016).

Upsamples by learning feature maps at the LOW resolution and only expanding
to full resolution in the final pixel-shuffle layer, rather than upsampling
first and convolving at high resolution (which SRCNN does, and which wastes
compute). This keeps inference cheap, which matters for the KLA
end-to-end-throughput scoring axis.
"""
import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, in_channels: int = 3, upscale_factor: int = 2, num_features: int = 64):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.to_subpixel = nn.Conv2d(
            num_features // 2, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: low-resolution (denoised) image. Returns image upscaled by
        `upscale_factor`, i.e. restored to the expected ground-truth
        resolution."""
        feat = self.feature_extract(x)
        feat = self.to_subpixel(feat)
        return self.pixel_shuffle(feat)
