"""
src/models/cnn.py
==================
2-D Convolutional Neural Network for spatiotemporal PDE surrogate modeling.

Architecture
------------
The CNN uses a stack of residual 2-D conv blocks to map the flattened context
tensor to the next time-step prediction.  Residual connections stabilize
training and help the model learn the identity mapping (small δu from previous
state) without vanishing gradients.

    Input [B, C*initial_step, H, W]
    → stem Conv2d → BN → act
    → N residual blocks (3×3 conv, BN, act, 3×3 conv, BN + skip)
    → head Conv2d(base_ch, C, 1) → output [B, C, H, W]

The spatial dimensions are preserved throughout (padding=1 on all 3×3 convs).
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock2d(nn.Module):
    """
    2-D residual block: Conv → BN → Act → Conv → BN + skip.

    If ``in_ch != out_ch``, a 1×1 projection is used for the skip connection.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip  = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


# ---------------------------------------------------------------------------
# CNN surrogate model
# ---------------------------------------------------------------------------

class CNN2d(nn.Module):
    """
    Residual 2-D CNN surrogate model.

    Parameters
    ----------
    in_channels  : int   C * initial_step
    out_channels : int   C
    base_channels: int   feature map width (default 32)
    n_layers     : int   number of residual blocks (default 5)
    kernel_size  : int   convolution kernel size (default 3)
    dropout      : float dropout probability (default 0)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        base_channels: int = 32,
        n_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        pad = kernel_size // 2

        # Stem: lift to base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        # Residual blocks (all at base_channels width)
        self.blocks = nn.Sequential(
            *[
                ResBlock2d(base_channels, base_channels, kernel_size, dropout)
                for _ in range(n_layers)
            ]
        )

        # Projection head: base_channels → out_channels
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : FloatTensor [B, C*initial_step, H, W]

        Returns
        -------
        out : FloatTensor [B, C, H, W]
        """
        out = self.stem(x)
        out = self.blocks(out)
        return self.head(out)

    # ------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: dict, sample_xx: torch.Tensor) -> "CNN2d":
        """Build from config and a sample input."""
        B, H, W, T, C = sample_xx.shape
        mcfg = cfg.get("cnn", {})
        return CNN2d(
            in_channels   = C * T,
            out_channels  = C,
            base_channels = mcfg.get("base_channels", 32),
            n_layers      = mcfg.get("n_layers", 5),
            kernel_size   = mcfg.get("kernel_size", 3),
            dropout       = mcfg.get("dropout", 0.0),
        )
