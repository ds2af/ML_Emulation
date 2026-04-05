"""
src/models/cnn_lstm.py
======================
CNN + ConvLSTM surrogate model for spatiotemporal PDE forecasting.

Input convention
----------------
Forward expects channels-first flattened context:
    x : [B, C*T, H, W]

The tensor is reshaped to [B, T, C, H, W], encoded per time step with a
shared CNN stem, and aggregated temporally with a ConvLSTM cell.

Output
------
    out : [B, C, H, W]
"""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell operating on 2-D feature maps."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class CNNLSTM2d(nn.Module):
    """CNN encoder + ConvLSTM temporal aggregator + conv head."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_steps: int,
        base_channels: int = 32,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_steps = context_steps

        if in_channels != out_channels * context_steps:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) * "
                f"context_steps ({context_steps})"
            )

        pad = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(out_channels, base_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.temporal = ConvLSTMCell(base_channels, hidden_channels, kernel_size=kernel_size)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C*T, H, W] -> out: [B, C, H, W]."""
        b, _, h, w = x.shape
        x = x.reshape(b, self.context_steps, self.out_channels, h, w)

        h_t = torch.zeros(b, self.temporal.hidden_ch, h, w, device=x.device, dtype=x.dtype)
        c_t = torch.zeros_like(h_t)

        for t in range(self.context_steps):
            feat = self.encoder(x[:, t])
            h_t, c_t = self.temporal(feat, h_t, c_t)

        return self.head(h_t)

    @staticmethod
    def from_config(cfg: dict, sample_xx: torch.Tensor) -> "CNNLSTM2d":
        b, h, w, t, c = sample_xx.shape
        _ = (b, h, w)
        mcfg = cfg.get("cnn_lstm", {})
        return CNNLSTM2d(
            in_channels=c * t,
            out_channels=c,
            context_steps=t,
            base_channels=mcfg.get("base_channels", 32),
            hidden_channels=mcfg.get("hidden_channels", 32),
            kernel_size=mcfg.get("kernel_size", 3),
            dropout=mcfg.get("dropout", 0.0),
        )
