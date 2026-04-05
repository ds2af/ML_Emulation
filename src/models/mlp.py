"""
src/models/mlp.py
==================
Coordinate-based MLP surrogate -- the data-only counterpart to the PINN.

Architecture
------------
Identical backbone to the PINN:

    Input  [N, 3]  (x, y, t) physical coordinates
                   x, y in (-2.5, 2.5),  t in (0, 1.0)
    Hidden [h0, h1, ...]  Tanh activations
    Output [N, 1]  water height h only

Differs from the PINN ONLY in the loss function:

    L_MLP  = MSE(h_pred(x,y,t), h_data)                     (this model)
    L_PINN = MSE(h_pred, h_data) + lambda_pde*R_PDE + lambda_ic*R_IC

This design allows a clean ablation: PINN with physics constraints vs.
MLP with pure data fitting, same network capacity and training distribution.

Initialisation: Glorot/Xavier normal (same as PINN reference).
Physical coords: x, y in (-2.5, 2.5),  t in (0, 1.0)  (PDEBench geometry).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """
    Coordinate-based MLP surrogate for h(x, y, t).

    Parameters
    ----------
    hidden_dims : list  hidden layer widths, e.g. [512, 512, 512, 256]
    activation  : str   "tanh" | "gelu" | "relu"
    """

    _ACTIVATIONS = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}

    def __init__(
        self,
        hidden_dims: List[int] = (512, 512, 512, 256),
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        act_cls = self._ACTIVATIONS.get(activation, nn.Tanh)

        layers: list[nn.Module] = []
        prev = 3  # (x, y, t)
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, 1))  # predict h only

        self.net = nn.Sequential(*layers)
        self._init_glorot()

    def _init_glorot(self) -> None:
        """Glorot/Xavier normal init -- same as PINN reference."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : FloatTensor [N, 3]  (x, y, t) physical coordinates

        Returns
        -------
        h : FloatTensor [N, 1]  water height
        """
        return self.net(coords)

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        data_coords: torch.Tensor,
        data_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Data-only MSE loss.

            L = MSE(h_pred(x,y,t), h_data)

        No PDE residual, no initial-condition constraint.
        """
        pred = self(data_coords)  # [N, 1]
        return F.mse_loss(pred, data_h)

    # ------------------------------------------------------------------
    def predict_on_grid(
        self,
        H: int,
        W: int,
        t: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Evaluate on a full H x W spatial grid at physical time t.

        Spatial range matches PDEBench reference: x, y in (-2.5, 2.5).
        Returns h: [H, W, 1].

        Parameters
        ----------
        t : float  physical time in [0, 1.0]
        """
        xs = torch.linspace(-2.5, 2.5, H, device=device)
        ys = torch.linspace(-2.5, 2.5, W, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        ts = torch.full_like(gx, t)
        coords = torch.stack([gx, gy, ts], dim=-1).reshape(-1, 3)  # [H*W, 3]

        with torch.no_grad():
            out = self(coords)  # [H*W, 1]
        return out.reshape(H, W, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: dict, sample_xx: Optional[torch.Tensor] = None) -> "MLP":
        """Construct from config dict. sample_xx is unused (kept for API compat)."""
        mcfg = cfg.get("mlp", {})
        return MLP(
            hidden_dims=mcfg.get("hidden_dims", [40, 40, 40, 40, 40, 40]),
            activation=mcfg.get("activation", "tanh"),
        )
