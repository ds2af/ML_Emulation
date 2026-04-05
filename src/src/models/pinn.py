"""
src/models/pinn.py
==================
Physics-Informed Neural Network (PINN) for the 2-D radial dam-break
shallow-water equation surrogate, matching the reference PDEBench setup
(``Reference_PDEBench/pdebench/models/pinn/train.py::setup_swe_2d``).

Implementation
--------------
Pure-PyTorch re-implementation of the PDEBench PINN -- no DeepXDE dependency.
Automatic differentiation via ``torch.autograd.grad`` computes spatial and
temporal first-order derivatives of the network outputs.

PDE formulation -- 2-D Shallow-Water Equations (non-conservative form)
-----------------------------------------------------------------------
As in ``pde_definitions.py::pde_swe2d`` with g = 1.0:

    eq1 = dh/dt + (dh/dx)*u + h*(du/dx) + (dh/dy)*v + h*(dv/dy) = 0
    eq2 = du/dt + u*(du/dx) + v*(du/dy) + g*(dh/dx)             = 0
    eq3 = dv/dt + u*(dv/dx) + v*(dv/dy) + g*(dh/dy)             = 0

Outputs: (h, u, v) -- water height, x-velocity, y-velocity.
Data supervision: h only (u=0, v=0 enforced as initial condition at t=0).

Architecture -- matches reference exactly
-----------------------------------------
    Input  [N, 3]  (x, y, t) physical coordinates
                   x, y in (-2.5, 2.5),  t in (0, 1.0)
    Hidden [40] x 6  with Tanh activation
    Output [N, 3]  (h, u, v)

    Initialisation: Glorot/Xavier normal  (reference: "Glorot normal")

Training loss
-------------
    L = lambda_data * MSE(h_pred, h_data)
      + lambda_ic  * MSE([u_pred, v_pred] at t=0, 0)
      + lambda_pde * mean(eq1^2 + eq2^2 + eq3^2)

predict_on_grid returns h only (channel 0) for compatibility with the
single-channel dataset.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# PINN backbone MLP
# ---------------------------------------------------------------------------

class PINNBackbone(nn.Module):
    """
    Coordinate-based MLP backbone.

    Maps (x, y, t) -> field values u(x, y, t) in R^C.

    Uses Tanh activations -- the standard choice for physics-informed networks
    because Tanh is twice-differentiable everywhere and has bounded range,
    which helps when computing higher-order derivatives via autograd.

    Parameters
    ----------
    in_dim      : int    input dimension (3 for x, y, t)
    out_dim     : int    number of output channels C
    hidden_dims : list   widths of hidden layers
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        hidden_dims: List[int] = (40, 40, 40, 40, 40, 40),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : FloatTensor [N, 3]  (x, y, t) -- all in [0, 1]

        Returns
        -------
        u : FloatTensor [N, C]
        """
        return self.net(coords)


# ---------------------------------------------------------------------------
# PINN with PDE residual loss
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """
    Physics-informed neural network for the 2-D radial dam-break SWE.

    Matches ``setup_swe_2d`` in PDEBench reference:
      - Network: FNN([3] + [40]*6 + [3], Tanh, Glorot normal)
      - PDE: non-conservative 2-D SWE with g = 1.0
      - Coordinates: x,y in (-2.5, 2.5),  t in (0, 1.0)
      - Outputs: (h, u, v)
      - Data supervision on h only; u=0, v=0 enforced as IC at t=0

    Parameters
    ----------
    hidden_dims  : list  hidden layer widths -- default [40]*6
    g            : float gravitational constant (1.0 in PDEBench)
    lambda_pde   : float weight of PDE residual term
    lambda_data  : float weight of supervised data (h) term
    lambda_ic    : float weight of initial-condition (u=v=0) term
    """

    def __init__(
        self,
        hidden_dims: List[int] = (40, 40, 40, 40, 40, 40),
        g: float = 1.0,
        lambda_pde: float = 1.0,
        lambda_data: float = 1.0,
        lambda_ic: float = 1.0,
    ) -> None:
        super().__init__()
        # 3 outputs: h, u, v  (matches reference FNN [...] + [3])
        self.out_channels = 3
        self.backbone     = PINNBackbone(3, self.out_channels, hidden_dims)
        self.g            = g
        self.lambda_pde   = lambda_pde
        self.lambda_data  = lambda_data
        self.lambda_ic    = lambda_ic

        # Glorot (Xavier) normal initialisation -- matches reference "Glorot normal"
        self._init_glorot()

    def _init_glorot(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Query the network at coordinates (x, y, t)."""
        return self.backbone(coords)

    # ------------------------------------------------------------------
    def pde_residual(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2-D SWE residual (non-conservative form, g=1.0).

        Matches ``pde_swe2d`` in PDEBench pde_definitions.py exactly:

            eq1 = h_t + h_x*u + h*u_x + h_y*v + h*v_y
            eq2 = u_t + u*u_x + v*u_y + g*h_x
            eq3 = v_t + u*v_x + v*v_y + g*h_y

        Parameters
        ----------
        coords : FloatTensor [N, 3]  (x, y, t)  -- will have requires_grad set

        Returns
        -------
        residual : FloatTensor [N, 3]  (eq1, eq2, eq3)
        """
        coords = coords.requires_grad_(True)
        out = self.backbone(coords)          # [N, 3]: (h, u, v)

        h = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]

        # First-order derivatives of each output w.r.t. (x, y, t)
        def _jac(scalar_field):
            return torch.autograd.grad(
                scalar_field, coords,
                grad_outputs=torch.ones_like(scalar_field),
                create_graph=True,
                retain_graph=True,
            )[0]                             # [N, 3]: (d/dx, d/dy, d/dt)

        dh = _jac(h);  h_x, h_y, h_t = dh[:, 0:1], dh[:, 1:2], dh[:, 2:3]
        du = _jac(u);  u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
        dv = _jac(v);  v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]

        eq1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
        eq2 = u_t + u * u_x + v * u_y + self.g * h_x
        eq3 = v_t + u * v_x + v * v_y + self.g * h_y

        return torch.cat([eq1, eq2, eq3], dim=1)   # [N, 3]

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        data_coords: torch.Tensor,
        data_h: torch.Tensor,
        colloc_coords: torch.Tensor,
        ic_coords: torch.Tensor,
        ic_h_coords: Optional[torch.Tensor] = None,
        ic_h_vals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total training loss.

        Matches reference PDEBench setup_swe_2d which enforces:
          - ic_h  : h(x,y,0) = 2.0 if r <= dam_radius else 1.0  (PointSetBC / IC)
          - ic_u  : u(x,y,0) = 0                                  (IC)
          - ic_v  : v(x,y,0) = 0                                  (IC)
          - bc_data : 30% of seed data as PointSetBC on h          (data supervision)
          - pde   : SWE residual over domain collocation points

        Parameters
        ----------
        data_coords   : [N_d, 3]   (x,y,t) supervised data points (30% of seed)
        data_h        : [N_d, 1]   observed h at those points
        colloc_coords : [N_c, 3]   PDE collocation points (num_domain=1000)
        ic_coords     : [N_ic, 3]  t=0 points for u=v=0  (num_initial=5000)
        ic_h_coords   : [N_ih, 3]  t=0 points for h IC constraint (optional)
        ic_h_vals     : [N_ih, 1]  exact h(x,y,0) values for those points
        """
        # -- Data loss on h (supervised, matches reference bc_data) -----
        pred_data = self.backbone(data_coords)          # [N_d, 3]
        data_loss = F.mse_loss(pred_data[:, 0:1], data_h)

        # -- IC loss: u = 0, v = 0 at t = 0 (matches reference ic_u, ic_v) --
        pred_ic = self.backbone(ic_coords)              # [N_ic, 3]
        zeros   = torch.zeros_like(pred_ic[:, 1:3])
        ic_uv_loss = F.mse_loss(pred_ic[:, 1:3], zeros)

        # -- IC loss: h(x,y,0) = 2 inside dam, 1 outside (matches reference ic_h) --
        if ic_h_coords is not None and ic_h_vals is not None:
            pred_ic_h = self.backbone(ic_h_coords)      # [N_ih, 3]
            ic_h_loss = F.mse_loss(pred_ic_h[:, 0:1], ic_h_vals)
        else:
            ic_h_loss = torch.tensor(0.0, device=data_coords.device)

        ic_loss = ic_uv_loss + ic_h_loss

        # -- PDE residual loss ------------------------------------------
        residual = self.pde_residual(colloc_coords)     # [N_c, 3]
        pde_loss = (residual ** 2).mean()

        total = (self.lambda_data * data_loss
                 + self.lambda_ic  * ic_loss
                 + self.lambda_pde * pde_loss)
        return total, {
            "data_loss":  data_loss.item(),
            "ic_uv_loss": ic_uv_loss.item(),
            "ic_h_loss":  ic_h_loss.item() if isinstance(ic_h_loss, torch.Tensor) else 0.0,
            "pde_loss":   pde_loss.item(),
        }

    # ------------------------------------------------------------------
    def predict_on_grid(
        self,
        H: int,
        W: int,
        t: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Evaluate the PINN on a full H x W spatial grid at physical time t.

        Spatial range matches reference geometry: x, y in (-2.5, 2.5).
        Returns h only (channel 0) for compatibility with the single-channel
        dataset; shape [H, W, 1].

        Parameters
        ----------
        t : float  physical time in [0, 1.0]
        """
        xs = torch.linspace(-2.5, 2.5, H, device=device)
        ys = torch.linspace(-2.5, 2.5, W, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        ts     = torch.full_like(gx, t)
        coords = torch.stack([gx, gy, ts], dim=-1).reshape(-1, 3)   # [H*W, 3]

        with torch.no_grad():
            out = self.backbone(coords)    # [H*W, 3]
        h = out[:, 0:1]                    # water height only
        return h.reshape(H, W, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: dict, sample_xx: torch.Tensor) -> "PINN":
        mcfg = cfg.get("pinn", {})
        return PINN(
            hidden_dims = mcfg.get("hidden_dims", [40, 40, 40, 40, 40, 40]),
            g           = mcfg.get("g", 1.0),
            lambda_pde  = mcfg.get("lambda_pde", 1.0),
            lambda_data = mcfg.get("lambda_data", 1.0),
            lambda_ic   = mcfg.get("lambda_ic",   1.0),
        )
