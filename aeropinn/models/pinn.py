"""Geometry-conditioned PINN architecture (Plan Section 9/10).

Input:  [x, y, sdf, m, p, t, alpha_deg, Re]  (8 scalars)
Output: [u_star, v_star, p_star]              (nondimensional, Plan Section 7)

Fourier-feature encoding on (x, y) only (not on geometry/condition scalars, which
are low-frequency conditioning signals) to combat spectral bias near the leading
edge / thin boundary layer. tanh activations throughout (C^2, required for the
second-order viscous residual terms).
"""
import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_frequencies: int = 8, sigma: float = 4.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_frequencies) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class InputNormalizer(nn.Module):
    """Fixed (non-trainable) per-dimension affine normalization to ~[-1, 1].

    Ranges reflect the Plan Section 6 validity envelope and the domain used for
    collocation sampling — NOT learned from data, so the same normalization is
    reproducible across training runs and at inference time on the deployed demo.
    """

    def __init__(self):
        super().__init__()
        # [x, y, sdf, m, p, t, alpha_deg, Re]
        lo = torch.tensor([-3.0, -3.0, -1.0, 0.0, 0.10, 0.08, -6.0, 1.0e4])
        hi = torch.tensor([4.0, 3.0, 1.0, 0.06, 0.60, 0.20, 8.0, 1.0e5])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0


class GeometryConditionedPINN(nn.Module):
    def __init__(self, hidden_layers: int = 8, hidden_width: int = 128, n_fourier: int = 8):
        super().__init__()
        self.normalizer = InputNormalizer()
        self.fourier = FourierFeatures(in_dim=2, n_frequencies=n_fourier)  # on (x, y) only
        cond_dim = 6  # sdf, m, p, t, alpha, Re
        in_dim = 2 * n_fourier + cond_dim  # FourierFeatures outputs [sin(proj), cos(proj)], each width n_fourier

        layers = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_width))
            layers.append(nn.Tanh())
            d = hidden_width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 3)  # u_star, v_star, p_star

        for m in self.backbone:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor):
        """x, y: (N, 1). cond: (N, 6) = [sdf, m, p, t, alpha_deg, Re]. Returns (u, v, p), each (N, 1)."""
        raw = torch.cat([x, y, cond], dim=-1)  # (N, 8), matches InputNormalizer's ordering
        normed = self.normalizer(raw)
        xy_normed = normed[:, 0:2]
        cond_normed = normed[:, 2:8]

        ff = self.fourier(xy_normed)
        h = torch.cat([ff, cond_normed], dim=-1)
        h = self.backbone(h)
        out = self.head(h)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
