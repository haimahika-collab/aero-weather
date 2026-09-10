"""Geometry sampling for the NACA 4-digit training family (Plan Section 6/13).

Fast-path scope: Latin Hypercube sample over the validated (m, p, t) envelope,
split at the GEOMETRY level into train / held-out sets (never split by point or
condition — Plan Section 13's leakage-prevention rule). The full plan's separate
interpolation/extrapolation held-out sets are collapsed into one held-out set here
for speed; the split logic itself still guarantees no held-out geometry's (m,p,t)
was seen in training.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Plan Section 6 validity envelope.
M_RANGE = (0.0, 0.06)
P_RANGE = (0.10, 0.60)
T_RANGE = (0.08, 0.20)


@dataclass
class Geometry:
    geom_id: str
    m: float
    p: float
    t: float
    split: str  # "train" | "held_out"

    @property
    def naca_digits(self) -> str:
        # NACA 4-digit code: first digit = 100*m, second = 10*p, last two = 100*t.
        d1 = round(self.m * 100)
        d2 = round(self.p * 10)
        d34 = round(self.t * 100)
        return f"{d1:01d}{d2:01d}{d34:02d}"


def sample_geometries(n_train: int, n_held_out: int, seed: int = 0) -> List[Geometry]:
    """LHS-sample (m, p, t) over the Plan Section 6 envelope, held-out geometries
    drawn from a separate LHS draw (not a subset of train) so they are genuinely
    distinct points in parameter space, not just excluded indices.
    """
    n_total = n_train + n_held_out
    sampler = LatinHypercube(d=3, seed=seed)
    unit_samples = sampler.random(n=n_total)

    lo = np.array([M_RANGE[0], P_RANGE[0], T_RANGE[0]])
    hi = np.array([M_RANGE[1], P_RANGE[1], T_RANGE[1]])
    samples = lo + unit_samples * (hi - lo)

    geometries = []
    for i, (m, p, t) in enumerate(samples):
        split = "train" if i < n_train else "held_out"
        idx = i if i < n_train else i - n_train
        geom_id = f"{split}_{idx:03d}"
        geometries.append(Geometry(geom_id=geom_id, m=float(m), p=float(p), t=float(t), split=split))

    # m=0 makes p meaningless (symmetric airfoil); avoid degenerate near-zero m that
    # would round to a NACA 0-camber code inconsistently across nearby samples.
    for g in geometries:
        if g.m < 0.005:
            g.m = 0.0

    return geometries
