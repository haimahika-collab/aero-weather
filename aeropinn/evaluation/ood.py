"""Out-of-distribution flag (Plan Section 21, fast-path scope: Mahalanobis distance
in the (m, p, t, alpha, Re) parameter space only — the full plan's small-ensemble
epistemic-uncertainty component is deferred; this alone is still a real, meaningful
safety check, not a placeholder, and the deployed demo must never present an
out-of-envelope query as equally trustworthy as an in-envelope one).
"""
import json
from pathlib import Path

import numpy as np

from aeropinn.data.generate_sweep import PROCESSED_DIR


class MahalanobisOOD:
    def __init__(self, mean: np.ndarray, inv_cov: np.ndarray, threshold: float):
        self.mean = mean
        self.inv_cov = inv_cov
        self.threshold = threshold

    def distance(self, query: np.ndarray) -> float:
        d = query - self.mean
        return float(np.sqrt(d @ self.inv_cov @ d))

    def is_out_of_distribution(self, query: np.ndarray) -> bool:
        return self.distance(query) > self.threshold

    def save(self, path: Path):
        np.savez(path, mean=self.mean, inv_cov=self.inv_cov, threshold=self.threshold)

    @classmethod
    def load(cls, path: Path) -> "MahalanobisOOD":
        d = np.load(path)
        return cls(d["mean"], d["inv_cov"], float(d["threshold"]))


def fit_from_training_geometries(percentile: float = 97.5) -> MahalanobisOOD:
    """Fit on the (m, p, t, alpha, Re) points actually used in training (one row per
    sweep case, not per spatial point), calibrating the flag threshold on the SAME
    data's own distance distribution — a held-out-based calibration (Plan Section 21's
    stated ideal) is deferred; noted explicitly rather than silently skipped.
    """
    with open(PROCESSED_DIR / "geometries.json") as f:
        geometries = {g["geom_id"]: g for g in json.load(f)}

    rows = []
    for npz_path in sorted(PROCESSED_DIR.glob("*.npz")):
        d = np.load(npz_path)
        geom_id = str(d["geom_id"])
        if geometries[geom_id]["split"] != "train":
            continue
        rows.append([float(d["m"]), float(d["p"]), float(d["t"]), float(d["alpha_deg"]), float(d["reynolds"])])

    X = np.array(rows)
    mean = X.mean(axis=0)
    cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    inv_cov = np.linalg.inv(cov)

    distances = np.array([np.sqrt((x - mean) @ inv_cov @ (x - mean)) for x in X])
    threshold = float(np.percentile(distances, percentile))

    ood = MahalanobisOOD(mean, inv_cov, threshold)
    from aeropinn.training.train_pinn import CHECKPOINT_DIR
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ood.save(CHECKPOINT_DIR / "ood_mahalanobis.npz")
    print(f"OOD threshold (p{percentile}): {threshold:.3f}, fit on {len(X)} training points", flush=True)
    return ood


if __name__ == "__main__":
    fit_from_training_geometries()
