"""NACA 4-digit airfoil geometry: thickness/camber distributions and surface coordinates.

Standard closed-form NACA 4-digit parameterization (open trailing edge).
m = max camber (fraction of chord), p = location of max camber (fraction of chord),
t = max thickness (fraction of chord). Chord length c = 1 (nondimensional).
"""
import numpy as np

_THICKNESS_COEFFS = (0.2969, -0.1260, -0.3516, 0.2843, -0.1015)


def naca_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """Half-thickness distribution y_t(x) for 0 <= x <= 1."""
    x = np.asarray(x, dtype=np.float64)
    a0, a1, a2, a3, a4 = _THICKNESS_COEFFS
    return 5.0 * t * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)


def naca_camber(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """Camber line y_c(x). Returns zeros identically if m == 0 (symmetric airfoil)."""
    x = np.asarray(x, dtype=np.float64)
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x)
    front = x < p
    yc = np.empty_like(x)
    yc[front] = (m / p**2) * (2 * p * x[front] - x[front] ** 2)
    yc[~front] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[~front] - x[~front] ** 2)
    return yc


def naca_camber_slope(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """dy_c/dx, used to rotate thickness normal to the camber line."""
    x = np.asarray(x, dtype=np.float64)
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x)
    front = x < p
    dyc = np.empty_like(x)
    dyc[front] = (2 * m / p**2) * (p - x[front])
    dyc[~front] = (2 * m / (1 - p) ** 2) * (p - x[~front])
    return dyc


def _cosine_spacing(n_points: int) -> np.ndarray:
    """Cosine-spaced x in [0, 1], denser near leading and trailing edge."""
    beta = np.linspace(0.0, np.pi, n_points)
    return 0.5 * (1.0 - np.cos(beta))


def naca4_surface(m: float, p: float, t: float, n_points: int = 200):
    """Upper and lower surface coordinates for a NACA 4-digit airfoil.

    Returns (upper_xy, lower_xy), each an (n_points, 2) array, ordered leading-edge
    to trailing-edge, cosine-spaced.
    """
    x = _cosine_spacing(n_points)
    yt = naca_thickness(x, t)
    yc = naca_camber(x, m, p)
    dyc = naca_camber_slope(x, m, p)
    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.stack([xu, yu], axis=-1)
    lower = np.stack([xl, yl], axis=-1)
    return upper, lower


def naca4_boundary_loop(m: float, p: float, t: float, n_points: int = 200) -> np.ndarray:
    """Closed boundary polygon: lower surface (LE->TE) then upper surface reversed (TE->LE).

    Returns an (2*n_points - 1, 2) array with no duplicated leading-edge point across
    the join (both branches start at the same LE point x=0).
    """
    upper, lower = naca4_surface(m, p, t, n_points)
    return np.concatenate([lower, upper[::-1][1:]], axis=0)
