"""Shipped forward dispersion model (Gaussian plume by stability class)."""
import numpy as np

STABILITY_CLASSES = ["A", "B", "C", "D", "E", "F"]

# Pasquill-Gifford sigma coefficients: sigma = a * x**b (meters), x in m
# (a_y, b_y, a_z, b_z) per class, approximate
SIGMA_COEF = {
    "A": (0.22, 0.894, 0.20, 0.894),
    "B": (0.16, 0.894, 0.14, 0.894),
    "C": (0.11, 0.894, 0.10, 0.894),
    "D": (0.08, 0.894, 0.08, 0.894),
    "E": (0.06, 0.894, 0.06, 0.894),
    "F": (0.04, 0.894, 0.04, 0.894),
}

def sigma_yz(x, stability):
    x = np.asarray(x, dtype=float)
    a_y, b_y, a_z, b_z = SIGMA_COEF[stability]
    sy = a_y * np.power(np.maximum(x, 1.0), b_y)
    sz = a_z * np.power(np.maximum(x, 1.0), b_z)
    # floor
    sy = np.maximum(sy, 15.0)
    sz = np.maximum(sz, 15.0)
    return sy, sz

def plume_rise(Q, exit_vel, stack_h, u, stability):
    # Briggs-style, depends on Q (emission) and exit velocity; simplified
    # F ~ g * (T_s-T_a)/T_a * V_s * d^2 /4 ; approximate with Q*exit_vel
    F = 1e-3 * Q * max(exit_vel, 0.1)
    if stability in ("E", "F"):
        s = 0.02  # stable
        dh = (F / (u * s + 1e-6)) ** (1.0 / 3.0)
    else:
        dh = 1.6 * (F / (u + 1e-6)) ** (1.0 / 3.0)
    dh = np.clip(dh, 0.0, 50.0)
    return stack_h + dh

def rotate_coords(src, rec, wind_dir_deg):
    # returns downwind distance x and crosswind y for receptor relative to source
    theta = np.deg2rad(wind_dir_deg)
    # wind vector components
    wx, wy = np.cos(theta), np.sin(theta)
    dx = rec[0] - src[0]
    dy = rec[1] - src[1]
    x = dx * wx + dy * wy   # downwind
    y = -dx * wy + dy * wx  # crosswind
    return x, y

def concentration_field(source_xy, receptor_xy, cohort_of_source, theta,
                       stack_height, exit_velocity, u, wind_dir, stability):
    """Return concentration at each receptor (R,). theta: (K,) log-emissions."""
    K = len(theta)
    Q_per_source = np.exp(theta[cohort_of_source])  # (S,)
    S = len(source_xy)
    R = len(receptor_xy)
    C = np.zeros(R)
    for s in range(S):
        Q = Q_per_source[s]
        H = plume_rise(Q, exit_velocity[s], stack_height[s], u, stability)
        for r in range(R):
            x, y = rotate_coords(source_xy[s], receptor_xy[r], wind_dir)
            if x <= 0:
                continue
            sy, sz = sigma_yz(x, stability)
            # scalar sigma (if arrays)
            sy = np.atleast_1d(sy).ravel()[0] if np.ndim(sy) > 0 else float(sy)
            sz = np.atleast_1d(sz).ravel()[0] if np.ndim(sz) > 0 else float(sz)
            term = (Q / (np.pi * u * sy * sz)) * np.exp(- (y**2) / (2 * sy**2)) * np.exp(- (H**2) / (2 * sz**2))
            C[r] += term
    return C
