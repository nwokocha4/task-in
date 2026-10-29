"""Shared dispersion physics (not agent-visible)."""
import numpy as np

SIGMA_COEF = {
    "A": (0.22, 0.894, 0.20, 0.894),
    "B": (0.16, 0.894, 0.14, 0.894),
    "C": (0.11, 0.894, 0.10, 0.894),
    "D": (0.08, 0.894, 0.08, 0.894),
    "E": (0.06, 0.894, 0.06, 0.894),
    "F": (0.04, 0.894, 0.04, 0.894),
}

def sigma_yz(x, stability):
    a_y, b_y, a_z, b_z = SIGMA_COEF[stability]
    sy = a_y * max(x, 1.0) ** b_y
    sz = a_z * max(x, 1.0) ** b_z
    return max(sy, 15.0), max(sz, 15.0)

def plume_rise(Q, exit_vel, stack_h, u, stability):
    F = 1e-3 * Q * max(exit_vel, 0.1)
    if stability in ("E", "F"):
        dh = (F / (u * 0.02 + 1e-6)) ** (1.0 / 3.0)
    else:
        dh = 1.6 * (F / (u + 1e-6)) ** (1.0 / 3.0)
    return stack_h + min(dh, 50.0)

def rotate(source, receptor, wind_dir_deg):
    th = __import__("math").radians(wind_dir_deg)
    import math
    wx, wy = math.cos(th), math.sin(th)
    dx = receptor[0] - source[0]
    dy = receptor[1] - source[1]
    return dx*wx + dy*wy, -dx*wy + dy*wx

def concentration_field(source_xy, receptor_xy, cohort_of_source, theta, stack_height, exit_velocity, u, wind_dir, stability):
    import numpy as np
    K = len(theta)
    Q = np.exp(theta[np.asarray(cohort_of_source)])
    C = np.zeros(len(receptor_xy))
    for s in range(len(source_xy)):
        H = plume_rise(float(Q[s]), float(exit_velocity[s]), float(stack_height[s]), float(u), stability)
        for r in range(len(receptor_xy)):
            x, y = rotate(source_xy[s], receptor_xy[r], float(wind_dir))
            if x <= 0: continue
            sy, sz = sigma_yz(float(x), stability)
            C[r] += float(Q[s])/(np.pi*float(u)*sy*sz) * np.exp(-y*y/(2*sy*sy)) * np.exp(-H*H/(2*sz*sz))
    return C
