"""Shared channel-routing physics (generator + verifier only; not shipped to /app/data)."""
import math, numpy as np

def build_H(source_xy, receptor_xy, branch_of_source, attenuation_rate, dilution_exponent, flow, K=None, topology=None):
    """Routing matrix H (R x K): concentration per unit grouped load."""
    S = len(source_xy); R = len(receptor_xy)
    if K is None:
        K = int(np.max(branch_of_source)) + 1 if len(branch_of_source) else 1
    H = np.zeros((R, K))
    for s in range(S):
        k = int(branch_of_source[s])
        for r in range(R):
            dx = float(receptor_xy[r,0] - source_xy[s,0])
            dy = float(receptor_xy[r,1] - source_xy[s,1])
            dist = math.hypot(dx, dy) / 1000.0
            atten = math.exp(-attenuation_rate * dist)
            dil = (max(flow, 0.1) ** (-dilution_exponent))
            H[r, k] += atten * dil
    return H

def concentration_field(source_xy, receptor_xy, branch_of_source, m_exp, attenuation_rate, dilution_exponent, flow):
    H = build_H(source_xy, receptor_xy, branch_of_source, attenuation_rate, dilution_exponent, flow)
    return H @ np.asarray(m_exp, dtype=float)
