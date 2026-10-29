# Routing and estimator specification

## Transport
Channel routing is dilution + first-order attenuation along the network. Attenuation rate and dilution rating-curve exponent are fixed and supplied in `network.npz` / `sigma` table; only grouped loads are free. The routing operator H is built deterministically from the network topology and the supplied coefficients — no fixed operator is shipped as importable code.

## Estimator (unique solution)
Free parameters: grouped contaminant loads m (K unknowns). Fixed: attenuation rate, rating-curve exponent, network routing coefficients, topology, priors, and noise factors — not estimated. The task is a bounded convex quadratic program with a unique solution.

Objective (penalty weight fixed at 1, no free lambda):
  min_m  || L^-1 (y - H m) ||^2  +  || P^-1/2 (m - m0) ||^2
where L is the supplied whitening factor (from `noise_factor.npz`), P^-1/2 is the supplied prior factor (from `prior.json`), y is the blank-corrected observation vector (see qa_policy.md), H is the disclosed routing operator, m0 is the supplied prior mean, and P is diagonal with entries sigma_j^2.

Box constraints: m_j in [m0_j - 3 sigma_j, m0_j + 3 sigma_j] in log-load space, recoverable from prior.json.

Convergence: KKT residual < 1e-10 relative.
Rounding: round-half-away-from-zero on the final float to the stated integer units.

## Output
Receptor concentrations are 1-hour averages in the withheld episode. Receptor list and output ordering are as in network.npz receptor_xy. Compliance threshold is 50 ug/L (stated as 5000 in the x100 integer units); averaging window is the withheld episode. Covariance-factor L units are (ug/L)^2 in variance space.
