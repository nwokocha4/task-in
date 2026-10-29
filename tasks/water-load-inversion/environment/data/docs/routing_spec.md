# Routing and estimator specification

## Transport
Channel routing is a Muskingum-style advection-dispersion along the supplied network topology (network.npz: flow_topology as parent indices per node, with attenuation_rate as first-order loss per unit travel time and dilution_exponent as the rating-curve dilution). Travel time between a source and a receptor is the sum of reach lengths along the network path from source to receptor divided by a reach velocity derived from flow via the rating curve; H is built deterministically from this topology, reach lengths, attenuation_rate, and dilution_exponent supplied in network.npz. No fixed operator is shipped as importable code; the operator must be constructed from this specification and the network files.

## Estimator (unique solution)
Free parameters: grouped contaminant loads m (K unknowns). Fixed: attenuation_rate, rating-curve exponent, network routing coefficients (reach lengths), topology, priors, and noise factors — not estimated. The task is a bounded convex quadratic program with a unique solution.

Objective (penalty weight fixed at 1, no free lambda):
  min_m  || L^-1 (y - H m) ||^2  +  || P^-1/2 (m - m0) ||^2
where L is the supplied whitening factor (from noise_factor.npz), P^-1/2 is the supplied prior factor (from prior.json), y is the blank-corrected observation vector (see qa_policy.md), H is the routing operator defined above, m0 is the supplied prior mean, and P is diagonal with entries sigma_j^2.

Box constraints: m_j in [m0_j - 3 sigma_j, m0_j + 3 sigma_j] in log-load space, recoverable from prior.json.

Convergence: KKT residual < 1e-10 relative.
Rounding: round-half-away-from-zero on the final float to the stated integer units.

## Output
Receptor concentrations are 1-hour averages in the withheld episode. Receptor list and output ordering are as in network.npz receptor_xy. Compliance threshold is 50 ug/L (stated as 5000 in the x100 integer units); averaging window is the withheld episode. Covariance-factor L units are (ug/L)^2 in variance space. The withheld episode uses a flow regime withheld from calibration; extrapolating the rating-curve dilution to that regime is part of the task.
