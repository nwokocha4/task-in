# emission-source-inversion

## Difficulty

Recover grouped emission source strengths from sparse downwind monitoring and predict exposure under a withheld meteorology. The core challenge is telling emission structure the monitoring record supports apart from structure that merely fits noise. Requires Gaussian-plume dispersion physics, Briggs plume rise, stability-class sigma parameterisation, whitening under a supplied factored covariance that hides a per-snapshot background offset, and regularized (MAP) inversion with identifiability analysis. Held-out prediction under an unseen stability class and low-wind regime distinguishes physically coherent estimates from overfit ones.

## Solution

Rebuilds the dispersion operator from supplied case files, detects a mislabelled stability class, whitens residuals with the supplied covariance factor, solves a non-negative MAP problem under the supplied log-emission prior, and forward-simulates the withheld meteorology. Expert hours: 4-8.

## Verification

Verifier re-solves dispersion from submitted coefficients and compares predicted withheld concentrations against withheld monitor values within a covariance-derived bound; checks admissibility (prior mean +/- 3 sigma in log space), non-negativity, finite values, shapes/keys, and internal consistency (max_residual). Six held-out cases with independently sampled geometry.