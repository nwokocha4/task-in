# Field dictionary

network.npz:
- source_xy (S,2): source (x,y) in metres.
- receptor_xy (R,2): receptor (x,y) in metres.
- cohort_of_source (S,): cohort index per source (0..K-1).
- K: number of cohorts.
- stack_height (S,): physical stack height (m).
- exit_velocity (S,): exit velocity (m/s).
- monitor_idx (M,): indices into receptor_xy that are monitors.

meteorology.npz:
- calib_wind_speed (T,), calib_wind_dir (T,): calibration snapshots.
- calib_stability (T,): stability class labels (one may be wrong).
- withheld_wind_speed, withheld_wind_dir, withheld_stability: withheld episode.

observations.npz:
- y (T,M): monitor readings (ug/m3).
- monitor_idx (M,): monitor receptor indices.

noise_factor.npz:
- L (T*M, T*M): lower-triangular Cholesky factor of Sigma_eps.

prior.json:
- theta_mean (K,), theta_sigma (K,): log-emission prior.
- admissible_sigma: 3.0 (prior mean +/- 3 sigma in log space).

Outputs:
- reconstruction.npz: emission_coefficients (K,), emission_per_source (S,), predicted_concentrations (R,).
- diagnostics.json: weighted_calibration_misfit, max_residual, max_predicted_concentration, limit_exceeded.