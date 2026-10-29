# Field dictionary

network.npz: source_xy (S,2), receptor_xy (R,2), cohort_of_source (S), K, stack_height (S), exit_velocity (S), monitor_idx (M).
meteorology.npz: calib_wind_speed (T), calib_wind_dir (T), calib_stability (T), withheld_wind_speed, withheld_wind_dir, withheld_stability.
observations.npz: y (T,M).
noise_factor.npz: noise covariance Sigma_eps and its factor L (see dispersion_spec.md).
prior.json: theta_mean (K), theta_sigma (K).
Outputs: reconstruction.npz with emission_coefficients (K), emission_per_source (S), predicted_concentrations (R); diagnostics.json with weighted_calibration_misfit, max_residual, max_predicted_concentration, limit_exceeded.
