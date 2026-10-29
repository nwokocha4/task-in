# Monitoring network policy

A monitor reading equals the true plume concentration at that receptor plus a per-snapshot background offset plus correlated noise with covariance Sigma_eps = L L^T. The background offset is encoded as a near-rank-one component of L and is not a separately named parameter; whiten residuals with L.

Include a monitor in the inversion only when its siting note does not mark it as a background monitor; a "background-corrected" monitor is still an emission monitor (the correction has already been applied). A "credited offset" reduces the reading; an "offset credit" does not. A "fugitive (non-point)" source belongs to the fugitive cohort; a "point (stack)" source belongs to the stack cohort. These mappings are derivable from this policy alone.