# Field dictionary

- `network.npz`: source_xy (S,2) source coordinates, receptor_xy (R,2) receptor coordinates, branch_of_source (S,) branch index per source, K number of grouped loads, attenuation_rate (scalar, 1/km), dilution_exponent (scalar), monitor_idx (M,) indices into receptor_xy that are monitors, flow_topology (S,R adjacency).
- `meteorology.npz` (hydrology): calib_flow (T,) calibration flow magnitudes, calib_flow_regime (T,) regime label per episode, withheld_flow, withheld_regime.
- `observations.npz`: y_raw (T,M) raw lab readings, remark (T,M) free-text QA remarks per reading (strings), batch_id (T,M) batch index per reading.
- `noise_factor.npz`: L (T*M, T*M) supplied whitening factor.
- `qa_factor.npz`: blank_factor (B,) per-batch blank correction scalar.
- `prior.json`: m0 (K,) prior mean in log-load space, sigma (K,) prior stdev in log-load space.
- Output JSON: `grouped_loads` (K integers, g/day), `receptor_concentrations` (R integers, 0.01 ug/L units — i.e. value×100, so 5000 = 50.00 ug/L), `compliance` ({limit_exceeded: bool at 5000 in x100 units}, averaging window is the withheld episode).
