# water-load-inversion

## Difficulty

Sparse water-quality monitoring for contaminant load compliance against a withheld hydrological regime. The agent must reconcile per-batch blank corrections and QA remarks against a disclosed policy, rebuild a channel-routing operator (dilution rating curve + attenuation), and solve a convex regularized inverse problem for grouped loads. A withheld high-flow/low-flow episode lies outside the calibrated regime; correct loads and compliance prediction require physical coherence, not just training-fit quality.

## Solution

Rebuild the routing operator H from network files per routing_spec.md, apply the disclosed per-batch blank correction from qa_policy.md, whiten residuals with the supplied prior/covariance factors, and solve the bounded convex QP for grouped loads (free parameters are loads only). Forward-simulate the withheld hydrology to receptor concentrations and a compliance flag.

## Verification

Six held-out catchments with different topologies/names. Each case output must exactly match sealed integers: grouped_loads (K grouped loads in g/day), receptor_concentrations (R at 0.01 ug/L precision), and compliance (limit_exceeded). Per-case all-or-nothing; task pass requires all six. Oracle 1, empty 0. Prior-mean/constant/unweighted baselines fail by margin (calibration only).
