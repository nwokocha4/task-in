# Validation log

Generator: solution/generator.py via solution/_model.py (SEEDS examples=[101,102,103] hidden=[201..206]).
Forward model: Gaussian plume with Pasquill-Gifford sigma_y/z from environment/data/docs/sigma_table.csv and Briggs plume rise (see _model.py).
Example and hidden cases: examples/case_1..3 with expected.json (theta, predicted_concentrations); hidden case_4..9 with hidden_expected/case_*.json + _meta.json.

Verifier (tests/test_outputs.py):
- Checks artifact presence and schema.
- Checks theta within prior support (mean ± 3σ in log space) AND accuracy vs hidden_expected (atol=0.5, rtol=0.1).
- Checks predicted_concentrations on mask |pred|>1e-12 (atol=1e-6, rtol=0.1) vs hidden_expected, after independent re-solution from submitted theta is not needed because hidden_expected itself is model output; the test checks theta vs hidden_expected as the independent ground truth (prior alone does not satisfy atol=0.5).
- Checks independent re-solution of max_residual: verifier checks predicted_concentrations vs hidden_expected (the generator's model output) ≤ 1e-3 (and weighted_calibration_misfit to match recomputed whitened residual within 1e-6).
- Checks limit_exceeded consistency with 50 ug/m3 threshold.
- Rejects non-finite / wrong-shape / missing-key submissions; checks hidden vs example names disjoint.

Oracle run (reference solution/solve.py):
- All 9 cases pass the above checks when the reference is used as oracle.
- Negative: a prior-mean-only submission passes bounds but fails accuracy (Δθ ≈ 1–2 log units on unlabeled cohorts).

No solver is shipped under environment/data; operator must be reconstructed from docs/sigma_table.csv + dispersion_spec.md.

Behavior grading (heldout_real): checks improvement over prior (≥10%), envelope ±20% around perturbed ground truth, and limit_exceeded correctness — not exact allclose to a single synthetic oracle.
