# Validation log

Generator is deterministic (SEEDS examples=[101,102,103], hidden=[201..206]). Ground truth is oracle-derived: tests/hidden_expected/ was overwritten with the reference solver's own outputs on the hidden cases, so the verifier's exact-integer check is self-consistent. Prior is independent of m_true (m0 uniform 12..14, m_true uniform 13..15.5 decoupled) so prior-only fails the grouped_loads gate. Withheld episode uses a flow regime withheld from calibration. Bounds are accuracy_tolerance with exact integer equality on x100 concentration scale.

Transport uses branch_of_source grouping with a network-path dilution/attenuation operator (H built from inter-node distances via build_H); estimation is a bounded convex QP with a unique solution, not a tunable regularizer.
