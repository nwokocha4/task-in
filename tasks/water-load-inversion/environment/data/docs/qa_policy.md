# Chain-of-custody / lab QA policy

Purpose: decide, per lab remark, whether a result enters the inversion and, if so, with what per-batch blank correction.

Rules (applied verbatim; no undisclosed convention):
- A result whose remark contains the standalone word "blank" (e.g. "batch blank") is a blank and is excluded from y; its batch factor is used as the correction for non-blank results in that batch.
- A result whose remark contains "blank-corrected" is a non-blank result that has already had the batch blank subtracted — it enters y without further correction.
- A result whose remark contains "hold-exceeded" or "preservation-failed" is censored and excluded.
- A result whose remark contains "hold-flag" (without "exceeded") or "dilution:" is a valid result that enters y (with blank correction applied).
- For results that enter y, subtract the batch-specific blank factor stored in `qa_factor.npz` (one scalar per batch) before forming y. The factor file is `qa_factor.npz` with array `blank_factor` (B,). This is the required correction — not just include/exclude.
- Near-miss wording that the policy resolves unambiguously: "blank-corrected" vs "blank", and "offset credit" vs "credited offset" are distinct and handled only by the rules above.

Every graded case contains at least one batch requiring this correction and at least one near-miss remark, both witnessed in the example evidence.
