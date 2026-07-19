# MIT License. Independent implementations of false-discovery-rate control:
#   Benjamini-Hochberg (BH)   — Benjamini & Hochberg 1995 (JRSS-B 57)
#   Benjamini-Yekutieli (BY)  — Benjamini & Yekutieli 2001 (Ann. Statist. 29)
# e-BH (Wang & Ramdas 2022, JRSS-B) is described in the README decision
# rule; it substitutes e-values for p-values and needs a streaming setup.
#
# The demo: 80 parallel drift checks per deploy (4 metrics x 10 flows x 2
# detectors).  3 checks sit on REAL regressions by construction; the other
# 77 are null.  Naive per-check alpha=0.05 fires a pile of phantoms; BH
# keeps the true ones and controls the expected fraction of false alarms.
#
# p-values here come from the same OLS slope test as s4 — that is the
# pipeline: detectors produce per-check p-values, FDR decides which alarms
# survive.
"""Run: python s5_fdr.py"""

import random
from math import log

from s4_drift_detectors import BASELINE_MEAN, BASELINE_SIGMA, ols_slope_test

SEED = 42
N_CHECKS = 80
TRUE_REGRESSIONS = {7, 33, 61}  # by construction: these three checks drift
ALPHA = 0.05


def benjamini_hochberg(p_values: list[float], alpha: float = ALPHA) -> set[int]:
    """Indices of discoveries under BH step-up at level alpha."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    cutoff = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / m:
            cutoff = rank
    return set(order[:cutoff]) if cutoff > 0 else set()


def benjamini_yekutieli(p_values: list[float], alpha: float = ALPHA) -> set[int]:
    """BY = BH at level alpha / H(m); valid under arbitrary dependence.

    Use when the checks share data — e.g. every check reads the same eval
    history, which is exactly the situation in an eval pipeline.
    """
    m = len(p_values)
    harmonic = sum(1.0 / i for i in range(1, m + 1))
    return benjamini_hochberg(p_values, alpha / harmonic)


def simulate_p_values(seed: int = SEED) -> list[float]:
    rng = random.Random(seed)
    p_values: list[float] = []
    for check in range(N_CHECKS):
        drift = -0.004 if check in TRUE_REGRESSIONS else 0.0
        scores = [BASELINE_MEAN + drift * i + rng.gauss(0.0, BASELINE_SIGMA) for i in range(30)]
        p_values.append(ols_slope_test(scores, BASELINE_SIGMA).p_value)
    return p_values


if __name__ == "__main__":
    ps = simulate_p_values()
    naive = {i for i, p in enumerate(ps) if p <= ALPHA}
    bh = benjamini_hochberg(ps)
    by = benjamini_yekutieli(ps)
    for name, hits in [("naive alpha=0.05", naive), ("BH", bh), ("BY", by)]:
        true_hits = hits & TRUE_REGRESSIONS
        print(f"{name:17s} fires {len(hits):2d} alarms — {len(true_hits)}/3 true, "
              f"{len(hits) - len(true_hits)} phantom")
    harmonic = sum(1 / i for i in range(1, N_CHECKS + 1))
    print(f"(expected phantoms under naive: ~{(N_CHECKS - 3) * ALPHA:.1f} per deploy; "
          f"BY penalty H({N_CHECKS}) = {harmonic:.2f} ~ ln(m)+0.577 = {log(N_CHECKS) + 0.577:.2f})")
