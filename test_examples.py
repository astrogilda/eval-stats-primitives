# MIT License. Paste-into-pytest examples for the five method scripts.
# Run: python -m pytest test_examples.py -q     (needs only pytest)
"""These are the shapes you'd drop into an existing test suite tonight."""

from s1_pass_at_k import pass_at_k, pass_pow_k
from s2_wilson import wilson_interval
from s3_beta_binomial import beta_cdf, credible_interval, posterior
from s4_drift_detectors import (
    BASELINE_MEAN,
    BASELINE_SIGMA,
    cusum_alarm,
    ewma_alarm,
    make_scenarios,
    ols_slope_test,
    shewhart_alarm,
)
from s5_fdr import TRUE_REGRESSIONS, benjamini_hochberg, simulate_p_values


def test_capability_and_reliability_are_different_questions() -> None:
    # 92% per step: best-of-5 looks near-perfect, 5-step chains fail a third of users.
    assert 1 - (1 - 0.92) ** 5 > 0.999
    assert pass_pow_k(0.92, 5) < 0.66
    # Unbiased pass@k from raw counts (n samples, c passing).
    assert pass_at_k(25, 23, 5) == 1.0


def test_reliability_gate_you_can_ship() -> None:
    """The actual gate: require the INTERVAL FLOOR, not the point estimate."""
    k, n = 23, 25
    lo, _hi = wilson_interval(k, n)
    assert lo > 0.70, "eval too small to support the claimed rate"


def test_wilson_boundary_never_lies_at_100_percent() -> None:
    lo, hi = wilson_interval(25, 25)
    assert hi > 0.999  # upper end approaches 1 but the FLOOR is what matters
    assert lo < 0.90, "25/25 does NOT license a >90% floor"
    assert round(lo, 3) == 0.867


def test_posterior_sharpens_with_data() -> None:
    lo25, hi25 = credible_interval(23, 25)
    lo100, hi100 = credible_interval(92, 100)
    assert (hi100 - lo100) < (hi25 - lo25)
    a, b = posterior(23, 25)
    assert 0.4 < 1.0 - beta_cdf(0.90, a, b) < 0.6  # "92%" ~ a coin flip vs the 90% bar


def test_accumulators_ignore_flakes_and_catch_regressions() -> None:
    s = make_scenarios()
    glitch, cliff, slide = s["one-run glitch"], s["cliff"], s["slow slide"]
    # A fixed threshold screams at one flaky run...
    assert shewhart_alarm(glitch, BASELINE_MEAN, BASELINE_SIGMA) is not None
    # ...the accumulators do not...
    assert ewma_alarm(glitch, BASELINE_MEAN, BASELINE_SIGMA) is None
    assert cusum_alarm(glitch, BASELINE_MEAN, BASELINE_SIGMA) is None
    # ...and they catch the real defects.
    assert cusum_alarm(cliff, BASELINE_MEAN, BASELINE_SIGMA) is not None
    assert ewma_alarm(slide, BASELINE_MEAN, BASELINE_SIGMA) is not None


def test_ols_recovers_the_injected_trend() -> None:
    slide = make_scenarios()["slow slide"][20:]
    t = ols_slope_test(slide, BASELINE_SIGMA)
    assert -0.0013 < t.slope < -0.0007  # injected: -0.0010/run
    assert t.p_value < 1e-6


def test_bh_kills_the_phantom_alarms() -> None:
    ps = simulate_p_values()
    naive = {i for i, p in enumerate(ps) if p <= 0.05}
    bh = benjamini_hochberg(ps)
    assert bh == TRUE_REGRESSIONS  # every discovery real, none missed
    assert len(naive - TRUE_REGRESSIONS) >= 3  # naive fires phantoms every deploy
