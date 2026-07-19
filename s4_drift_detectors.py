# MIT License. Independent implementations of three sequential drift
# detectors over an eval-score history, per the published methods:
#   EWMA control chart  — Lucas & Saccucci 1990 (Technometrics 32(1))
#   CUSUM               — Page 1954 (Biometrika 41)
#   OLS slope test      — ordinary least squares, e.g. Montgomery,
#                         "Introduction to Statistical Quality Control"
# plus the Shewhart individuals chart (Shewhart 1931) — which is what a
# fixed dashboard threshold IS, whether you call it that or not.
#
# The point the demo makes: the DETECTOR FAMILY, not the brand, decides
# what you catch.  A fixed threshold (Shewhart) screams at one flaky run
# and is pure luck on real regressions; the accumulators (EWMA, CUSUM)
# ignore the flake and catch both the cliff and the slow slide; the OLS
# slope test confirms and SIZES the trend, and its p-value is the raw
# material FDR control (s5) consumes.
#
# All numbers deterministic: SEED = 7, scenario shapes mirror widely
# used quality-control test patterns (stable 0.90 baseline with +/-0.01
# noise; one-run glitch -3.5 sigma; cliff 0.90 -> 0.88; slide -0.001/run).
"""Run: python s4_drift_detectors.py"""

import random
from dataclasses import dataclass
from math import erf, sqrt

SEED = 7
BASELINE_MEAN = 0.90
BASELINE_SIGMA = 0.01
EWMA_LAMBDA = 0.2  # standard smoothing weight (Lucas-Saccucci table range)
EWMA_L = 3.0  # control-limit width in sigmas
CUSUM_K = 0.5  # reference value, in sigmas (ARL0 ~ 339 with h=4)
CUSUM_H = 4.0  # decision interval, in sigmas


def shewhart_alarm(scores: list[float], mu0: float, sigma: float, L: float = 3.0) -> int | None:
    """Index of the first single point below mu0 - L*sigma, else None.

    This is every fixed "alert if score < X" dashboard rule, formalized.
    """
    for i, x in enumerate(scores):
        if x < mu0 - L * sigma:
            return i
    return None


def ewma_alarm(scores: list[float], mu0: float, sigma: float) -> int | None:
    """Index of the first EWMA lower-limit crossing, else None."""
    z = mu0
    for i, x in enumerate(scores):
        z = EWMA_LAMBDA * x + (1 - EWMA_LAMBDA) * z
        # Time-varying limit per Lucas-Saccucci eq. for early observations.
        var = (EWMA_LAMBDA / (2 - EWMA_LAMBDA)) * (1 - (1 - EWMA_LAMBDA) ** (2 * (i + 1)))
        if z < mu0 - EWMA_L * sigma * sqrt(var):
            return i
    return None


def cusum_alarm(scores: list[float], mu0: float, sigma: float) -> int | None:
    """Index of the first one-sided (downward) CUSUM crossing, else None."""
    s_minus = 0.0
    for i, x in enumerate(scores):
        s_minus = max(0.0, s_minus + (mu0 - x) / sigma - CUSUM_K)
        if s_minus > CUSUM_H:
            return i
    return None


@dataclass(frozen=True)
class SlopeTest:
    slope: float
    z: float
    p_value: float  # one-sided, H1: slope < 0


def ols_slope_test(scores: list[float], sigma: float) -> SlopeTest:
    """OLS slope over run index, with a normal test on the slope.

    With n >= 20 the t distribution is close enough to normal that we use
    the z approximation and keep the file dependency-free; the known noise
    sigma comes from the baseline period, as in Phase-I/Phase-II control
    charting.
    """
    n = len(scores)
    if n < 3:
        raise ValueError("need at least 3 points")
    xbar = (n - 1) / 2
    sxx = sum((i - xbar) ** 2 for i in range(n))
    slope = sum((i - xbar) * y for i, y in enumerate(scores)) / sxx
    se = sigma / sqrt(sxx)
    z = slope / se
    p_one_sided = 0.5 * (1 + erf(z / sqrt(2)))  # P(Z <= z): small when slope << 0
    return SlopeTest(slope=slope, z=z, p_value=p_one_sided)


def make_scenarios(seed: int = SEED) -> dict[str, list[float]]:
    """Four 60-run histories; every defect starts at run 20."""
    scenarios: dict[str, list[float]] = {}
    for name in ("stable", "one-run glitch", "cliff", "slow slide"):
        rng = random.Random(seed)  # fresh stream per scenario: same baseline noise
        xs = [BASELINE_MEAN + rng.gauss(0.0, BASELINE_SIGMA) for _ in range(20)]
        for i in range(40):
            if name == "one-run glitch":
                mean = BASELINE_MEAN - (0.035 if i == 5 else 0.0)  # one flaky run, then recovery
            elif name == "cliff":
                mean = 0.88  # a real 2-sigma regression
            elif name == "slow slide":
                mean = BASELINE_MEAN - 0.001 * (i + 1)  # provider quietly degrading
            else:
                mean = BASELINE_MEAN
            xs.append(mean + rng.gauss(0.0, BASELINE_SIGMA))
        scenarios[name] = xs
    return scenarios


if __name__ == "__main__":
    print(f"{'scenario':16s}  {'threshold':>10s}  {'EWMA':>6s}  {'CUSUM':>6s}   OLS on runs 20+")

    def lat(a: int | None) -> str:
        return f"+{a - 20}" if a is not None else "-"

    for name, scores in make_scenarios().items():
        s_ = shewhart_alarm(scores, BASELINE_MEAN, BASELINE_SIGMA)
        e = ewma_alarm(scores, BASELINE_MEAN, BASELINE_SIGMA)
        c = cusum_alarm(scores, BASELINE_MEAN, BASELINE_SIGMA)
        t = ols_slope_test(scores[20:], BASELINE_SIGMA)
        print(f"{name:16s}  {lat(s_):>10s}  {lat(e):>6s}  {lat(c):>6s}   slope {t.slope:+.4f}/run  p={t.p_value:.1e}")
    print("(latency = runs after the defect at run 20; '-' = never fires in 40 runs)")
