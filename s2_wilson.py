# MIT License. Independent implementation of the Wilson score interval
# (Wilson 1927), the recommended binomial interval per Brown, Cai &
# DasGupta 2001, "Interval Estimation for a Binomial Proportion"
# (Statistical Science 16(2)), which shows the textbook Wald interval
# misbehaves badly — especially at the boundaries k == 0 and k == n.
"""Run: python s2_wilson.py"""

from math import sqrt

# z for a 95% two-sided interval. Hard-coded so the file has zero deps;
# swap for statistics.NormalDist().inv_cdf(0.975) if you want other levels.
Z95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials.

    Boundary handling: unlike Wald, this formula needs no special-casing
    at k == 0 or k == n — the interval stays inside (0, 1) and never
    collapses to a zero-width lie.  (boundary: k==0 or k==n is exactly
    where Wald returns [0,0] or [1,1].)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def wald_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The textbook interval — shown here only to demonstrate its failure."""
    p = k / n
    half = z * sqrt(p * (1 - p) / n)
    return (max(0.0, p - half), min(1.0, p + half))


if __name__ == "__main__":
    for k, n in [(23, 25), (25, 25)]:
        lo_w, hi_w = wald_interval(k, n)
        lo, hi = wilson_interval(k, n)
        print(f"{k}/{n} passed  ({k / n:.0%})")
        print(f"  Wald   : [{lo_w:.1%}, {hi_w:.1%}]" + ("   <- a lie" if k == n else ""))
        print(f"  Wilson : [{lo:.1%}, {hi:.1%}]")
