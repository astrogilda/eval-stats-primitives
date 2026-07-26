# MIT License. Independent implementation of Bayesian pass-rate estimation
# via Beta-Binomial conjugacy: with a Beta(a, b) prior and k successes in
# n trials, the posterior is Beta(a + k, b + n - k): no sampling, no
# framework, just conjugate arithmetic.  This is the binary special case of
# the Dirichlet-categorical framework applied to pass@k estimation in
# Hariri et al. (ICLR 2026).
#
# Why bother beyond Wilson (s2): the posterior is a full curve, not a
# range: you can read off P(true rate > 0.9), and it sharpens honestly
# as data arrives.
#
# The prior is a modelling choice, and the credible floor moves with it
# by whole points at small n.  A method that hides that choice is quietly
# lying about its own precision, so here the prior is a named, selectable
# argument and the demo prints the common choices side by side.
"""Run: python s3_beta_binomial.py"""

from math import exp, lgamma, log

# The two standard reference priors for a binomial rate.  Uniform
# (Bayes-Laplace) treats every rate as equally likely a priori; Jeffreys
# is the invariance-derived reference prior recommended by Brown, Cai &
# DasGupta (2001).  Both are defensible, which is exactly why the choice
# must be visible in the output rather than baked into a default.
PRIORS: dict[str, tuple[float, float]] = {
    "uniform": (1.0, 1.0),
    "jeffreys": (0.5, 0.5),
}


def log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def beta_pdf(x: float, a: float, b: float) -> float:
    if not 0.0 < x < 1.0:
        return 0.0
    return exp((a - 1) * log(x) + (b - 1) * log(1 - x) - log_beta(a, b))


def beta_cdf(x: float, a: float, b: float, steps: int = 20_000) -> float:
    """Regularized incomplete beta via plain trapezoid integration.

    Deliberately the dumbest correct thing: for teaching-scale a, b this
    matches scipy to ~1e-6 and keeps the file dependency-free.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    h = x / steps
    total = 0.5 * (beta_pdf(1e-12, a, b) + beta_pdf(x, a, b))
    for i in range(1, steps):
        total += beta_pdf(i * h, a, b)
    return min(1.0, total * h)


def posterior(k: int, n: int, a: float = 1.0, b: float = 1.0) -> tuple[float, float]:
    """Posterior Beta parameters after k successes in n trials under a Beta(a, b) prior."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    return (a + k, b + n - k)


def credible_interval(k: int, n: int, mass: float = 0.95, a: float = 1.0, b: float = 1.0) -> tuple[float, float]:
    """Equal-tailed credible interval by bisection on the posterior CDF.

    The prior Beta(a, b) defaults to uniform but is a real argument: at
    small n the interval's lower edge shifts visibly between defensible
    priors, and that sensitivity is part of the answer.
    """
    post_a, post_b = posterior(k, n, a, b)
    lo_q, hi_q = (1 - mass) / 2, 1 - (1 - mass) / 2

    def inv_cdf(q: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if beta_cdf(mid, post_a, post_b) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return (inv_cdf(lo_q), inv_cdf(hi_q))


if __name__ == "__main__":
    print("95% equal-tailed credible intervals; the floor is the interval's lower edge.")
    for k, n in [(23, 25), (92, 100), (920, 1000), (25, 25)]:
        for name, (prior_a, prior_b) in PRIORS.items():
            post_a, post_b = posterior(k, n, prior_a, prior_b)
            lo, hi = credible_interval(k, n, a=prior_a, b=prior_b)
            p_above_90 = 1.0 - beta_cdf(0.90, post_a, post_b)
            print(
                f"{k}/{n} ({name:>8} prior): posterior Beta({post_a:.1f},{post_b:.1f})"
                f"  CI [{lo:.1%}, {hi:.1%}]  floor {lo:.1%}  P(rate>90%)={p_above_90:.1%}"
            )
    print(
        "The 25/25 floor moves between priors; the method-dependence is the disclosure.\n"
        "Compare the Wilson floor from s2_wilson.py on the same 25/25 data: numerically\n"
        "close to the uniform-prior credible floor by coincidence, not by identity."
    )
