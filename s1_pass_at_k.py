# MIT License. Independent implementation of the unbiased pass@k estimator
# from Chen et al. 2021, "Evaluating Large Language Models Trained on Code"
# (arXiv:2107.03374), plus the pass^k reliability metric.
#
# pass@k answers: "if I let the model try k times, how often does at least
# one attempt succeed?"  (capability)
# pass^k answers: "if k users each get ONE attempt, how often do ALL of
# them succeed?"  (reliability)
# One number, "92% pass rate", silently mixes these two questions.
"""Run: python s1_pass_at_k.py"""

from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k given n samples with c successes.

    Computes 1 - C(n-c, k)/C(n, k) without enumerating combinations of
    failures, exactly as in Chen et al. 2021, Section 2.1.
    """
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c} n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k} n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k(p: float, k: int) -> float:
    """Reliability: probability that k independent single attempts ALL pass.

    Assumes independent attempts; real agent chains are correlated, which
    is exactly why the interval methods (s2, s3) matter: they tell you what
    the point estimate can and cannot support.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"need 0 <= p <= 1, got {p}")
    return p**k


if __name__ == "__main__":
    # The talk example: a step that passes 92% of the time.
    p = 0.92
    print(f"per-step pass rate      : {p:.0%}")
    print(f"pass@5  (best of 5)     : {1 - (1 - p) ** 5:.4%}   # capability")
    print(f"pass^5  (5-step chain)  : {pass_pow_k(p, 5):.1%}     # reliability")
    # Unbiased pass@k from raw counts: 25 eval cases, 23 passed, k=5.
    print(f"pass@5 from n=25, c=23  : {pass_at_k(25, 23, 5):.4f}")
