# Eval-stats primitives — "Your AI Eval Is Lying To You" (PyCon Colombia 2026)

Reference implementations of the five statistical methods from the talk, each a
standalone MIT-licensed pure-Python file with zero dependencies (pytest only for
the examples). Paste into an existing pytest suite tonight.

| Script | Method | Published source |
|---|---|---|
| `s1_pass_at_k.py` | pass@k (capability) vs pass^k (reliability) | Chen et al. 2021, arXiv:2107.03374 |
| `s2_wilson.py` | Wilson score interval, honest at the k=n boundary | Wilson 1927; Brown, Cai & DasGupta 2001 |
| `s3_beta_binomial.py` | Bayesian pass rate via Beta-Binomial conjugacy | Hariri et al., ICLR 2026 |
| `s4_drift_detectors.py` | Shewhart threshold vs EWMA/CUSUM accumulators vs OLS trend sizing | Shewhart 1931; Lucas & Saccucci 1990; Page 1954; Montgomery |
| `s5_fdr.py` | BH / BY false-discovery-rate control over 80 parallel drift checks | Benjamini & Hochberg 1995; Benjamini & Yekutieli 2001 |

Run any script directly (`python s4_drift_detectors.py`) — every number in the
talk reproduces from these files with the seeds as committed.
`python -m pytest test_examples.py -q` runs the paste-able test shapes.

## Five questions to ask your eval tonight

1. **Capable or reliable?** Does your "92%" mean best-of-k or every-single-time?
   A 92%-per-step agent completes a 5-step chain ~66% of the time.
2. **What's the interval?** 23/25 is not "92%" — it's honestly [75%, 98%] (Wilson).
   At 25/25 the textbook interval says [100%, 100%]; the honest floor is 86.7%.
3. **Enough samples?** The interval width at n=25 vs n=100 vs n=1000 is a
   computable design choice, not a vibe.
4. **Would you notice a slow slide?** A fixed dashboard threshold false-alarms on
   one flaky run and catches a real regression only by luck; accumulating
   detectors (EWMA/CUSUM) ignore the flake and catch both the cliff and the
   slide; an OLS slope test then sizes the trend.
5. **How many of your alarms are noise?** 80 checks at per-check α=0.05 → ~4
   phantom alarms every deploy. BH keeps the real ones.

## Which FDR correction? (decision rule)

- **Independent checks** (rare in practice): **BH** at your target FDR.
- **Checks sharing data** — same eval history feeding many metrics/detectors
  (the normal case): **BY** (BH with an ~ln(m) penalty; valid under arbitrary
  dependence).
- **Streaming / optional-stopping** — you look at the dashboard whenever you
  feel like it: **e-BH** (Wang & Ramdas 2022) — replaces p-values with
  e-values so peeking doesn't invalidate the guarantee. Not implemented here;
  the point of the rule is knowing when the p-value versions stop being safe.

These files are teaching-scale independent implementations of the published
methods above. The talk previews a pytest plugin (private beta) that packages
production-grade versions into one pipeline with SARIF reporting and baseline
regression; these primitives stand alone regardless.
