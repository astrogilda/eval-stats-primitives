<p align="center">
  <img src="logo-mark.png" alt="eval-stats-primitives" width="88">
</p>

# Eval-stats primitives: "Your AI Eval Is Lying To You" (PyCon Colombia 2026)

[![eval-gate](https://github.com/astrogilda/eval-stats-primitives/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/astrogilda/eval-stats-primitives/actions/workflows/eval-gate.yml)

Most AI eval results are reported as a single point estimate: one pass rate, one score
on a dashboard, a green check. That number usually carries no confidence interval, no
correction for running many checks at once, and no test for whether it is drifting over
time. So a headline like "92%" can mean far less than it looks like, or hide a real
regression, and nothing on the dashboard tells you which.

These five files are the statistical fixes, one method per file. Each is a standalone
MIT-licensed pure-Python file with no dependencies (pytest is needed only to run the
examples), so you can paste one into an existing test suite today. They accompany the
PyCon Colombia 2026 talk of the same name, but every method and every number here stands
on its own.

| Script | Method | Published source |
|-|-|-|
| s1_pass_at_k.py | pass@k (capability) vs pass^k (reliability) | Chen et al. 2021, arXiv:2107.03374 |
| s2_wilson.py | Wilson score interval, honest at the k=n boundary | Wilson 1927; Brown, Cai & DasGupta 2001 |
| s3_beta_binomial.py | Bayesian pass rate via Beta-Binomial conjugacy, the binary special case of the Dirichlet-categorical framework; the prior is named and selectable (uniform vs Jeffreys, printed side by side) | Hariri et al., ICLR 2026 |
| s4_drift_detectors.py | Shewhart threshold vs EWMA/CUSUM accumulators vs OLS trend sizing | Shewhart 1931; Lucas & Saccucci 1990; Page 1954; Montgomery |
| s5_fdr.py | BH / BY false-discovery-rate control over 80 parallel drift checks | Benjamini & Hochberg 1995; Benjamini & Yekutieli 2001 |

Run any script directly, for example:

```
python s4_drift_detectors.py
python -m pytest test_examples.py -q
```

Every number quoted below reproduces from these files with the seeds committed, and the
pytest command runs the paste-able test shapes.

## Five questions to ask your eval today

Start with whether you are measuring capability or reliability. Does your "92%" mean
best-of-k, or every single time? A 92%-per-step agent completes a 5-step chain about
66% of the time. Next, what is the interval? 23/25 is not "92%"; honestly it is
[75%, 98%] by Wilson, and at 25/25 the textbook interval says [100%, 100%] where the
honest floor is 86.7%. Then ask whether you have enough samples, since the interval
width at n=25 vs n=100 vs n=1000 is a design choice you can compute in advance. A fourth
question is whether you would notice a slow slide: a fixed dashboard threshold
false-alarms on one flaky run and catches a real regression only by luck, whereas
accumulating detectors like EWMA and CUSUM ignore the flake and catch both the cliff
and the slide, and an OLS slope test then sizes the trend. Last, how many of your alarms
are noise? Eighty checks at a per-check alpha of 0.05 means about 4 phantom alarms every
deploy, and BH keeps the real ones.

## Which FDR correction?

For independent checks, which are rare in practice, use BH at your target FDR. When
checks share data, meaning the same eval history feeds many metrics or detectors, which
is the normal case, use BY: it is BH with a penalty of about ln(m), valid under
arbitrary dependence. If you stream results and stop whenever you happen to look at the
dashboard, the p-value versions stop being safe and you want e-BH (Wang & Ramdas 2022),
which swaps p-values for e-values so that peeking does not invalidate the guarantee.
e-BH is not implemented here; the point is knowing where the safe boundary is.

These files are teaching-scale independent implementations of the methods cited above.
The talk previews a pytest plugin (private beta) that packages production-grade versions
into one pipeline with SARIF reporting and baseline regression, but the primitives stand
on their own.

## The competitive survey behind the "empty quadrant" claim

One quadrant of the eval-tooling market is empty, and the talk's central claim rests on
that gap. Here is the survey behind it, so you can check it yourself. It sorts platforms
on two axes.

The first axis is when a platform tests. A point-in-time run compares one batch against
a baseline. A sequential or streaming platform scores a continuously arriving stream of
production traffic and surfaces the metric over time. The second axis is whether the
platform corrects for multiplicity, meaning whether a documented multiple-comparison or
false-discovery-rate procedure (Benjamini-Hochberg, Benjamini-Yekutieli, Storey
q-values, Holm, e-BH, an online-FDR method) is applied across the family of scores,
metrics, slices, or rubric criteria under evaluation. A per-metric threshold, a count of
drifted columns, or one confidence interval on one score does not qualify.

|  | Point-in-time | Sequential / streaming |
|-|-|-|
| FDR-controlled | (empty) | (empty, except the approach this talk presents) |
| Uncorrected | LangSmith, DeepEval, Promptfoo | Arize Phoenix, Evidently, Galileo, Fiddler, Braintrust, Langfuse, W&B Weave |

All ten platforms sit in the bottom, uncorrected row, surveyed July 2026. Six are open
source and were searched at source level for "benjamini", "hochberg", "false discovery
rate", "mSPRT", "always-valid", and related terms, with zero substantive hits; the rest
were checked against current product documentation. The strongest single artifact is
Evidently's open-source dataset-drift decision, an uncorrected per-column test at
p < 0.05 that is counted, with dataset drift declared once the share of tripped columns
clears one half.

The statistics here are not new. What keeps the claim true is a single qualifier.
Optimizely's Stats Engine has combined sequential testing with Benjamini-Hochberg FDR
control since 2015, a shipped and documented decade-old product that cites Benjamini &
Hochberg 1995 and Benjamini & Yekutieli 2001 directly. That machinery lives in web and
product A/B testing: binary conversions and unbounded revenue-style metrics, in a
randomised treatment-versus-control design. What has not happened is anyone carrying
that discipline across to bounded judge scores, a judge score on [0, 1] or a 1-5 rubric,
often with no control arm, where the many hypotheses are metrics, slices, and rubric
criteria rather than variations. The claim is narrow on purpose and holds only with that
last qualifier in place: none combines sequential testing with FDR control on bounded
scoring scales. Drop the "bounded scoring scales" clause and Optimizely already occupies
the quadrant.
