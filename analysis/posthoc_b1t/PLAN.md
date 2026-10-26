# Post-hoc B1 extension T: the identical-policy local comparison at trained policies

Written 2026-10-23 before any run below existed and before any result of extensions B1C, B1G, B1E or B1N had been read,
after the B1 and B2 results of record and an independent review (review_A.md C3, E6) that objects that B1 probes only
untrained initialisations, a regime training never visits. The manuscript names this study as its first future step.
Not part of any frozen family; labelled post hoc.

## Questions
T. At the policies HQ actually reached in early training (B2 final checkpoints, about 30,000 decisions), from the
   identical starting policy given by the exact twin and at equal mean forward KL, does the circuit-realised natural
   step still gain less than the twin's coefficient-space steps (the B1 contrasts), and than the same direction applied
   linearly?
TD. At C-matched's own trained policies (B2 final checkpoints of the same blocks), does its natural step through its
   nonlinear layer gain less than the same direction applied linearly (the classical counterpart of T's path contrast)?

## Units and data
Part T: the 36 B2 HQ runs (blocks 0-35, learning rate 3e-3); each unit is the run's final checkpoint (stored SHA-256
checked before use), loaded into the B1 model class. Its exact twin is built and certified with B1's static parity
checks (fail closed). Part TD: the 36 B2 C-matched runs' final checkpoints of the same blocks.
Both parts collect their sets with `dq_b1.collect_set` under the seeds of two-layer B1 unit i = block (common random
numbers with B1): part T the fit, oracle, calibration and held-out sets (B1's full instrument), part TD the fit,
calibration and held-out sets. Only the circuit angles (T) or the inner Linear(4, 4) layer (TD) are changed by the
arms; everything else of the trained policy is frozen, as in B1.

## Arms
Part T: every B1 arm, computed by the frozen `dq_b1.unit_instruments` unchanged. Part TD: C-Fisher, C-TAN and C-plain,
computed by the functions of extension G (idea_dq_posthoc_b1g/runner.py, part D) unchanged.
kappa in {0.001, 0.01} (and 0.05 where B1's instrument computes it, descriptive only).

## Analysis (fixed now)
Paired over units: mean, 95% paired t interval, two-sided paired t p-value (one-sided for T2 as in B1's M-H1),
mean/sd, share below zero.
Family T (kappa = 0.01, Holm over the four at familywise 0.05):
- T1: G_H(HQ-Fisher) − G_H(TAN) at trained HQ policies.
- T2: G_H(SPAN) − G_H(HQ-Fisher) (B1's M-H1 at trained policies; one-sided, H1: mean > 0).
- T3: G_H(HQ-Fisher) − mean G_H(RND-k) (B1's A-H1 at trained policies; two-sided).
- TD1: G_H(C-Fisher) − G_H(C-TAN) at trained C-matched policies.
Descriptive only: the same at kappa = 0.001; every arm's mean gain; retained shares (natural / linear) with percentile
bootstrap intervals for trained HQ, trained C-matched, and untrained HQ (B1) and C-matched (extension G); the Jacobian's
(s1/sk)^2 and the theta-Fisher condition number at the trained angles; the calibrated scales and quadratic errors.

## Reading rules (fixed now; "resolved" = rejected by Holm)
- T1 resolved negative: the path difference is present at the policies reached by training, not only at
  initialisation. T1 not resolved: not resolved at trained policies (not evidence of no difference); the manuscript then
  states that the B1 path result was found at initialisation and not resolved at the trained policies.
- T2, T3: read as B1's M-H1 and A-H1 at trained policies (same directions).
- TD1 resolved negative: a classical nonlinear head also shows the path difference at its trained policies.
  TD1 not resolved: not resolved for the classical head (not evidence of equality). Retained shares are compared
  descriptively only.

## Reporting commitment
Whatever the outcome, family T and the descriptive values are reported in the manuscript (Sections 4.4, 5.1, 5.3 (c),
Appendix A) with this plan's SHA-256; scripts and records go to the repository.
