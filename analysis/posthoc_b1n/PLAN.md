# Post-hoc B1 extension N: fit-set size (estimation noise against the path difference)

Written 2026-10-23 before any run below existed and before any result of extensions B1C, B1G or B1E had been read,
after the B1 results of record, the B1 extension of 2026-10-22 and two independent reviews (review_A.md M2, review_B.md
M6) that object that gradient-estimation noise is confounded with curvature (split-half cosine of the update direction
about 0; batch size never varied). Not part of any frozen family; labelled post hoc.

## Question
N. Does the size of the fit set from which the gradient and the Fisher are estimated change (i) the difference between
   the circuit-realised natural step and the same direction applied linearly (HQ-Fisher − TAN), and (ii) the shortfall
   of the free-coefficient step against random subspaces (SPAN − RND-k), which the manuscript reads as estimation error?

## Units and data
The 36 two-layer B1 units (primary stratum), excluding any unit that extension C excluded. For each unit: the
calibration set K and the held-out set H are the rows extension C regenerated and saved (unchanged, so every fit-set
size is calibrated and evaluated on the same K and H). The fit set is regenerated from its frozen B1 seeds (reproduction
check: identical per-episode J), and 768 further fit episodes are collected with the unit's untrained policy
(96 per cell), environment seeds start(dq_fit) + 50,000 + (slot × 8 + cell) × 96 + offset (B1's fit seeds lie below
start + 18,432), action streams DQ_STUDY.stream(SET_TAG fit, 1000 + slot, cell, 1000 + offset).
Fit sets, nested: F64 (B1 offsets 0-7 in every cell), F128 (offsets 0-15), F256 (= B1's fit set), F1024 (F256 plus the
768 new episodes). Advantages are built inside each fit set exactly as B1 builds them (leave-one-episode-out per-cell
mean of returns-to-go).

## Arms (B1's definitions, at every fit-set size; KL calibrated on K to within 1e-3 kappa; G_H on H)
HQ-Fisher, TAN, SPAN and RND-k (the same 20 Haar subspaces as B1 for the unit, from B1's frozen RND stream), at
kappa in {0.001, 0.01}. Reproduction check: at F256 the G_H of HQ-Fisher and TAN equal extension C's values for the unit
(relative 1e-6); a failing unit is excluded.

## Analysis (fixed now)
Paired over units: mean, 95% paired t interval, two-sided paired t p-value, mean/sd, share below zero.
Family N (kappa = 0.01, Holm over the two at familywise 0.05):
- N1: [G_H(HQ-Fisher) − G_H(TAN)] at F1024 minus the same at F256.
- N2: [G_H(SPAN) − G_H(RND-k)] at F1024 minus the same at F256.
Descriptive only: every arm and both contrasts at F64, F128, F256, F1024 and at kappa = 0.001; HQ-Fisher − TAN at F1024
itself; the split-half cosine of the circuit's natural direction at F1024 (two halves of 512 episodes) against B1's at
F256.

## Reading rules (fixed now; "resolved" = rejected by Holm)
- N1 resolved positive: a four times larger fit set shrinks the path difference; estimation noise contributes to it.
  N1 not resolved: no resolved change between 256 and 1,024 episodes (not evidence of independence); the manuscript
  then reports whether HQ-Fisher − TAN at F1024 is itself below zero. N1 resolved negative: the path difference grows.
- N2 resolved positive: SPAN's shortfall against random subspaces shrinks with data, supporting the estimation-error
  reading of the rank ladder. N2 not resolved: that reading is not supported by this test.

## Reporting commitment
Whatever the outcome, family N and the descriptive ladder are reported in the manuscript (Sections 4.4, 5.1, 5.3 (e),
Appendix A) with this plan's SHA-256; scripts and records go to the repository.
