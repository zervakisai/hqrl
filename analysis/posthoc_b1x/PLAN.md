# Post-hoc B1 extension: re-linearised circuit steps and a damping ladder

Written 2026-10-22 before any run below existed, after the B1 results of record were known. Not part of any frozen
confirmatory family; every result here is labelled post hoc in the manuscript.

## Question
B1 found that the circuit's damped natural step realised through the circuit (HQ-Fisher) gained less held-out surrogate
improvement than the same initial direction applied linearly (TAN), at equal mean forward KL on K. The manuscript
offers one interpretation and names its test: the single natural step, computed with a Fisher of condition number
about 5e6, moves far along directions that change the policy little to first order but much at higher order, so the
KL calibration shrinks the whole step; smaller re-linearised steps to the same final KL would then recover the gain.
A second open point is damping: only delta = 1e-6 (B1) and its delta -> infinity limit (HQ-plain) were measured.

## Units and data
The 72 B1 units (36 two-layer "shipped", primary; 36 three-layer "heldout", consistency, sharing trunk
initialisations), each rebuilt by `dq_b1.build_unit`. The fit set F, calibration set K and held-out set H of each unit
are regenerated from their frozen B1 seeds by `dq_b1.collect_set` (code tree unchanged). Reproduction is checked per
unit: identical row counts and per-episode J for F, K, H, and the re-computed HQ-Fisher and TAN gains equal to the
stored B1 values within relative 1e-6. A unit failing the check is reported and excluded from every contrast.

## Arms (all from the unit's untrained policy, outcome G_H on H as in B1, KL matched on K to within 1e-3 kappa)
- HQ-Fisher@delta and TAN@delta for delta in {1e-6 (= B1), 1e-4, 1e-2, 1}: B1's two arms with relative damping delta.
- HQ-Fisher-m (m = 4, 16): re-linearised circuit steps. At every sub-step the importance-weighted surrogate gradient
  and the Fisher are recomputed on F at the current policy, the circuit's damped (delta = 1e-6) natural step is taken
  through the circuit, scaled so that KL(current || next) on K = kappa / m^2, and the sub-step that crosses the target
  total KL(initial || new) = kappa on K is bisected to the first crossing (at most 25 m sub-steps).
- TAN-16: the same iteration inside the fixed initial tangent span, applied linearly (descriptive).
kappa in {0.001, 0.01} for the iterated arms; {0.001, 0.01, 0.05} for the damping ladder.

## Analysis (fixed now)
Two-layer stratum, kappa = 0.01, paired over units (n = 36 minus excluded units):
- X1: G_H(HQ-Fisher-m16) - G_H(HQ-Fisher@B1)   (does re-linearised realisation improve the circuit update?)
- X2: G_H(HQ-Fisher-m16) - G_H(TAN@B1)         (does it close the gap to the linear path?)
Mean, 95% paired t interval, two-sided paired t p-value, mean/sd; Holm over X1 and X2 at familywise 0.05; labelled
post hoc. The three-layer stratum repeats X1 and X2 as a consistency check (same statistics, not pooled).
Descriptive only (means and individual 95% t intervals, no tests): the step ladder m = 1, 4, 16 at both kappa;
HQ-Fisher@delta - TAN@delta across the damping ladder and HQ-plain (stored B1 arm) at every kappa; TAN-16 - TAN@B1;
sub-steps used; cosine between the net coefficient displacement of HQ-Fisher-m and the TAN direction.

## Reading rules (fixed now; "resolved" = rejected by Holm in the two-layer stratum)
- X1 resolved positive and X2 not resolved: re-linearised circuit steps recover the linear-path gain; the B1 shortfall
  is attributed to realising one natural step through the curved circuit map (not an equivalence claim).
- X1 resolved positive and X2 resolved negative: re-linearisation recovers part of the shortfall; part remains.
- X1 not resolved positive and X2 resolved negative: re-linearised steps do not recover the shortfall.
- X2 resolved positive: re-linearised circuit steps exceed the linear path.
- X1 resolved negative: re-linearisation makes the circuit update worse.
Damping ladder: if HQ-Fisher@delta - TAN@delta stays negative (interval below zero) at every delta at kappa = 0.01,
the shortfall is not specific to near-zero damping; otherwise the delta at which it is unresolved is reported.

## Reporting commitment
Whatever the outcome, X1, X2, the step ladder and the damping ladder are reported in the manuscript (Section 4.4 and
Appendix A) as a post-hoc extension, with this plan's SHA-256, and the scripts and records go to the repository.
