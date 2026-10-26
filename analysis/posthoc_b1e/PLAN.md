# Post-hoc B1 extension E: optimised circuit steps against the best linear tangent step, and calibration checks

Written 2026-10-23 (before any result of extensions C and G had been read), after the B1 results of record, the B1
extension of 2026-10-22 and two independent pre-submission reviews (reviewer_audit/work/review_A.md, review_B.md). Not
part of any frozen confirmatory family; every result is labelled post hoc in the manuscript.

## Questions
B1's linear control TAN moves the coefficients along the circuit's initial direction, c0 + a A d, which in general no
circuit can represent. A reviewer therefore objects that "path" and "reachable set" are not separated: the circuit step
may gain less because no circuit policy at that KL gains as much, not because the natural step takes a poor path.
E. At equal mean forward KL on K, does the best circuit step that a local optimiser finds gain as much held-out
   surrogate improvement as the best linear step in the tangent span, and more than the natural circuit step?
Two checks of B1's instruments are added: whether the calibrated HQ-Fisher scale is the first crossing of the KL
target along the ray (the KL of a periodic circuit need not be monotone in the scale), and how far the calibrated
circuit steps move the angles (wrap-around beyond pi).

## Units and data
The 72 B1 units. Their fit set F, calibration set K and held-out set H are the rows regenerated from the frozen B1 seeds
by extension C (idea_dq_posthoc_b1c/rows/, saved before any of its results were read; each file's SHA-256 is in the
extension's unit record), loaded with the Fourier features rebuilt by qc.basis. Units that extension C excluded by its
reproduction check, or whose rows were not saved, are excluded here.

## Arms (from the unit's untrained policy; kappa in {0.001, 0.01}; G_H on H as in B1)
- OPT-CIRC: maximise the importance-weighted fit-set surrogate L_F(theta) = mean_F (pi_theta / pi_0 - 1) Ahat over the
  non-inert circuit weights subject to mean forward KL(pi_0 || pi_theta) on K <= kappa, by SLSQP (scipy) with exact
  gradients (torch), from five starts: the calibrated HQ-Fisher step, the calibrated HQ-plain step, and three random
  directions (seeded by SHA-256 of "b1e|<circuit>|<index>|<start>") scaled to KL kappa/4. At most 200 iterations per
  start; the feasible end point (KL <= kappa (1 + 1e-3)) with the largest L_F is kept.
- OPT-TAN: the same optimisation over linear steps in the tangent span, c0 + A u, from the calibrated TAN step and three
  random starts.
- TAN-c: the damped natural step inside the tangent span with coefficient-space damping (Eq. (3) with B the orthonormal
  tangent basis), applied linearly and calibrated as in B1 (descriptive; TAN damps in theta coordinates).
- Re-computed HQ-Fisher and TAN (B1), whose G_H must equal extension C's values for the unit (relative 1e-6).

## Checks (descriptive)
- First crossing: KL on K along the HQ-Fisher ray at 4,000 equally spaced scales in [0, alpha_B1]; a unit is flagged if
  KL exceeds kappa (1 + 1e-3) before alpha_B1, and then G_H at the first crossing is reported beside B1's.
- Angle displacement of the calibrated HQ-Fisher and HQ-plain steps: max_i |alpha d_i| and mean_i |alpha d_i| over the
  non-inert weights, and the share of units with max_i |alpha d_i| > pi.

## Analysis (fixed now)
Paired over units: mean, 95% paired t interval, two-sided paired t p-value, mean/sd, share of units below zero.
Family E (two-layer stratum, kappa = 0.01, Holm over the two at familywise 0.05):
- E1: G_H(OPT-CIRC) - G_H(OPT-TAN)
- E2: G_H(OPT-CIRC) - G_H(HQ-Fisher)
The three-layer stratum repeats E1 and E2 as a consistency check (not pooled, not in the family). Descriptive: both
contrasts at kappa = 0.001; G_H(OPT-TAN) - G_H(TAN); G_H(TAN-c) - G_H(TAN); fit-set objective values of every arm; the
first-crossing flags and the displacement summaries.

## Reading rules (fixed now; "resolved" = rejected by Holm in the two-layer stratum)
- E2 resolved positive and E1 not resolved: optimised circuit steps reach held-out gains not distinguishable from the
  best linear tangent step found; the B1 shortfall then lies in the natural step's path, not in what circuit policies
  can reach at this KL (not an equivalence claim).
- E1 resolved negative: even optimised circuit steps gain less than the best linear tangent step; the circuit's
  reachable policies limit the gain at this KL (with the caveat that the optimiser is local).
- E2 not resolved: optimisation did not improve on the natural circuit step.
- E1 resolved positive: optimised circuit steps exceed the best linear tangent step found.
A local optimiser can miss better circuit steps, so E1 negative bounds what was found, not what exists.

## Reporting commitment
Whatever the outcome, family E, the checks and the descriptive values are reported in the manuscript (Sections 3.7, 4.4,
5.1, Appendix A) as a post-hoc extension with this plan's SHA-256; scripts and records go to the repository.
