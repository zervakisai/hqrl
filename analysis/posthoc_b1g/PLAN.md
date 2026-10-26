# Post-hoc B1 extension G: the path comparison at generic circuit angles and on the classical C-matched head

Written 2026-10-23 before any run below existed, after the B1 results of record and the B1 extension of 2026-10-22 were
known, and after two independent pre-submission reviews (reviewer_audit/work/review_A.md, review_B.md) raised the
questions below. Not part of any frozen confirmatory family; every result is labelled post hoc in the manuscript.
Runs alongside the separately planned extension C (idea_dq_posthoc_b1c, matched-tangent classical charts).

## Questions
B1 found that the circuit's damped natural step realised through the circuit (HQ-Fisher) gained less held-out
surrogate improvement than the same initial direction applied linearly (TAN), at equal mean forward KL on K.
G. The B1 units start the circuit at small angles (0.1 x N(0, 1), the initialisation used in training). Near zero
   angles the non-final RZ directions are flat to first order but curved, so the coefficient Jacobian has weak
   directions (relative singular values about 0.04-0.10 in the two-layer circuit). Does the path difference persist
   when the same units start from generic circuit angles?
D. The manuscript says the same comparison applies to any nonlinear head, including a classical one, and was not run.
   Does C-matched, the classical head with the same 4-dimensional bottleneck, show a difference between its own damped
   natural step realised through its nonlinear layer and the same initial direction applied linearly?

## Units and data
Part G (36 units): each two-layer B1 unit (`dq_b1.build_unit('shipped', i)`, i = 0..35) with only its 16 circuit
angles replaced by independent draws from U(-pi, pi), generator seed = first 8 bytes of SHA-256 of "b1g|shipped|<i>".
Part D (36 units): C-matched unit i is `RecurrentActorCritic(..., actor_head='bottleneck')` built under the B1
initialisation seed of unit i; its encoder, GRU, critic and first head layer are then identical to B1 unit i (checked
before this plan). Its head is Linear(64, 4)-Tanh-Linear(4, 4)-Tanh-Linear(4, 27); only the middle Linear(4, 4) (20
parameters, the counterpart of the 16 circuit angles) is changed by the arms; the trunk, first layer, readout and critic
are frozen, as in B1.
Both parts collect their sets with `dq_b1.collect_set` under the seeds of two-layer B1 unit i (common random numbers
with B1): part G the fit, oracle, calibration and held-out sets (B1's full instrument needs O), part D the fit,
calibration and held-out sets. Advantages, KL calibration (first crossing by B1's doubling-then-bisection rule, to
1e-3 kappa) and the held-out surrogate gain G_H are B1's.

## Arms
Part G: every B1 arm, computed by the frozen `dq_b1.unit_instruments` unchanged (HQ-Fisher, TAN, SPAN, RND-k, HQ-plain,
etc.), at kappa in {0.001, 0.01, 0.05}.
Part D (written for this plan; the B1 functions cannot take a non-Fourier head): with u_i = tanh(first layer(h_i)),
z_i(theta) = tanh(M u_i + b), theta = (M, b), logits_i = W_r z_i + b_r:
- C-Fisher: the damped (delta = 1e-6, lambda = delta tr(F_theta) / 20) natural step d = (F_theta + lambda I)^-1 g_theta
  on the fit set, realised through the layer: theta0 + a d.
- C-TAN: the same initial direction applied linearly in the head output: z_i + a J_i d, J_i = dz_i / dtheta.
- C-plain: the plain gradient g_theta realised through the layer (descriptive).
g_theta and F_theta are the exact gradient of the surrogate and the exact Fisher of the mean forward KL on the fit set,
computed in closed form and checked against automatic differentiation before the launch.

## Analysis (fixed now)
Paired over units: mean, 95% paired t interval, two-sided paired t p-value, mean/sd, share of units below zero.
Family G (kappa = 0.01, Holm over the two at familywise 0.05):
- G1: G_H(HQ-Fisher) - G_H(TAN) in part G   (does the circuit's path difference persist at generic angles?)
- D1: G_H(C-Fisher) - G_H(C-TAN) in part D  (does a classical nonlinear head show a path difference?)
Descriptive only: both contrasts at kappa = 0.001; G1 at 0.05 (subject to the extension's caveat on step scales); the
other B1 contrasts in part G (SPAN - HQ-Fisher, HQ-Fisher - RND-k, TAN - RND-k); C-plain - C-TAN; every arm's mean gain;
the retained share mean G_H(natural through the head) / mean G_H(linear), with a percentile bootstrap over units
(20,000 resamples), for HQ at B1's angles (stored B1 records, same units), HQ at generic angles and C-matched;
conditioning (Jacobian (s1/sk)^2, Fisher condition number on the tangent), calibrated scales, quadratic relative error
and parameter displacement of each natural arm.

## Reading rules (fixed now; "resolved" = rejected by Holm)
- G1 resolved negative: the circuit's path difference persists at generic angles; it is not only a property of the
  small-angle initialisation. G1 not resolved: not resolved at generic angles (not evidence of no difference); the
  manuscript then states that the B1 path difference was found at the small-angle initialisation and was not resolved
  at generic angles. G1 resolved positive: at generic angles the circuit path gains more than the linear one.
- D1 resolved negative: a classical nonlinear head also shows a path difference between its natural step and its
  linearisation, so the effect is not specific to the circuit. D1 not resolved: no resolved difference for this
  classical head (not evidence of equality), reported with the retained-share comparison. D1 resolved positive:
  reported as such.
The retained shares are compared descriptively only; no test across heads.

## Reporting commitment
Whatever the outcome, family G, the descriptive values and the conditioning diagnostics are reported in the manuscript
(Sections 4.4, 5.1 and 5.3, Appendix A) as a post-hoc extension with this plan's SHA-256; scripts and records go to the
repository.
