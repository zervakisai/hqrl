# Post-hoc B1 extension C: matched-tangent classical charts and the mission effect of local updates

Written 2026-10-23 before any run below existed, after the B1 results of record and the B1 extension of 2026-10-22
(`idea_dq_posthoc_b1x`, plan SHA-256 efd8b1d7...) were known. Not part of any frozen confirmatory family; every result
is labelled post hoc in the manuscript. Decisions informed by earlier results are listed under "Design choices".

## Questions
B1 and its extension found that the circuit's damped natural step realised through the circuit (HQ-Fisher) gained less
held-out surrogate improvement than the same initial direction applied linearly (TAN), at equal mean forward KL on K,
and that re-linearised circuit sub-steps recovered about a quarter of that gap. The manuscript states that this path
effect is not shown to be specific to the circuit: the same comparison applies to any nonlinear parameterisation, and
none was tested (limitation (e)). Two questions follow.

A. Does a classical nonlinear parameterisation with the same starting policy, the same tangent, the same pulled-back
   Fisher and gradient, hence the same initial natural direction, and the same Fisher-weighted second-order magnitude,
   show a comparable shortfall against the linear path?
C. Do the local surrogate differences appear in the directly measured mission score, with enough episodes to resolve
   an effect of the size that the surrogate predicts to first order?

## Units and data
The 72 B1 units (36 two-layer "shipped", primary; 36 three-layer "heldout", consistency, sharing trunk
initialisations), each rebuilt by `dq_b1.build_unit` from the B1 code tree of record (a66f8979, the tree the B1
extension used). The fit set F, calibration set K and held-out set H of each unit are regenerated from their frozen B1
seeds by `dq_b1.collect_set`. Reproduction check per unit: identical row counts and per-episode J for F, K and H, and
the re-computed HQ-Fisher and TAN gains equal to the stored B1 values within relative 1e-6 at kappa = 0.001 and 0.01.
A unit failing the check is reported and excluded from every contrast.

## Charts (theta are the circuit weights; inert weights stay inert in every chart)
All arms start from c0 = C(theta0) and move only the head coefficients on the support S. With A = dC/dtheta at
theta0 (restricted to S), every chart below has Jacobian A at its origin, so its pulled-back gradient A^T g and Fisher
A^T F A, and therefore its damped (delta = 1e-6) natural direction d, are those of the circuit exactly.
- HQ-Fisher: c(a) = C(theta0 + a d) (B1).
- TAN: c(a) = c0 + a A d (B1).
- CTRIG (primary classical chart): c(phi) = c0 + A sin(phi) + s R (1 - cos(W phi)), phi = a d; sin and cos act
  componentwise. W (m x n_weights, m = 64) has independent entries uniform on {-1, 0, 1}, with the columns of the inert
  weights set to zero; R (|S| x m) has independent N(0, 1) entries. Like the circuit's coefficient map C(theta), the
  chart is then a trigonometric polynomial of degree at most 1 in each weight (periodic and bounded), its Jacobian at
  the origin is A, and its Hessian is H_ij = s sum_k R_k W_ki W_kj. The scale s is set so that
  the Fisher-weighted second-order magnitude matches the circuit's: sum_ij H_ij^T F H_ij = sum_ij Hc_ij^T F Hc_ij, with
  Hc the circuit's Hessian d^2 C / dtheta_i dtheta_j at theta0 (exact, by automatic differentiation) and F the
  coefficient Fisher on the fit set at c0. 20 independent draws of (W, R) per unit; their gains are averaged within the
  unit, as for RND-k. Draw seeds: the first 8 bytes of SHA-256 of "b1c|<circuit>|<index>|<draw>".
- Descriptive charts on the same draws: QTAY, the circuit's own second-order Taylor curve c0 + a A d + (a^2/2) Hc[d, d];
  CQR, CTRIG's second-order Taylor curve c0 + a A d + (a^2/2) H[d, d]; CTRIG with s multiplied by 0.1 and by 10
  (curvature ladder); CTRIG-16, the CTRIG analogue of the extension's HQ-Fisher-16 (re-linearised natural sub-steps
  through the chart with per-sub-step KL kappa/256 and gradient and Fisher recomputed on F, first crossing bisected,
  at most 400 sub-steps), for draw 0 at kappa = 0.01 only.
Each arm's scale is calibrated on K to mean forward KL kappa within 1e-3 kappa (B1's first-crossing rule);
kappa in {0.001, 0.01}. Outcome: held-out surrogate gain G_H on H, as in B1.

## Mission evaluation (question C)
For each unit, at kappa = 0.01: BASE (the unit's twin at c0), HQ-Fisher, TAN and CTRIG draw 0, each as an exact
Fourier twin with the arm's coefficients, flown on the same 160 fresh common-random-number episodes (20 per cell, 8
cells): environment seed = start(dq_fresh_eval) + 50000 + (slot x 8 + cell) x 20 + offset (B1's own fresh episodes
occupy the first 5,760 seeds of this namespace), action stream DQ_STUDY.stream(SET_TAG fresh_eval, 1000 + slot, cell,
1000 + offset). Outcome: cell-weighted J (p4.aggregate_evaluation) in RE (x 18); also timely-credited reports, false
reports, return fraction and correction attempts per episode.

## Analysis (fixed now)
Paired over units, mean, 95% paired t interval, two-sided paired t p-value, mean/sd, share of units below zero.
Family A (two-layer stratum, kappa = 0.01, Holm over the two at familywise 0.05):
- Y1: G_H(CTRIG) - G_H(HQ-Fisher)    (does a matched classical chart lose less than the circuit?)
- Y2: G_H(CTRIG) - G_H(TAN)          (does the matched classical chart also fall short of the linear path?)
Family C (two-layer stratum, Holm over the two at familywise 0.05), in RE per episode:
- W1: J(TAN) - J(HQ-Fisher)          (does the path difference appear in the measured mission score?)
- W2: J(TAN) - J(BASE)               (does the strongest local update change the measured mission score?)
The three-layer stratum repeats Y1, Y2, W1 and W2 with the same statistics as a consistency check (not pooled, not
part of the Holm families). For W1 and W2 the minimum detectable effect at 80% power (two-sided 0.05, from the observed
sd) is reported with the result.
Descriptive only (means and individual 95% t intervals, no tests): every arm's G_H at both kappa; QTAY - HQ-Fisher,
CQR - CTRIG, the CTRIG curvature ladder, CTRIG-16 - CTRIG and CTRIG-16 - TAN, and HQ-Fisher-16 from the extension's
stored records on the same units; the calibrated scales and the quadratic relative error of each arm's calibration;
curvature diagnostics (the share of Hc[d, d] in the Fisher-orthogonal tangent, and the ratio of the Fisher-weighted
curvature along d to its mean over 200 random unit directions in the active weights, for the circuit and for CTRIG);
mission-score differences of every arm against BASE with their first-order surrogate predictions
(G_H x decisions per episode x 18) and the per-unit correlation between predicted and measured differences.

## Reading rules (fixed now; "resolved" = rejected by Holm in the two-layer stratum)
Family A:
- Y2 resolved negative and Y1 not resolved: a classical chart matched in start, tangent, Fisher and second-order
  magnitude shows a shortfall against the linear path that is not distinguishable from the circuit's; the B1 path
  effect is then reproduced by a generic classical nonlinear parameterisation and is not specific to the circuit.
- Y2 resolved negative and Y1 resolved positive: the matched classical chart also falls short of the linear path, but
  by less than the circuit; a generic component and a circuit-specific excess, at this matching.
- Y2 not resolved and Y1 resolved positive: the matched classical chart shows no resolved shortfall while the circuit
  does; the path effect is then specific to the circuit relative to this control.
- Y1 resolved negative: the matched classical chart gains less than the circuit.
- Y2 resolved positive: the matched classical chart gains more than the linear path.
- Neither resolved: inconclusive at this sample size.
Family C:
- W1 resolved positive: the path difference has a measurable mission consequence. W1 not resolved: not resolved at
  the reported minimum detectable effect; this is not evidence of equality. W1 resolved negative: reported as such.
- W2 is read in the same way for the strongest local update against the unchanged policy.

## Design choices informed by earlier results (disclosed)
- kappa = 0.05 is not used: in the extension, HQ-Fisher at 0.05 reached step scales near 6e8 (numerically ill-posed).
- The matching of second-order magnitude in the Fisher metric, the periodic chart form and m = 64 were chosen before
  any classical-chart gain was computed; the curvature ladder and QTAY/CQR report how much the matching matters.
- The 160 mission episodes per arm and unit were chosen from Table 8's spread (sd about 0.09 RE per unit at 80
  episodes) so that the first-order prediction for TAN - HQ-Fisher (about 0.05 RE per episode) is resolvable.

## Reporting commitment
Whatever the outcome, families A and C, the three-layer consistency values and the descriptive quantities are reported
in the manuscript (Sections 4.4, 4.5, 5.1 and 5.3, Appendix A) as a post-hoc extension with this plan's SHA-256, and the
scripts and records go to the repository.
