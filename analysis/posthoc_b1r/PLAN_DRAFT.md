# B1R: prespecified replication of three B1 local contrasts on 36 fresh two-layer units

DRAFT, written 2026-10-25 before any fresh-unit episode existed. The lead reviews it, renames it PLAN.md and hashes
it (PLAN.sha256, PLAN.utc); nothing below may change after the hash. Written after the B1 results of record and
after the post-hoc B1 extensions X, C, G, E, N and T (`idea_dq_posthoc_b1x`, `_b1c`, `_b1g`, `_b1e`, `_b1n`, `_b1t`)
were known. Their results on the original units fixed the choice of contrasts, the test directions and the margin
(listed under "Design choices informed by earlier results").

## Question
B1's local comparison takes one untrained policy (a unit), estimates the update direction on a fit set F, calibrates
each update's step to the same mean forward KL on a calibration set K, and measures the importance-weighted surrogate
gain G_H on a held-out set H. On the 36 two-layer units at kappa = 0.01 the stored results are (x 1e-5, paired over
units, 95% t interval):

| contrast | source | mean [95% CI] | dz | status |
|---|---|---|---|---|
| HQ-Fisher - TAN (path) | B1 | -5.70 [-8.21, -3.19] | -0.77 | not prespecified: TAN was descriptive in B1's protocol; path reading post hoc (extension X) |
| CTRIG - TAN (matched classical chart) | B1C, Y2 | -4.31 [-5.75, -2.88] | -1.02 | post hoc, Holm-rejected |
| OPT-CIRC - OPT-TAN (optimised steps) | B1E, E1 | -0.60 [-2.21, +1.02] | -0.13 | post hoc, not rejected |

Every extension re-used the same 72 B1 units (36 two-layer, primary), and the contrasts the manuscript emphasises
were chosen after those units' results were known. B1R asks whether the three findings hold as prespecified tests on
36 new two-layer units drawn by the same construction.

## Units: fresh, by B1's own rules at indices no study has used
A fresh unit is B1's two-layer unit at index i in 100..135 (36 units). Everything except i is B1's:
- policy: `dq_b1.build_unit('shipped', i)`, unchanged (seed `Study('dq').stream(50, i)`; B1 used i = 0..35);
- episode seed: namespace.start + (i x 8 + cell) x 32 + offset in dq_fit, dq_calibration and dq_heldout. This is B1's
  rule with slot = i, giving [start + 25,600, start + 34,816). B1 used [start, start + 18,432) for both circuits, and
  extension N used dq_fit [start + 50,000, start + 77,648). Extensions X, C, G and T reuse B1's own seeds, and no
  other code draws from these namespaces;
- action stream: `Study('dq').stream(SET_TAG[set], 1000 + i, cell, offset)`, stream units 1100..1135. B1, B1N and
  B1C used 1000..1071; D-6 used stream units below 100 in dq_development;
- CTRIG draws: SHA-256 of 'b1c|shipped|<i>|<draw>' (B1C's rule). OPT random starts: SHA-256 of
  'b1e|shipped|<i>|<kind><j>|<kappa>' (B1E's rule). These are new realisations because i is new.
`runner.py seeds` checks all of this statically, with no episode run:
- no fresh seed equals any seed B1's rule gives for its 72 units, or any seed stored in B1's 72 unit records;
- no fresh seed falls in extension N's range;
- the 36 initialisation seeds and the 36 circuit-weight hashes are distinct from B1's and from the training seeds
  `stream(block, 0)`, blocks 0-199;
- for i = 0..35, this runner's rule equals `dq_b1.episode_seed` / `dq_b1.stream_unit`.
The fresh units share with B1 the 8 frozen core cells, the namespaces, the time model, the executive and the policy
class with its initialisation distribution.

## Data per unit
F, K and H have 256 episodes each (8 cells x 32 offsets), flown by the unit's untrained policy through
`p4_train.run_learned_episode`. Each episode passes B1's namespace gate `dq_runtime.require_open`. Rows and advantages
come from `dq_b1.rows_of` (unchanged). The oracle set O is not collected: no arm below uses it. An infrastructure or
invalid-row episode fails the unit (B1's rule). Each unit is certified by B1's checks, which fail closed:
- static parity P1, P2, P5-P7 (`dq_b1.certify_unit`);
- row parity P3 on F (`dq_transplant.certify_rows`);
- trajectory parity P4: the twin re-flies F (cell 0, offset 0).

## Arms (code imported unchanged; B1's settings)
Every arm starts from the unit's untrained policy and moves only the 16 circuit angles, or the head coefficients on
the support. kappa is 0.001 or 0.01. Each step is scaled to mean forward KL kappa on K within 1e-3 kappa (B1's
doubling and bisection). The outcome is G_H on H.
- Extension C's `unit_arms(..., draws=20, sub_steps=False)` (runner a712ee54f556) computes:
  - HQ-Fisher and TAN (B1's definitions, delta = 1e-6);
  - CTRIG: 20 draws of the matched trigonometric chart (m = 64, Fisher-weighted second-order magnitude matched to the
    circuit's). Its unit value is the mean G_H over the draws whose status is ok;
  - descriptive only: QTAY, CQR, CTRIGx0.1 and CTRIGx10. CTRIG-16 is not computed.
- Extension E's `unit` (runner 909d6f54346d) runs on the rows the unit has just saved:
  - OPT-CIRC: SLSQP from 5 starts (calibrated HQ-Fisher, calibrated HQ-plain, 3 random at KL kappa/4), at most 200
    iterations each. The best feasible fit-set surrogate is kept;
  - OPT-TAN: the same over linear tangent steps, from 4 starts (calibrated TAN and 3 random);
  - descriptive only: HQ-plain, TAN-c, the first-crossing grid and the angle displacement.
- Not run: SPAN, RND-k, FIT-k, the mission evaluation of B1C (family C), the three-layer stratum and kappa = 0.05.

## Precheck (before the launch; it never touches a fresh unit)
`runner.py precheck` runs B1 unit shipped 0 (published values) through this runner's own code path, then checks:
- (a) all 768 F/K/H episodes equal the stored B1 episode records (seed, J, decisions), and the row counts match;
- (b) the rows the runner saves equal, bitwise, the rows extension C saved for unit 0 (angles, mask, action,
  advantage, episode and readout);
- (c) every extension C value equals extension C's record within relative 1e-9: HQ-Fisher, TAN, QTAY, and for all 20
  CTRIG draws CTRIG, its ladder and CQR (statuses, alpha, KL, G_H, diagnostics; CTRIG-16 excluded);
- (d) every extension E value equals extension E's record within relative 1e-9 (OPT-CIRC and OPT-TAN end points,
  iterations, messages, G_H and L_F; HQ-Fisher, HQ-plain, TAN, TAN-c; the first crossing);
- (e) HQ-Fisher and TAN G_H are within relative 1e-6 of B1's own instrument (B1C's criterion);
- (f) the certificate passes;
- (g) the static seed checks pass;
- the B1 code tree's live copy is a66f8979.
`runner.py launch` refuses to start unless PLAN.md matches PLAN.sha256 and precheck.json is the precheck of record,
passed under this plan's SHA-256. It also refuses if fewer than 8 h remain before the DQ calendar cut-off (below).

## Primary analysis (fixed now)
The analysis is paired over fresh units: each unit gives one difference. Family R is two-layer, kappa = 0.01, Holm
over three at familywise alpha = 0.05:
- **R1**: G_H(HQ-Fisher) - G_H(TAN). One-sided paired t, H0: mean >= 0 against H1: mean < 0 (direction from B1).
- **R2**: G_H(CTRIG) - G_H(TAN). One-sided paired t, H0: mean >= 0 against H1: mean < 0 (direction from B1C Y2).
- **R3**: G_H(OPT-CIRC) - G_H(OPT-TAN). Non-inferiority, one-sided paired t: H0: mean <= -delta against
  H1: mean > -delta, with **delta = 2.85 x 10^-5**. This is half of B1's HQ-Fisher - TAN mean on the original units
  (5.70 x 10^-5). The claim under test is extension E's reading: optimised circuit steps are not limited by what
  circuit policies can reach at this KL. Its confirmatory form is that they lose less than half the natural step's
  shortfall against the best linear tangent step found.
Each contrast reports n, mean, sd, the 95% two-sided t interval, the one-sided 95% bound, t, p, dz = mean/sd and the
share of units below zero. Values are printed in units of 1e-5.

Secondary family S (Holm over two; reported as secondary, never as primary):
- S1: CTRIG - HQ-Fisher, two-sided (B1C's Y1 was not resolved);
- S2: OPT-CIRC - HQ-Fisher, one-sided > 0 (B1E's E2).
R3's two-sided difference test and its TOST at +-delta are also reported, outside any family.

Descriptive only (no test is read):
- every contrast at kappa = 0.001, and every arm's mean;
- QTAY - HQ-Fisher, CQR - CTRIG, the CTRIG ladder, CTRIG#0 - TAN, OPT-TAN - TAN, OPT-CIRC - TAN, TAN-c - TAN and
  HQ-plain - TAN;
- the first-crossing counts, the angle displacement, the optimisers' best-start counts and iterations, and the CTRIG
  within-unit draw sd;
- side by side with the 36 original B1 units (stored values only): the difference between the two studies' means
  (Welch, descriptive) and the 72-unit pooled mean. Pooling is never used for a claim.

## Exclusions and missing data (fixed now)
- A unit whose record is not 'complete' is listed and excluded from every contrast: an episode failure, a certificate
  failure, a refused namespace, or any exception. No unit is replaced. Only a unit interrupted by a crash or reboot is
  rerun; it starts again from scratch, and its partial staging files are moved to interrupted/. The pipeline is
  deterministic, so a rerun gives the same values.
- A unit is excluded from every contrast if extension E's recomputed HQ-Fisher or TAN G_H differs from extension C's by
  more than relative 1e-6 at either kappa (an internal consistency check, expected never to trigger).
- An arm whose status is not ok at kappa = 0.01 drops its unit from the contrasts that use that arm, and the counts
  are reported. Not ok means a calibration failure, or no feasible end point for OPT-CIRC or OPT-TAN. CTRIG uses the
  mean of its ok draws, and the number of units with fewer than 20 ok draws is reported.
- If any primary contrast has fewer than 30 units, the report states that the replication is incomplete. The test is
  unchanged.

## Power (from the stored two-layer values at kappa = 0.01; disclosed, not part of the tests)
With n = 36, one-sided, at Holm's smallest level 0.05/3 (and at 0.05):
- R1 (sd 7.42 x 1e-5): power 0.99 (0.998) at B1's effect; 0.89 at 75% of it; 0.54 (0.73) at 50%. Minimum detectable
  effect at 80% power: 3.8 x 1e-5.
- R2 (sd 4.24): above 0.999 at B1C's effect; 0.80 (0.91) at 50%.
- R3 (sd 4.77): non-inferiority power 0.73 (0.87) if the true mean is B1E's -0.60; 0.91 (0.97) if it is 0. If the
  sd is 6.0, power is 0.52 (0.71). R3 is the weakest test of the three. 48 units would raise it to 0.86 (0.94) at
  -0.60 (lead's decision, not part of this draft).

## Reading rules (fixed now; "rejected" means rejected by Holm in family R)
- **R1 rejected**: the path shortfall is replicated. The circuit-realised natural step gains less than the same
  initial direction applied linearly, on units that were never used, in a prespecified test.
  **R1 not rejected**: not replicated at this sample size. The manuscript must state that the B1 path shortfall was
  not replicated in a prespecified replication, report B1 and B1R side by side, and qualify every path claim in the
  abstract, the summary and the conclusions. No pooling and no post-hoc analysis may replace R1.
- **R2 rejected**: the matched classical trigonometric chart also falls short of the linear path, replicated; the
  generic (not circuit-specific) reading of the path effect is confirmed for this matching.
  **R2 not rejected**: B1C's matched-chart shortfall stays a single post-hoc finding. The manuscript must not call the
  path effect generic; it says that a matched classical chart fell short in a post-hoc study and that this was not
  replicated.
- **R3 rejected**: optimised circuit steps were non-inferior to the best linear tangent step found, at margin
  2.85 x 10^-5. Caveat: the optimiser is local and five-start.
  **R3 not rejected**: a reachability limit of 2.85 x 10^-5 or more is not excluded, and whether the shortfall lies in
  the path or in the reachable set stays unresolved. If R3's two-sided difference is negative at 0.05, that is
  reported as secondary evidence of a reachability limit.
- **Joint reading of R1 and R3**:
  - both rejected: the B1 shortfall replicates and, at margin delta, lies in the natural step's path rather than in
    what circuit policies can reach (local optimiser);
  - R1 only: the shortfall replicates; path against reachability is not settled;
  - R3 only: optimised steps are non-inferior, but no natural-step shortfall replicated for them to explain;
  - neither: neither finding is shown.
- **S1 and S2** are read only as secondary, with the same direction wording as B1C's Y1 and B1E's E2.
- In every case B1's original units remain the study of record for B1. B1R is reported beside B1, never merged with
  it, and its p values are never combined with B1's.

## Design choices informed by earlier results (disclosed)
- The three contrasts, the one-sided directions of R1 and R2, the non-inferiority form of R3 and delta were all chosen
  knowing the results of B1, B1C and B1E on the original units. On those units R3 at this delta would give p = 0.0038
  (one-sided 95% lower bound -1.94 x 1e-5). The replication's evidence comes only from the fresh units.
- kappa = 0.01 is primary (B1's primary value). kappa = 0.05 is not used: extension X found its steps ill-posed, with
  scales up to 6e8.
- Only the two-layer circuit is run (B1's primary stratum). CTRIG-16, the mission evaluation, SPAN, RND-k and FIT-k
  are left out for cost. They are not part of any claim.
- No hyperparameter was selected or tuned for B1R. Every setting is B1's, B1C's or B1E's frozen value, and no fresh
  unit's data was looked at before this plan.

## Order of operations
1. The lead reviews this draft, renames it PLAN.md, and writes PLAN.sha256 and PLAN.utc.
2. The SHA-256 of runner.py and b1r_analysis.py is recorded in PLAN_LOG.md.
3. `nice -n 10 .venv/bin/python -B runner.py precheck` runs (one process, about 30-60 min). Only status, reproduction
   and timing fields of B1 unit 0 are read.
4. `nice -n 10 .venv/bin/python -B runner.py launch --workers 10` runs. The data collection must finish before the
   DQ calendar cut-off of 2026-10-31T00:00Z, after which `dq_runtime.require_open` refuses the B1 namespaces. A
   relaunch skips completed units.
5. `b1r_analysis.py` runs once all 36 records exist (it refuses partial data unless given --provisional), unchanged
   from its recorded hash.

## Reporting commitment
Whatever the outcome, families R and S, the exclusions, the descriptive values and the side-by-side with the
original units are reported in the manuscript (Section 4.4, Appendix A) as a prespecified replication, with this
plan's SHA-256. The scripts and records go to the repository.
