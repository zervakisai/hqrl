# B1R: prespecified replication of B1's local path contrasts on 36 fresh two-layer units

Final plan, written 2026-10-25 before any fresh-unit episode existed. It is PLAN_DRAFT.md (kept unchanged for the
record) with every fix the lead decided after the adversarial review (REVIEW.md; the fixes are listed item by item in
README_BUILD.md, "Fixes after review (2026-10-25)"). The lead hashes this file (PLAN.sha256, PLAN.utc). Nothing below
may change after the hash. Any later change is a deviation and is logged in PLAN_LOG.md.

This plan was written after the B1 results of record were known, and after the post-hoc B1 extensions X, C, G, E, N
and T (`idea_dq_posthoc_b1x`, `_b1c`, `_b1g`, `_b1e`, `_b1n`, `_b1t`). Their results on the original units fixed the
choice of contrasts, the test directions and the R3 margin (see "Design choices informed by earlier results").

## Question
B1's local comparison works on one untrained policy (a unit). It estimates the update direction on a fit set F,
calibrates each update's step to the same mean forward KL on a calibration set K, and measures the
importance-weighted surrogate gain G_H on a held-out set H. On the 36 two-layer units at kappa = 0.01 the stored
results are (x 1e-5, paired over units, 95% t interval):

| contrast | source | mean [95% CI] | dz | status before B1R | role in B1R |
|---|---|---|---|---|---|
| HQ-Fisher - TAN (path) | B1 | -5.70 [-8.21, -3.19] | -0.77 | not prespecified: TAN was descriptive in B1's protocol; path reading post hoc (extension X) | R1, primary |
| CTRIG - TAN (matched classical chart) | B1C, Y2 | -4.31 [-5.75, -2.88] | -1.02 | post hoc, Holm-rejected | R2, primary |
| OPT-CIRC - OPT-TAN (optimised steps) | B1E, E1 | -0.60 [-2.21, +1.02] | -0.13 | post hoc, not rejected | R3, secondary margin test |
| OPT-CIRC - OPT-TAN on the fit-set objective L_F | B1E records (recomputed) | -7.98 [-9.34, -6.62] | -1.98 | post hoc, descriptive | descriptive; reported with every R3 reading |

Every extension reused the same 72 B1 units (36 two-layer, primary). The contrasts the manuscript emphasises were
chosen after those units' results were known. B1R asks whether the path findings hold as prespecified tests on 36 new
two-layer units, drawn by the same construction.

## Units: fresh, by B1's own rules at indices no study has used
A fresh unit is B1's two-layer unit at an index i in 100..135 (36 units). Everything except i is B1's:
- **Policy:** `dq_b1.build_unit('shipped', i)`, unchanged. Its seed is `Study('dq').stream(50, i)`; B1 used
  i = 0..35.
- **Episode seed:** namespace.start + (i x 8 + cell) x 32 + offset, in dq_fit, dq_calibration and dq_heldout.
  - This is B1's rule with slot = i. It gives [start + 25,600, start + 34,816).
  - B1 used [start, start + 18,432) for both circuits.
  - Extension N used dq_fit [start + 50,000, start + 77,648).
  - Extensions X, C, G and T reuse B1's own seeds. No other code draws from these namespaces.
- **Action stream:** `Study('dq').stream(SET_TAG[set], 1000 + i, cell, offset)`, i.e. stream units 1100..1135.
  B1, B1N and B1C used 1000..1071; D-6 used stream units below 100 in dq_development.
- **CTRIG draws:** SHA-256 of 'b1c|shipped|<i>|<draw>' (B1C's rule).
- **OPT random starts:** SHA-256 of 'b1e|shipped|<i>|<kind><j>|<kappa>' (B1E's rule).
- The CTRIG draws and OPT starts are new realisations, because i is new.

`runner.py seeds` checks all of this statically, with no episode run:
- no fresh seed equals any seed that B1's rule gives for its 72 units, or any seed stored in B1's 72 unit records;
- no fresh seed falls in extension N's range;
- the 36 initialisation seeds and the 36 circuit-weight hashes differ from B1's and from the training seeds
  `stream(block, 0)`, blocks 0-199;
- for i = 0..35, this runner's rule equals `dq_b1.episode_seed` / `dq_b1.stream_unit`.

The adversarial review also checked the 32-bit action-stream seeds directly. There are 0 collisions between the
27,648 fresh seeds and B1's 79,487 distinct seeds, and none with the 36 fresh initialisation seeds.

The fresh units share with B1:
- the 8 frozen core cells;
- the namespaces;
- the time model;
- the executive;
- the policy class and its initialisation distribution.

FRESH = 100..135 and n = 36 are fixed in runner.py and in b1r_analysis.py, and a unit test checks that they agree.

**Approval (disclosed).** B1R draws 27,648 new episode seeds (9,216 in each of dq_fit, dq_calibration and dq_heldout)
from the B1-gated namespaces, under B1's Codex approval (`dq_runtime.GATED` / `require_open`). That approval covered
B1's seed manifest (slots 0..71). These slots were not part of it. No new approval was sought. The precedent is
extension N's extra fit seeds (dq_fit [50,000, 77,648)), which were drawn the same way.

## Data per unit
F, K and H have 256 episodes each (8 cells x 32 offsets). They are flown by the unit's untrained policy through
`p4_train.run_learned_episode`, and each episode passes B1's namespace gate `dq_runtime.require_open`. Rows and
advantages come from `dq_b1.rows_of` (unchanged). Its row diagnostics (`row_info`: baseline fallback rows, replay
log-probability error) are kept in the unit record, as B1 kept them. The oracle set O is not collected, because no arm
below uses it. An infrastructure or invalid-row episode fails the unit (B1's rule).

Each unit is certified by B1's checks, in B1's fail-closed order (`dq_b1.run_unit`). No arm is computed before the
certificate has passed:
1. F, K, H are collected.
2. Static parity P1, P2, P5-P7 (`dq_b1.certify_unit`).
3. Rows.
4. Row parity P3 on F (`dq_transplant.certify_rows`).
5. Trajectory parity P4: the twin re-flies F (cell 0, offset 0). The re-flight passes `require_open`, so the DQ
   calendar cut-off (2026-10-31T00:00Z) applies to the certificate as well as to the data.
6. Then the staged rows and the arms.

Each unit record carries the SHA-256 of the runner that produced it (`runner_sha256`).

## Arms (code imported unchanged; B1's settings)
Every arm starts from the unit's untrained policy and moves only the 16 circuit angles, or the head coefficients on
the support. kappa is 0.001 or 0.01. Each step is scaled to mean forward KL kappa on K, within 1e-3 kappa (B1's
doubling and bisection). The outcome is G_H on H.

Extension C's `unit_arms(..., draws=20, sub_steps=False)` (runner a712ee54f556) computes:
- HQ-Fisher and TAN (B1's definitions, delta = 1e-6);
- CTRIG: 20 draws of the matched trigonometric chart (m = 64, Fisher-weighted second-order magnitude matched to the
  circuit's). Its unit value is the mean G_H over the draws whose status is ok;
- descriptive only: QTAY, CQR, CTRIGx0.1 and CTRIGx10. CTRIG-16 is not computed.

Extension E's `unit` (runner 909d6f54346d) runs on the rows the unit has just saved. It computes:
- OPT-CIRC: SLSQP from 5 starts (calibrated HQ-Fisher, calibrated HQ-plain, and 3 random at KL kappa/4), at most 200
  iterations each. The best feasible fit-set surrogate L_F is kept;
- OPT-TAN: the same over linear tangent steps, from 4 starts (calibrated TAN and 3 random);
- descriptive only: HQ-plain, TAN-c, the first-crossing grid and the angle displacement.

Not run:
- SPAN, RND-k and FIT-k;
- the mission evaluation of B1C (family C);
- the three-layer stratum;
- kappa = 0.05.

## Precheck of record (before the launch; it never touches a fresh unit)
The lead runs `runner.py precheck` after hashing this plan. It runs B1 unit shipped 0 (published values) in full, all
768 F/K/H episodes, through this runner's own code path. It then checks:
- (a) all 768 F/K/H episodes equal the stored B1 episode records (seed, J, decisions), and the row counts match;
- (b) the rows the runner saves equal, bitwise, the rows extension C saved for unit 0 (angles, mask, action,
  advantage, episode and readout). The rows file's SHA-256 is reported against B1C's;
- (c) every extension C value equals extension C's record within relative 1e-9:
  - HQ-Fisher, TAN and QTAY;
  - for all 20 CTRIG draws: CTRIG, its ladder and CQR (statuses, alpha, KL, G_H, diagnostics);
  - CTRIG-16 is excluded;
- (d) every extension E value equals extension E's record within relative 1e-9:
  - the OPT-CIRC and OPT-TAN end points, iterations, messages, G_H and L_F;
  - HQ-Fisher, HQ-plain, TAN and TAN-c;
  - the first crossing;
- (e) the HQ-Fisher and TAN G_H are within relative 1e-6 of B1's own instrument (B1C's criterion);
- (f) the certificate passes;
- (g) the static seed checks pass;
- (h) the unit record carries this runner's SHA-256.

The B1 code tree's live copy must be a66f8979. In every comparison a NaN matches only a NaN; a NaN against a number is
a mismatch. Reported, but not a pass criterion: B1's row diagnostics of the same sets (`row_info`).

The precheck writes precheck.json and keeps a copy of the runner it ran (`precheck/runner_snapshot.py.txt`). Only its
pass flags, statuses, reproduction differences and timings are read. If it fails, failure rule 4 applies.

`runner.py launch` refuses to start unless all of the following hold:
- PLAN.md matches PLAN.sha256;
- runner.py's SHA-256 is written in this plan, or, after a disclosed deviation, in PLAN_LOG.md;
- the latest precheck of record (precheck.json, or precheck_2.json, ... after a rerun) passed under this plan's
  SHA-256;
- that precheck was run by the same runner.py, with the same reused runners and the B1 tree of record, which is still
  the live copy;
- at least 8 h remain before the DQ calendar cut-off.

## Primary analysis: family R (fixed now)
The analysis is paired over fresh units: each unit gives one difference. Family R is two-layer, kappa = 0.01, with
Holm over two at familywise alpha = 0.05:
- **R1**: G_H(HQ-Fisher) - G_H(TAN). One-sided paired t, H0: mean >= 0 against H1: mean < 0 (direction from B1).
- **R2**: G_H(CTRIG) - G_H(TAN). One-sided paired t, H0: mean >= 0 against H1: mean < 0 (direction from B1C, Y2).

Each contrast reports:
- n, mean and sd;
- the 95% two-sided t interval and the one-sided 95% bound;
- t, p and dz = mean/sd;
- the share of units below zero.

Values are printed in units of 1e-5.

## Secondary analysis: family S (fixed now; reported as secondary, never as primary)
Holm over three at familywise alpha = 0.05:
- **S1**: CTRIG - HQ-Fisher, two-sided (B1C's Y1 was not resolved).
- **S2**: OPT-CIRC - HQ-Fisher, one-sided, H1: mean > 0 (B1E's E2).
- **R3**: G_H(OPT-CIRC) - G_H(OPT-TAN). A one-sided paired t against a margin, H0: mean <= -delta against
  H1: mean > -delta, with **delta = 2.85 x 10^-5**. In statistical form this is a non-inferiority test.

R3 was primary in the draft. It is secondary here (lead's decision, option (c) of the review), so S grows from two
tests to three.

**R3's scale (disclosed; it goes with every report of R3).** On the original units at kappa = 0.01, the mean arm gains
(x 1e-5) are TAN 4.38, OPT-TAN 3.99, OPT-CIRC 3.39 and HQ-Fisher -1.31. So:
- delta = 2.85 x 10^-5 is fixed at half of B1's natural-step shortfall on the original units (HQ-Fisher - TAN,
  5.70 x 10^-5). It is not rescaled to B1R's own shortfall, which may be smaller or larger;
- delta is also about 71% of OPT-TAN's mean held-out gain;
- passing the margin therefore means only that OPT-CIRC keeps at least about 29% of OPT-TAN's held-out gain.

The margin was chosen knowing that the original units pass it (p = 0.0038; one-sided 95% lower bound
-1.94 x 10^-5). R3 is reported with its interval and with this scale. It is never worded as "non-inferior" or
"replicated" in a headline: the abstract, highlights, summary, section titles or conclusions. In the body, its
statistical form may be named only together with its interval and the scale sentence.

**R3 tests held-out gain, not reachability.** Reachability concerns the fit-set objective L_F (see "Descriptive
only"), and G_H also mixes in overfitting. R3 and its readings therefore speak only of the held-out gap between the best
optimised circuit step and the best linear tangent step found (OPT-CIRC - OPT-TAN on G_H; local optimiser).
- This gap is never called "the reachability gap", and no conclusion about what circuit policies can reach is drawn
  from G_H alone.
- Every R3 reading and both joint branches in which R1 and R3 are rejected report OPT-CIRC - OPT-TAN on L_F (mean and
  95% interval, descriptive) in the same sentence, together with delta's anchor (B1's original shortfall). The joint
  branch with R3 only gives the L_F difference; the branches with R1 only and with neither give neither, because the
  R3 reading reported beside them carries both.
- On the original units, that L_F difference was -7.98 [-9.34, -6.62] x 10^-5 (n = 36), about 2.8 delta, while the
  G_H difference was -0.60 [-2.21, +1.02] x 10^-5.

Re-runs (rules 1 and 4) write their logs under new names (precheck_2.log, launch_canary_2.log), so no earlier
log is overwritten. A launcher is stopped with Ctrl-C (SIGINT to its process group), and the worker pids are
confirmed gone before any relaunch.

Outside any family, also reported:
- R3's two-sided difference test (alpha 0.05), used by the reading rules below;
- R3's TOST at +-delta (descriptive).

## Descriptive only (no test is read)
- Every contrast at kappa = 0.001, and every arm's mean.
- **OPT-CIRC - OPT-TAN on the fit-set objective L_F**, the optimisers' own objective. Reachability concerns this
  objective, while G_H also mixes in overfitting: on the original units, OPT-TAN - TAN = -0.39 x 10^-5 on H, and
  OPT-CIRC - OPT-TAN = -7.98 [-9.34, -6.62] x 10^-5 on L_F. Reported in the same sentence as every R3 reading and
  as the joint branches listed under the R3 family above, and beside R3 together with OPT-TAN - TAN on L_F and each arm's mean L_F. No test is read on it.
- delta as a share of the fresh units' mean OPT-TAN gain (beside the 71% of the original units).
- These differences:
  - QTAY - HQ-Fisher, CQR - CTRIG, the CTRIG ladder and CTRIG#0 - TAN;
  - OPT-TAN - TAN, OPT-CIRC - TAN, TAN-c - TAN and HQ-plain - TAN.
- The first-crossing counts, the angle displacement, the optimisers' best-start counts and iterations, and the
  within-unit sd of the CTRIG draws.
- Side by side with the 36 original B1 units (stored values only), for R1's, R2's and R3's differences and the L_F
  difference:
  - the difference between the two studies' means (Welch, descriptive);
  - the 72-unit pooled mean.
  Pooling is never used for a claim.

## Exclusions and missing data (fixed now)
**Failed units.** A unit whose record is not 'complete' is listed with its failure class and excluded from every
contrast. Failures include:
- an episode failure;
- a certificate failure;
- a refused namespace;
- any exception.

No unit is replaced by another index. Units are rerun only under the failure rules below.

**Internal consistency** (expected never to trigger). A unit is excluded from every contrast if, at either kappa:
- extension E's recomputed HQ-Fisher or TAN has a different calibration status from extension C's; or
- their G_H differ by more than relative 1e-6.
Extension E's own reproduction figure skips an arm when either side lacks G_H. The status rule closes that gap.

**Arms that are not ok.** An arm whose status is not ok at kappa = 0.01 drops its unit from the contrasts that use that
arm, and the counts are reported. Not ok means a calibration failure, or no feasible end point for OPT-CIRC or
OPT-TAN. CTRIG uses the mean of its ok draws, and the number of units with fewer than 20 ok draws is reported.

**Incomplete replication.** If R1 or R2 has fewer than 30 units, the analysis output carries `incomplete: true` and
the statement "The replication is incomplete: ... (fewer than 30). The tests are unchanged." The manuscript reports
that statement with the results. R3's n is reported in every case.

**Which records are analysed.** The analysis refuses a record set in any of these cases:
- records run under another PLAN.md;
- records from more than one runner.py;
- records from a runner whose hash is in neither this plan nor PLAN_LOG.md;
- an analysis script whose hash is in neither.

It also refuses fewer than 36 records, unless run with --provisional, whose output is labelled as such.

## Failure rules (fixed now)
The pipeline is deterministic given a unit's index: its seeds, streams, draws and starts, with one torch thread per
process. A rerun from scratch therefore reproduces the same values and cannot select them. A failed record's arm
values do not exist, or are never read.

1. **Canary.** Fresh unit 100 runs first, alone (`runner.py launch --canary`). Only its status, certificate and
   timing are read (`runner.py status`); no arm value and no episode J is read. Then the other 35 are launched.
   The full launch refuses until the canary has a record. If the canary failed, the full launch refuses until the
   lead has done the following:
   - diagnosed it from its status, failure class, error and traceback only;
   - logged the classification in PLAN_LOG.md;
   - applied rule 2 or 3 (an environmental canary failure is rerun with `runner.py launch --canary
     --rerun-environmental`), or, for a unit-level failure by B1's rules (an infrastructure or invalid-row episode, a
     certificate failure that is not a code fault), relaunched with `--canary-failure-logged`. Unit 100 then stays
     failed and excluded.
2. **Systematic failure.** If two or more units fail with the same exception type, the launch stops: the launcher
   terminates its own workers, and a relaunch refuses.
   - The cause is diagnosed without reading any arm value. The record's status, failure class, error, traceback,
     certificate and timing may be read, as may the code.
   - A reproduction may run on B1 unit 0. A fault that appears only at a fresh index may be reproduced on the failing
     fresh unit with arms disabled (`run_unit(..., with_arms=False)`: episodes, rows and certificate only). Only status
     and error are read, outputs go to debug/, and they are never analysed.
   - The fix and the new runner hash go into PLAN_LOG.md as a disclosed deviation.
   - `runner.py supersede --deviation-logged` then moves every record and staged file to superseded/<UTC>/; nothing is
     deleted. Superseded records are never analysed, and only their statuses are reported.
   - A changed runner needs a new passing precheck of record (rule 4) before the launch accepts it.
   - Every unit is then rerun from scratch, canary first.
   - If the diagnosis finds unit-level failures by B1's rules and no code fault, the log says so, and the launch
     continues with `--failures-diagnosed`.
   - If it finds environmental failures (rule 3) and no code fault, the log says so, and the launch continues with
     `--failures-diagnosed --rerun-environmental`, which reruns each of them once from scratch.
   - `--failures-diagnosed` covers only the failed units that the diagnosis read. A later failure of the same type is
     undiagnosed and stops the launch again.
3. **Environmental failure.** A failure caused by memory exhaustion or by a failed write, wherever it was raised, is
   environmental (failure class 'environmental'). The unit may be rerun once from scratch
   (`runner.py launch --rerun-environmental`).
   - The runner classes a failure as environmental by its exception type (MemoryError, OSError and their subclasses)
     or by its recorded error text. The text is needed because torch reports these failures as RuntimeError:
     - a failed CPU allocation: "DefaultCPUAllocator: can't allocate memory";
     - a torch.save on a full disk: "unexpected pos ... vs ...", "iostream error" or "file write failed".
     Extensions C and E also pass on only the text: `save_rows` returns it, and the runner raises "rows not saved:
     ..."; E's `unit` records it, and the runner raises "extension E arms failed: MemoryError ...". The patterns are
     `ENVIRONMENTAL_NAMES` and `ENVIRONMENTAL_TEXT` in runner.py. The record's `environmental_match` names the pattern
     that applied.
   - **Fallback.** A failure that the runner did not class as environmental, but whose recorded error or traceback
     shows memory exhaustion or a failed write, is classified by the lead from its status, error and traceback only.
     The lead logs the classification in PLAN_LOG.md and moves the record by hand to failed_environmental/<UTC>/. The
     next launch then reruns the unit once from scratch as interrupted (`runner.py launch --canary` if it is the
     canary).
   - The failed record is kept in failed_environmental/<UTC>/ and reported.
   - A second environmental failure of the same unit is final.
   - A unit left without a record by a crash, a reboot or a killed worker (for example by the OOM killer) counts as
     interrupted. The next launch moves its staged files to interrupted/<UTC>/ and reruns it from scratch. If the
     same unit is interrupted twice, the lead diagnoses (rule 2's limits) before a third attempt, and logs it.
4. **Failed precheck of record.** No launch.
   - The cause is diagnosed (on B1 unit 0 only), and the fix is logged in PLAN_LOG.md with the new runner hash if the
     runner changed.
   - `runner.py precheck` is then rerun. It writes precheck_2.json (then precheck_3.json, ...) and never overwrites
     an earlier one.
   - The launch accepts only the latest precheck, and only if it passed.

**Cut-off.** The DQ calendar cut-off of 2026-10-31T00:00Z closes the B1 namespaces (`dq_runtime.require_open`). It
applies to every episode, including the certificate's P4 re-flight, which now follows the data directly. A unit whose
data and certificate are not complete by then fails, and no rerun is possible after it. The full launch therefore
starts no later than 2026-10-30T00:00Z, so that a stop under rule 2 or a rerun under rule 3 still fits. The code
refuses any launch with less than 8 h left. Arms run after the certificate and are not gated.

## Power (from the stored two-layer values at kappa = 0.01; disclosed, not part of the tests)
With n = 36, one-sided:
- **R1** (sd 7.42 x 1e-5), at Holm's smallest level 0.025 (and at 0.05):
  - power 0.99 (0.998) at B1's effect;
  - 0.92 (0.96) at 75% of it;
  - 0.61 (0.73) at 50% of it;
  - minimum detectable effect at 80% power: 3.6 (3.1) x 1e-5.
- **R2** (sd 4.24), at 0.025 (and at 0.05): above 0.999 at B1C's effect; 0.84 (0.91) at 50% of it.
- **R3** (sd 4.77; secondary), at family S's smallest level 0.05/3 (and at 0.05):
  - margin-test power 0.73 (0.87) if the true mean is B1E's -0.60;
  - 0.91 (0.97) if the true mean is 0;
  - 0.52 (0.71) if the sd is 6.0.

## Reading rules (fixed now)
"Rejected" means rejected by Holm in the contrast's family: R for R1 and R2, S for S1, S2 and R3. "Opposite" means
that the two-sided 95% t interval lies entirely above 0. `b1r_analysis.readings()` prints these readings.

**R1**
- **Rejected:** replicated. On fresh units, in a prespecified test, the circuit-realised natural step (HQ-Fisher)
  gains less than the same initial direction applied linearly (TAN).
- **Opposite:** this contradicts B1. The manuscript states that the prespecified replication contradicts B1, and it
  removes the path claim (B1's claim that the circuit-realised natural step falls short of the linear tangent step).
  It does not qualify it.
- **Otherwise:** not replicated at this sample size. The manuscript must do all of the following:
  - state that the B1 path shortfall was not replicated in a prespecified replication;
  - report B1 and B1R side by side;
  - qualify every path claim in the abstract, the summary and the conclusions.
  No pooling and no post-hoc analysis may replace R1.

**R2**
- **Rejected with R1:** replicated. The matched classical trigonometric chart also falls short of the linear tangent
  step on fresh units.
  - This is an existence claim: a matched classical chart also shows the shortfall, so the path effect is not
    specific to the circuit, for this matching.
  - It does not say that the chart and the circuit fall short by the same amount: S1 not rejected is not equivalence.
- **Rejected without R1:** shown for the matched chart only, conditional on R1. The matched classical chart falls short
  of the linear tangent step on fresh units, but the circuit's own shortfall did not replicate, or was contradicted
  (R1).
  - The generic reading is not confirmed: there is no replicated circuit shortfall for the chart result to
    generalise.
  - R2 is reported as a finding about the matched chart only. S1 not rejected is not equivalence.
- **Opposite:** this contradicts B1C. The manuscript states that the replication contradicts B1C, and it removes the
  matched-chart path claim (B1C's claim that a matched classical chart also falls short, i.e. the generic reading of
  the path effect). It does not qualify it.
- **Otherwise:** B1C's matched-chart shortfall stays a single post-hoc finding. The manuscript must not call the path
  effect generic. It says that a matched classical chart fell short in a post-hoc study and that this was not
  replicated.

**R3 (secondary; held-out gain only)**

Every R3 reading carries R3's scale sentence above. In the same sentence as the reading, it also gives delta's anchor
(fixed at half of B1's original shortfall, 5.70 x 10^-5, not rescaled to the fresh units' own) and OPT-CIRC - OPT-TAN
on L_F (descriptive).
- **Rejected, and the two-sided difference negative at 0.05:** "within the margin, yet worse". On held-out gain,
  OPT-CIRC's mean is shown to be less than delta below OPT-TAN's. Its two-sided difference is also negative, so the
  best optimised circuit step found gains less on H than the best linear tangent step found (local optimiser), by
  less than delta.
- **Rejected otherwise:** the margin is passed. On held-out gain, OPT-CIRC's mean is shown to be less than delta below
  OPT-TAN's. So the held-out gap between the best optimised circuit step and the best linear tangent step found is
  smaller than delta. The two-sided difference is reported as not resolved, or as positive at 0.05.
- These two readings end: "this is a held-out reading: reachability concerns L_F and is not tested".
- **Not rejected:** a held-out gap of delta or more between the best optimised circuit step and the best linear
  tangent step found is not excluded.
  - If the two-sided difference is negative at 0.05, this is reported as secondary evidence that the best optimised
    circuit step found gains less on H than the best linear tangent step found. It is not read as a reachability
    limit, because reachability concerns L_F.
  - If it is not resolved, how the shortfall splits between the natural step's path and what circuit policies can
    reach stays unresolved.
  - A two-sided difference that is positive at 0.05 is reported as computed.
- The word "shown" is used, never "replicated": B1E's R3 was a null two-sided result. The optimiser is local
  (SLSQP; five starts for OPT-CIRC, four for OPT-TAN).
- R3's G_H difference is never called "the reachability gap".

**R1 with the secondary R3 (a fixed, deterministic rule: `b1r_analysis.at_most_half`)**

delta is anchored to B1's original shortfall, not to B1R's. R1 can be rejected with a fresh shortfall well short of
5.70 x 10^-5: its minimum detectable effect at 80% power is 3.6 x 10^-5. A gap below delta can then be most of the
fresh shortfall. So when R1 and R3 are both rejected, the reading depends on B1R's own R1 mean (its point estimate; no
further test is read).

Both branches open in the same way. "The B1 shortfall replicates (R1). Secondary: on held-out gain, the best optimised
circuit step found comes within delta of the best linear tangent step found (local optimiser)". If R3's two-sided
difference is negative at 0.05, this is added: "but gains significantly less, so a held-out gap smaller than delta is
shown". delta's anchor and the L_F difference follow in the same sentence. Then:
- **Both, and R1's mean at or below -2 delta (-5.70 x 10^-5):** delta is at most half of the fresh shortfall's point
  estimate. So on held-out gain this gap accounts for less than half of the fresh shortfall. The shortfall lies
  mainly outside the gap between the best optimised steps found ("mainly, not wholly" when R3's two-sided difference
  is negative at 0.05).
  - The rest of the shortfall is HQ-Fisher - OPT-CIRC plus OPT-TAN - TAN on H. The reading does not apportion it
    further.
  - It ends: "this is a held-out reading: reachability concerns L_F and is not tested".
- **Both, and R1's mean above -2 delta:** delta is more than half of the fresh shortfall's point estimate. So how the
  fresh shortfall splits between the natural step's path and what circuit policies can reach is not settled.
- **Wording in both branches.** The readings never use the words "rather than", and use "mainly" only in the first
  branch. No reading places the shortfall in the natural step's path as opposed to what circuit policies can reach,
  because G_H does not test reachability.
- **R1 only:** the shortfall replicates; path against reachability is not settled.
- **R3 only:** the natural-step shortfall did not replicate. The secondary held-out margin test passed (the L_F
  difference is given in the same sentence), with no replicated shortfall for it to explain.
- **Neither:** neither is shown.

The analysis output records the rule's inputs (`joint_R1_R3_rule`: R1's mean, the limit -5.70 and whether R1's mean
is at or below it).

**Generic versus circuit-specific**
- **R1 and R2:** the shortfall is shown for the circuit and for a matched classical chart (an existence claim; S1 not
  rejected is not equivalence).
- **R1 only:** shown for the circuit only; the generic reading is not confirmed.
- **R2 only:** conditional, as above.
- **Neither:** neither is shown.

**S1 and S2 (secondary)**
- S1 rejected: the direction is stated (CTRIG gains more, or less, than HQ-Fisher).
- S1 not rejected: "not resolved", which is not evidence that the two gain the same.
- S2 rejected: OPT-CIRC gains more than HQ-Fisher, in the direction of B1E's E2.
- S2 not rejected: not resolved.

**In every case:**
- An incomplete replication carries its statement (see the exclusion rules).
- B1's original units remain the study of record for B1.
- B1R is reported beside B1, never merged with it, and its p values are never combined with B1's.
- Secondary results are never read as primary.

## Design choices informed by earlier results (disclosed)
- The contrasts, the one-sided directions of R1 and R2, the margin form of R3 and delta were all chosen knowing the
  results of B1, B1C and B1E on the original units. On those units R3 at this delta gives p = 0.0038 (one-sided 95%
  lower bound -1.94 x 1e-5). R3 is secondary for that reason, and because of its scale. The replication's evidence
  comes only from the fresh units.
- Also known before this plan: on the original units, OPT-CIRC - OPT-TAN on L_F is -7.98 [-9.34, -6.62] x 1e-5
  (n = 36), about 2.8 delta. The round-3 verifier computed it, and it was recomputed with `b1r_analysis` on the stored
  B1E records. By this plan's definition of reachability (L_F), the original units do not support placing the
  shortfall in the natural step's path rather than in what circuit policies can reach. R3 is therefore read on
  held-out gain only. The joint rule's limit (R1's mean at or below -2 delta) was fixed before any fresh unit existed.
- kappa = 0.01 is primary (B1's primary value). kappa = 0.05 is not used: extension X found its steps ill-posed, with
  scales up to 6e8.
- Only the two-layer circuit is run (B1's primary stratum).
- CTRIG-16, the mission evaluation, SPAN, RND-k and FIT-k are left out for cost. They are not part of any claim.
- No hyperparameter was selected or tuned for B1R. Every setting is B1's, B1C's or B1E's frozen value. No fresh
  unit's data was looked at before this plan.

## Limitations (stated with the results)
- **Shared code.** The replication uses the same instrument code as B1C and B1E (imported unchanged), and B1's data
  code. It tests new policies and new episodes, so it cannot detect an implementation error that B1C, B1E and B1R
  share.
- **Single scene.** The fresh units share B1's 8 frozen core cells. The replication covers new policies and new
  episodes, not new scenes, so the single-scene limitation stands.
- **R3.** R3 rests on a local optimiser and a margin chosen after the original results, with the scale stated above.
  It tests held-out gain only. It does not test reachability, which concerns L_F, and L_F is only described.
- **Shared approval.** The units draw under B1's namespace approval, as disclosed above.

## Run instructions (order of operations)
All commands run from /home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b1r with
PY=/home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python. Every process runs CPU only under `nice -n 10` and
`-B`.

Values do not depend on the worker count. Each worker runs one unit with one torch thread, and every unit is
deterministic given its index. The count is the operator's choice.

1. The lead verifies the script hashes below (`sha256sum runner.py b1r_analysis.py`). The lead then writes
   PLAN.sha256 (`sha256sum PLAN.md > PLAN.sha256`) and PLAN.utc, and starts PLAN_LOG.md.
2. Optional: `$PY -B -m pytest -q -p no:cacheprovider test_b1r.py` (no episode is run).
3. Precheck of record: `nice -n 10 $PY -B runner.py precheck > precheck.log 2>&1`. It is one process and takes 30-60
   min. Only pass flags, statuses, reproduction differences and timings are read. If it fails, rule 4 applies.
4. Canary: `nice -n 10 $PY -B runner.py launch --canary > launch_canary.log 2>&1`. It is one process and takes 25-50
   min. Then `$PY -B runner.py status`: only status, certificate and timing are read (rule 1).
5. The other 35: `nice -n 10 $PY -B runner.py launch --workers 5 > launch.log 2>&1`.
   - Use 4-6 workers while other campaigns run, up to 10 on an idle machine.
   - Each unit takes about 25-50 min, depending on load. Measured on 2026-10-25 for a single process: 1.1 s per
     episode and 483 s for extension E's arms at load 3-8; 2.0-2.5 s and 750-780 s at load 12-18.
   - Expect about 3-7 h wall time for the 35 at 4-6 workers.
   - Start no later than 2026-10-30T00:00Z.
   - A relaunch skips units with a record.
6. **Watchdog.** Run `$PY -B runner.py status` every hour or so. It lists the units started without a record, with
   their age and whether their worker pid is alive.
   - Investigate any unit without a record for more than 2x the canary's wall time (2 h if in doubt), or whose worker
     pid is gone. A SIGKILLed worker hangs `multiprocessing.Pool`.
   - In that case, stop the launcher (the operator's own process) and relaunch. The unit counts as interrupted under
     rule 3.
7. **Failures.** `runner.py status` shows each failed unit's failure class and error. For an environmental failure it
   also shows the pattern that matched (`environmental_match`). Reruns follow failure rules 1-3 only:
   - `launch --rerun-environmental` for an environmental failure;
   - with `--canary` added for the canary;
   - with `--failures-diagnosed` added after a rule-2 stop that PLAN_LOG.md diagnoses as environmental.
8. When all 36 records exist: `nice -n 10 $PY -B b1r_analysis.py > analysis_stdout.txt 2>&1`. The analysis runs once,
   with the hash below unchanged.

## Scripts executed, by SHA-256 (computed after the last edit of each file)
| file | role | sha256 |
|---|---|---|
| runner.py | seeds, precheck, launch, status, supersede | 90ac63deff4b6445c5dda0d02eb09627c0f4487337bfecc298eab4021032e62e |
| b1r_analysis.py | the analysis | ef5c05152cfe078b0d11c0a0b321ff3aaca1433b9f5311c47b86c08d7a91dc97 |
| test_b1r.py | unit tests (not run by the runs or the analysis; for the record) | 64910192246cb51fd9505f7b0964e2f511a421852f7920c6c268a0742dbce693 |
| ../idea_dq_posthoc_b1c/runner.py | extension C, imported unchanged | a712ee54f5568ab9e9cb4c34e719f9865bce9d3feea8ddb109d4b26e0f0d7d14 |
| ../idea_dq_posthoc_b1e/runner.py | extension E, imported unchanged | 909d6f54346d81f85ddf6c3cd1be1e6f0121347178557fdbed8bab1d6a12e446 |
| ../idea_dq_b1_tree_a66f8979 | B1 code tree of record (`p4_train.live_copy_tree()`) | a66f89793664b63bdf6e1ece843667972d30d142dc0fd5b8c2f684b7fbeecb56 |

runner.py refuses to import either reused runner if its hash differs. The launch refuses unless runner.py's hash is
the one above, or one logged in PLAN_LOG.md as a deviation. The analysis refuses unless its own hash is the one above,
or one so logged.

**Development history (m11).** The runner of development quick-check run 1 (sha 5b7fc101...) was not archived, so the
claim that only the quick criterion changed between runs 1 and 2 cannot be checked against code. Both runs' JSON are
kept. From this plan on, every precheck archives the runner it ran (`<precheck>/runner_snapshot.py.txt`), and every
launch archives it too (`launch_snapshots/<UTC>_runner.py.txt`).

## Reporting commitment
Whatever the outcome, the manuscript (Section 4.4, Appendix A) reports the following as a prespecified replication,
with this plan's SHA-256:
- families R and S;
- the exclusions and the failure log;
- the descriptive values;
- the side-by-side with the original units.

The scripts and records go to the repository.
