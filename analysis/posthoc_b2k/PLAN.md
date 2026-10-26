# Post-hoc check E3: early training at matched per-update KL (B2)

Status: final plan, written 2026-10-25 from PLAN_DRAFT.md (kept unchanged for the record, sha256
ce68473d5e5f0ef289fcdb6c9f48a17439a8bbe55d46c09abca1099d4d7f4eb9) with the fixes decided by the lead after the
independent review (REVIEW.md) and the corrections of the independent verifier (round 2: the "not resolved after
Holm" reading applies to K-H1 as well). The lead hashes this file (PLAN.sha256, PLAN.utc) before `runner.py launch`;
no main run exists when it is written. The calibration pilot (Stage 1) has already run under the rule that PLAN_DRAFT.md fixed
before it started (snapshot `pilot/PLAN_DRAFT_at_pilot_launch_2026-10-25T075118Z.md`); its outcome and the two selected
learning rates are recorded below and are not changed by this plan.

Written after the B2 results of record and the post-hoc extensions (matched learning rates `idea_dq_posthoc_lr`, HQ at
1e-2 `idea_dq_posthoc_hqlr`, C-matched at 1e-2 `idea_dq_posthoc_cmlr`, generic angles `idea_dq_posthoc_b2g`, frozen
circuit angles `idea_dq_posthoc_b2f`) were known. Not part of any frozen family; every result is labelled post hoc.

## Question
B2's primary contrasts compared arms whose realised update sizes differed. B2's logged per-update KL (statistic below)
averaged 0.00782 for HQ (lr 3e-3), 0.00234 for C-matched (3e-3) and 0.00993 for the twin (1e-3). The matched-
learning-rate check showed that a common nominal learning rate does not equalise it (HQ 0.0031 / 0.0078, twin
0.0099 / 0.028 at 1e-3 / 3e-3). A reviewer can therefore read B-H2 (HQ − twin, −0.0078) as the twin taking larger
updates, and B-H1 (HQ − C-matched, −0.0087) as an update-size effect as well. This check re-trains one arm of each
contrast at a constant learning rate calibrated on development seeds so that its mean per-update KL equals the stored
comparator's, and repeats the contrast:
- K-H2: AUC(HQ, stored B2, lr 3e-3) − AUC(twin, new, KL matched to stored B2 HQ).
- K-H1: AUC(HQ, new, KL matched to stored CMLR C-matched) − AUC(C-matched, stored CMLR, lr 1e-2).

Interpretation note on K-H2 (added by the lead after the B2F result of 2026-10-25, before any E3 main run). HQ with
frozen circuit angles is the same model with the same gradients as the twin with its coefficients (surrogate layer)
frozen (B2F identity check on blocks 0, 7 and 21: the same 16 trainable tensors, max |Δlogit| ≤ 4.0e-15, max |Δgrad|
≤ 2.1e-15), and B2F found HQ with frozen
angles equivalent to HQ trained (F1 = +0.00058, 90% interval [−0.00082, +0.00197] inside B2's ±0.0056; B2F PLAN
a5e573b3…, PLAN_LOG 2026-10-25T11:44Z). K-H2 therefore compares, approximately, the fixed-feature model (HQ's
features frozen, i.e. the twin with frozen coefficients) with the same model whose coefficients are also trained, at
HQ's update size (the twin's mean per-update KL matched to HQ's). "Approximately" because the B2F equivalence is an
equivalence within a margin, not an identity, and because what is matched is the per-update policy KL, not the
per-layer step: the classical layers' learning rates differ (HQ 3e-3, twin 7.79e-04). One descriptive contrast (below, no test) makes this comparison
directly against the B2F runs.

## The matched statistic (fixed before the pilot)
For one run: the arithmetic mean, over the run's update records (`metrics.jsonl`, kind `update`), of
`metrics.approx_kl`. That value is `smdp_ppo.ppo_update`'s mean over the update's optimiser steps (12 in B2) of the
minibatch estimate mean((r − 1) − log r), computed before each step against the rollout log-probabilities (the first
step's value is about 0). For an arm: the arithmetic mean of the run statistic over its blocks. It is a within-update
average, not the KL of the policy after the update, not a per-state KL and not a parameter distance; matching its mean
does not match its median or its distribution over updates (both are reported).

Targets (stored records, blocks 0-35 in ascending order; recomputed by the precheck, exact):
- T_H2 = 0.007823255686871039: stored B2 HQ (quantum, lr 3e-3), the comparator of K-H2.
- T_H1 = 0.00419994415996206: stored CMLR C-matched (bottleneck, lr 1e-2), the comparator of K-H1.

## Stage 1: calibration pilot (development only; rule fixed before any pilot run; completed)
- Study object: `dq_b2.STUDY_SMOKE` (salt (20260923, 3), training namespace `dq_development`), not the B2 salt, not
  `dq_e2_training`, not `dq_e2_evaluation`. Development blocks 90-98, each used once per arm (disjoint per candidate).
- No evaluation episode is run (`evaluation` = None): the pilot produces no evaluation J, no learning curve and no
  AUC. The selection reads only the update records' `approx_kl` (`runner.py pilot-select`); the trainer still writes
  training-episode records, which nothing reads.
- Everything else as in the main runs: B2's PPO settings, entropy 0, constant learning rate, 30,000 nominal decisions,
  `p4.train_arm` called directly (never `dq_b2.run_one/launch/smoke`).
- Budget: the full 30,000 decisions. In the stored runs the running mean over a shorter budget differs from the
  full-run mean by an arm- and learning-rate-dependent factor (ratio of means at 15,000 decisions: HQ 3e-3 0.90,
  HQ 1e-3 1.03, twin 1e-3 1.03, HQ 1e-2 1.39, C-matched 1e-2 1.20), so no shorter budget estimates the statistic.
- Candidates (log-spaced around the log-log interpolation of the stored points of the same arm):
  - twin (surrogate625): 5e-4 (blocks 90-92), 7e-4 (93-95), 1e-3 (96-98); stored-point prediction 7.76e-4;
  - HQ (quantum): 1e-3 (blocks 90-92), 1.4e-3 (93-95), 2e-3 (96-98); stored-point prediction 1.43e-3.
  18 pilot runs. Each candidate uses its own blocks: the run statistic varies between blocks with a coefficient of
  variation of 0.33-0.39 (stored HQ and twin) and is only moderately correlated across learning rates within a block
  (r ≈ 0.53-0.63), so disjoint blocks give the level estimate more independent information per run. Resampling the
  stored per-block values (with the 36-block sampling error of the main runs added) gave P(achieved main-run mean
  within ±25% of target) ≈ 0.92-0.94 for 9 disjoint pilot runs per arm.
- Disclosure (review M4). The fixed slopes s and the candidate centres (stored-point predictions) come from stored
  *training-update* KL (`approx_kl` of update records) of the runs on the B2 blocks 0-35, whose training seeds are the
  main runs' own (`dq_e2_training`); they do not come from development data. No AUC, J or evaluation datum entered
  them, the targets, the candidates or the selection; this is the intent of the "development seeds/suites only" rule,
  which the pilot itself (development blocks, no evaluation) follows literally.
- Selection rule (per arm; `runner.select`):
  1. K̄(lr) = arithmetic mean of the run statistic over the candidate's completed pilot runs.
  2. With x = ln lr and y = ln K̄ over the arm's candidates, and x̄, ȳ their means:
     lr* = exp(x̄ + (ln T − ȳ) / s), with s fixed from the stored points of the same arm between lr 1e-3 and 3e-3:
     s = 0.9385004171529107 (twin), 0.8421691057010121 (HQ). The pilot fixes only the level.
  3. lr* is rounded to 3 significant digits (`format(lr, '.2e')`); that value is the main runs' learning rate.
  4. Range guard: if lr* lies outside [min candidate / 1.5, max candidate × 1.5], stage 1b runs for that arm only
     (candidates round3(lr*) × {1/1.4, 1, 1.4} on the same blocks, rule 1-3 on the stage-1b runs only, no further
     iteration).
  5. A pilot run that fails is dropped from its candidate's mean; fewer than two candidates of an arm with a completed
     run means no learning rate and no main set for that arm.
  6. Reported, not used: K̄ per candidate and per run, the OLS slope and the lr it would select, the predicted KL at
     the selected lr, and the pilot's K̄ at the stored learning rates versus the stored B2-block values.

Pilot outcome (2026-10-25T07:51Z-11:40Z; runner.py 96bb8b08…, archived as
`precheck_prev_20261025T120553Z/runner.py.96bb8b08`). 18 of 18 runs completed, no failure, 56-57 updates each, no
evaluation episode. `runner.py pilot-select` read only the update records' approx_kl:

| arm | candidate: K̄ (3 blocks) | lr* (fixed slope) | range guard | selected |
|---|---|---|---|---|
| twin | 5e-4: 0.005001; 7e-4: 0.008656; 1e-3: 0.008353 | 7.786e-4 | [3.3e-4, 1.5e-3], inside | **7.79e-04** |
| HQ | 1e-3: 0.003991; 1.4e-3: 0.005183; 2e-3: 0.005814 | 1.163e-3 | [6.7e-4, 3e-3], inside | **1.16e-03** |

No stage 1b. Descriptive, not used: pilot OLS slopes 0.73 (twin) and 0.54 (HQ), which would select 8.01e-4 and
1.05e-3; transfer check at the stored learning rate 1e-3: pilot K̄ / stored B2-block mean = 0.835 (twin: 0.008353 vs
0.009926) and 1.287 (HQ: 0.003991 vs 0.003101). The independent review recomputed the selection from the pilot
metrics (7.786e-4 → 7.79e-04; 1.1635e-3 → 1.16e-03). The selected rates are final (`selection.json`, sha256
f819d71257824c0c0262c2a0350577772c7b65948fd5b387487e5694954f163f); they are not adjusted for anything learned since.

## Stage 2: main runs (72, all new, none retried; a run interrupted by a crash is restarted from the beginning)
B2 study object unchanged (`dq_b2.STUDY_B2`: salt, initialisation, training schedule, action and minibatch streams;
the 80 common-random-number evaluation episodes at the 11 nominal points 0, 3,000, …, 30,000; entropy 0; 30,000
nominal decisions; blocks 0-35). The B2 code tree (ffe9dfef…) is imported unchanged; the runner sets only Adam's
constant learning rate (no decay, no freeze, no KL controller) and adds the candidate name to the study's candidate
table in its own process.
- T2 (K-H2 arm): twin (`surrogate625`, the exact transplant of the block's untrained HQ via `dq_b2.dq_build_model`,
  static parity P1, P2, P5-P7 certified at build) at **lr*_twin = 7.79e-04** (selection.json). Before training, each
  run re-flies the block's first training episode with HQ and twin (`dq_b2.twin_trajectory_parity`, P4, |Δ log π| ≤
  1e-9, fail closed: a parity error becomes a failure record before training) and records it with the static record,
  as in B2.
- T1 (K-H1 arm): HQ (`quantum`) at **lr*_HQ = 1.16e-03** (selection.json).
Each new run is paired with the stored run of its comparator on the same block (same schedule, same evaluation
episodes; for T2 also the same initial policy). Task order: blocks ascending, HQ first within a block.

Launch (review M12). `runner.py launch --workers 8` at most while other campaigns share the machine (16 threads); a
larger worker count only on a machine otherwise idle. Expected wall time about 4-9 h at 8 workers, depending on load
(about 65 worker-hours under the heavy load measured in the pilot; the final precheck ran its rollouts about twice as
fast under the lighter load of 12:05Z).

## Precheck (passed before this plan was written; `runner.py precheck`, 2 processes)
Final precheck 2026-10-25T12:05-12:08Z with the final runner.py (sha256 below), after the runner was edited for M7/M8;
the earlier precheck (runner 96bb8b08…, 07:47-07:51Z) is kept in `precheck_prev_20261025T120553Z/`. Both passed with
identical check results.
(a) HQ, block 0, at B2's stored 3e-3, through the launch code path (`run_one`), 3,000 decisions, grid [0, 3,000]:
    evaluation (2), update (6) and episode (77) events and the arm_start model/identity/seeds identical to stored B2
    `quantum__b0`, and the initial checkpoint identical tensor by tensor (17 tensors).
(b) twin, block 0, at B2's stored 1e-3, same path with P4 parity: evaluation (2), episode (80) events, arm_start and
    the initial checkpoint (18 tensors) identical; update metrics identical (maximum relative difference 0.0); P4
    1.3322676295501878e-15, no static failure; the parity and static record equal the stored B2 record exactly.
(c) The 72 main configurations differ from `dq_b2.config_for(arm, block, STUDY_B2)` only in ppo.learning_rate,
    candidate, note and protocol_freeze_sha256, and pass E1's validator (`p4.validate_config`). The precheck checked
    them at the pilot's middle candidates; review M6: `validate_main_configs.py` re-validated all 72 at the selected
    7.79e-04 / 1.16e-03 (output `main_configs_validation_placeholder.json`: 72 of 72 passed, same four differing keys;
    plan hash placeholder, since this plan was not yet hashed).
(d) Pilot configurations: study dq_b2_smoke, evaluation None, budget 30,000, B2 PPO settings except the learning
    rate; every seed blocks 90-98 can draw lies in dq_development and outside dq_e2_training / dq_e2_evaluation.
    Declared (review M5): the pilot configurations bypass `p4.validate_config`; budget 30,000, grid [0] and blocks
    90-98 lie outside STUDY_SMOKE's frozen role (6,000 decisions, grid [0, 3,000, 6,000], block 90). This is harmless
    because `p4.train_arm` is called directly and the checks in (d) replace the validator for the pilot.
(e) T_H1, T_H2 and (f) the two slopes recomputed from the stored records equal the values above.
(g) `p4.build_model` is `dq_b2.dq_build_model` in the runner's process; (h) the live idea_dq tree is ffe9dfef….
The launch refuses without PLAN.md matching PLAN.sha256, without a ready selection.json whose `lr_str` values appear
in PLAN.md, if runner.py differs from the one that passed the precheck, or if PLAN.md does not record the sha256 of
runner.py and of b2k_analysis.py (review B3).

## Outcome and analysis (fixed now; `b2k_analysis.py`, sha256 below)
Outcome: B2's AUC (sample read-out, trapezoidal learning-curve area over the nominal grid / 30,000), recomputed from
each new run's raw evaluation events (see Integrity); stored comparators from their summaries.
- K-H2 = AUC(HQ, 3e-3; stored B2 `quantum__b`) − AUC(twin, 7.79e-04; new T2), paired over the 36 blocks.
- K-H1 = AUC(HQ, 1.16e-03; new T1) − AUC(C-matched, 1e-2; stored CMLR `C1_bottleneck__b`), paired over the 36 blocks.
For each: mean, 95% paired t interval, two-sided paired t p-value, dz, share of blocks below zero. Holm over
{K-H1, K-H2} at familywise 0.05. Secondary, outside the Holm family: TOST at B2's margin ±0.0056 (90% paired t
interval inside ±0.0056 → "within B2's equivalence margin"). Precision (stored paired SDs of B-H2, B-H1 and Z1):
about ±0.0045 (K-H2) and ±0.0052-0.0057 (K-H1) for the 95% half-width.

Failures, dropped blocks and degenerate cases (review B1; B2's frozen rules, `dq_b2_analysis.py`, copied):
- A run whose record status is `algorithmic_failure` (from the trainer) or `failure` (from the runner, e.g. a P4
  parity refusal) is an outcome: it is reported as failed and its block is dropped from every contrast that needs it
  (counted and listed).
- More than 3 of 36 blocks dropped (or fewer than 3 paired blocks): the contrast is NOT ESTIMABLE (not "not
  rejected"); it is not tested.
- A not-estimable contrast is left out of the Holm ordering, but the Holm denominator stays at the family size 2.
- Degenerate case (sd = 0): mean 0 → t = 0, p = 1; a non-zero constant effect → t = ±inf, p = 0.
- The TOST of a not-estimable contrast is reported as "not estimable".

Integrity (review B2; any failure is an IntegrityError: the analysis refuses and writes nothing; it is never converted
into a dropped block):
- FROZEN: PLAN.md matches PLAN.sha256 and records the sha256 of b2k_analysis.py and runner.py; precheck.json passed
  with that runner.py; selection.json is ready and its lr strings are in PLAN.md; the live idea_dq tree is ffe9dfef….
- IDENTITY: records/ holds only `T1_quantum__b<0-35>.json` / `T2_surrogate625__b<0-35>.json`; each record names its
  set, arm, block and kind `main`; its learning_rate equals selection.json's; plan_sha256 = PLAN.sha256's content;
  runner_sha256 = the precheck's; code_tree_sha256 = ffe9dfef…; its config equals
  `runner.config_main(arm, block, selected lr, PLAN hash)` (canonical JSON).
- RAW EVENTS (completed runs): exactly one arm_start (first) and one arm_end (last); update indices 1..U contiguous
  without duplicates, U = the arm_end's and the record's update count (duplicated update lines → refuse); each GRID
  point evaluated exactly once, in order; finite J; AUC recomputed with `p4_train.learning_curve_auc` equal to the
  arm_end's and the record's stored value; status, decisions, episodes and updates agree; one final checkpoint; the
  record's KL statistic equals the one recomputed from the verified update events.
- OUTCOME: a failure record never coexists with a completed arm_end or a final checkpoint and carries its error text.
- TWIN (B2's rule): a completed twin run counts only with a failure-free static certificate and P4 parity ≤ 1e-9; a
  completed twin run without them is a contradiction (the runner fails closed before training), so the analysis
  refuses.
- A record that does not parse counts as no record (see Crashes); the final read-out waits for 72 parsable records.

Achieved-KL check (per new arm): A = arithmetic mean over its completed runs of the run statistic; matched iff
0.75 T ≤ A ≤ 1.25 T. Reported with R = A / T, the 95% t interval of A over blocks, the median of the run statistic and
the median and 90th percentile of the pooled per-update values, beside the same for the stored comparator.
Expectation (review M10, stated before the runs and not acted on): on the B2 blocks the stored-point interpolation
predicts R ≈ 0.84 for HQ (0.0031 × 1.16^0.842 ≈ 0.0035 vs 0.0042; the pilot-based rule predicts 0.0042) and R ≈ 1.00
for the twin; HQ is the arm closer to the lower edge (a rough 5-10% chance of R < 0.75). The selected rates are not
corrected for this: that would be selecting on B2-block data. A miss is reported as it is.

Reading rules (fixed now). Direction of a contrast, the same for K-H1 and K-H2:
- "not estimable" (rules above);
- "below zero" / "above zero" = Holm-rejected with negative / positive mean;
- not rejected and the unadjusted 95% interval contains 0 → "not resolved";
- not rejected although the unadjusted 95% interval excludes 0 → "not resolved after Holm", with no update-size
  statement and no TOST add-on. This is possible when the contrast is tested at Holm's 0.025 (it has the smaller
  p-value) and not rejected, or when Holm stops at the other, unrejected contrast and this one is not tested.
For K-H2 additionally (review M2), with b = B-H2 of record = −0.007772148908780312 (stored B2, 36 blocks; 95% interval
[−0.0122, −0.0033]) used as a fixed reference value (its own sampling error is not propagated; the paired measure D
below shows the change directly):
- Holm-rejected below zero with the whole 95% interval above b → "partial reduction" (instead of "below zero");
- not rejected and the 95% interval contains both 0 and b → "inconclusive" (instead of "not resolved");
- not rejected, the interval contains 0 and lies above b → "not resolved".
K-H2's 95% interval is always reported beside b.

If the arm's KL is matched:
- K-H2 below zero: at matched mean per-update KL HQ still learned less than its exact twin, and the interval does not
  lie above b; B-H2 is not explained by the twin's larger updates as measured by this statistic.
- K-H2 partial reduction: at matched mean per-update KL HQ still learned less than its exact twin, and the whole 95%
  interval of K-H2 lies above B-H2 of record (b; b's own sampling error is not propagated); the twin's larger updates account for part of B-H2, not all of it, as measured by this statistic.
- K-H2 inconclusive: the check neither removes nor confirms B-H2 (not evidence of equality).
- K-H2 not resolved: at matched per-update KL the HQ − twin difference is not resolved and its interval excludes b;
  the B-H2 difference is consistent with an update-size account (not evidence of equality). If the secondary TOST
  passes, the difference at matched KL is also reported as within B2's ±0.0056 margin.
- K-H2 above zero: at matched per-update KL HQ learned more than its twin; B-H2 reverses and is reported as dependent
  on update size.
- K-H1 below zero: at matched mean per-update KL HQ learned less than C-matched; B-H1 is not explained by update size
  as measured by this statistic.
- K-H1 not resolved: at matched per-update KL the HQ − C-matched difference is not resolved; B-H1 is consistent with
  a dependence on update size (not evidence of equality); TOST as for K-H2.
- K-H1 above zero: at matched per-update KL HQ learned more than C-matched; B-H1 is reported as dependent on update
  size.
- K-H1 or K-H2 not resolved after Holm: reported as not resolved after Holm, with its unadjusted 95% interval; no
  update-size statement and no TOST add-on (the TOST result itself is still reported as a secondary result).
- Not estimable: reported as not estimable, with the dropped blocks; no statement.

If an arm's KL is not matched (R outside [0.75, 1.25]), its contrast is still computed, tested and reported, labelled
"KL not matched (R = …)", and only these one-sided statements are made ("below zero" here means Holm-rejected with a
negative mean, which includes "partial reduction"):
- K-H2 with R < 0.75 (twin's updates smaller than HQ's) and K-H2 below zero: HQ learned less than a twin whose updates
  were smaller; otherwise no update-size statement.
- K-H1 with R > 1.25 (HQ's updates larger than C-matched's) and K-H1 below zero: HQ learned less although its updates
  were larger; otherwise no update-size statement.

Premise of the one-sided statements (review M3, stated here so that they answer B-H1's own alternative "HQ's larger
updates hurt it"): within an arm, AUC does not decrease as updates get larger over the range concerned. Stored
evidence (differences of the paired means of the stored-run table, 36 blocks, descriptive): HQ 3e-3 − 1e-3 = +0.0055
and 1e-2 − 3e-3 = +0.0032; twin 3e-3 − 1e-3 = +0.0023; C-matched 1e-2 − 3e-3 = +0.0036. The evidence is in terms of
learning rate: between 1e-3 and 3e-3 a larger rate also meant a larger KL for HQ (0.0031 → 0.0078) and the twin
(0.0099 → 0.0278), but HQ's KL at 1e-2 (0.0070) is below its KL at 3e-3 while its AUC is higher. The new twin rate
(7.79e-04) lies slightly below the twin's lowest stored rate (1e-3), so there the premise is an extrapolation. The
premise is not re-tested by E3; the descriptive D and HQ_new − HQ_1e-3 are reported beside any one-sided statement, so
a reader can see whether E3's own runs contradict it.

K-H1 and the Holm family (review M9). K-H1 is largely predicted by stored data: the stored pairing HQ 1e-3 −
C-matched 1e-2 gives −0.0178 [−0.0230, −0.0125] at a KL ratio of 0.74, and HQ 3e-3 − C-matched 1e-2 gives −0.0122
[−0.0182, −0.0063]; the new HQ rate (1.16e-03) lies between them. K-H1 is therefore mostly a confirmation, and its
likely rejection with the smaller p-value means that Holm then tests K-H2 at 0.05 (if K-H2 had the smaller p-value,
it would be tested first at 0.025; if K-H1 had the smaller p-value and were not rejected at 0.025, Holm would stop
and K-H2 would not be tested). This is legitimate under Holm and the paper says so. The K-H1
comparator is C-matched at 1e-2 (KL 0.0042) because it keeps every arm at a tested learning rate: C-matched never
reached HQ's 0.0078 at any tested rate (0.0023 at 3e-3, 0.0042 at 1e-2); log-log extrapolation of its two stored
points puts the needed rate at about 3e-2 or more (3.6e-2), which is untested. K-H1 therefore lowers HQ's update size
to C-matched's instead of raising C-matched's.

No time cut-off; the read-out waits for all 72 records (`--provisional` only for monitoring, never reported).

Descriptive (no tests, no p-values; review M1):
- AUC, J at the first and final checkpoints, per arm.
- K-H2's 95% interval beside B-H2 of record (above).
- D = AUC(twin new, 7.79e-04) − AUC(twin B2, 1e-3), paired, same initial policy and streams, only the learning rate
  differs: the direct descriptive measure of how much of B-H2 the twin's step size explains. Per block D = B-H2 − K-H2,
  so a negative D is the part of B-H2 that the twin's larger step accounts for; the point share mean(D) / mean(B-H2)
  is reported without an interval.
- AUC(twin new, KL-matched) − AUC(HQ with frozen circuit angles; B2F runs `idea_dq_posthoc_b2f/runs/F_quantum__b`,
  same blocks), paired: the twin with trained coefficients against the same model with frozen coefficients (see the
  interpretation note). The classical layers' learning rates differ (7.79e-04 vs 3e-3), so it is not an exact
  decomposition. B2F runs are read only through their metrics.jsonl summary line, as `b2f_analysis.py` reads them.
- Within-arm learning-rate effects AUC(HQ new) − AUC(HQ B2, 3e-3) and − AUC(HQ, 1e-3; idea_dq_posthoc_lr L2).
- B-H1 and B-H2 of record on the same blocks; the paired per-block KL ratio of each contrast; P4 and static parity
  summary of the twin runs.

## Stored-run descriptive table (no new runs; `b2k_analysis.py --stored-table`, `stored_kl_table.json`)
From stored runs only, labelled descriptive post hoc, with no p-values and no decisions: AUC(HQ) − AUC(C-matched)
at every stored learning-rate pairing (HQ 1e-3 L2, 3e-3 B2, 1e-2 H1 × C-matched 3e-3 B2, 1e-2 C1), and
AUC(HQ) − AUC(twin) at every stored pairing (HQ 1e-3, 3e-3, 1e-2 × twin 1e-3 B2, 3e-3 L1): n, mean, 95% paired t
interval, share of blocks below zero, dz, each arm's mean per-update KL and the KL ratio. The generic-angle HQ runs
(b2g) are excluded (another initialisation). The final b2k_analysis.py reproduces the stored table bitwise.

## Crashes, failures and relaunch (reviews M7, M8)
- Completed records are kept. A run with a parsable record of any status is never run again; a failure stays a
  failure.
- Records are written atomically (temporary file in records/, fsync, os.replace). A record that does not parse counts
  as no record: the record, the run directory and any stale temporary record are moved unchanged to
  `interrupted/<UTC>/` and the run restarts from the beginning. A run directory without a record (killed by a crash) is
  moved the same way. `runner.py launch` applies this rule without looking at any partial result and logs every move
  in launches.jsonl.
- Single instance: precheck, pilot and launch hold an exclusive flock on `runner.lock` for their whole life; the
  fork-pool workers inherit it, so while any worker of an earlier launch is alive (for example after only the launcher
  was killed) a new launch is refused.
- Relaunch procedure: stop the launcher's whole process group (`kill -- -<PGID>`), confirm that no `runner.py`
  process has this folder as its working directory, then start `runner.py launch` again (the lock refuses it
  otherwise).

## Declared limits (review M11)
- No post-update full-batch KL is measured (no stored comparator has it); the matched statistic is the within-update
  mean of the pre-step minibatch KL.
- Matching the mean of the statistic does not match its distribution over updates (C-matched's per-update KL is
  strongly skewed: median 0.0010 vs mean 0.0023 at 3e-3).
- K-H1 and K-H2 reuse stored comparator runs, so they are correlated with B-H1, B-H2 and Z1 and are not independent
  replications; they add 2 p-values (plus 2 TOST results) to the paper.

## Reporting commitment
Whatever the outcome, K-H1, K-H2, their Holm decisions and estimability, the TOST results, K-H2 beside B-H2, the
achieved KLs with R and the matched flags, the pilot's KL readings and selected learning rates, the descriptive
measures above and the stored-run table are reported in Section 4.7 next to the matched-learning-rate checks
(Table 11), with this plan's SHA-256. Scripts and records go to the repository.

## Code and data pinned by this plan (review B3; sha256 computed after the last edit of each file)
| File | Role | sha256 |
|---|---|---|
| `runner.py` | precheck, pilot, pilot-select, launch (the runs execute it; the analysis imports it) | 210923c4e93c9fa6f0cb67e74ac0b0f28f83246a7a4f351040c6f1b0c2acc3f9 |
| `b2k_analysis.py` | the analysis, stored table, selftest | 98fe023328dfc909ee3e9b44e9b5bf2122373873c45986b519564c3dbc0e13af |
| `validate_main_configs.py` | review M6 re-validation of the 72 main configurations | 28104f64a8a31d8de20cffb5148f6b8cfd120e2625ed16356573486554fac7eb |
| `test_b2k.py` | unit tests (no run) | ae6299aa60ec79f8f71906bb7a1e31526a97fe96f94b67bbc1b36e41bf80810c |
| `selection.json` | selected learning rates (7.79e-04 twin, 1.16e-03 HQ) | f819d71257824c0c0262c2a0350577772c7b65948fd5b387487e5694954f163f |
| `precheck.json` | final precheck (passed, runner.py above) | 641fb8fb9012b72f09f34e4558982e46e2c1503b9873ec8ce2dc4008f21f07d3 |
| `main_configs_validation_placeholder.json` | M6 output | 1278516fca7899a2a7da3f4d27ad81d9a9aeab496c4d8028c1dbe5724e975a55 |

Runner that produced the pilot and selection.json (archived, not executed again):
`precheck_prev_20261025T120553Z/runner.py.96bb8b08`, sha256
96bb8b088c1221605d7f9400e407a93a896adcc77d8d7400dfeb5c8434d31e68. The edits since then (lock, atomic records,
resume rule, launch guard, default workers) do not touch the configuration, seeding or training path; the final
precheck reproduced stored B2 block 0 bitwise again.
