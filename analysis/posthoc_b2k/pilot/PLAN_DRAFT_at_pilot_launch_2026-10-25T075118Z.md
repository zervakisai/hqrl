# Post-hoc check E3: early training at matched per-update KL (B2) — DRAFT

Status: draft for the lead's review. Not hashed. After review and after the calibration pilot, the lead writes PLAN.md
from this text, inserts the two selected learning rates exactly as spelled in `selection.json` (`lr_str`), hashes it
(PLAN.sha256, PLAN.utc), and only then may `runner.py launch` start. No main run exists when this is written.

Written 2026-10-25, after the B2 results of record and the post-hoc extensions (matched learning rates
`idea_dq_posthoc_lr`, HQ at 1e-2 `idea_dq_posthoc_hqlr`, C-matched at 1e-2 `idea_dq_posthoc_cmlr`, generic angles
`idea_dq_posthoc_b2g`) were known. Not part of any frozen family; every result is labelled post hoc.

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

## The matched statistic (fixed now)
For one run: the arithmetic mean, over the run's update records (`metrics.jsonl`, kind `update`), of
`metrics.approx_kl`. That value is `smdp_ppo.ppo_update`'s mean over the update's optimiser steps (12 in B2) of the
minibatch estimate mean((r − 1) − log r), computed before each step against the rollout log-probabilities (the first
step's value is about 0). For an arm: the arithmetic mean of the run statistic over its blocks. It is a within-update
average, not the KL of the policy after the update, not a per-state KL and not a parameter distance; matching its mean
does not match its median or its distribution over updates (both are reported).

Targets (stored records, blocks 0-35 in ascending order; recomputed by the precheck, exact):
- T_H2 = 0.007823255686871039: stored B2 HQ (quantum, lr 3e-3), the comparator of K-H2.
- T_H1 = 0.00419994415996206: stored CMLR C-matched (bottleneck, lr 1e-2), the comparator of K-H1.

## Stage 1: calibration pilot (development only; fixed before any pilot run)
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
  (r ≈ 0.53-0.63), so disjoint blocks give the level estimate below more independent information per run. Resampling
  the stored per-block values (with the 36-block sampling error of the main runs added) gives P(achieved main-run mean
  within ±25% of target) ≈ 0.92-0.94 for 9 disjoint pilot runs per arm, ≈ 0.86-0.89 for 6, and ≈ 0.77 for 3 learning
  rates on 2 shared blocks.
- Selection rule (per arm; `runner.select`):
  1. K̄(lr) = arithmetic mean of the run statistic over the candidate's completed pilot runs.
  2. With x = ln lr and y = ln K̄ over the arm's candidates, and x̄, ȳ their means:
     lr* = exp(x̄ + (ln T − ȳ) / s), with s fixed now from the stored points of the same arm between lr 1e-3 and
     3e-3: s = 0.9385004171529107 (twin), 0.8421691057010121 (HQ). The pilot fixes only the level; a slope fitted to
     three noisy points 1.4× apart would have a standard error of about 0.3.
  3. lr* is rounded to 3 significant digits (`format(lr, '.2e')`); that value is the main runs' learning rate.
  4. Range guard: if lr* lies outside [min candidate / 1.5, max candidate × 1.5], stage 1b runs for that arm only:
     candidates round3(lr*) × {1/1.4, 1, 1.4} on the same blocks, rule 1-3 applied to the stage-1b runs only. No
     further iteration; the stage-1b result is used whatever it is.
  5. A pilot run that fails (algorithmic or infrastructure) is dropped from its candidate's mean. If fewer than two
     candidates of an arm have a completed run, no learning rate is selected for that arm, its main set is not
     launched, and this is reported.
  6. Reported, not used: K̄ per candidate and per run, the OLS slope over the candidates and the lr it would select,
     the predicted KL at the selected lr, and the comparison of the pilot's K̄ at the stored learning rates (twin 1e-3,
     HQ 1e-3) with the stored B2-block values (a transfer check, descriptive).

## Stage 2: main runs (72, all new, none retried; a run interrupted by a machine crash is restarted from the beginning)
B2 study object unchanged (`dq_b2.STUDY_B2`: salt, initialisation, training schedule, action and minibatch streams;
the 80 common-random-number evaluation episodes at the 11 nominal points 0, 3,000, …, 30,000; entropy 0; 30,000
nominal decisions; blocks 0-35). The B2 code tree is imported unchanged; the runner sets only Adam's constant
learning rate and adds the candidate name to the study's candidate table in its own process.
- T2 (K-H2 arm): twin (`surrogate625`, the exact transplant of the block's untrained HQ via `dq_b2.dq_build_model`,
  static parity P1, P2, P5-P7 certified at build) at lr*_twin = [from selection.json]. Before training, each run
  re-flies the block's first training episode with HQ and twin (`dq_b2.twin_trajectory_parity`, P4, |Δ log π| ≤
  1e-9, fail closed) and records it with the static record, as in B2 and the matched-learning-rate runner.
- T1 (K-H1 arm): HQ (`quantum`) at lr*_HQ = [from selection.json].
Each new run is paired with the stored run of its comparator on the same block (same schedule, same evaluation
episodes; for T2 also the same initial policy). Task order: blocks ascending, HQ first within a block.

## Precheck (before the launch; `runner.py precheck`, 2 processes)
(a) HQ, block 0, at B2's stored 3e-3, through the launch code path (`run_one`), 3,000 decisions, grid [0, 3,000]:
    evaluation, update and episode events and the arm_start model/identity/seeds identical to stored B2
    `quantum__b0`, and the initial checkpoint identical tensor by tensor.
(b) twin, block 0, at B2's stored 1e-3, same path with P4 parity: evaluation and episode events, arm_start and the
    initial checkpoint identical; update metrics identical, or (as in the matched-learning-rate precheck) equal to a
    maximum relative difference ≤ 1e-12, which is then reported as summation-order rounding; P4 ≤ 1e-9 and no static
    failure; equality with the stored B2 parity record reported.
(c) The 72 main configurations differ from `dq_b2.config_for(arm, block, STUDY_B2)` only in ppo.learning_rate,
    candidate, note and protocol_freeze_sha256, and pass E1's validator (`p4.validate_config`).
(d) Pilot configurations: study dq_b2_smoke, evaluation None, budget 30,000, B2 PPO settings except the learning
    rate; every seed blocks 90-98 can draw lies in dq_development and outside dq_e2_training / dq_e2_evaluation.
(e) T_H1, T_H2 and (f) the two slopes recomputed from the stored records equal the values above.
(g) `p4.build_model` is `dq_b2.dq_build_model` in the runner's process; (h) the live idea_dq tree is ffe9dfef….
The launch also refuses without PLAN.md matching PLAN.sha256, without a ready selection.json whose `lr_str` values
appear in PLAN.md, or if runner.py differs from the one that passed the precheck.

## Outcome and analysis (fixed now; `b2k_analysis.py`)
Outcome: B2's AUC (sample read-out, trapezoidal learning-curve area over the nominal grid / 30,000) from each run's
summary. A failed run is reported as failed and its block is dropped from that contrast only.
- K-H2 = AUC(HQ, 3e-3; stored B2 `quantum__b`) − AUC(twin, lr*_twin; new T2), paired over the 36 blocks.
- K-H1 = AUC(HQ, lr*_HQ; new T1) − AUC(C-matched, 1e-2; stored CMLR `C1_bottleneck__b`), paired over the 36 blocks.
For each: mean, 95% paired t interval, two-sided paired t p-value, dz, share of blocks below zero. Holm over
{K-H1, K-H2} at familywise 0.05. Secondary, outside the Holm family: TOST at B2's margin ±0.0056 (90% paired t
interval inside ±0.0056 → "within B2's equivalence margin"). Precision (stored paired SDs): about ±0.0045 (K-H2) and
±0.0052 (K-H1) for the 95% half-width.

Achieved-KL check (per new arm): A = arithmetic mean over its completed runs of the run statistic; matched iff
0.75 T ≤ A ≤ 1.25 T. Reported with R = A / T, the 95% t interval of A over blocks, the median of the run statistic and
the median and 90th percentile of the pooled per-update values, beside the same for the stored comparator.

Reading rules (fixed now). "Below zero" / "above zero" = Holm-rejected with negative / positive mean; otherwise "not
resolved". If the arm's KL is matched:
- K-H2 below zero: at matched mean per-update KL HQ still learned less than its exact twin; B-H2 is not explained by
  the twin's larger updates as measured by this statistic.
- K-H2 not resolved: at matched per-update KL the HQ − twin difference is not resolved; the B-H2 difference is
  consistent with an update-size account (this is not evidence of equality). If the secondary TOST passes, the
  difference at matched KL is also reported as within B2's ±0.0056 margin.
- K-H2 above zero: at matched per-update KL HQ learned more than its twin; B-H2 reverses and is reported as dependent
  on update size.
- K-H1 below zero: at matched mean per-update KL HQ learned less than C-matched; B-H1 is not explained by update size
  as measured by this statistic.
- K-H1 not resolved: at matched per-update KL the HQ − C-matched difference is not resolved; B-H1 is consistent with
  a dependence on update size (not evidence of equality); TOST as for K-H2.
- K-H1 above zero: at matched per-update KL HQ learned more than C-matched; B-H1 is reported as dependent on update size.
If an arm's KL is not matched (R outside [0.75, 1.25]), its contrast is still computed, tested and reported, labelled
"KL not matched (R = …)", and only these one-sided statements are made:
- K-H2 with R < 0.75 (twin's updates smaller than HQ's) and K-H2 below zero: HQ learned less than a twin whose updates
  were smaller; otherwise no update-size statement.
- K-H1 with R > 1.25 (HQ's updates larger than C-matched's) and K-H1 below zero: HQ learned less although its updates
  were larger; otherwise no update-size statement.
No time cut-off; the read-out waits for all 72 records (`--provisional` only for monitoring, never reported).

Descriptive (no tests): AUC, J at the first and final checkpoints, per arm; within-arm learning-rate effects
AUC(twin new) − AUC(twin B2, 1e-3), AUC(HQ new) − AUC(HQ B2, 3e-3) and − AUC(HQ, 1e-3; idea_dq_posthoc_lr L2);
B-H1 and B-H2 of record on the same blocks; the paired per-block KL ratio of each contrast; P4 and static parity
summary of the 36 twin runs.

## Stored-run descriptive table (no new runs; `b2k_analysis.py --stored-table`)
From stored runs only, labelled descriptive post hoc, with no p-values and no decisions: AUC(HQ) − AUC(C-matched)
at every stored learning-rate pairing (HQ 1e-3 L2, 3e-3 B2, 1e-2 H1 × C-matched 3e-3 B2, 1e-2 C1), and
AUC(HQ) − AUC(twin) at every stored pairing (HQ 1e-3, 3e-3, 1e-2 × twin 1e-3 B2, 3e-3 L1): n, mean, 95% paired t
interval, share of blocks below zero, dz, each arm's mean per-update KL and the KL ratio. The generic-angle HQ runs
(b2g) are excluded (another initialisation).

## Crashes and failures
Completed records are kept. A run with a record of any status is never run again; a failure stays a failure. A run
directory without a record was killed by a crash: it is moved unchanged to `interrupted/<UTC>/` and the run restarts
from the beginning (`runner.py launch` / `pilot` do this; every launch is logged in launches.jsonl). The rule is
applied without looking at any partial result.

## Reporting commitment
Whatever the outcome, K-H1, K-H2, their Holm decisions, the TOST results, the achieved KLs with R and the matched
flags, the pilot's KL readings and selected learning rates, and the stored-run table are reported in Section 4.7 next
to the matched-learning-rate checks (Table 11), with this plan's SHA-256. Scripts and records go to the repository.
