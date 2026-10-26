# E3 build record: early training at matched per-update KL (B2 protocol)

Folder: `/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b2k`. Built 2026-10-25 by a subagent. Nothing outside
this folder and the session scratchpad was written; idea_dq was imported read-only (live tree
`ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952` before and after, checked by every precheck and
pilot launch). No main run was started. PLAN_DRAFT.md is not hashed.

## Files
| File | What it is |
|---|---|
| `PLAN_DRAFT.md` | Draft plan: question, matched statistic and targets, pilot and selection rule (fixed before the pilot), 72 main runs, precheck, contrasts, tests, KL tolerance, reading rules, stored-run table, crash rule, reporting commitment |
| `runner.py` | `precheck`, `pilot [--workers N] [--stage 1|1b]`, `pilot-select [--stage …]`, `launch [--workers N]`; b2g/hqlr pattern (fork pool, maxtasksperchild=1, records/ skip, interrupted/ move, launches.jsonl) |
| `b2k_analysis.py` | Main analysis (refuses < 72 records unless `--provisional`, which writes a separately named file), `--stored-table`, `--selftest` |
| `test_b2k.py` | 17 unit tests (selection rule, configs, KL reader, paired t / Holm against scipy, reading rules, refusal on partial data, calendar-gate independence) |
| `precheck.json`, `precheck/`, `precheck.log` | Final precheck (runner.py `96bb8b08…`): passed |
| `precheck_attempt1_assessment_crashed/`, `precheck_attempt1.log` | First precheck: both runs completed but the assessment code crashed (`KeyError: 'compared'`, a bug in my comparison helper); kept for the record; produced by an earlier runner.py |
| `stored_kl_table.json`, `stored_kl_table_stdout.txt` | Descriptive table from stored runs only |
| `pilot/`, `pilot.log` | Calibration pilot (completed): `launches.jsonl`, the PLAN_DRAFT.md snapshot taken at launch, `runs/`, `records/` |
| `selection_stage1.json`, `selection.json`, `pilot_select_stdout.txt` | Output of `runner.py pilot-select` (the pre-declared rule) |

## What I ran (UTC), and what passed
1. Stored-record statistics, KL only (scratchpad `b2k/kl_stats.py`, `b2k/pilot_sim.py`, `b2k/pilot_sim2.py`,
   `b2k/pilot_sim3.py`; python -B, read-only). Findings used in the plan:
   - targets (blocks 0-35 ascending): stored B2 HQ 3e-3 = 0.007823255686871039; stored CMLR C-matched 1e-2 =
     0.00419994415996206;
   - between-block CV of the run statistic 0.33-0.39 (HQ, twin), 0.60-0.67 (C-matched 3e-3/1e-2, HQ 1e-2); within-
     block correlation of ln KL across learning rates 0.53-0.63 (1e-3 vs 3e-3); log-log slopes (1e-3 → 3e-3):
     twin 0.9385, HQ 0.8422; stored-point predictions: twin 7.76e-4 for 0.00782, HQ 1.43e-3 for 0.00420;
   - shorter budgets are biased (ratio of the running mean at 15,000 decisions to the 30,000 mean: 0.90-1.39 by arm and
     learning rate), so the pilot uses 30,000 decisions;
   - resampling of stored per-block KLs: P(achieved 36-block mean within ±25% of target) ≈ 0.77 with 3 learning rates
     × 2 shared blocks (the brief's example), 0.86-0.89 with 6 disjoint-block runs per arm, 0.92-0.94 with 9. The plan
     uses 9 per arm (3 candidates × 3 disjoint development blocks, blocks 90-98).
2. `runner.static_checks()` (no training): all pass after the declared targets were set to the ascending-order
   values (the first constants differed from the recomputation in the last bit, from another summation order).
3. 07:34-07:41Z, precheck attempt 1 (2 processes, nice 10): both runs completed; the assessment step crashed (bug,
   fixed). An ad hoc comparison of its outputs already showed bitwise reproduction for both arms.
4. 07:42-07:45Z, pilot code-path smoke test in the scratchpad (development block 99, which the pilot does not use,
   about 1,050 decisions, both arms, one process): both completed; no evaluation event; summary AUC `{}`; seeds
   20262020000+ (dq_development); twin P4 parity 1.3e-15 on the development study.
5. `pytest test_b2k.py`: 16 passed; after the pilot, with the 17th test (calendar-gate independence) added: 17 passed.
6. `b2k_analysis.py --selftest`: passed (B-H1 −0.00866, p = 0.0017; B-H2 −0.00777, p = 0.0012; both targets equal).
7. `b2k_analysis.py --stored-table` (after PLAN_DRAFT.md fixed the pilot rule): table below.
8. 07:47-07:51Z, final precheck (runner.py `96bb8b088c1221605d7f9400e407a93a896adcc77d8d7400dfeb5c8434d31e68`, 2
   processes, nice 10): **passed**.
   - (a) HQ block 0 at 3e-3: 2 evaluations, 6 updates, 77 episodes, arm_start and the 17 initial-checkpoint tensors
     identical to stored B2 `quantum__b0`.
   - (b) twin block 0 at 1e-3: 2 evaluations, 6 updates (bitwise: max relative difference 0.0), 80 episodes,
     arm_start, 18 tensors identical; P4 parity 1.3322676295501878e-15 and the static record equal to the stored B2
     record exactly. (The matched-learning-rate precheck of idea_dq_posthoc_lr had 3.5e-14 update differences; there
     the parity episode ran before `torch.set_num_threads(1)`; here `run_one` sets it first, as `dq_b2.run_one` did.)
   - (c) the 72 main configs differ from `dq_b2.config_for` only in ppo.learning_rate, candidate, note,
     protocol_freeze_sha256 and pass `p4.validate_config`; (d) pilot configs dq_b2_smoke, evaluation None, 30,000
     decisions, seeds in dq_development only; (e) targets and (f) slopes recomputed exactly; (g) build path is
     `dq_b2.dq_build_model`; (h) live tree ffe9dfef….
9. 07:51:18Z-11:40Z, calibration pilot (18 runs, 2 workers, nice 10): all completed; then `runner.py pilot-select`
   (see the pilot section). While it ran, only file edits and second-long reads were made (plus a one-off dry run of
   `b2k_analysis.main` on symlinked stored stand-in runs in the scratchpad, which exercised the full read-out path and
   found a division by zero for identical paired runs, now guarded).
10. After the pilot: `pytest` 17 passed; `b2k_analysis.py --selftest` passed; `--stored-table` re-run with the final
    script (same numbers); live idea_dq tree still ffe9dfef…; runner.py unchanged since the passed precheck
    (`96bb8b08…`); PLAN_DRAFT.md edited after the pilot launch only outside Stage 1 (diff against the snapshot: the
    K-H1 precision sentence, the two selected lrs in Stage 2, the appendix).

## Timings measured under the current load (load average 16-19 on 16 threads; another 10-worker campaign at nice 5)

Precheck (block 0, first 3,000 decisions):

| | HQ (quantum) | twin (surrogate625) |
|---|---|---|
| Rollout wall, first 3,200 decisions of block 0 | 30.2 s / 1,000 decisions | 18.8 s / 1,000 decisions |
| Rollout CPU, same | 29.0 s / 1,000 | 18.2 s / 1,000 |
| Update wall | 0.86 s / 1,000 | 0.44 s / 1,000 |
| Evaluation wall (points 0 and 3,000) | 1.04 s / episode | 0.66 s / episode |
| Precheck run (3,000 decisions + 160 evaluation episodes) | 270 s | 183 s |

Against the stored B2 run of the same block and span, rollouts were about ×1.4 (HQ) and ×1.3 (twin) slower. Block
0 is a cheap block: across B2 the mean rollout was 0.41 h (HQ) and 0.34 h (twin) per 30,000-decision run, and
evaluation about 1.9 s per episode over all 11 points. Projection: pilot run (no evaluation) ≈ 0.45-0.6 h each, 18
runs at 2 workers ≈ 4.5-5.5 h (measured: 3 h 49 min).

Measured in the pilot (30,000 decisions, no evaluation, 2 workers, same load):

| | HQ (9 runs) | twin (9 runs) |
|---|---|---|
| Rollout wall per 1,000 decisions | mean 53.7 s (21.3-76.0) | mean 41.2 s (10.5-63.3) |
| Rollout CPU per 1,000 decisions | mean 52.0 s | mean 39.8 s |
| Update wall per 1,000 decisions | 0.80 s | 0.40 s |
| Run wall | mean 1,652 s (678-2,317) | mean 1,273 s (343-1,951) |
| Training episodes per run | mean 650 (575-737) | mean 592 (526-723) |

Main-run projection (training as measured plus 880 evaluation episodes at about 1.9-2.5 s each): about 1.0 h per HQ
run and 0.8 h per twin run under this load, so 72 runs ≈ 65 worker-hours, about 5.5-6.5 h wall at 12 workers.

## Stored-run descriptive table (no new runs; descriptive, post hoc, no p-values)
AUC(HQ) − AUC(comparator), 95% paired t interval, 36 blocks; KL = mean per-update approx_kl.

| Contrast | HQ KL | comparator KL | KL ratio | ΔAUC [95% CI] | HQ lower |
|---|---|---|---|---|---|
| HQ 1e-3 − C-matched 3e-3 | 0.0031 | 0.0023 | 1.33 | −0.0142 [−0.0186, −0.0098] | 86% |
| HQ 3e-3 − C-matched 3e-3 (B-H1) | 0.0078 | 0.0023 | 3.34 | −0.0087 [−0.0138, −0.0035] | 75% |
| HQ 1e-2 − C-matched 3e-3 | 0.0070 | 0.0023 | 2.98 | −0.0055 [−0.0105, −0.0005] | 69% |
| HQ 1e-3 − C-matched 1e-2 | 0.0031 | 0.0042 | 0.74 | −0.0178 [−0.0230, −0.0125] | 86% |
| HQ 3e-3 − C-matched 1e-2 | 0.0078 | 0.0042 | 1.86 | −0.0122 [−0.0182, −0.0063] | 75% |
| HQ 1e-2 − C-matched 1e-2 (Z1) | 0.0070 | 0.0042 | 1.66 | −0.0090 [−0.0147, −0.0034] | 69% |
| HQ 1e-3 − twin 1e-3 (L2) | 0.0031 | 0.0099 | 0.31 | −0.0133 [−0.0178, −0.0089] | 92% |
| HQ 3e-3 − twin 1e-3 (B-H2) | 0.0078 | 0.0099 | 0.79 | −0.0078 [−0.0122, −0.0033] | 83% |
| HQ 1e-2 − twin 1e-3 | 0.0070 | 0.0099 | 0.70 | −0.0046 [−0.0096, +0.0004] | 72% |
| HQ 1e-3 − twin 3e-3 | 0.0031 | 0.0278 | 0.11 | −0.0156 [−0.0193, −0.0120] | 97% |
| HQ 3e-3 − twin 3e-3 (L1) | 0.0078 | 0.0278 | 0.28 | −0.0101 [−0.0133, −0.0068] | 94% |
| HQ 1e-2 − twin 3e-3 | 0.0070 | 0.0278 | 0.25 | −0.0069 [−0.0120, −0.0018] | 75% |

Every HQ − C-matched interval is below zero, including the two pairings where HQ's KL is within ×0.74-1.33 of
C-matched's. Every stored HQ − twin pairing has the twin at the larger KL (ratio 0.11-0.79), which is why K-H2 needs
new runs; one of those intervals (HQ 1e-2 − twin 1e-3) includes zero.

## Calibration pilot
Launched 2026-10-25T07:51:18Z (`setsid nohup nice -n 10 …/python -B runner.py pilot --workers 2`), finished
11:40Z (3 h 49 min wall at 2 workers). 18 of 18 runs completed, no failure, 56-57 updates and 30,032-30,506
decisions each; every summary has `auc {}` and `learning_curve {}` (no evaluation episode ran). The launcher has
exited. `runner.py pilot-select` (reads only update records' approx_kl) wrote `selection_stage1.json` and
`selection.json` (`ready: true`); stdout in `pilot_select_stdout.txt`.

Run statistic (mean per-update approx_kl over the run; median over the run's updates in brackets):

| arm | candidate | block: KL [median] | K̄ |
|---|---|---|---|
| twin | 5e-4 | 90: 0.004458 [0.00379]; 91: 0.005006 [0.00408]; 92: 0.005539 [0.00448] | 0.005001 |
| twin | 7e-4 | 93: 0.008481 [0.00820]; 94: 0.009155 [0.00783]; 95: 0.008332 [0.00798] | 0.008656 |
| twin | 1e-3 | 96: 0.010063 [0.00868]; 97: 0.003105 [0.00044]; 98: 0.011892 [0.01082] | 0.008353 |
| HQ | 1e-3 | 90: 0.003977 [0.00294]; 91: 0.003368 [0.00199]; 92: 0.004629 [0.00363] | 0.003991 |
| HQ | 1.4e-3 | 93: 0.006099 [0.00588]; 94: 0.004777 [0.00381]; 95: 0.004675 [0.00423] | 0.005183 |
| HQ | 2e-3 | 96: 0.008165 [0.00657]; 97: 0.002008 [0.00093]; 98: 0.007270 [0.00631] | 0.005814 |

Selection (pre-declared fixed-slope rule): **twin lr* = 7.786e-4 → 7.79e-04** (target 0.0078233; range guard
[3.3e-4, 1.5e-3], inside); **HQ lr* = 1.163e-3 → 1.16e-03** (target 0.0042; guard [6.7e-4, 3e-3], inside). No stage
1b. Rule-predicted KL at the rounded lrs: 0.007827 (twin), 0.004189 (HQ).

Descriptive only (not used by the rule, reported as the plan says):
- OLS slopes over the three candidate means: 0.73 (twin), 0.54 (HQ) (stored: 0.94, 0.84); with them the rule would
  give 8.01e-4 and 1.05e-3.
- Transfer check at the stored learning rate 1e-3: pilot K̄ / stored B2-block mean = 0.84 (twin, 0.008353 vs
  0.009926) and 1.29 (HQ, 0.003991 vs 0.003101); with a between-block CV of about 0.36 a 3-block mean has a relative
  SE of about 0.21, so these are about −0.8 and +1.4 SE.
- Block 97 gave the lowest KL in both arms (twin 1e-3: 0.0031, median 0.00044; HQ 2e-3: 0.0020); the two are
  plausibly linked, since the twin is the same block's transplanted HQ on the same schedule. It lowers the third candidate's mean in
  both arms. Sensitivity (not a rule): without block 97 the rule would give 7.07e-4 (twin) and 1.04e-3 (HQ); if those
  were right, the selected lrs would give R ≈ 1.10 for both arms. If instead the stored-point interpolation were right
  for B2 blocks, R ≈ 1.00 (twin) and 0.84 (HQ). All of these lie inside the ±25% tolerance; HQ is the arm closer to
  the lower edge.
- The stored-point predictions were 7.76e-4 (twin, agrees with the pilot) and 1.43e-3 (HQ, the pilot selected 19%
  lower).

## Next steps for the lead
1. Review PLAN_DRAFT.md; write PLAN.md from it with the two `lr_str` values of `selection.json` written verbatim
   (the launch searches PLAN.md for them); write PLAN.sha256 (`sha256sum PLAN.md > PLAN.sha256`) and PLAN.utc.
2. If runner.py is edited in review: move `precheck/` and `precheck.json` aside and re-run `runner.py precheck`
   (≈ 5 min, 2 processes); the launch refuses a precheck made by another runner.py.
3. `nice -n 10 …/python -B runner.py launch --workers N` (resumable; a crash-killed run is moved to interrupted/).
4. After all 72 records: `b2k_analysis.py` (refuses partial data), then report per the plan.

## Open risks
- Calibration can still miss: P(achieved within ±25%) ≈ 0.92-0.94 per arm before the pilot, by resampling stored
  values. After the pilot, HQ is the riskier arm: plausible R spans about 0.84-1.10 (pilot section). A miss is
  reported as "KL not matched" with only the pre-declared one-sided statements.
- HQ's selected 1.16e-3 is 19% below the stored-point prediction (1.43e-3) because the development blocks showed
  higher HQ KL at 1e-3 than the B2 blocks (ratio 1.29, 3 blocks). The plan uses the development pilot only, as
  required; the lead should be aware that the stored B2 points alone would have suggested a higher rate.
- Transfer from development to B2 blocks is assumed unbiased (same cells, weights and generative process; other salt
  and seed namespace). The pilot includes each arm's stored learning rate (twin 1e-3, HQ 1e-3) on 3 blocks, a
  descriptive transfer check only (noisy at 3 blocks).
- The fixed slope comes from stored points at 1e-3 and 3e-3; HQ's KL is not monotone above 3e-3 (0.0070 at 1e-2). Its
  error is second order while the candidates bracket the target; the range guard triggers stage 1b otherwise.
- The matched statistic is B2's within-update mean of the pre-step minibatch KL, not the post-update KL; matching its
  mean does not match its median or its distribution (C-matched's per-update KL is strongly skewed: median 0.0010 vs
  mean 0.0023 at 3e-3). The plan states this; a post-update full-batch KL was not added (no stored comparator has it).
- The comparator sets are stored runs reused from B2 and CMLR, so K-H1/K-H2 are correlated with B-H1, B-H2 and Z1
  and are not independent replications; they add 2 p-values (plus 2 TOST results) to the paper.
- Additions beyond the brief, for the lead to keep or strip: a secondary TOST at B2's ±0.0056; one-sided readings for
  unmatched KL; 9 pilot runs per arm on disjoint blocks 90-98 instead of the brief's example of blocks 90-91; a range
  guard with one stage-1b iteration.
- Machine load: the pilot shares the CPU with another 10-worker campaign and other agents' prechecks (load 16-18 on
  16 threads); wall times above are inflated by this.
- Calendar: `dq.require_open` is called only by dq_b2.run_one/launch/smoke and dq_transplant's CLI, none of which this
  runner uses; the 17th unit test checks the per-run build/transplant/parity path with the gate forced to refuse.
- No resume within a run (train_arm saves no generator state); a main run takes about 1 h under load.

## Fixes after review (2026-10-25)
Applied 2026-10-25, 12:00-12:20Z, by a subagent, following the lead's binding decisions on REVIEW.md. Nothing outside
this folder and the session scratchpad was written; idea_dq was only imported (live tree ffe9dfef… before and after);
no main run was started; PLAN.sha256 / PLAN.utc were NOT written (the lead hashes PLAN.md and launches). No outcome
value of any pilot run was read. PLAN_DRAFT.md is unchanged (ce68473d…).

Item by item:
- **B1 (dropped-block ceiling, not estimable, degenerate).** `b2k_analysis.py` now carries B2's frozen rules from
  `dq_b2_analysis.py`, copied: `differences` lists the dropped blocks; `t_test` returns NOT ESTIMABLE with more than
  `MAX_DROPPED = 3` of 36 blocks dropped (or n < 3), p = 1 for sd = 0 with mean 0, p = 0 for a non-zero constant;
  `holm` orders only estimable contrasts and keeps the denominator at the family size 2; `tost` reports "not
  estimable"; the reading of a not-estimable contrast is "not estimable" (no nan is sorted any more). The rules are in
  PLAN.md "Outcome and analysis". Tests: `test_more_than_three_dropped_blocks_is_not_estimable`,
  `test_not_estimable_is_left_out_of_holm_but_the_denominator_stays_two`, `test_degenerate_sd_zero`,
  `test_failed_runs_drop_blocks_and_four_make_k_h2_not_estimable`.
- **B2 (integrity of the new runs).** New `frozen_identity`, `load_records`, `prove`, `check_identity`,
  `check_completed_run`, `check_outcome`, `check_twin`; any failure raises `IntegrityError` before anything is written.
  Checked: PLAN.md = PLAN.sha256 and PLAN.md records the sha256 of b2k_analysis.py and runner.py; precheck.json passed
  with the imported runner.py; selection.json ready, lr strings in PLAN.md; live tree ffe9dfef…; records/ holds only
  the 72 expected names; each record's kind/set/arm/block, learning_rate = selection.json, plan_sha256 = PLAN.sha256,
  runner_sha256 = the precheck's, code tree ffe9dfef…, config = `runner.config_main(arm, block, selected lr, PLAN
  hash)` (canonical JSON); raw events: one arm_start first, one arm_end last, update indices 1..U contiguous with U =
  arm_end's and record's count, the GRID evaluated exactly once in order, finite J, AUC recomputed with
  `p4.learning_curve_auc` equal to arm_end's and record's, decisions/episodes/updates agree, one final checkpoint, the
  record's KL statistic equal to the one recomputed from the verified update events; a failure never coexists with a
  completed arm_end or a final checkpoint; a completed twin needs static_failures == [] and P4 ≤ 1e-9 (B2's TWIN rule,
  refusing as B2's analysis does). AUC, J and KL of the new runs now come from the proven events. Tests: a complete
  synthetic 72-record campaign passes and matches a hand computation (`test_clean_synthetic_campaign_…`); 14 doctored
  copies are refused for the intended reason (duplicated update line, duplicated grid point, two arm_starts, AUC off
  by 1e-12, config at another lr, wrong plan/runner/tree hash, record KL changed, twin P4 2e-9, twin static failure,
  PLAN.md edited after hashing, PLAN.md without the analysis hash, foreign record) plus a failure record beside a
  completed run; `test_raw_event_check_on_the_real_precheck_runs` runs the checker on the real trainer output of the
  new precheck (config equality and exact AUC recomputation hold).
- **B3 (code pinned).** PLAN.md records the sha256 of runner.py, b2k_analysis.py, validate_main_configs.py and
  test_b2k.py (and of selection.json, precheck.json and the M6 output), computed after the last edit, and contains
  `7.79e-04` and `1.16e-03` verbatim. `runner.py launch` now also refuses unless PLAN.md contains the sha256 of
  runner.py and of the current b2k_analysis.py, and unless selection.json's lr equals float(lr_str)
  (`test_launch_refuses_a_plan_that_does_not_pin_the_code`, which also checks the 72 tasks, T1 first, at the selected
  lrs, with pool_run replaced so that nothing runs).
- **M1.** Descriptive contrasts use `interval()` (no p-value); `test_descriptive_interval_has_no_p_value` and the
  synthetic campaign test check that no `p` key appears anywhere under `descriptive`.
- **M2.** `B_H2_RECORD = -0.007772148908780312` (B-H2 of record, 36 blocks; `--selftest` and a unit test recompute it
  exactly). K-H2's 95% interval is reported beside it (`K-H2_beside_B-H2_of_record`). New directions for K-H2:
  "partial reduction" (Holm-rejected below zero, whole interval above B-H2 of record), "inconclusive" (not rejected,
  interval contains 0 and B-H2 of record), "not resolved" (contains 0, above B-H2 of record; the update-size and TOST
  texts), "not resolved after Holm" (unadjusted interval excludes 0 but Holm does not reject; no statement). D =
  AUC(twin new) − AUC(twin B2, 1e-3) is named the direct descriptive measure, with a point share mean(D)/mean(B-H2).
  Correction to REVIEW.md M2: per block D = B-H2 − K-H2 (not K-H2 − B-H2), so −D is the part of B-H2 removed; PLAN.md
  and the code use the corrected sign. Tests: `test_reading_rules` (rewritten), `test_beside_b_h2`.
- **M3.** PLAN.md states the within-arm monotonicity premise of the one-sided readings with the stored values (paired
  mean differences of the stored-run table): HQ 3e-3 − 1e-3 = +0.0055, 1e-2 − 3e-3 = +0.0032; twin 3e-3 − 1e-3 =
  +0.0023; C-matched 1e-2 − 3e-3 = +0.0036; and its limits (evidence by learning rate; HQ's KL at 1e-2 below that at
  3e-3; the new twin rate is below the twin's lowest stored rate).
- **M4.** Stage 1 discloses that the slopes and candidate centres come from stored training-update KL on B2 blocks 0-35
  (dq_e2_training), with no AUC, J or evaluation datum.
- **M5.** Precheck (d) in PLAN.md declares that the pilot configs bypass `p4.validate_config` (budget 30,000, grid [0],
  blocks 90-98 outside STUDY_SMOKE's frozen role 6,000 / [0, 3,000, 6,000] / block 90; verified from dq_b2.STUDY_SMOKE)
  and why that is harmless.
- **M6.** New `validate_main_configs.py` (no training): the 72 main configs at the selected 7.79e-04 / 1.16e-03 pass
  `p4.validate_config`, differ from `dq_b2.config_for` only in ppo.learning_rate, candidate, note,
  protocol_freeze_sha256, have lr = float(lr_str), entropy 0, candidate `e3_lr<lr_str>_ent0`, B2's block, arms, budget,
  grid, evaluation, study and role. Output `main_configs_validation_placeholder.json` (72 of 72 ok; plan hash
  placeholder because PLAN.md is not hashed yet) and its stdout. The lead may re-run it after hashing (it then writes
  `main_configs_validation_plan-<hash8>.json`; it never overwrites).
- **M7 + M8 (runner.py edited, precheck re-run).** runner.py: `single_instance()` takes an exclusive non-blocking flock
  on `runner.lock` for precheck, pilot and launch (fork-pool workers inherit it, so a relaunch is refused while any
  worker of an earlier launch lives); `write_json_atomic()` writes every record via a temporary file, fsync and
  os.replace; `prepare()` applies the resume rule: a parsable record of any status is skipped, an unparsable record,
  its run directory and any stale temporary record are moved unchanged to `interrupted/<UTC>/` and the run restarts;
  moves are logged in launches.jsonl. The run path (configs, seeds, build, P4, train_arm) is untouched. Tests:
  `test_single_instance_lock_refuses_a_second_holder`, `test_lock_survives_in_a_forked_child_after_the_parent_side_closes`,
  `test_write_json_atomic_leaves_no_temporary_file`, `test_prepare_resume_rule`,
  `test_unparsable_record_counts_as_no_record` (analysis side). Precheck: the old outputs (precheck/, precheck.json,
  precheck.log) were moved to `precheck_prev_20261025T120553Z/` together with an archival copy of the old runner
  (`runner.py.96bb8b08`, which produced the pilot and selection.json); `nice -n 10 …/idea/.venv/bin/python -B
  runner.py precheck` ran 12:05:57-12:08:07Z (2 workers) with runner.py 210923c4…: **passed**, bitwise reproduction
  again: (a) HQ evaluation 2/2, updates 6/6 (max relative difference 0.0), episodes 77/77, arm_start, 17 initial
  tensors identical; (b) twin evaluation 2/2, updates 6/6 (0.0), episodes 80/80, arm_start, 18 tensors identical, P4
  1.3322676295501878e-15, static record equal to stored B2; static checks (c)-(h) identical to the previous precheck.
  Timing under the lighter load: HQ rollout 14.6 s / 1,000 decisions, evaluation 0.50 s / episode (run 129 s); twin
  8.8 s / 1,000, 0.31 s / episode (85 s).
- **M9.** PLAN.md states that K-H1 is largely predicted by the stored pairings (HQ 1e-3 − C-matched 1e-2 −0.0178
  [−0.0230, −0.0125]; HQ 3e-3 − C-matched 1e-2 −0.0122) and that Holm then tests K-H2 at 0.05; and why the comparator is
  C-matched at 1e-2: C-matched never reached 0.0078 at a tested rate; log-log extrapolation of its two stored points
  needs about 3.6e-2 (untested).
- **M10.** Not changed (selected rates final); PLAN.md states the expectation R ≈ 0.84 for HQ by stored-point
  interpolation (computed 0.8368; twin 1.0037) and that a miss is reported as it is.
- **M11.** Declared in PLAN.md "Declared limits".
- **M12.** PLAN.md: launch with at most 8 workers while other campaigns run; runner.py's default `--workers` is now 8.
- **Lead, after B2F.** PLAN.md carries the interpretation note (HQ with frozen angles = the twin with frozen
  coefficients, same model and gradients; B2F F1 equivalent within ±0.0056; K-H2 therefore compares, approximately, the
  fixed-feature model with the same model with trained coefficients, at HQ's update size) and ONE descriptive paired
  contrast `twin_new_minus_HQ_frozen_B2F` = AUC(twin new) − AUC(HQ frozen, `idea_dq_posthoc_b2f/runs/F_quantum__b*`,
  same blocks), no test, labelled not an exact decomposition (classical learning rates 7.79e-04 vs 3e-3). B2F runs are
  read only through the last (summary) line of metrics.jsonl, as `b2f_analysis.summary` does (`load_b2f`;
  `test_b2f_reader_uses_only_the_summary_line`); `--selftest` checks that 36 B2F runs are readable (count only).

Other changes and checks:
- `interval()` reproduces the earlier arithmetic: `--stored-table` written to the scratchpad by the final script gives
  rows and stdout identical, bit for bit, to `stored_kl_table.json` / `stored_kl_table_stdout.txt` (not rewritten here).
- `b2k_analysis.py` now imports `runner` (hence dq_b2 / p4_train from the unchanged idea_dq tree) for the expected
  configs and `learning_curve_auc`; run it with `python -B`.
- `pytest test_b2k.py` (python -B, nice 10, basetemp in the scratchpad): **48 passed**.
  `b2k_analysis.py --selftest`: passed (B-H1 −0.00866,
  p 0.0017; B-H2 −0.00777 = B_H2_RECORD exactly, p 0.0012; targets exact; 36 B2F runs readable).
  Count: of the 17 earlier items, 2 were rewritten for the new API (`test_paired_and_holm_against_scipy` →
  `test_t_test_against_scipy_and_holm_family_of_two`; `test_reading_rules`); 31 items are new, 14 of them the
  parametrised doctored cases.
- The FROZEN check was simulated on a scratchpad copy of PLAN.md + precheck.json + selection.json with a PLAN.sha256
  written there: accepted (plan b40f60f7…, runner 210923c4…, analysis 8daa84d3…). In this folder `runner.py launch`
  currently refuses only because PLAN.sha256 does not exist.
- `b2k_analysis.py` on this folder refuses ("incomplete: 0 of 72 parsable records"), as it must.

Round 2 (independent verifier, 2026-10-25; three items, all confirmed as defects and fixed):
- **R2-1 (K-H1 had no "not resolved after Holm"; b2k_analysis.py `direction()` / `READINGS`, PLAN.md "Reading
  rules" and the matched-case list).** Reproduced: with K-H1 p = 0.030 first in the Holm order (threshold 0.025, not
  rejected) and 95% interval [−0.0076, −0.0004], the old `direction()` returned "not resolved" and the reading said
  "B-H1 is consistent with a dependence on update size", while K-H2 in the same situation got "not resolved after
  Holm" with no statement. Fix: `direction()` applies one rule to both contrasts (Holm-rejected → below / above zero;
  not rejected, unadjusted 95% interval contains 0 → not resolved; not rejected, interval excludes 0 → not resolved
  after Holm); K-H2's M2 refinements (partial reduction, inconclusive) are unchanged. `READINGS` has
  ('K-H1', 'not resolved after Holm') with "no update-size statement"; `reading()` adds the TOST text only to "not
  resolved", so neither contrast gets a TOST add-on after Holm (the TOST result itself is still reported). PLAN.md
  states the direction rule once for both contrasts and lists "K-H1 or K-H2 not resolved after Holm" among the
  matched-case readings. A scratchpad grid check (5,166 cases, not a test) confirmed that K-H2's direction is
  unchanged and that K-H1's changes only in the not-rejected, interval-excludes-0 cell; the old code gave the
  verifier's misreading. Tests: `test_reading_rules` extended (K-H1 and K-H2, both signs, TOST passing, matched and
  not matched); new `test_not_resolved_after_holm_through_holm_both_orders` runs the verifier's two reproductions
  through `holm()`.
- **R2-2 (PLAN.md:201 parenthetical "possible only if Holm's threshold for K-H2 is 0.025" was false).** The case also
  arises when K-H1 has the smaller p-value, is not rejected at 0.025 and Holm stops (K-H2: holm_threshold None,
  rejected False). New wording: "possible when the contrast is tested at Holm's 0.025 (it has the smaller p-value) and
  not rejected, or when Holm stops at the other, unrejected contrast and this one is not tested". The M9 paragraph now
  also says that, if K-H1 had the smaller p-value and were not rejected at 0.025, Holm would stop and K-H2 would not
  be tested. Covered by the new test (K-H2 p = 0.040, interval [−0.0059, −0.0001], threshold None → "not resolved
  after Holm").
- **R2-3 (procedure).** b2k_analysis.py 8daa84d3… → 98fe0233…, test_b2k.py f949b3a3… → ae6299aa…, updated in PLAN.md
  "Code and data pinned by this plan" and in the table below. runner.py (210923c4…), validate_main_configs.py,
  selection.json, precheck.json and the M6 output are unchanged; no runner or precheck-path file was edited, so the
  precheck stays valid and was not re-run. Checks after the last edit: `pytest test_b2k.py` **49 passed** (python -B,
  nice 10, basetemp in the scratchpad; supersedes the count of 48 above); `b2k_analysis.py --selftest` passed (same
  values as above); `--stored-table` written to the scratchpad reproduces stored_kl_table.json and its stdout byte for
  byte; on a scratchpad copy of PLAN.md + precheck.json + selection.json with a PLAN.sha256 written there,
  `b2k_analysis.frozen_identity` accepted (plan 5daf07a4…, runner 210923c4…, analysis 98fe0233…, tree ffe9dfef…;
  supersedes the round-1 simulation values above) and `runner.launch(8)` accepted with pool_run replaced (72 tasks,
  T1 first, lr 0.00116 / 0.000779, nothing run); a copy whose PLAN.md carries the old analysis hash 8daa84d3… is
  refused by both; the analysis on the empty scratch campaign refuses ("incomplete: 0 of 72"). PLAN.md's status line
  names the round-2 corrections. No PLAN.sha256 / PLAN.utc was written in this folder.

Files changed or added (sha256 after the last edit):
| File | Before | After |
|---|---|---|
| `runner.py` | 96bb8b088c1221605d7f9400e407a93a896adcc77d8d7400dfeb5c8434d31e68 | 210923c4e93c9fa6f0cb67e74ac0b0f28f83246a7a4f351040c6f1b0c2acc3f9 |
| `b2k_analysis.py` | 8349b5869ad1d424c092d8a3fe3d3ec12c43a8728fdb54dbaa76339cf2e72d33 | 98fe023328dfc909ee3e9b44e9b5bf2122373873c45986b519564c3dbc0e13af (round 1: 8daa84d3…) |
| `test_b2k.py` | 793e1ab1d220e47563b5057f494c148ba2da8303538e6bf9313dedf1d5d99437 | ae6299aa60ec79f8f71906bb7a1e31526a97fe96f94b67bbc1b36e41bf80810c (round 1: f949b3a3…) |
| `validate_main_configs.py` | new | 28104f64a8a31d8de20cffb5148f6b8cfd120e2625ed16356573486554fac7eb |
| `PLAN.md` | new | 5daf07a4acb47549882a2476ec30c0a663c15fed0ab1108fc03c92d9cc7eda96 (as delivered after round 2; round 1: b40f60f7…; the lead's PLAN.sha256 is authoritative) |
| `precheck.json` | 1420470974aa9b3100c5d002888dfb7471a7988801ab06246eb468deb09a122b (moved) | 641fb8fb9012b72f09f34e4558982e46e2c1503b9873ec8ce2dc4008f21f07d3 |
| `precheck.log` | moved | bede08fcf09d748eb8e7d67c0534083dc583c1099113d4a27acd7732b816d67b |
| `precheck/` | moved | new precheck runs and records |
| `main_configs_validation_placeholder.json` | new | 1278516fca7899a2a7da3f4d27ad81d9a9aeab496c4d8028c1dbe5724e975a55 |
| `main_configs_validation_placeholder_stdout.txt` | new | 230c54a71b79ae02daca4d515f6c855085e5bfe7a9e4576f017a837b13815540 |
| `runner.lock` | new | lock file (content: pid/time of the last holder; only the flock state matters) |
| `precheck_prev_20261025T120553Z/` | new | the previous precheck/, precheck.json, precheck.log and `runner.py.96bb8b08` (archival) |
| `README_BUILD.md` | 6d79f65b80a2ec557b1e37037f7084afb6b7c8b2bb7b627d5c228568dd7babf1 | this section appended (round 2 added above) |

Unchanged: PLAN_DRAFT.md (ce68473d…), REVIEW.md, selection.json (f819d712…), selection_stage1.json (d3e7dc7e…),
stored_kl_table.json (610181a6…), pilot/, pilot.log, precheck_attempt1*.

Next steps for the lead (supersede "Next steps" above):
1. Read PLAN.md; then, in this folder: `sha256sum PLAN.md > PLAN.sha256` and `date -u +%Y-%m-%dT%H:%M:%SZ > PLAN.utc`.
   Do not edit runner.py or b2k_analysis.py afterwards (the launch and the analysis refuse if PLAN.md does not record
   their current sha256; an edit to runner.py also needs a new precheck).
2. Optional (M6 with the real hash): `nice -n 10 /home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python -B
   validate_main_configs.py` (writes `main_configs_validation_plan-<hash8>.json`).
3. Launch: `cd /home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b2k && setsid nohup nice -n 10
   /home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python -B runner.py launch --workers 8 >> launch.log 2>&1 &`
   (at most 8 workers while other campaigns run; idea_dq_posthoc_long2's pilot was running with 2 workers at 12:00Z).
4. Relaunch only after stopping the launcher's whole process group (`kill -- -<PGID>`) and confirming that no
   `runner.py` process has this folder as its cwd (`for p in $(pgrep -f "runner.py launch"); do readlink
   /proc/$p/cwd; done`; a bare `pgrep -af runner.py` also lists other folders' runners). The lock refuses it otherwise.
5. After all 72 records: `nice -n 10 …/python -B b2k_analysis.py` (refuses partial or inconsistent data), then report
   per PLAN.md.
