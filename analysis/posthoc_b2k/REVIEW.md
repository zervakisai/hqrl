# Independent review of E3 (matched per-update KL, B2 protocol): idea_dq_posthoc_b2k

Reviewer: independent subagent, 2026-10-25 (about 11:50Z). Read-only. I edited none of the builder's files; this
REVIEW.md is the only file I wrote here. My scratch scripts are in
`/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad/b2k_review/`.

**Verdict: FIX FIRST.** The fixes are small and all belong before PLAN.md is hashed. The pilot, the selected learning
rates (7.79e-04 twin, 1.16e-03 HQ) and the precheck are sound and do not need redoing. The precheck only needs re-running
if runner.py is edited (that edit is optional, see M7/M8).

## What I checked, and the result

| Check | Result |
|---|---|
| runner.py sha256 = precheck.json `runner_sha256` = pilot launches.jsonl = every pilot record | `96bb8b08…` everywhere: OK |
| Live idea_dq tree (`code_version.tree_hash`, which excludes `__pycache__`) | `ffe9dfef…`: OK. No `runs/` directory exists, so no main run has started |
| Stored comparators: B2 `quantum__b*`, CMLR `C1_bottleneck__b*`, LR `L1/L2` | 36/36 completed each. Same configuration (study dq_b2, role dq_b2, budget 30,000, 11-point GRID, suite dq_b2_evaluation, sample read-out, 512 transitions, entropy 0). Same tree `ffe9dfef` (B2 records, cmlr/lr launch logs) |
| Main configs vs plan | `config_main` = `dq_b2.config_for(arm, b, STUDY_B2)`, changing only `ppo.learning_rate`, `candidate`, `note` and `protocol_freeze_sha256`. Evaluation, grid and budget are B2's. Constant Adam lr; no decay, no freeze, no KL controller. T1 = quantum, T2 = surrogate625, lrs taken from selection.json. Order: blocks ascending, HQ first |
| Twin | Built by `dq_b2.dq_build_model` (exact transplant of the block's HQ). P4 parity (`twin_trajectory_parity`) runs before `train_arm`; any parity error becomes a failure record before training (fail closed). The static record is attached, as in B2 |
| Seeds do not depend on candidate/note | The precheck reproduced B2 block 0 bitwise with the new candidate label: evaluation 2/2, updates 6/6 (max relative difference 0.0), episodes 77/80, arm_start, and 17/18 initial tensors |
| Pilot (18 records, re-read) | All completed. Study dq_b2_smoke, `evaluation` None, **no `evaluation` event in any metrics.jsonl**. Training seeds 20262011000-20262019736, which lie in dq_development (index 10: [20261921000, 20262021000)). Blocks and lrs exactly as the plan's table |
| Selection, recomputed independently from the pilot metrics | twin raw 7.786e-4 -> `7.79e-04`; HQ raw 1.1635e-3 -> `1.16e-03`. Both identical to selection.json |
| PLAN_DRAFT.md vs the pilot-launch snapshot (`diff`) | Only these differ: the two lr placeholders, the K-H1 precision sentence and the appendix. The reading rules and analysis text are unchanged since 07:51Z |
| Outcome reads before selection | The builder's scratch scripts (`b2k/kl_stats.py`, `pilot_sim*.py`) read no AUC, J or evaluation data. `pilot-select` reads update `approx_kl` only. Stored AUCs were read only for the selftest and the stored table (results of record, not main-run data) |
| Calendar cut-off | `require_open` appears only in `dq_b2.run_one/launch/smoke`, in `dq_transplant.main` (the CLI) and in other modules this runner does not call. `train_arm`, `dq_build_model`, `make_twin`, `certify_static` and `twin_trajectory_parity` contain none. Test 17 also covers this |
| `pytest test_b2k.py` (python -B, nice 10, basetemp in scratchpad) | 17 passed in 6 s |
| `b2k_analysis.py --selftest` | Passed: B-H1 -0.00866 (p 0.0017), B-H2 -0.00777 (p 0.0012), targets exact |
| Timings in README_BUILD | Recomputed from the records. Pilot rollout 53.7 s (HQ) and 41.2 s (twin) per 1,000 decisions; precheck 30.2 / 18.8 s per 1,000 and 1.04 / 0.66 s per evaluation episode. All match |

## Blocking: fix before PLAN.md is hashed

**B1. No ceiling on dropped blocks; no not-estimable or degenerate rule.**
- The plan says only that a failed run's block is "dropped from that contrast only". `b2k_analysis.paired` tests any n.
- B2's frozen analysis, which these contrasts re-examine, fixes three rules (dq_b2_analysis.py lines 30-32, `MAX_DROPPED = 3`, `t_test`, `holm`):
  - more than 3 of 36 blocks dropped makes the contrast NOT ESTIMABLE (not rejected);
  - a non-estimable contrast is left out of the Holm ordering, but the Holm denominator stays at the family size;
  - degenerate sd = 0: mean 0 gives p = 1; a non-zero constant gives p = 0.
- E3 instead returns `p = nan` and sorts on it (`holm()` with a nan key).
- Survivor bias is a real risk without a ceiling: for example, several algorithmic failures at a new learning rate would still be tested on the remaining blocks.
- Fix: copy B2's three rules verbatim into PLAN "Outcome and analysis" and into `b2k_analysis.py` (TOST: "not estimable"), and add a unit test.

**B2. The analysis does not verify the new runs' integrity, so a crash/relaunch accident would corrupt the data silently.**
- `b2k_analysis.main` reads AUC from the last line of metrics.jsonl and averages every `update` line. It never checks:
  - (i) that each T1/T2 record's config equals `runner.config_main(arm, b, lr_selected, PLAN_SHA256)` (canonical), with `ppo.learning_rate` = selection.json, `plan_sha256` = PLAN.sha256, `runner_sha256` = precheck's, and the tree `ffe9dfef`;
  - (ii) the raw events: exactly one `arm_start` (first) and one `arm_end` (last); update indices 1..U contiguous with U = arm_end `updates`; evaluation points = GRID, once each; AUC recomputed with `p4.learning_curve_auc` equal to arm_end's and the record's; record status = summary status;
  - (iii) B2's TWIN rule: a completed twin run enters K-H2 only with a failure-free static certificate and P4 <= 1e-9. Today `twin_parity()` only summarises.
- Why it matters (question 4):
  - `JsonLines.write` re-opens `metrics.jsonl` by path for every event (p4_train.py:524-531). `launch` moves a run directory that has no record to `interrupted/` and restarts the run at the same path.
  - Suppose only the launcher parent is killed. With `setsid nohup`, a SIGTERM or SIGKILL to the parent does not stop the fork-pool workers. Or suppose `launch` is started twice. Either way, the surviving worker keeps appending into the restarted run's metrics.jsonl and checkpoints.
  - The runs are deterministic, so AUC would stay numerically right. But duplicated `update` lines bias the KL statistic, which decides "matched", and the event file is corrupt. The analysis as written would not notice.
  - A machine crash or hard reboot does NOT cause this; pairing by block name is otherwise robust.
- Fix: add (i)-(iii) as refusals (IntegrityError style, as in dq_b2_analysis), with a test on a doctored copy. This changes only the analysis, not runner.py, so the precheck stays valid.

**B3 (lead, at hashing). Pin the code in PLAN.md.**
- Write the sha256 of the fixed `b2k_analysis.py` (and of runner.py, `96bb8b08…` unless it is edited) into PLAN.md.
- The plan says the outcome and analysis are "fixed now; `b2k_analysis.py`", but only PLAN.md is hashed. The analysis script was edited during the pilot (09:13Z, the divide-by-zero guard) and nothing pins it.

## Non-blocking: should fix or state

- **M1. Descriptive contrasts carry p-values.**
  - The plan says "Descriptive (no tests)". But `twin_new_minus_twin_B2_1e-3`, `HQ_new_minus_*` and `B-H*_of_record_same_blocks` come from `paired()`, which keeps `p`; lines 197-199 remove only lo90/hi90.
  - The dry run in the scratchpad shows `p` in the output. Use `interval()` or remove `p`.
- **M2. The K-H2 "not resolved" reading is one-sided.**
  - "Not resolved" also covers an underpowered result whose 95% CI still contains B-H2 (-0.0078).
  - Add: report K-H2's CI beside B-H2. If it contains both 0 and -0.0078, the check is *inconclusive*: it neither removes nor confirms B-H2.
  - Add one sentence for "below zero but clearly smaller than B-H2" (partial attenuation).
  - Name the direct quantity, `twin_new - twin_B2_1e-3`: paired, the same initial policy, which is K-H2 minus B-H2 block by block. It shows how much of B-H2 the twin's step size accounts for.
  - The current wording leans against the paper's claim, so this is accuracy, not integrity.
- **M3. The one-sided readings for unmatched KL rest on an unstated premise.**
  - The premise: within an arm, AUC does not decrease as updates get larger over this range.
  - The stored data support it (from stored_kl_table differences): HQ 1e-3 < 3e-3 < 1e-2; twin 1e-3 < 3e-3 by about 0.0023; C-matched 3e-3 < 1e-2 by about 0.0035.
  - Without this premise, "HQ learned less although its updates were larger" is no answer to B-H1's own update-size alternative ("HQ's larger updates hurt it"). State the premise and the evidence for it in the plan.
- **M4. Disclose where the slope and candidate centres came from.**
  - The fixed slope `s` and the candidate centres come from stored *training-update* KL on the B2 blocks (dq_e2_training blocks 0-35, the main runs' own training seeds). They do not come from development data.
  - No AUC, J or evaluation data entered, which fits the intent of the brief's "development seeds/suites only" rule. Say so explicitly in Stage 1.
- **M5. Pilot configs bypass `p4.validate_config`.** Budget 30,000, grid [0] and blocks 90-98 are outside STUDY_SMOKE's frozen role (6,000, [0, 3000, 6000], block 90). That is harmless because `train_arm` is called directly, but precheck item (d) should state it.
- **M6. The precheck validated the configs at the wrong learning rates.** Precheck (c) validated the 72 configs at the pilot candidates (7e-4, 1.4e-3), not at 7.79e-4 / 1.16e-3, and `launch` does not re-validate. It is trivially equivalent; optionally re-run `static_checks`-style validation at the selected lrs from a separate script before launch.
- **M7. No single-instance lock and no relaunch procedure.** Add to README_BUILD: before any relaunch, stop the whole process group (`kill -- -<PGID>`) and confirm that `pgrep -af "runner.py (launch|pilot)"` is empty. An O_EXCL PID lock in `launch` is better, but it changes runner.py, so the precheck would have to be re-run (about 5 min, 2 processes).
- **M8. Records are written non-atomically (`write_text`).**
  - If a crash hits mid-write, the truncated record makes `launch` skip the run forever and makes the analysis crash on `json.loads`.
  - Declare in the plan: "a record that does not parse counts as no record; record and run directory are moved to interrupted/". Alternatively write a temporary file and `os.replace` it (a runner change: re-run the precheck).
- **M9. K-H1 is largely predicted by stored data.**
  - The stored pairing HQ 1e-3 - C-matched 1e-2 gives -0.0178 [-0.0230, -0.0125] at a KL ratio of 0.74, so K-H1 is mostly a confirmation. Its near-certain rejection lets K-H2 be tested at 0.05 under Holm. That is legitimate, but the paper should say it.
  - Also state why the K-H1 comparator is C-matched at 1e-2 (KL 0.0042). It keeps every arm at tested learning rates; C-matched cannot reach HQ's 0.0078 without an untested lr of about 3e-2.
- **M10. Expect HQ's achieved KL to land below target.**
  - On B2 blocks the stored-point prediction is R ≈ 0.84 (0.0031 × 1.16^0.842 ≈ 0.0035 vs 0.0042). That is inside the ±25% band, with a rough 5-10% chance of falling below 0.75.
  - This must not be "corrected" now: that would be selecting on B2-block data. Report it as is.
- **M11. Other declared limits, acceptable as stated:**
  - no post-update full-batch KL (the feasibility critique, missing[6]);
  - matching the mean does not match the distribution;
  - K-H1 and K-H2 are correlated with B-H1, B-H2 and Z1 through the reused stored runs.
- **M12. Load.** Launching 12 workers while the 10-worker campaign runs would oversubscribe the 16 threads. The wall-time estimate assumes a free machine.

## Answers to the five questions

1. **Does the code do what the plan says?** Yes: pairing, seeds, grid, evaluation suite, arms and lrs, the twin build and P4, constant lr, entropy 0. Segment rewards, freeze and decay do not apply to E3. The gaps are in the analysis's guards (B1, B2), not in the run path.
2. **Does any selection read forbidden outcome data?** No. The pilot ran no evaluation and the selection reads only `approx_kl`; I recomputed the selection independently. The disclosure in M4 is still needed.
3. **Are the contrasts, tests, margins and reading rules fixed, complete and honest?** Mostly. Missing: the dropped-block ceiling and the degenerate and non-estimable rules (B1). To improve: the "not resolved" and unmatched readings (M2, M3), and the descriptive p-values (M1).
4. **Would a crash or resume corrupt pairing?** A machine crash: no. Restarts are deterministic and pairing is by block name. A parent-only kill or a double launch followed by a relaunch: yes, silently for the KL statistic (B2, M7). A truncated record would block the run (M8).
5. **What would make a reviewer reject the result?** The analysis script not pinned to the plan (B3). Survivor bias without B2's ceiling (B1). An "update-size" reading that does not state its monotonicity premise (M3). The others are declared limits.
