# E4 build record: idea_dq_posthoc_long2

Built 2026-10-25 between 07:31Z and 08:20Z by a workflow subagent. **No main run was launched.** The development pilot
was launched after the precheck passed, as the task allowed, with 2 workers at nice 10. It is still running.
The idea_dq tree was never edited. Its live hash was ffe9dfef2b24… before the build, after the precheck, and at 08:13Z.
Every Python process ran with `-B`, so no bytecode was written into any tree.

## Files
| file | what it is |
|---|---|
| `PLAN_DRAFT.md` | Draft plan for the lead to review and hash. **Not hashed by me**: no PLAN.md or PLAN.sha256 exists. |
| `e4lib.py` | Patch layer (sha256 9d72adf6…). It swaps 4 p4_train globals per process and sets `p4.BLOCK_STRIDE = 10_000`. Contents: `WideLayout`/`WIDE` (254+16), `EvidenceEncoder`, `InfoRLMissionPolicy`, `build_model_e4` (LONG's model plus 16 zero columns), `ppo_update_e4` (linear learning-rate decay), `configure()`/`restore_originals()`, and the `e4_dev` development study. |
| `runner.py` | Subcommands `precheck`, `pilot [--workers 2]`, `pilot-select`, `launch [--workers 12]` and `status` (sha256 ae36bc0e…). |
| `rule_reference.py` | The paired rule reference (subcommands `run` and `verify`). |
| `rule_reference.json` | **Result written now**: the rule on the 80 dq_e2_evaluation episodes (sha256 5b8b8d71…). |
| `e4_analysis.py` | Planned analysis. Refuses partial data unless `--provisional` (output then goes to `e4_results_PROVISIONAL.json`). `--deadline-readout` is allowed only after the plan's deadline. `--selftest` reads no E4 run. |
| `tests/test_e4.py` | 13 unit tests. Output in `tests/pytest_output.txt`. |
| `precheck.json`, `precheck.log`, `precheck/` | Precheck result, log and runs. |
| `pilot/` | Pilot runs, records, `launches.jsonl` and a byte copy of PLAN_DRAFT.md as it stood at launch. |

## What was run and what passed
1. **Unit tests** (`pytest tests/`, 42 s, 1 process): **13 passed.**
   - The base build is bitwise equal to LONG's stored `update00000` checkpoint, for both heads.
   - The widened model equals the base model at init. With random extra inputs, max |dlogit| is 1.4e-16 and max |dvalue| is 4.9e-17. On real observations, replayed with each decision's own GRU state, the difference is exactly 0.
   - The global torch RNG state after the widened build equals the state after the base build.
   - The encoder was checked on a real rule episode (`uncertainty_fix` σ 1.5, dq_development realisation 620; 153 decisions, 8 reports, 18 inspections):
     - features 0-253 equal the frozen `obs-encoding/4-gnss` at every decision;
     - the 16 features equal the rule's own `_credited` and `_inspections` state;
     - at every rule REPORT the reported track's slot shows ≥ 2 viewpoints;
     - at every rule INSPECT the inspected track's slot shows fewer inspections than the cap.
   - The encoder was checked on learned episodes (C-strong 3 episodes, 129 decisions, 9 executed inspections; C-matched 2 episodes, 78 decisions, 7 inspections). The features match an independent re-implementation of the crediting and inspection counting.
   - The learning-rate wrapper:
     - is bitwise the frozen `ppo_update` in pass-through;
     - follows lr0·(1 − D/H) exactly and reaches 0;
     - logs `entropy_coef`.
   - The pilot selection rule is tested on synthetic cases: winner, tie band, collapse guard, guard lifted, failures.
2. **Rule reference** (`rule_reference.py run`, 07:36-07:54Z, 1,105 s, 1 process): **all checks passed.**
   - **J_rule = 0.211329** (cell-weighted). Per target-present cell: d2 nominal 0.2955, d2 tight 0.2452, d4 nominal 0.2935, d4 tight 0.2224. No-human cells: 0. Half the rule is 0.10566.
   - Pairing: the (cell_id, seed) sequence equals that of every stored evaluation event, with 0 mismatches. That covers B2 (108 runs, 1,188 events), LONG (12 runs, 132 events) and LONG-HQ (6 runs, 66 events). Every row also carries its `crn_sha256`.
   - Determinism: two episodes flown again were identical.
   - The same code path reproduced 2 stored E1 rule rows (e1_evaluation, indices 0 and 10) exactly: J, crn_sha256, decisions and fixes.
3. **Precheck** (`runner.py precheck --workers 2`, 07:55-08:10Z, 935 s): **passed** (`precheck.json`; runner ae36bc0e…, e4lib 9d72adf6…).
   - **(a) All changes off:** for both heads, stored LONG block 4 was reproduced exactly for 3,000 decisions. That is the point-0 evaluation, 6/6 update records, 74 (C-strong) and 69 (C-matched) episode records, and the arm_start model, identity and seeds.
   - **(b) Init:** LONG's initial model was rebuilt bitwise. Every tensor of the widened model equals the base except the zero columns. Max |dlogit| is 2.2e-16, and the RNG state is unchanged.
   - **(c) All changes on** (entropy 0.01, lr0 3e-3 decaying over 300,000), both heads completed, and:
     - point-0 evaluation, the 12 and 11 episodes of update 1, and the cell and seed schedule all equal LONG's; all seeds are in [start+40,000, start+50,000);
     - every update logged exactly lr0·(1 − D_u/300,000), from 0.003 at update 1 to 0.00297337 at update 6, with entropy_coef 0.01;
     - evidence features were populated: 713 of 3,211 C-strong decisions had a nonzero viewpoint feature and 435 a nonzero inspection feature; there were 0 feedback errors;
     - the final checkpoint has obs_dim 270 and optimiser lr equal to the last logged lr; the extra encoder columns became nonzero (max |w| 0.10 and 0.08);
     - the 3,000-decision evaluation ran with 80 episodes.
4. **Analysis**:
   - `--selftest` passed. It recomputed LONG's stored 300k means exactly (C-strong 0.0055126, C-matched 0.0437964) and verified the rule reference.
   - A dry run of the full analysis in the scratchpad used LONG runs as stand-ins for E4 runs, so no E4 data was involved. The complete, partial-refusal, provisional and deadline-guard paths all work. The collapse indicator flags LONG C-strong blocks 4, 5, 7, 8 and 9, which matches LONG's forensic finding.
   - On the real folder the analysis refuses, because 0 of 12 runs exist.
5. **Pilot launched** at 2026-10-25T08:12:07Z: `runner.py pilot --workers 2`, nice 10, detached, log `pilot.log`.
   - It runs 8 runs, block 6 first. My processes are the parent 2689581 (idle) and 2 workers.
   - Monitor it with `python -B runner.py status`.
   - To stop it, kill the parent and its workers. A relaunch skips completed runs and moves an interrupted run to `pilot/interrupted/` to restart it from scratch.
   - Health check at 08:15Z: both block-6 C-strong runs have obs_dim 270, salt (20260923, 41) and first training seed
     20261981000 (dq_development start + 60,000). Update 1 logged lr 0.003 and entropy_coef 0.003 and 0.01; rollout ran at
     about 56 ms per decision.
   - **I have not run `pilot-select` and have read no pilot evaluation J.**

## Timings measured under the current load
The load average was 17-20 on 16 threads: the B2F launch with 10 workers at nice 5, plus other agents' prechecks. My processes ran at nice 10.

| item | measured |
|---|---|
| training rollout, untrained to 3k (≈45-50 decisions per episode) | C-strong 50.3 s, C-matched 43.5 s per 1,000 decisions |
| PPO update | 0.56 s and 0.36 s per 1,000 decisions |
| learned-policy evaluation | 1.75-2.22 s per episode, i.e. 140-178 s per 80-episode point |
| rule episodes | 21.8 s per target-present episode, 4.0 s per no-human episode (mean 12.9 s; 135 decisions per target-present episode) |

Estimates, all inferred from the rates above:

| item | estimate |
|---|---|
| one pilot run | about 1.5-2.5 h |
| pilot (8 runs, 2 workers) | about 6-10 h, so it should finish roughly 14:00-18:00Z on 2026-10-25 |
| one main run at today's per-process speed | about 4.5-8.5 h (300 × 47-80 s, plus 11 × 160-600 s of evaluation) |

The main-run estimate grows if:
- 12 workers share the machine with other experiments (a 12 → 16 concurrency slowdown of about 1.34 was measured earlier in the study);
- the E4 policy becomes more rule-like: rule episodes are about 10× longer in wall time than current learned ones. In the worst case evaluation adds up to about 3 h per run, and training cost per decision about doubles.

## For the lead: how to go on
1. Review PLAN_DRAFT.md, then write PLAN.md and run `sha256sum PLAN.md > PLAN.sha256`. The launcher checks both. The stopping-rule deadline (2026-10-30T12:00Z) is a proposal; it also appears as `DEADLINE_UTC` in e4_analysis.py.
2. When the pilot's 8 records exist, run `python -B runner.py pilot-select`. It writes `pilot/selection.json` once.
3. Run `nice -n 10 python -B runner.py launch --workers 12`. It refuses unless all of these hold:
   - PLAN.md and PLAN.sha256 are present and match;
   - precheck.json passed for the identical runner.py and e4lib.py;
   - rule_reference.json passed its checks;
   - pilot/selection.json exists.
4. After the 12 runs, run `python -B e4_analysis.py` to write e4_results.json.
Any edit to runner.py or e4lib.py invalidates the precheck (re-run time is about 15 min at 2 workers). Edits to the plan do not.

## Open risks
- **The information match is partial.** k = 16: credited viewpoints and executed inspections per track slot. Not added:
  - the per-view novelty filter;
  - sticky abandonment;
  - the fix-attempt counter.

  The interface asymmetries also stay as they are: free RELOCALIZE (the "relocalize trap", 15-21 fixes per episode against the rule's 2.1) and voluntary RETURN masked. The feasibility scouts found these asymmetries, not the missing counts, to be the dominant gap. Even with the counts, E4 may stay far below the rule.
- **Confounding and power.** Information and re-tuning are confounded, and n = 6. LONG C-strong collapsed in 5 of 6 blocks, so P1 for C-strong may mostly measure whether collapse is avoided.
- **Pilot limits.**
  - Only the entropy coefficient is piloted; lr0 is fixed at 3e-3. The justification is in the plan, and it is a judgment call.
  - A 100k pilot (the first third of the schedule) cannot see collapse after 100k. The entropy guard (≥ 0.30 nats) is only a proxy.
  - The selection is noisy: 2 blocks × 2 points × 80 episodes.
  - The tie band of 0.0056 J favours 0.003.
- **The pilot ran before plan review.** If the lead changes the pilot section, the pilot must be discarded and re-run. PLAN_DRAFT now notes this. I recorded the draft's SHA-256 in precheck.json and in pilot/launches.jsonl (66520936…), and kept a byte copy. These are provenance only, not the plan freeze.
- **Development seed overlap.** The pilot seeds (`e4_dev`, salt (20260923, 41); dq_development training [start+60,000, start+80,000); realisations 600-609) avoid every dq_development range documented in idea_dq: D-1..D-6 realisations 0-21 and 100-431, and smoke block 90 with realisations 950-951. Other agents' concurrent post-hoc folders (gate, b1r, b2k) might use dq_development ranges I did not inspect. Any overlap is development-only, not an integrity issue.
- **No resume.** A crash restarts a 300k run from zero. Runs take several hours each, so a crash during the main campaign costs up to one wave. Avoid Chrome video during runs (the BAR1 freeze note).
- **Checkpoints need e4lib.** E4 checkpoints have obs_dim 270 and the identity `obs-encoding/4-gnss+ev16/1`. Re-evaluating them requires `e4lib.configure(True, …)` in the process.
- **How "paired" the rule is.** The rule reference is deterministic and is a single value, paired with E4 at the episode level. Intervals for J_E4 − J_rule therefore treat J_rule as fixed.
- **Two half-rule thresholds.** The paired threshold is 0.10566; LONG used 0.103. Both are reported. A block between them would read differently under each.
- **Read-out mode.** Only the sampled read-out is used, as in B2 and LONG. The rule is deterministic, and no argmax read-out is added.
- **Calendar cut-off.** `dq_runtime.require_open` stops working after 2026-10-31. It is never called on these paths: train_arm and run_policy_episode are called directly.
- **Measurement conditions.** All timings are from an oversubscribed machine at nice 10. They are not clean-machine costs.
