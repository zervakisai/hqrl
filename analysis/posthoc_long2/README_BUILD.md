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


## Fixes after review (2026-10-25)
A workflow subagent applied these fixes between about 12:00Z and 12:35Z. The review is REVIEW.md, and the lead's
decision on every item is binding.

**Hard limits kept throughout:**
- No main run was launched, and the pilot was not stopped or signalled. The pilot is launcher PID 2689581 with 2 workers,
  still running on the original code in memory.
- No pilot or main-run outcome was read: no J, reports, entropy, S or AUC. Only counts, status and timing were looked at.
- The idea_dq tree was not touched (ffe9dfef… before and after).
- No PLAN.sha256 or PLAN.utc was written.
- At most 2 processes of this agent ran at once, at nice 10. This includes the precheck re-run with 1 worker.

The final plan is **PLAN.md**, which the lead hashes. PLAN_DRAFT.md is unchanged.

Provenance copies of the files as they stood before the fixes are in `provenance/`:
- runner.py ae36bc0e… — the code the pilot runs. It was reconstructed byte-exactly from the edited file and its hash
  verified.
- e4_analysis.py 0b87a35c…
- tests/test_e4.py e03a9892…
- README_BUILD.md cf8ef0c1…

### Blocking items
- **B1. Pilot eligibility.** Done, in a new `pilot_select.py`; the pilot code path is untouched.
  - Eligibility is now behavioural: a candidate must complete both development blocks, and its development-suite
    timely-credited reports per episode must exceed 0.012 at 80,000 and at 100,000 decisions in both blocks.
    PLAN.md cites LONG's PLAN_LOG of 2026-10-24T16:37:30Z for the 0.012 level.
  - S is as before.
  - Two eligible candidates within 0.0056 → 0.01, the larger entropy.
  - None eligible → 0.01, a fallback fixed now.
  - The mean entropy over the last 20 updates is descriptive only.
  - PLAN.md corrects the collapse text:
    - collapsed blocks 5, 7 and 8 ended at 0.312, 0.372 and 0.308 nats, and non-collapsed block 6 at 0.077 (I checked
      these against long_descriptive_post.json);
    - "collapsed to a near-deterministic policy" is replaced by the behavioural statement.
  - PLAN.md records the change as a deviation from the draft's "fixed before the pilot starts" clause. The rule changed
    on 2026-10-25 after the review, while the pilot was running and before any pilot result was read, and the original
    rule is quoted.
  - `runner.py pilot-select` is retired: it refuses and points to pilot_select.py.
  - The old rule's code (`select_entropy`, `pilot_summary`, `pilot_select`) was removed from runner.py and is kept in
    provenance/.
  - pilot_select.py has two further guards:
    - it refuses unless PLAN.md and PLAN.sha256 match and PLAN.md names its SHA-256;
    - it checks the integrity of every pilot run before computing anything: raw events, learning-rate schedule, entropy
      coefficient, observation version, the record's code and task, the 80 development evaluation episodes, and seeds
      below dq_development start + 7,000.
  - `python -B pilot_select.py --check` runs these checks alone and prints pass/fail only. Run on the real pilot with
    5 records standing, it gave **5 of 5 standing runs passed (3 runs still running, listed as absent)**. It computes and prints no selection statistic, and I
    read none.
- **B2. Threshold.** Done in e4_analysis.py.
  - The holds / restated / withdrawn readings use θ = min(0.103, J_rule/2) = 0.103.
  - P3 reports the blocks at or above 0.103 and at or above 0.10566.
  - The same reading at the paired 0.10566 is also written, labelled descriptive and not of record.
- **B3. Failure rules.** Done in runner.py and e4_analysis.py.
  - (a) A P4Error "block b leaves the <namespace> namespace", for the run's own block, is classed `namespace_end`: an
    algorithmic failure, never restarted (`runner.classify`).
  - (b) Any other non-algorithmic failure: the run and its record are moved to interrupted/<utc>/ (kept), logged in
    restarts.jsonl, and restarted once, at once, from scratch. A second failure stands and counts as a code failure. A
    relaunch also restarts a first failure left by a dead launcher, and the analysis refuses while one is pending.
  - (c) The deadline read-out T is computed only over runs running or completed at the deadline.
- **B4. Freeze.** Done.
  - PLAN.md records the SHA-256 of runner.py, e4lib.py, pilot_select.py, e4_analysis.py and rule_reference.json (and
    rule_reference.py).
  - `launch`, pilot_select.py and e4_analysis.py all refuse unless PLAN.md names them.
  - DEADLINE_UTC is still 2026-10-30T12:00:00+00:00, equal to the plan's deadline; a test checks this.
  - e4_analysis.py now refuses on any failure of these checks:
    - raw-event integrity: one arm_start, arm_end iff ended through train_arm, contiguous updates and episodes,
      consistent cumulative counts, no duplicated line, each grid point once and in order, each evaluation after its
      checkpoint;
    - (i) point-0 evaluation bitwise equal to LONG's;
    - (ii) training (cell_id, seed) sequence equal to LONG's on the overlapping episodes;
    - obs_version and 270 inputs;
    - entropy equal to pilot/selection.json, in the record, the task and every update;
    - learning-rate schedule on every update;
    - protocol_freeze_sha256 equal to sha256(PLAN.md), in the config, the task and the arm_end;
    - record code hashes equal to precheck.json's;
    - the logged launches.

### Minor items, as decided
- **Item 1. Timing.**
  - PLAN.md re-estimates 20-30 h of wall time for the 12-run wave under load.
  - Checked against LONG's records: 18.9-26.6 h of wall time per run, 3.5-10.1 h of CPU time, and 15.6-18.3 (summed walls) h to
    100,000 decisions.
- **Item 2. Single instance.**
  - `runner.py pilot` and `runner.py launch` take a non-blocking exclusive flock on runner.lock, which their workers
    inherit.
  - Before that, they scan /proc for any process running this folder's runner.py with pilot or launch, whatever its
    code version. The scan is read-only and signals nothing. It therefore also catches the current pilot launcher,
    which predates the lock.
  - PLAN.md states the relaunch procedure.
  - `runner.py pilot` also refuses once pilot/selection.json exists.
- **Item 3. OOM and stall watch.**
  - PLAN.md has the watch note.
  - `runner.py status` now shows counts, status and timing only: records by class, the age of the last write, the last
    evaluated point and the live launchers.
  - The old `status` printed the last update's entropy, so I did not run it.
- **Item 4. Peak versus final.**
  - The final checkpoint (300,000, or T under the deadline read-out) is of record.
  - Intermediate checkpoints at or above θ or J_rule/2, and the highest block J at any checkpoint, are reported
    descriptively.
  - There is no checkpoint selection.
- **Item 5. Deadline and full read-out.**
  - If every run ended by the deadline, the full read-out is of record. Otherwise the deadline read-out is of record.
    Its T comes from the files (checkpoint mtime plus the evaluation's wall_s; arm_end utc or the record's mtime), so the
    time at which the analysis is run does not change it.
  - A later full read-out is supplementary (`--supplementary-full`).
  - e4_analysis.py refuses to overwrite an existing output unless given `--force`; the old file is kept, and every
    write is logged in e4_analysis_log.jsonl.
  - Every reading string uses T.
  - `--provisional` is refused on this folder.
- **Item 6. Holm.** With only one head having P1, Holm runs over it with the family size kept at 2, i.e. at 0.025.
- **Item 7. Precheck (c) values.** These are now disclosed in PLAN.md.
- **Item 8. Wording.**
  - "information-matched" became "evidence-count-augmented" in the plan, in runner.py's docstring and in the analysis
    reading strings.
  - e4lib.py keeps one code comment with the old word. It is unchanged so that the pilot and the main runs share
    e4lib 9d72adf6.
  - No run directory name changed.
  - P2 is labelled an estimate, not a test.
  - Added: per-head t intervals for J_T(E4) − θ and J_T(E4) − J_rule/2.
- **Item 9. Records.** The checkpoint identity's feature_names was **not** fixed: it would change e4lib.py, which the
  running pilot uses, so it is disclosed in PLAN.md along with the lr-schedule and `e4lib.configure` notes.
- **Item 10. Kept copies.** Interrupted and failed copies are kept (move_aside) and logged in restarts.jsonl. The
  analysis compares each with the restart over the common prefix of events, without timing fields, and reports the
  result.
- **Item 11. Process.** Noted; the lead accepted the pilot design.

### Tests and checks run
- `pytest tests/`: **42 passed, 36 s**.
  - The 12 e4lib tests from test_e4.py remain.
  - test_e4.py's draft selection test was removed; the rule moved.
  - New tests are in tests/test_fixes.py:
    - runner: classes, the lock, the launcher scan, restart once / second failure stands / namespace end not
      restarted, crash and first-failure restart on relaunch, counts-only status;
    - pilot_select: every branch of the amended rule, plus end-to-end on a synthetic pilot (PLAN refusal, selection,
      no overwrite, the pending-restart refusal, the standing second failure, integrity refusals);
    - e4_analysis:
      - integrity on stored LONG runs and mutations, the schedule check on the precheck (c) run, partial lines;
      - B2 thresholds, T in readings, Holm with family size 2, peak versus final, the deadline constant;
      - end to end on LONG stand-ins in a temporary copy: the full read-out, the no-overwrite and --force log,
        provisional refused here, seven refusal mutations, the deadline read-out at 180,000 with the before-deadline
        refusal, namespace_end with LOCF, and kept-copy prefix comparisons.
- `e4_analysis.py --selftest`: passed: LONG's stored 300k means recomputed exactly (0.0055126, 0.0437964); rule reference verified; every LONG run passes the new integrity checker; reading threshold 0.103.
- **Precheck re-run.** Old outputs are in precheck_prev_2026-10-25T121716Z/. The re-run was `runner.py precheck
  --workers 1`, nice 10, alongside the pilot, from 12:17Z, 724 s: **passed (a), (b) and (c) for both heads**.
  - runner 9601f88d8f88934ceb61445c1de9a134c385302f2a52872932c623628bd0046e, e4lib 9d72adf6….
  - Bitwise determinism against the earlier precheck: every logged event of the four precheck runs equals the first precheck's in every trajectory field. Only the timing fields and the checkpoint files' SHA-256 differ, because torch.save output is not byte-reproducible; the analysis's copy comparison ignores both (tested).

### Files changed (new SHA-256)
| file | change | SHA-256 |
|---|---|---|
| `PLAN.md` | new: the final plan (the lead hashes it; no PLAN.sha256/PLAN.utc written) | `b718b4f9597b3c1aecca59df3885055631c62c2c1bf7367702f56509d1bbb7dc` |
| `runner.py` | lock and launcher scan, classify (namespace_end), restart once, counts-only status, pilot-select retired, launch checks; per-run code path byte-identical | `9601f88d8f88934ceb61445c1de9a134c385302f2a52872932c623628bd0046e` |
| `pilot_select.py` | new: the amended selection rule (B1) and the pilot integrity checks | `c47cb2521fd6975472190ac847602f56d64c07354ad912539ff113a6e2b621ca` |
| `e4_analysis.py` | B2 threshold, B3 classes and the deadline T, B4 checks, the minor items | `b4402cf435b81f8ca6f1f7d63a60a557c9093c90c59aade867bb88c03c47687b` |
| `tests/test_fixes.py` | new: 30 tests of the fixes | `6028493b5e6d2fda571b40d890c77c0a7217f139d0d5c4e212528b86818f6831` |
| `tests/test_e4.py` | the draft selection test removed (rule moved); 12 e4lib tests unchanged | `226b77ab96b02e076bfdd01cc57cb748cb2f99148593d2cf59d3e67c3dabfe92` |
| `tests/pytest_output.txt` | new: 42 passed (the builder's output kept as tests/pytest_output_before_fixes.txt) | `c3f54708d4795a861f03e91e818647ba254a350dc13e8bcb1fbd923cd4126f2e` |
| `precheck.json` | precheck re-run of record (old one in precheck_prev_2026-10-25T121716Z/) | `ddbb7cd6329162129fc4b29139c1c3d0c75592bbe43fe2682775d6cc724bc067` |
| `precheck.log` | its log | `a63511e3d52c6fd0fee170cfd9707e9b8422b1fc3167b34456a8bc80809dfefe` |
| `provenance/runner_at_pilot_launch_ae36bc0e.py` | new: byte copy of the runner.py the pilot runs | `ae36bc0ea97d06f86ec0dd50b8b943594b314b2320177cd7e576b6f6ab63829a` |
| `provenance/e4_analysis_draft_0b87a35c.py` | new: byte copy of the draft analysis | `0b87a35c2b0904ad1cb10a03c9a8d62ffe4f48ce0eb7f585ad4eba99077ba916` |
| `provenance/test_e4_draft_e03a9892.py` | new: byte copy of the draft tests | `e03a989206178e08ad74b95a82086a2360861f2654dc1b3197a26fbaaeb2a16a` |
| `provenance/README_BUILD_before_fixes_cf8ef0c1.md` | new: byte copy of this file before this section | `cf8ef0c105d2a1532c192aeb51165f303b084d1f87f42607a81bbb20b62e8e78` |
| `e4lib.py` | unchanged | `9d72adf6b5df54fa3c3f5d2074430ba037a097ee8bbda972d5b0923ef1b8d6c6` |
| `rule_reference.json` | unchanged | `5b8b8d71a44518c48452154b083dd5eeff809cf9e078214e7136ec38d0269a3d` |
| `rule_reference.py` | unchanged | `ae4dd5856e6d06972fa79f020d886f077e1810afc67f3c0e503eeebab6ef7cf2` |
| `PLAN_DRAFT.md` | unchanged | `40892991683d73fc554781041c5b42592d6db3b0eb9e8337b7dbbb9b6fc90bc6` |

Rows superseded in round 2 (below): `PLAN.md`, `e4_analysis.py`, `tests/test_fixes.py` and `tests/pytest_output.txt`.
The hashes in the table above are those of round 1.

### Round 2: defects found by the independent verifier (2026-10-25, about 12:50Z-13:10Z)
A workflow subagent fixed all four defects. It kept the same hard limits:
- No main run was launched, and the pilot was not signalled.
- No pilot or main-run outcome was read. The pilot was looked at only through `runner.py status`: 6 of 8 records
  completed, two runs in progress.
- No PLAN.sha256 or PLAN.utc was written.
- At most 2 processes of this agent ran at once, at nice 10.
- The idea_dq tree, runner.py, e4lib.py, pilot_select.py and rule_reference.json were not touched.
- The files as they stood before round 2 are in the scratchpad folder fix_e4_long2/round2_before/. The round-1
  e4_analysis.py (b4402cf4…) was never frozen.

**Defect 1 (B2/B3, P3 under failures).** The systems reading ignored blocks and heads with no value at T. The
verifier's two synthetic cases both read "holds": C-strong failing in all 6 blocks after 0.12 at 60,000, and two
C-strong blocks failing after 0.15. Reproduced, then fixed.
- PLAN.md now has the rule "P3 when blocks lack a value at T":
  - "holds" requires all 12 blocks to have a value at T, every one below θ;
  - with any failed block, the three readings are applied to the values at T (over the heads that have one) and with
    the failed blocks' last evaluated J carried forward (LOCF), and the more severe of the two is of record
    (withdrawn > restated > holds);
  - a failed block whose last evaluated J is at or above θ counts as a block that reached half the rule;
  - if both give "holds", the reading is "not computable" (a head with no value at T) or "qualified", never "holds";
  - every such reading names n per head, each failed block (class, last evaluated point and J, or "no evaluated
    point"), every last J at or above θ, and both underlying readings;
  - the same rule gives the descriptive reading at J_rule / 2.
- e4_analysis.py: new `three_way` and `systems_reading(at_T, without, θ, T)` returning (category, text). There are new
  output fields `systems_reading_category`, `blocks_without_value_at_T`, `failed_blocks_with_value_at_T` and, in P3,
  the failed blocks whose last evaluated J is at or above each threshold.
- The verifier's cases now give "withdrawn" (case 1, on LOCF, with "C-strong n = 0 of 6" and "not computable at T for
  C-strong") and "restated" (case 2, naming blocks 4 and 5 at 0.15 at 270,000).

**Defect 2 (deadline read-out, failed run with J_T).** Plan and code disagreed, and the code was inconsistent with
itself (the run was in `ok` and also counted toward the 3-failure rule).
- Decided rule, stated in PLAN.md under "Failures": a block has a value at T when its run evaluated T (by the deadline
  under the deadline read-out), even if the run later failed. That evaluation belongs to the same deterministic
  trajectory, just as a run still running at the deadline enters with its J_T.
- A "failed block" is one without a value at T. P1's 3-failure rule, the LOCF sensitivity and the P3 failure rule count
  only such blocks.
- The main-runs rule (b) now says a standing code failure is excluded only at read-out points it did not reach.
- The deadline stopping rule notes that a run failed before the deadline still has its value at T.
- e4_analysis.py:
  - `no_value` is every block without T. It refuses if such a block is not a failure, as an invariant.
  - P1 is computed iff fewer than 3 blocks lack a value.
  - The P1 string is "failed in k of 6 blocks (no value at T decisions)".
  - Every run's curve now stops at T in every read-out. Before, only running and completed runs were cut under the
    deadline read-out.

**Defect 3 (slope).** The plan listed E4 − LONG on the last-three-checkpoint slope, but the code had only E4's.
- The code now computes `slope_last3_per_30k` = {checkpoints, E4, LONG, E4_minus_LONG}. It is the least-squares slope
  per 30,000 decisions over the last three grid points up to T, as long_analysis.py computes it.
- PLAN.md says so.
- A test checks that LONG's slope reproduces long_results.json's `slope_last3_per_30k` mean when the stand-ins are LONG.

**Defect 4 (deviation record).** PLAN.md now quotes the draft header clause: "The pilot section below must not change
once any pilot result has been read; if it is changed before that, the pilot is discarded and re-run." By the lead's
decision it is read as applying to the pilot runs, which are unchanged, and not to the selection statistic, so the pilot
is not re-run.

**Tests and checks (round 2).**
- `pytest tests/`: **47 passed** (tests/pytest_output.txt; round 1's 42-pass output is kept as
  tests/pytest_output_round1.txt).
- New tests in tests/test_fixes.py:
  - `test_p3_failures_never_give_holds`: the verifier's two cases, qualified, not computable, a failure without any
    evaluated point, withdrawn on the values at T with a failure, and plain "holds" unchanged;
  - `test_three_way_core`;
  - `test_failed_run_with_a_value_at_T_counts`: 3 failures after T give n = 6 and P1 computed; 3 failures before T give
    P1 not computed and "qualified"; a non-failed block without T is refused;
  - `test_slope_last3_for_E4_LONG_and_difference`;
  - `test_deadline_readout_failed_run_keeps_its_value_at_T`: end to end on LONG stand-ins. A namespace-end failure
    before the deadline after 240,000 keeps its J at T = 180,000, does not enter T, and its curve stops at T.
- Two existing tests were extended: the namespace-end test now asserts the reading is not "holds" and names
  "C-matched n = 5 of 6", and the full read-out test checks the LONG slope against long_results.json. The P1 string
  assertion was updated.
- `e4_analysis.py --selftest` passed: LONG's 300k means recomputed exactly; rule reference verified; reading threshold
  0.103.
- On this folder, the analysis still refuses (no PLAN.sha256) and wrote nothing.
- `runner.plan_names_frozen_hashes()` is True for all five frozen files.
- **No precheck re-run was needed.** No runner.py, e4lib.py or other precheck-path file changed; e4_analysis.py is not
  on the precheck path. precheck.json (ddbb7cd6…) stands for runner.py 9601f88d… and e4lib.py 9d72adf6….

**Files changed in round 2 (new SHA-256)**
| file | change | SHA-256 |
|---|---|---|
| `e4_analysis.py` | defects 1-3 | `a24467a540f43faacf32b0b89e5e33cba972d8a3a21f9ce4c8a895f27de257ee` |
| `PLAN.md` | defects 1-4, the new e4_analysis.py hash, and the header note (the lead hashes it) | `a0d4ad7c11741fb351512c270f626df9c96688afb2d9830205d85998b6582cfd` |
| `tests/test_fixes.py` | 5 new tests, 2 extended | `2ca7bb81e7c26877d8ee59a9ed9823acde5debb8ed416f9227556526f378f778` |
| `tests/pytest_output.txt` | 47 passed | `6774b1d634ccb397fe7934d77915330f790b3b5b20c8837f4c35d41611bda0e2` |
| `tests/pytest_output_round1.txt` | new: byte copy of round 1's output | `c3f54708d4795a861f03e91e818647ba254a350dc13e8bcb1fbd923cd4126f2e` |
| `runner.py`, `e4lib.py`, `pilot_select.py`, `rule_reference.json`, `rule_reference.py` | unchanged | as in the table above |

### For the lead
1. Read PLAN.md, then hash it: `sha256sum PLAN.md > PLAN.sha256` and `date -u +%Y-%m-%dT%H:%M:%SZ > PLAN.utc`.
2. When the pilot has its 8 records and its launcher has exited, run `python -B pilot_select.py`.
3. Then run `nice -n 10 python -B runner.py launch --workers 12`.
4. After the runs, run `python -B e4_analysis.py`.

Any edit to a frozen file after hashing invalidates PLAN.md. The launch, pilot_select.py and the analysis then refuse.
