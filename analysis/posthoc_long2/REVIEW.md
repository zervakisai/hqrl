# Independent review of E4 (idea_dq_posthoc_long2), 2026-10-25 ~08:30Z

Reviewer: workflow subagent, adversarial review. Read: PLAN_DRAFT.md, README_BUILD.md, runner.py, e4lib.py,
e4_analysis.py, rule_reference.py, tests/test_e4.py, precheck.json, pilot/launches.jsonl, and the relevant idea_dq
code (p4_train.train_arm / schedule / evaluate / build_model, dq_b2.config_for / ARMS / STUDY_B2,
executive.EvidencePolicy, rl_policy.RLMissionPolicy, encoding.ObservationEncoder, smdp_ppo.ppo_update). I also read
the stored LONG records and metrics, which are already-analysed data.

What I ran: `e4_analysis.py --selftest`, 5 s, nice 10, 1 process. It passed and reproduced LONG's stored 300k means
exactly. I also ran sha256sum, a diff of the plan against its pilot-launch snapshot, and read-only parses of stored
LONG metrics. **I read no pilot J and no pilot entropy value.** From the live pilot I read only decision and episode
counts and rollout rates. Nothing was edited except this file.

Verdict: **FIX FIRST.** The code does what the plan says on pairing, and no selection reads the evaluation suite.
Four plan and analysis defects must be fixed before PLAN.md is hashed. All four are cheap. None of them requires
re-running the pilot, as long as nobody reads pilot results before the fix.

---

## Verified correct

- **Pairing with LONG.**
  - Main runs use STUDY_B2: the same salt, the INIT, ACTION and MINIBATCH streams, and the schedule stream.
  - BLOCK_STRIDE is 10,000, so training seeds are start + 10,000·b + episode, blocks 4-9, the same as LONG.
  - Evaluation uses the same 80 dq_e2_evaluation episodes and the same GRID. EVAL_STREAM is keyed by point index.
  - Compared with LONG's record, the precheck (c) config differs only in ppo.entropy_coef, the candidate/note
    labels, the freeze label and the budget/grid. The lr schedule and the observation change live in e4lib state and
    are logged per update and in the identity.
- **Patching.** p4_train resolves LAYOUT_V4_GNSS, RLMissionPolicy, build_model and ppo_update as module globals at
  call time, so the four swaps reach train_arm, run_learned_episode, evaluate and RolloutBuffer.
  - With maxtasksperchild=1 and fork, every run gets a fresh process, and configure() resets SCHED['done'].
- **Learning-rate decay.** Every parameter group is set to lr0·max(0, 1 − D/300,000) before each update.
  - D equals the cumulative policy decisions before the update; check_c confirms this exactly.
  - Update 1 runs at lr0.
- **Evidence features.**
  - The bookkeeping is EvidencePolicy._update_evidence and _evidence_feedback, unchanged. _update_evidence is
    idempotent on a re-asked observation, so the resample path is safe.
  - The base encoder allocates `layout.dim` and writes only at fixed offsets, so features 0-253 are unchanged; the
    tests confirm this.
  - The zero columns make the policy identical at init; the precheck confirms this on real rollouts.
- **Pilot isolation.**
  - The pilot study e4_dev trains on dq_development seeds [start+60k, start+80k).
  - Its development evaluation seeds are start + 1000·sci + r with sci ∈ {0, 2, 4, 6} and r = 600-609, all below
    start + 7,000.
  - The study has no dq_e2_evaluation suite. pilot-select reads only pilot records.
  - The pilot section of PLAN_DRAFT.md is byte-identical to the launch snapshot (66520936…). The diff touches only the
    provenance paragraph and the wording of the rule-reference paragraph.
- **Hashes and records.**
  - Hashes match README_BUILD: runner ae36bc0e…, e4lib 9d72adf6…, rule_reference.json 5b8b8d71….
  - precheck.json passed (a), (b) and (c) for both heads.
  - The idea_dq tree is ffe9dfef… before and after.

---

## Blocking (fix before hashing PLAN.md / before reading any pilot result)

### B1. The pilot's collapse guard is not valid, and the plan's justification for it is factually wrong.

PLAN_DRAFT calls 0.30 nats "LONG's collapse level". LONG's own data contradict this
(long_descriptive_post.json, C-strong):

| block | collapsed? (no timely reports, from) | last-update entropy |
|---|---|---|
| 4 | yes, 120k | 0.048 |
| 5 | yes, 90k | **0.312** |
| 6 | **no** | 0.077 |
| 7 | yes, 270k | **0.372** |
| 8 | yes, ~240k | **0.308** |
| 9 | yes, 240k | 0.203 |

- Three of the five collapsed blocks ended above 0.30 nats. The one block that did not collapse ended at 0.077.
- Single-update entropy is very noisy. LONG mlp b8 read 0.45, 1.12 and 0.24 at 50k, 100k and 200k; b4 read 0.05 at
  100k and 0.21 at 200k.
- Three or four of the five collapse onsets came after the pilot's 100k horizon.

As written, the guard would have passed most of LONG's collapsing trajectories and could reject healthy ones. The
plan's sentence "collapsed to a near-deterministic policy in 5 of 6 blocks" is also inaccurate: collapse here is
behavioural (no timely reports), not reliably an entropy event. This matters, because the E4 fix (an entropy bonus)
targets entropy.

**Fix.**
- Make eligibility the development-suite behavioural indicator: mean timely-credited reports per episode > 0.012 at
  both 80k and 100k in both blocks, which is LONG's forensic cut.
- Optionally add mean entropy over the last 20 updates, about 10k decisions.
- Correct the plan text.

Both quantities are already logged by the running pilot, so no pilot re-run is needed. Amend the clause "if the pilot
section changes, the pilot is discarded and re-run" so it applies to changes in the pilot runs, not to the selection
statistic computed from logged data.

**Decide the tie direction explicitly as well.**
- The selection noise (2 blocks × 2 points × 80 episodes) probably exceeds the 0.0056 tie band, so the band will
  usually decide.
- As written it picks 0.003: the weaker regulariser, and the one more exposed to the post-100k collapse the pilot
  cannot see.
- For a best-effort test of an objection to the authors' own conclusion, a tie should go to 0.01, or the band should be
  removed so the larger S wins. Otherwise a reviewer can argue the rule leans toward the conclusion.

To implement:
- Either put the rule in a separate `pilot_select.py`, since launch() needs only pilot/selection.json, so runner.py's
  hash and the precheck stay valid;
- or edit runner.py and re-run the precheck (about 15 min). Editing runner.py on disk does not affect the running pilot
  workers: they fork from the parent's in-memory code.

### B2. The half-rule threshold moved in the direction that protects the conclusion.

- The E4 readings use J_rule/2 = 0.10566, the paired reference.
- LONG's planned threshold, already in the manuscript, is 0.103.
- A block with J between 0.103 and 0.1057 would read "holds" under E4 but "reached half the rule" under LONG's rule.

The paired reference is methodologically better, but it was introduced after LONG's results were known, and it is the
more lenient threshold for the authors' claim. This is not hypothetical: LONG C-strong block 9 reached 0.1061 at 60k.

**Fix.** For the "holds / restated / withdrawn" readings, use min(0.103, J_rule/2) = 0.103, or require a block to be
below both thresholds before "holds". Report both. This is a one-line change in e4_analysis.py, where `half` drives
the readings, plus the plan text.

### B3. Infrastructure (non-algorithmic) failures have no rule, and block 9 cannot raise ScheduleExhausted.

- run_one turns every non-algorithmic exception into status 'failure', and resumable() never retries a run that has a
  record. Such exceptions include an OSError, an OOM, a P4Error, or a bug in the E4 wrapper.
- The analysis then refuses forever, because that run has no arm_end.
- In `--deadline-readout`, the failed run counts as "unfailed": `reached` excludes only 'algorithmic_failure'. Its last
  checkpoint therefore becomes T for **all** 12 runs.
- There is a further asymmetry. dq_e2_training is exactly 100,000 seeds wide, so block 9's range ends at the namespace
  end. When block 9 exhausts its 10,000 episodes, CellRuntime.schedule raises
  P4Error('leaves the namespace') before it reaches the ScheduleExhausted check (p4_train.py:393-397).
- So the same algorithmic event is an algorithmic failure in blocks 4-8 and an infrastructure failure in block 9.
- The probability is low: it needs fewer than 30 decisions per episode on average, and LONG used 2.5-4.9k episodes. The
  handling must still be fixed in advance.

**Fix (plan and analysis).**
- (a) A P4Error "leaves the … namespace" raised by the schedule counts as ScheduleExhausted, i.e. an algorithmic
  failure.
- (b) Any other non-algorithmic failure is treated like a crash. Move the run to interrupted/ and restart it once from
  scratch; this is deterministic, so pairing is unaffected. If the same failure recurs, report it as a code failure,
  exclude it with n stated, and fix the code only under an addendum.
- (c) The deadline T is computed only over runs that are running or completed.

### B4. Freeze the analysis together with the plan.

- PLAN.md should record the SHA-256 of e4_analysis.py, and of rule_reference.json, runner.py and e4lib.py, so the
  analysis cannot change after results. LONG did this with long_analysis_v1_<hash>.py.
- DEADLINE_UTC must equal the plan's date.
- Before freezing, add the free integrity checks the analysis currently lacks:
  - (i) every E4 run's point-0 evaluation is bitwise equal to LONG's point-0 evaluation for the same block and head
    (this follows from the identical init);
  - (ii) the (cell_id, seed) sequence of E4's training episodes equals LONG's for the overlapping episode numbers;
  - (iii) the arm_start obs_version is `obs-encoding/4-gnss+ev16/1`, the config entropy equals pilot/selection.json,
    the config protocol_freeze_sha256 equals sha256(PLAN.md), and the record code hashes equal the precheck's.

  Today the analysis takes whatever sits in runs/.

---

## Important but not blocking

1. **The timing estimate is about 3× optimistic.** README_BUILD gives 4.5-8.5 h per main run, extrapolated from an
   untrained 3k-decision run. LONG's own records, with 12 workers and load 21-33, show:
   - 18.9-26.6 h wall per run;
   - 3.5-10.1 h CPU per run;
   - 15.6-17.6 h wall to reach 100k.

   Plan on about 20-30 h for the one 12-worker wave, more if B2F is still running. The pilot may also run past the
   14:00-18:00Z estimate. The Oct 30 12:00Z deadline still looks reachable if the main runs start by about Oct 26,
   but there is little slack in a one-week paper schedule.

2. **There is no lock against a second launcher.** Running `runner.py pilot` or `launch` again while a launcher is
   alive does damage:
   - it moves the live run directories to interrupted/ and restarts them;
   - the original workers then crash on their next write and write 'failure' records.

   README_BUILD tells the lead to relaunch in order to resume, and a pilot is live now. Until a flock-based lock is
   added, check `pgrep -af "runner.py (pilot|launch)"` together with each process's cwd (`readlink /proc/PID/cwd`)
   before any relaunch.

3. **Memory and hung pools.** Free memory is about 11 GB, and 2 GB of swap is already in use. Twelve more workers need
   about 6.6 GB. If the OOM killer takes a worker, Pool.imap_unordered hangs and never returns. Watch for a run whose
   metrics.jsonl stops growing. The recovery is to kill the launcher and relaunch, which works correctly. Avoid Chrome
   video during the runs (the BAR1 freeze note).

4. **Peak versus final checkpoint.** LONG C-strong block 9 peaked at 0.1061 at 60k and had collapsed by 300k. E4 may
   show the same pattern. Fix the reading in advance:
   - an intermediate-checkpoint block at or above the threshold is reported descriptively;
   - no checkpoint is selected on the evaluation suite;
   - the paper says so.

5. **Deadline read-out and the full read-out.**
   - State which one is of record if all runs finish after the deadline.
   - e4_analysis.py overwrites e4_results.json without refusing.
   - The reading strings hard-code "300,000 decisions" even when T < 300k.

6. **Holm with one head.** If one head has 3 or more failures, Holm runs over a single test. Say so.

7. **Precheck (c) evaluated the E4 path on dq_e2_evaluation at 3,000 decisions.** It recorded J 0.0149 (C-strong) and
   0.0087 (C-matched) in precheck.json. Nothing selects on these values and they are not a grid point, but they are an
   evaluation-suite read of an E4 configuration taken before selection. Disclose this in the plan.

8. **Wording.**
   - "Information-matched" overclaims: the novelty filter, sticky abandonment and fix counter are absent, and the
     interface asymmetries remain. Prefer "evidence-count-augmented".
   - P2 is listed as primary but has no test and no reading rule. Call it an estimate.
   - Consider adding a t interval for J300k(E4) − J_rule/2 per head, which is more honest than comparing means at
     n = 6.

9. **Records.**
   - The checkpoint identity's feature_names list holds 254 names for a 270-input model, from the base describe().
   - The lr schedule appears only in the candidate label and in the per-update metrics, not in train_config.

   This is cosmetic, but re-evaluating a checkpoint requires e4lib.configure(True, …), and the paper's repository
   notes should say so.

10. **A crash after arm_end but before the record is written** causes a full re-run. Keep the interrupted copy and
    compare its prefix with the restart, which documents determinism.

11. **Process.** The builder launched the pilot (2 workers, 6-16 h) before the lead had reviewed the plan. This is
    acceptable: the runs are development-only, and B1 changes only the selection statistic, not the runs. If the lead
    rejects the pilot design itself, stop it now.

## Order of operations for the lead

1. Do not read pilot/records or pilot metrics.
2. Fix B1-B4 in PLAN_DRAFT.md and e4_analysis.py, with the selection rule in a separate script or a precheck re-run.
3. Write PLAN.md with the file hashes, then hash it.
4. When the pilot finishes, run the selection.
5. Check that no other launcher is alive, then launch.
