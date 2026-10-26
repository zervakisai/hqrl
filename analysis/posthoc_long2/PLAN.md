# Post-hoc check E4: best-effort, evidence-count-augmented, re-tuned PPO for the classical heads (300,000 decisions)

Final plan, written on 2026-10-25 from PLAN_DRAFT.md. The draft is kept unchanged for the record (SHA-256
40892991683d…; the byte copy the pilot launched under is 66520936…). This plan applies every fix the lead decided after
the independent review (REVIEW.md, SHA-256 20679cce…). It also fixes the four defects that an independent verifier
found in its first version (README_BUILD.md, "Round 2"). The lead hashes it into PLAN.sha256 and PLAN.utc.

When it was written:
- after all results of record, and after B2, LONG (`idea_dq_posthoc_long`) and LONG-HQ were known;
- before any E4 main run existed;
- while the development pilot was running, and before any pilot result (J, reports, entropy, S) was read.

It is not part of any frozen family and is labelled post hoc in the manuscript. The idea_dq tree (live hash
ffe9dfef2b24…) is imported unchanged.

**Frozen files.** These are the SHA-256 hashes of every script in this folder that the runs or the analysis execute,
plus the rule reference they read, computed after the last edit:

| file | role | SHA-256 |
|---|---|---|
| `runner.py` | precheck, pilot, main launch, status | `9601f88d8f88934ceb61445c1de9a134c385302f2a52872932c623628bd0046e` |
| `e4lib.py` | E4 patch layer (unchanged since the pilot launch) | `9d72adf6b5df54fa3c3f5d2074430ba037a097ee8bbda972d5b0923ef1b8d6c6` |
| `pilot_select.py` | the pilot selection rule (B1) | `c47cb2521fd6975472190ac847602f56d64c07354ad912539ff113a6e2b621ca` |
| `e4_analysis.py` | the planned analysis | `a24467a540f43faacf32b0b89e5e33cba972d8a3a21f9ce4c8a895f27de257ee` |
| `rule_reference.json` | the paired rule reference (a result written before this plan) | `5b8b8d71a44518c48452154b083dd5eeff809cf9e078214e7136ec38d0269a3d` |
| `rule_reference.py` | wrote and verifies rule_reference.json; not run by the runs or the analysis | `ae4dd5856e6d06972fa79f020d886f077e1810afc67f3c0e503eeebab6ef7cf2` |

The checks that hold these files fixed:
- `runner.py launch` and `pilot_select.py` refuse unless PLAN.md names the first five hashes and PLAN.sha256 matches
  PLAN.md.
- `e4_analysis.py` refuses on the same conditions.
- `runner.py launch` refuses unless precheck.json passed for this exact runner.py and e4lib.py.

The precheck of record is precheck.json (re-run after the runner.py edits, see "Precheck").

## Question
LONG trained C-strong and C-matched for 300,000 decisions at their E1 configuration: Adam at a constant learning rate
of 3e-3, entropy coefficient 0. Both ended below half the tuned rule's J in every block.

Two objections remain (Read 2, problem 2):
- (i) The learned observation lacks evidence the rule uses. The rule reports only tracks with ≥ 4 detections from ≥ 2
  credited distinct viewpoints, and it stops inspecting a track after 3 inspections. The learned observation has
  neither count.
- (ii) The configuration was not re-tuned for 300,000 decisions. In LONG, C-strong ended with ≤ 0.012 timely-credited
  reports per episode in 5 of 6 blocks. This is a behavioural collapse, as recorded in LONG's PLAN_LOG,
  2026-10-24T16:37:30Z.
  - The median last-update policy entropy was 0.26 nats, but entropy does not mark the collapse. The collapsed blocks
    5, 7 and 8 ended at 0.312, 0.372 and 0.308 nats. The block that did not collapse (6) ended at 0.077
    (long_descriptive_post.json 0e34c221f622).

Does the systems conclusion survive a best-effort classical PPO that has the rule's evidence counts and a standard
long-run configuration (an entropy bonus and a learning rate decayed to 0)?

## What changes relative to LONG (and nothing else)
1. **Observation** `obs-encoding/4-gnss+ev16/1`, 270 = 254 + 16 features.
   - Features 0-253 are the frozen `obs-encoding/4-gnss`, byte for byte.
   - Appended, for each of the 8 track slots of the frozen encoder (unreported tracks, in its order):
     `min(credited distinct viewpoints, 4) / 4` and `min(executed inspections, 3) / 3` of that track. An empty slot
     gets 0.
   - The rule's own code keeps both counts, as `GnssRulePolicy` does: `executive.EvidencePolicy._update_evidence` on
     every `act` and `EvidencePolicy._evidence_feedback` on every `feedback`.
   - The counts come from public quantities only: public detection counts, the vehicle's local position and the
     feedback on executed inspections.
   - `RLMissionPolicy` inherits this bookkeeping but never updated it. The subclass `InfoRLMissionPolicy` only updates
     it and appends it.
   - Unchanged: the action space, masks, RETURN rule and rejection handling.
   - Not added (stated as a limitation): the rule's per-view novelty filter, sticky abandonment and the per-report
     fix-attempt counter. Nothing changes the interface asymmetries (free RELOCALIZE, masked voluntary RETURN).
   - The observation is therefore evidence-count-augmented, not information-matched.
2. **Model.** LONG's model is built exactly as in LONG, from the same initialisation stream. It is bitwise equal to
   LONG's stored initial checkpoint of the same block and head.
   - Its first linear layer then receives 16 zero input columns. At initialisation the E4 policy is therefore LONG's
     policy: logits equal to < 1e-12, and exactly 0 on real observations.
   - The first rollout (the point-0 evaluation and the episodes of update 1) is identical to LONG's (precheck c).
   - Record caveat, disclosed and not fixed: the checkpoint identity's `feature_names` lists the 254 base names for the
     270-input model, because it comes from the base encoder's `describe()`. Fixing it would change e4lib.py, which the
     running pilot uses.
   - The learning-rate schedule is in the per-update metrics and the candidate label, not in `train_config`.
   - Re-evaluating an E4 checkpoint requires `e4lib.configure(True, …)` in the process. The repository notes say so.
3. **PPO.**
   - Entropy coefficient e_h > 0, chosen per head from {0.003, 0.01} by the development pilot below.
   - The Adam learning rate decays linearly from lr0 = 3e-3 (each head's E1 selection) to 0 over the 300,000-decision
     budget: lr_u = lr0 × (1 − D_u / 300,000).
   - D_u is the number of policy decisions consumed by the updates before update u. lr_u is set in every parameter
     group before update u and logged with it. Update 1 runs at lr0, as in LONG.
   - lr0 is not piloted. E1's 30,000-decision tuning selected 3e-3 over 1e-3 for both classical heads (C-strong AUC
     0.0380 vs 0.0373; C-matched 0.0164 vs 0.0133). Linear decay already halves the mean step, and a 2 × 2 × 2
     lr0 × entropy pilot does not fit this week's compute cap.

Everything else is LONG's:
- **Study object.** The B2 study object: salt (20260923, 2), and the initialisation, action and minibatch streams.
- **Training seeds.** dq_e2_training with 10,000 seeds per block, on blocks 4-9. Each E4 run therefore meets exactly the
  training cells and disturbance seeds of LONG's run of the same block and head, in the same order.
- **PPO settings.**
  - 512 transitions per update, clip 0.2, value coefficient 0.5;
  - gradient-norm clip 0.5 per group, 4 epochs;
  - minibatches of ≤ 256 transitions, segments of ≤ 32;
  - advantage std floor 0.05;
  - γ = 1 with the time-aware λ (0.95, 60 s);
  - float64, one thread.
- **Evaluation.**
  - Sampled read-out on the 80 common-random-number episodes of dq_e2_evaluation (8 core cells × realisations 0-9).
  - LONG's 11 points: 0, 30,000, …, 300,000. Each is taken at the first update boundary at or after the point.
  - The evaluation action stream is keyed by the point index, as in LONG, so every E4 evaluation is paired with LONG's
    at the same point.

The two changes, information and re-tuning, are confounded by design. E4 is one best-effort arm, and n = 6 blocks cannot
separate their contributions. HQ is not run, because the question concerns the classical baselines.

## Development pilot (development seeds only)
**Runs (8).** These are unchanged since the pilot launch.
- {C-strong, C-matched} × entropy {0.003, 0.01} × development blocks {6, 7}, 100,000 decisions each.
- The main configuration: the observation and model above, and lr0 3e-3 decaying over the 300,000 horizon. The pilot
  is the first third of the main schedule.
- Study `e4_dev`, salt (20260923, 41).
- Training seeds: dq_development.start + 10,000 × block + episode, i.e. [start + 60,000, start + 80,000).
- Development evaluation suite: the 8 core cells × dq_development realisations 600-609. The seeds are
  start + 1,000 × seed_cell_index + r, all below start + 7,000.
- Sampled read-out at 0, 50,000, 80,000 and 100,000 decisions.
- No dq_e2_training or dq_e2_evaluation seed is used, and no pilot step reads anything from dq_e2_evaluation.
- The pilot was launched on 2026-10-25T08:12:07Z with 2 workers at nice 10, under runner.py ae36bc0e… and e4lib.py
  9d72adf6…. A byte copy of that runner.py is in provenance/. The pilot runs keep exactly that per-run code path.

**Selection rule of record**, per head. It is applied by `pilot_select.py` after PLAN.md is hashed and written to
pilot/selection.json once, before any main run.
- **Eligible.** A candidate e ∈ {0.003, 0.01} is eligible only if both of these hold:
  - it completed in both development blocks;
  - its development-suite timely-credited reports per episode exceed 0.012 at both 80,000 and 100,000 decisions, in
    both blocks. The measure is the mean tp_timely over the 80 development evaluation episodes.

  0.012 is the per-episode level at or below which LONG's PLAN_LOG (2026-10-24T16:37:30Z) classed the C-strong blocks
  as having ended without timely reports.
- **Score.** S(e) = mean weighted development J at 80,000 and 100,000 decisions over both development blocks.
- **Choice.**
  - One eligible value: it is chosen.
  - Two eligible values: the larger S. The exception is |S(0.01) − S(0.003)| < 0.0056 J (0.1 report-equivalent per
    episode, dq_b2.TOST_DELTA), which chooses 0.01, the larger entropy. The weaker regulariser is the one more prone to
    the late collapse that a 100,000-decision pilot cannot see.
  - No eligible value: 0.01. This fallback is fixed now.
- **Descriptive only.** The mean policy entropy over the last 20 PPO updates (about 10,000 decisions) of every pilot run
  is reported. It plays no role in the choice.
- **Pilot failures.** A pilot run with a non-algorithmic failure is restarted once from scratch (the rule of "Main
  runs"). A second failure stands, and that value has then not "completed in both development blocks".
- The pilot's runs, every S, the eligibility table and the entropies are reported whatever is selected. They are
  development material, not outcomes.

**Deviation from PLAN_DRAFT.md, recorded.** The draft said that this section is "fixed before the pilot starts".
- The selection rule was changed on 2026-10-25 after the review (item B1). The pilot was running at the time (4 of 8
  runs complete when the lead decided), and no pilot result had been read: only counts, status and timing.
- The pilot runs themselves are unchanged: the same code, seeds and configuration, all already started under the
  draft. Only the statistic computed from their logged data changed, and it uses quantities the pilot logs anyway.
- The draft's header also said: "The pilot section below must not change once any pilot result has been read; if it
  is changed before that, the pilot is discarded and re-run." By the lead's decision this clause is read as applying to
  the pilot runs, which are unchanged, and not to the selection statistic computed from their logged data (as REVIEW.md
  B1 asked). The pilot is therefore not re-run.
- The draft's rule, quoted for the record:

  > Eligible: completed in both blocks (no algorithmic failure) and last-update mean policy entropy ≥ 0.30 nats in
  > both blocks (LONG's collapse level). If no value is eligible, the entropy guard is lifted (failures stay
  > ineligible); if both values failed, e = 0.01. One eligible value: it is chosen. Two: the larger S, except that
  > |S(0.01) − S(0.003)| < 0.0056 J (0.1 report-equivalent per episode, dq_b2.TOST_DELTA) chooses 0.003 (the smaller
  > entropy).

- **Why it changed.** LONG's own data contradict the entropy guard. In C-strong, three of the five collapsed blocks ended
  above 0.30 nats, and the one block that did not collapse ended at 0.077. Single-update entropy is noisy. Three or
  four of the five collapse onsets came after the pilot's 100,000-decision horizon. The behavioural indicator is what
  LONG's forensic used.

| LONG C-strong block | ended without timely reports (≤ 0.012 per episode at 300,000)? | last-update entropy (nats) |
|---|---|---|
| 4 | yes (none from 120,000) | 0.048 |
| 5 | yes (from 90,000) | 0.312 |
| 6 | no (0.42 at 300,000) | 0.077 |
| 7 | yes (from 270,000) | 0.372 |
| 8 | yes (0.0119 at 300,000) | 0.308 |
| 9 | yes (from 240,000) | 0.203 |

- **Why ties go to 0.01.** The tie direction was 0.003 in the draft. The selection noise (2 blocks × 2 points × 80
  episodes) probably exceeds the band, so the band will often decide. For a best-effort test of an objection to the
  authors' own conclusion, ties go to the stronger regulariser.
- The retired `runner.py pilot-select` refuses and points to pilot_select.py. `runner.py launch` refuses unless
  pilot/selection.json carries the SHA-256 of the pilot_select.py on disk and this plan's SHA-256.

## Main runs (12, all new)
C-strong (`mlp`) and C-matched (`bottleneck`) × B2 blocks 4-9, at 300,000 nominal decisions. The entropy e_h comes
from the pilot, and lr0 3e-3 decays linearly to 0. The launch command is
`nice -n 10 python -B runner.py launch --workers 12`, run once the pilot has finished and pilot_select.py has run.

**Failures and restarts (B3)**, applied by runner.py and by the analysis:
- (a) **Algorithmic, never restarted.** FloatingPointError, ScheduleExhausted, and a P4Error "block b leaves the
  <namespace> namespace" raised by the schedule.
  - That P4Error is the namespace-end form of ScheduleExhausted. dq_e2_training is exactly 100,000 seeds wide, so
    block 9's range ends at the namespace end, and for block 9 the P4Error check in `CellRuntime.schedule`
    (p4_train.py:394-395) comes before the ScheduleExhausted check.
  - The same algorithmic event is therefore classed the same way in every block.
- (b) **Any other non-algorithmic failure** is moved, with its record, to interrupted/<utc>/ (kept) and restarted once,
  at once, from scratch. Examples: an OSError, a MemoryError, any other P4Error, or a bug in the E4 wrapper. The restart
  is deterministic (identical seeds), so pairing is unaffected.
  - If the same run fails again, the second failure stands. It is reported as a code failure, excluded with n stated,
    and the code is fixed only under an addendum. It is excluded only at read-out points it did not reach: an
    evaluation at T before the failure is a value at T (reading rules, "Failures").
  - A first failure that a dead launcher left unrestarted is restarted by the next launch. The analysis refuses while
    such a restart is pending.
- (c) **A run killed by a crash** (a run directory without a record) is moved to interrupted/<utc>/ and restarted from
  scratch by the next launch.
- (d) **Kept copies.** Every move is logged in restarts.jsonl. The analysis compares each kept copy with the restarted
  run over their common prefix of events, excluding timing fields, and reports the result. A difference is reported,
  not used to exclude a run. This documents determinism.

**Single instance and relaunch.**
- `runner.py pilot` and `runner.py launch` take an exclusive flock on runner.lock, which their workers inherit. They
  refuse while any other process runs this folder's runner.py with `pilot` or `launch`, whatever its code version. The
  first pilot launcher predates the lock and is caught by this scan.
- Relaunching is the only way to resume:
  1. Run `python -B runner.py status`. It shows records by class, the progress of every unfinished run (counts, the
     last evaluated point, seconds since the last write) and every live launcher of this folder. It shows no J and no
     entropy.
  2. If a launcher is alive and progressing, do nothing.
  3. If it is dead or stalled, stop the launcher parent first and then its workers. These are the processes from
     `pgrep -af "runner.py launch"` whose `readlink /proc/PID/cwd` is this folder. Confirm with `runner.py status` that
     none is alive.
  4. Run `nice -n 10 python -B runner.py launch --workers 12`. Records that stand are skipped. Cases (b) and (c) above
     are restarted.

**OOM and stall watch.**
- Free memory is about 11-15 GB, and 12 workers need about 6.6 GB.
- If the OOM killer takes a worker, the pool never returns, so a stalled run has to be recognised by hand. It is a run
  whose metrics.jsonl has not grown for more than 2 hours (`runner.py status`, age_s) while its worker is gone or dmesg
  shows an OOM kill. An evaluation of 80 episodes takes 3-10 minutes, and up to about 30 minutes if episodes become
  rule-like.
- Apply the relaunch procedure.
- Avoid Chrome video during the runs (the BAR1 freeze).

**Timing (re-estimated).**
- Plan on about 20-30 h of wall time for the single 12-worker wave under load, and more if other campaigns are running
  or the E4 policies make longer, rule-like episodes.
- The basis is LONG's records with 12 workers at load 21-33: 18.9-26.6 h of wall time and 3.5-10.1 h of CPU time per
  run, and 15.6-18.3 (summed rollout, update and evaluation walls) h of wall time to reach 100,000 decisions.
- The builder's estimate of 4.5-8.5 h per run, extrapolated from an untrained 3,000-decision run, was about 3× too
  optimistic.
- A launch on 2026-10-25 or 26 should end on 2026-10-27 or 28, which leaves about two days before the deadline.

**Precheck** (`runner.py precheck`, precheck.json). The launch refuses unless it passed for the exact runner.py and
e4lib.py being launched.
- (a) **Every change switched off.** The runner's code path, with all wrappers in pass-through, re-runs stored LONG block
  4 of both heads for 3,000 decisions. It reproduces LONG's point-0 evaluation and its update, episode and arm_start
  events exactly.
- (b) **Initialisation.**
  - LONG's initial model is rebuilt bitwise.
  - The widened model equals it in every tensor except the 16 zero columns.
  - Its logits and values equal the base model's to < 1e-12.
  - The global torch generator is left where the base build leaves it.
  - Unit tests add two checks. The encoder on real observations reproduces features 0-253 of the frozen encoder at every
    decision. The 16 features equal an independent re-implementation of the rule's bookkeeping.
- (c) **Every change switched on** (entropy 0.01), a 3,000-decision run of each head:
  - it completes;
  - its point-0 evaluation and the episodes of update 1 equal LONG's, and its schedule is LONG's;
  - every update logs lr_u exactly as defined, and the entropy coefficient;
  - the evidence features are populated;
  - the final checkpoint holds the 270-input model and the last learning rate.
- runner.py changed after review (lock, restart rule, status, launch checks; its per-run code path is byte-identical).
  The precheck was therefore re-run on 2026-10-25 with 1 worker, alongside the pilot: passed (precheck.json, started 2026-10-25T12:17:20Z, 724 s; (a) both heads reproduced stored LONG block 4 exactly for 3,000 decisions: point-0 evaluation, updates, episodes and arm_start; (b) LONG's initial model rebuilt bitwise, max |dlogit| 1.2e-16 (C-strong) and 2.2e-16 (C-matched), generator state unchanged; (c) every check above for both heads; runner.py 9601f88d8f88…, e4lib.py 9d72adf6…, idea_dq tree ffe9dfef… before and after). The earlier
  outputs are kept in precheck_prev_2026-10-25T121716Z/.
- **Disclosure.** Precheck (c) evaluated the E4 path on dq_e2_evaluation at 3,000 decisions. Weighted J was
  0.0149 for C-strong and 0.0087 for C-matched; the point-0 values −0.00545 and −0.00745 are LONG's own. These are
  evaluation-suite reads of an E4 configuration taken before selection. Nothing selects on them, and 3,000 is not a grid
  point. The re-run reproduced them exactly: every logged event of the four precheck runs equals the first precheck's in every trajectory field; only timing fields and the checkpoint files' SHA-256 (torch.save output is not byte-reproducible) differ.

## Paired rule reference (computed before this plan was hashed; not an outcome of E4)
- **What it is.** The tuned rule (`uncertainty_fix`, σ 1.5 m, `e1_baselines.p3_rule()`, run by
  `e1_baselines.run_policy_episode`) flown on exactly the 80 dq_e2_evaluation episodes of every B2, LONG, LONG-HQ and
  E4 evaluation (`rule_reference.py`, rule_reference.json).
- **Values.** J_rule = its cell-weighted J (p4_train.aggregate_evaluation) = 0.21133, so J_rule / 2 = 0.10566.
- **Pairing.** The rule draws no action stream, so this single set of 80 rows pairs with every evaluation point of every
  learned run (same cell, seed and CRN group).
- **Checks.**
  - The (cell_id, seed) of every stored B2, LONG and LONG-HQ evaluation event equals the reference's, in order.
  - Two episodes flown again are identical.
  - The same code path reproduces two stored E1 rule rows exactly (J, crn_sha256, decisions).

## Outcome and analysis (fixed now; `e4_analysis.py`)
**Outcome.** Weighted J (sampled read-out, B2/LONG aggregation) at each checkpoint. The read-out point is T, which is
300,000 except under the deadline read-out (see the stopping rule).

**Before anything is computed**, the analysis checks every E4 run and refuses on any failure (B4):
- **Raw events.** Exactly one arm_start, first. Exactly one arm_end, last, iff the run ended through train_arm.
  Contiguous update indices and episode numbers, consistent cumulative counts, and no duplicated line. Every grid point
  is evaluated once and in order, each evaluation preceded by its checkpoint, and a completed run has every grid point.
- **(i)** The point-0 evaluation is bitwise equal to LONG's of the same block and head.
- **(ii)** The training (cell_id, seed) sequence equals LONG's on the overlapping episode numbers.
- **(iii) Identity.**
  - arm_start obs_version `obs-encoding/4-gnss+ev16/1` with 270 inputs;
  - the entropy coefficient equals pilot/selection.json, in the record's config and task and in every update;
  - every update's learning rate equals the schedule;
  - protocol_freeze_sha256 equals sha256(PLAN.md), in the config, the task and the arm_end;
  - the record code hashes equal precheck.json's, and the idea_dq tree is ffe9dfef…;
  - every logged launch names this plan, the selected entropies and the prechecked code.
- **Pairing.** Every evaluation flies the rule reference's 80 episodes, and every stored weighted J is re-derived from
  its episodes.

**Primary**, per head h ∈ {C-strong, C-matched}, n = 6 blocks, paired by block with the stored LONG run:
- **P1_h** = J_T(E4) − J_T(LONG): the mean, 95% paired t interval, two-sided paired t p-value and dz.
  - Holm over {P1_C-strong, P1_C-matched} at familywise 0.05.
  - If only one head has P1, Holm runs over the available test with the family size kept at 2, i.e. at 0.025.
  - No equivalence test.
- **P2_h**, an estimate and not a test: the share of LONG's gap to the paired rule reference that E4 closed,
  Σ_b (J_E4 − J_LONG) / Σ_b (J_rule − J_LONG).
  - It comes with a 95% percentile bootstrap interval over blocks (20,000 resamples, seed 20261025), and per-block
    shares descriptively.
  - It has no reading rule.
- **P3 (half-rule check).** The readings use θ = min(0.103, J_rule / 2) = 0.103 (B2).
  - 0.103 is LONG's planned threshold, already in the manuscript. J_rule / 2 = 0.10566 is the paired reference, which
    was introduced after LONG's results were known and is the more lenient threshold for the authors' claim.
  - For each head, at both thresholds: the blocks with J_T(E4) at or above the threshold, the maximum block J_T and the
    mean, and the failed blocks (see Failures) whose last evaluated J is at or above the threshold.

**Descriptive** (no test, not in the Holm family):
- Mean J at every checkpoint (E4 and LONG), and E4 − LONG at each.
- E4 − LONG on: AUC over 0-T (LONG's trapezoid), J at 30,000, and the slope over the last three checkpoints up to T
  (least squares, per 30,000 decisions, as long_analysis.py computed it). E4's slope and LONG's slope over the same
  blocks are reported too.
- J_T(E4) − J_rule with its t interval, and J_T(E4) / J_rule.
- A per-head t interval for J_T(E4) − θ and for J_T(E4) − J_rule / 2. This is more honest than comparing means at
  n = 6.
- Components at T for E4, LONG and the rule: timely-credited reports, false reports, correction attempts and decisions
  per episode.
- **Collapse indicator:** the blocks with ≤ 0.012 timely-credited reports per episode at T (the level summarised in LONG's PLAN_LOG of 2026-10-24T16:37:30Z; LONG's descriptive script counted 'no timely
  reports' at 0.01), for E4
  and LONG. The mean policy entropy over the last 20 updates and the last-update entropy are reported per block,
  descriptively only. Entropy is not a collapse indicator.
- **Peak versus final:** every intermediate checkpoint of a block at or above θ, or at or above J_rule / 2, and the
  highest single-block J at any checkpoint.
- The evidence-feature diagnostics: decisions with a nonzero viewpoint or inspection feature.
- The restart prefix comparisons.
- The pilot table and the selection.

**Reading rules** (fixed now). Every reading names T.
- **P1, per head.**
  - Holm-rejected with the interval above 0: "the best-effort configuration (the rule's evidence counts, an entropy
    bonus, a decayed learning rate) ended higher than LONG at T decisions", reported with P2.
  - Holm-rejected with the interval below 0: "it ended lower than LONG at T decisions".
  - Otherwise: "not resolved at n = … (not evidence of equality)".
- **Systems conclusion (P3, with θ = 0.103).** The value of record is J_T; no checkpoint is selected on the evaluation
  suite. The three readings below apply as written when all 12 blocks have a value at T. When a block has none, the
  rule "P3 when blocks lack a value at T" (under Failures) decides.
  - All 12 blocks have a value at T, and every one is below θ: "the systems conclusion holds for best-effort,
    evidence-count-augmented, re-tuned classical PPO at T decisions". The scope is two classical heads, one scene, these
    16 evidence features and this budget.
  - Some block at or above θ, but both head means below θ: the conclusion is restated as "no learned controller
    approached the rule on average", naming the block(s) that reached half the rule. The application conclusion is
    then called budget- and configuration-specific.
  - A head's mean at or above θ: the systems conclusion is withdrawn for best-effort classical PPO at T decisions. It
    is restated as "under the E1/B2 configuration the learned controllers stayed far below the rule; with the rule's
    evidence counts and re-tuning, C-x reached [share] of the rule".
  - In every case:
    - a head whose 95% interval for J_T(E4) − J_rule reaches 0 is reported as "not shown to be below the rule";
    - the same reading at J_rule / 2 is reported alongside, descriptively, so a reader can see whether the threshold
      choice mattered;
    - an intermediate checkpoint at or above a threshold is reported descriptively, and the paper says that no
      checkpoint was selected.
- **Failures.** Collapse is not a failure (P4 protocol section 6): a collapsed run is completed and enters as it is.
  - **Value at T.** A block has a value at T when its run evaluated T, in the current copy or in a kept copy whose
    common prefix with it is identical. Under the deadline read-out, the evaluation must have come by the deadline.
    - This holds even if the run later ended with a failure. That evaluation belongs to the same deterministic
      trajectory, just as a run still running at the deadline enters with its J_T whatever happens to it later.
    - Every run's curve stops at T.
  - **Failed block.** A block without a value at T. Its run ended before evaluating T, with an algorithmic failure
    (FloatingPointError, ScheduleExhausted or its namespace-end form) or with a second, standing code failure.
    - A run that failed after evaluating T is listed among the failures but is not a failed block: it enters with its
      J_T.
    - e4_results.json lists every failed run per head under `failed_blocks` and the failed blocks in this sense under
      `blocks_without_value_at_T`.
  - P1 and P2 use the blocks that have a value at T (n stated). A sensitivity analysis of P1 carries each failed
    block's last evaluated J forward (LOCF).
  - With 3 or more failed blocks in a head, P1 is not computed for it, and the head is reported as "failed in k of 6
    blocks". Holm then runs as stated under Primary.
  - **P3 when blocks lack a value at T.** "Holds" requires all 12 blocks to have a value at T, every one below θ. With
    one or more failed blocks:
    - The three readings above are applied twice: (1) to the values at T, over the heads that have at least one;
      (2) with each failed block's last evaluated J carried forward (the LOCF sensitivity).
    - The reading of record is the more severe of the two (withdrawn > restated > holds). A failure therefore never
      makes the reading more favourable to the systems conclusion.
    - A failed block whose last evaluated J is at or above θ counts as a block that reached half the rule. It is named
      with its J and its point.
    - If both applications give "holds", the reading is "not computable" when a head has no value at T at all, and
      "qualified" otherwise. "Qualified" means that every value at T and every last evaluated J is below θ, but not
      every block has a value at T. Neither is stated as "holds".
    - Every such reading names n per head, each failed block (its class and its last evaluated point and J, or "no
      evaluated point"), every last evaluated J at or above θ, and the two underlying readings.
    - The same rule gives the descriptive reading at J_rule / 2.
- **Stopping rule, and which read-out is of record.** The deadline is 2026-10-30T12:00Z (`DEADLINE_UTC` in
  e4_analysis.py).
  - If every one of the 12 runs has ended (its record stands) by the deadline, the full read-out at T = 300,000 is of
    record.
  - Otherwise the deadline read-out is of record. T is the largest grid point that every run running or completed at
    the deadline had evaluated by the deadline.
    - The time of an evaluation is the mtime of the checkpoint saved just before it plus the evaluation's wall_s. A
      run's end is its arm_end utc, or else its record's mtime.
    - Runs that had ended with a failure by the deadline do not enter T. Such a run still has a value at T if it
      evaluated T before it failed (see Failures).
    - A kept copy (a crash or a failure) whose common prefix with the restarted run is identical may supply evaluated
      points.
    - The read-out is paired with LONG at T and stated as such.
    - It is computed from the files whenever the analysis is run after the deadline, so the time it is run does not
      change T. It stays of record if the runs finish later. The 300,000 read-out of those runs is then reported as
      supplementary (`--supplementary-full`, e4_results_full_after_deadline.json), labelled as after the deadline and
      not of record.
  - No run is extended, shortened or restarted for this reason.
  - Before the deadline, with a run unfinished, the analysis refuses.
  - The analysis refuses to overwrite e4_results.json unless given `--force`. The old file is then kept, and every write
    is logged in e4_analysis_log.jsonl.
  - `--provisional` works on a copy (`--root`) only and is refused on this folder, so no main-run result is read before
    the read-out of record.

## Reporting commitment
Whatever the outcome, these are reported in Section 4.2.1 (next to LONG) and in limitation (h), with this plan's
SHA-256:
- P1-P3 and the descriptive values above;
- both thresholds;
- the pilot table and the selection (with the deviation above);
- the precheck, including the (c) disclosure;
- the rule reference;
- the configuration caveat: one best-effort configuration, information and re-tuning confounded, n = 6;
- the evidence features that were not added, and the feature_names record caveat.

Scripts, records, the pilot, the restarts ledger and the kept copies go to the repository.

## Order of operations (lead)
1. Hash this file: `sha256sum PLAN.md > PLAN.sha256` and `date -u +%Y-%m-%dT%H:%M:%SZ > PLAN.utc`. Do not read pilot
   records or metrics before step 3.
2. Wait until the pilot has 8 records and no pilot launcher is alive (`python -B runner.py status`). If a pilot run
   recorded a non-algorithmic failure, `nice -n 10 python -B runner.py pilot --workers 2` restarts it once.
3. Run `python -B pilot_select.py`. It writes pilot/selection.json once.
4. Run `nice -n 10 python -B runner.py launch --workers 12`. It refuses while any launcher of this folder is alive.
5. Monitor with `python -B runner.py status` (counts, status and timing only), and relaunch only as described above.
6. When every run has ended, or after the deadline, run `python -B e4_analysis.py`, which writes e4_results.json.

## Lead's confirmations at hashing (2026-10-25)
- R2-1 confirmed: under failures, the more severe of the at-T and LOCF readings of P3 is of record.
- R2-2 confirmed: a run that failed after evaluating T keeps its J_T.
- The 'holds' reading, wherever it is reported, carries the scope stated above (two classical heads, one scene, the
  evidence-count features, this budget).
