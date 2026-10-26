# Post-hoc E1: a learned correction trigger inside the tuned rule's executive

DRAFT written 2026-10-25, before any main run, after all results of record and of the post-hoc extensions were
known. It becomes PLAN.md (hashed by the lead) only after review; until then no main run exists and `runner.py
launch` refuses. Not part of any frozen family; labelled post hoc in the manuscript.

## Question
Reviewers asked whether learning can earn a narrow role inside a handcrafted executive rather than replace it end to
end. Here a learned policy takes over exactly one decision of the tuned transparent rule (UncertaintyFix, sigma 1.5 m):
the correction trigger "relocalise now, or report now?", asked just before a report. Everything else is the rule's own
code. Does this learned trigger match, beat or fall short of the tuned rule on the E1 evaluation episodes?

## What is learned and what is not (fixed now)
- **Policy.** `gate_core.GatePolicy` subclasses `gnss_policies.UncertaintyFix` and is built with
  `e1_baselines.p3_rule()['parameters']` (sigma 1.5 m). `gnss_policies.py` is not edited (its SHA-256 is re-checked by
  `p3_rule()` in every process), and the idea_dq code tree stays at `ffe9dfef…` (checked before and after the
  precheck). Only `should_fix()` is replaced. The rule's own `act()` calls it at its usual place: the base action is the
  REPORT of the lowest-id rule-ready open track (at least 4 detections from at least 2 credited viewpoints), and the
  per-report fix allowance is left (at most one attempt per report). Gate action 1 (FIX) gives the rule's RELOCALIZE
  to its cheapest reachable anchor, preferring available ones. Gate action 0 gives the rule's REPORT. Search,
  inspection, evidence bookkeeping, report choice, merge radius and RETURN are unchanged.
  - When no anchor is reachable, the rule reports anyway. The gate is then not asked and nothing is recorded; such
    events are counted.
  - When the executive rejects a RELOCALIZE (none in the 120 stored E1 rule episodes), the same decision is resampled
    with FIX masked, as `RLMissionPolicy` resamples a rejected decision. The final mask is [1, 0] and the log-prob 0.
    Only in this never-observed case does a rule-imitating gate differ from the rule: the rule would re-propose the
    fix until the rejection cap is reached.
- **Observation** (24 public features, each scaled by a declared constant; T is the cell's mission budget and R its
  return reserve):
  - georeference: RMS error/3 m (the rule's only input), min(time since fix, T)/T, fixes attempted and fixes
    accepted (each capped at 10, divided by 10), last fix accepted, no fix yet;
  - time: elapsed/T, remaining/T, return estimate/T, slack (remaining − return estimate − R)/T, T/2700 s;
  - anchors: estimated duration of the cheapest reachable anchor/100 s, whether it is available, the share of
    anchors that are reachable and available;
  - reports so far (capped at 20)/20;
  - the candidate track: detections (cap 20)/20, credited viewpoints (cap 8)/8, max score, √(cov_xx + cov_yy)/10 m,
    age/T, inspections done/3;
  - the number of other rule-ready open tracks (cap 5)/5, the number of open tracks (cap 10)/10, and the fraction of
    the search plan still pending.

  The rule's own recommendation (rms ≥ 1.5) is *not* a feature: the gate gets the raw inputs only, so it is not a
  residual gate.
- **Model.** `GateActorCritic`, float64, no recurrence. The actor is Linear(24,64)-Tanh-Linear(64,64)-Tanh-Linear(64,2),
  with the last layer scaled by 0.01 so that the initial gate is close to a fair coin: 5,890 parameters. The critic is a
  separate Linear(24,64)-Tanh-Linear(64,64)-Tanh-Linear(64,1): 5,825 parameters. It is initialised under
  `torch.manual_seed(SeedSequence([20261025, 1, block, 0]))`.
- **SMDP segments** (`gate_core.build_gate_segments`; unit-tested). There is one row per executed gate decision.
  - Extent: segment k runs from gate decision k to gate decision k+1. The last segment runs to the end of the
    episode's last record.
  - Reward: the sum of the `reward-gnss-timely/1` events (`gnss_rewards.place_gnss_rewards`) on the segment's
    records. These are the report increments and the terminal correction, with γ_ref = 1.
  - Identity: the identity kept + pre-gate + invalid-segment reward = J − dropped is asserted on every episode
    (|residual| ≤ 1e-9).
  - Reward before the first gate was caused by no gate decision. It is excluded from training and counted. In the
    precheck it is 0 on all 80 episodes.
  - Duration: the next gate time minus this gate time. The last segment's duration runs to the end of the last valid
    record. Durations must exceed `smdp.MIN_DURATION_S`, which `RolloutBuffer.add` enforces; the minimum measured
    over the 80 precheck episodes is 1.36 s.
  - Terminal flag: the last segment of a terminated episode is terminal.
  - Infrastructure exits: a segment holding an invalid record is an invalid row (no reward, not terminal). So is the
    last segment of an episode that ended in an infrastructure exit without an invalid record. The valid row before
    it becomes a GAE cut, bootstrapped from its stored value.
  - Episodes with no gate decision (every no-human episode under the rule) give no row.
- **PPO** (`smdp_ppo.compute_advantages` and `ppo_update`, unchanged).
  - Optimiser: Adam with a constant learning rate of 1e-3, 10 epochs, and minibatches of 64 transitions.
  - Segments: max_segment_length = 1, because the model is feedforward, so minibatches are shuffled transitions.
  - Losses: clip 0.2, value coefficient 0.5, entropy coefficient 0, max grad norm 0.5 per group.
  - Advantage: std floor 0.0056 J (0.1 report-equivalent), γ_ref = 1, τ = 60 s, λ_ref = 0.95 in time-aware mode.
  - Rollout size: 256 gate decisions per update (whole episodes, so 256 to about 264).
  - Budget: **12,800 gate decisions** (50 updates). That is about 2,065 training episodes at the rule's measured
    6.2 gate decisions per training episode (7.75 per target-present episode × 0.8).
  - These values are fixed a priori, not tuned on any suite. Epochs and minibatch size follow common PPO defaults.
    The learning rate is raised from the common 3e-4 to 1e-3 because the budget allows only about 50 updates.
    Entropy 0 is the value of every E1/B2-selected arm. *Optional, lead's choice before hashing:* the development
    pilot described under "Optional development pilot" below.
- **Seeds.** Two new namespaces follow P3's rule (start = 20260921·1000 + index·100,000) and lie after dq_e2_evaluation
  (index 18): `gate_training`, index 19, [20262821000, 20262921000), and `gate_development`, index 20. `CellRuntime`
  refuses any overlap with the frozen, E1 or DQ ranges. I found no other use of index 19 or above in idea_dq or the
  post-hoc folders.
  - Salt: (20261025, 1) for the main runs, (20261025, 2) for development.
  - Training episode k of block b: seed = gate_training.start + 5,000·b + k. Its cell is drawn with the frozen
    weights (0.8 over the 4 positive core cells, 0.2 over the 4 no-human ones) from SeedSequence([20261025, 1, b, 3,
    k]).
  - Action sampling uses SeedSequence([…, b, 1]) and the minibatch order SeedSequence([…, b, 2]), as in p4_train.
  - The stride of 5,000 is about 2.5 times the expected episodes per run. If it is exhausted, the run is an
    algorithmic failure, as in p4.
- **Read-outs.** Every read-out is deterministic (argmax).
  - Learning curve: on the development suite `e1_development` (40 episodes; the rule's stored rows are paired) at 0,
    3,200, 6,400, 9,600 and 12,800 decisions. It is descriptive only: no checkpoint selection and no early stopping.
  - Final: the **final checkpoint** (no selection) is read once on `e1_evaluation`, 80 episodes (8 core cells × r0-9).
    Each episode is paired with the stored rule row of the same episode in
    `idea_e1/…/e1_campaign_v2/baselines/baselines.json` (SHA-256 00202039…); the pairing is checked on cell_id, seed
    and crn_sha256.
  - Nothing in this design was chosen by reading `e1_evaluation`.

## Runs (12, all new, none retried)
Blocks 0-11 of `gate_training`, one run each, with training and the final read-out in the same task. A run killed by
a machine crash is moved to `interrupted/` and restarted from the beginning, with the same seeds. Any other failure is
recorded and never retried. A completed run, whatever its curve, enters as it is. An algorithmic failure (non-finite
value, schedule exhausted) has no final checkpoint and is reported as such; that block contributes no D_b.
Infrastructure episodes stay in as scored and are counted.

## Precheck before launch (the launch refuses without it, for this plan's SHA-256 and this code)
(a) The idea_dq live tree equals ffe9dfef… before and after; `p3_rule()` passes (gnss_policies.py hash); the stored
baseline file matches its SHA-256.
(b) The unit tests of the segment builder and gate bookkeeping pass: reward sum = J − dropped, zero pre-gate reward,
durations, terminal flag, invalid rows and cuts, an invalid record mid-episode refused, abandoned decisions, a merged
forced RETURN, a re-asked FIX resampled with FIX masked, checkpoint round trip, and the model running through
`ppo_update`.
(c) A rule-imitating gate (FIX iff rms ≥ 1.5 m) run through this exact gate code path reproduces **all 80 stored
e1_evaluation rule rows exactly**. It must match every stored field except wall time: J (bitwise), fixes, decisions,
crn_sha256, TP/FP, return, termination and elapsed time. On every one of these episodes, the segment identity must
hold, the pre-gate reward must be 0, and the FIX rows must equal the fixes attempted.
(d) A training smoke on development seeds (block 19 of gate_development, about 30 episodes, 3 updates of ≥64 rows):
it must log every event kind, keep the identity on every episode and draw its seeds in range, and its reloaded final
checkpoint must reproduce its first evaluation episode exactly. Timings are recorded.

## Outcome and analysis (fixed now; `gate_analysis.py`)
- **P1 (primary).** Per block b, D_b = cell-weighted J(gate, final, argmax) − cell-weighted J(rule) over the 80
  paired e1_evaluation episodes. The weights are the frozen cell weights (0.2 per positive cell, 0.05 per no-human
  cell), through `p4_train.aggregate_evaluation`, the same J scale on which the rule's 0.2056 of record and the margin
  were defined. Across the completed blocks (12 planned):
  - mean, 95% paired t interval, two-sided paired t p-value, dz;
  - **TOST at ±0.0056 J** (0.1 report-equivalent per episode, dq_b2.TOST_DELTA): two one-sided t tests at 0.05,
    which equals a 90% interval inside the margin;
  - Holm over the primary family. The family has one member, so Holm is the identity. If the lead adds a second
    primary contrast, Holm is applied over the difference tests.

  The rule submits no report in any of the 40 no-human e1_evaluation episodes. The gate is therefore never asked
  there, so those episodes are identical to the rule's (the analysis checks this per block), and D_b comes from the
  positive cells.
- **Descriptive.**
  - D_b as the unweighted mean over the 80 episodes, with its interval;
  - per-cell mean differences;
  - fixes, false reports and timely true reports per positive episode (gate vs rule), return rate, infrastructure
    episodes, gate decisions per episode;
  - the gate's fix rate by RMS bin ([0,1), [1,1.5), [1.5,2), ≥2 m) against the rule's threshold;
  - the e1_development learning curve;
  - training statistics: episodes, approx_kl per update, P(fix) below and above 1.5 m after each update, pre-gate
    and dropped reward.
- **S1 (secondary, optional; no Holm coupling with P1).** If the lead keeps it, the 12 finals are read on the
  320-episode stress map (D600 0/2/4/8 m × 2 budgets × 2 presences × r0-19, namespace dq_stress_eval), paired with
  the stored rule job `freeze_a_campaign_v1/jobs/p3_rule__stress.json`. The read-out uses the map weights (0.4 per
  positive cell, 0.1 per no-human cell, normalised over the map), with the same statistics and the per-drift means.
  Drifts of 0 and 8 m are outside the training range. The low-level functions are called directly
  (`dq.require_open` stops after 2026-10-31). Before it runs, the rule-imitating gate must reproduce all 320 stored
  rule stress rows (`runner.py precheck-stress`).
- **A0.** The Freeze A replay ceiling (+0.40 report-equivalents, about 0.022 J) re-scores never_fix trajectories as if
  georeferencing were perfect. It is not an upper bound for a gate that changes timing, it is not called headroom,
  and it is quoted at most as a scale.

## Reading rules (fixed now; P1)
Let "rejected" mean that the two-sided paired t test is rejected at 0.05 after Holm, and "equivalent" mean that TOST
p < 0.05.
- **Equivalent and not rejected → "equivalent."** "A learned correction trigger inside the rule's executive, trained
  from scratch on 12,800 gate decisions, matches the tuned rule within 0.1 report-equivalent per episode." Learning
  recovers the tuned trigger; it does not improve on it.
- **Equivalent and rejected → "equivalent, with a detectable difference smaller than the margin"** (gain or loss,
  stated with its sign and interval).
- **Rejected with the interval above 0, not equivalent → "better."** The learned trigger exceeds the tuned rule by
  the stated amount, so the tuned trigger is not optimal and a narrow learned role inside the executive adds value.
  S1, if run, is reported beside it as the generalisation check. Nothing is claimed beyond this decision and these
  cells.
- **Rejected with the interval below 0, not equivalent → "worse."** Even with the rule's executive and a superset
  of its information, PPO at this budget did not recover the tuned trigger.
- **Neither → "not resolved."** This is not evidence of equality, and the interval is reported.
- **Fewer than 8 completed blocks → "not resolved (insufficient completed blocks)."**

In every case the comparator is one tuned rule, the result is post hoc, and the learned role is deliberately narrow:
7.75 gate decisions per target-present episode (measured) and none in no-human episodes. A0 is not an upper
bound.

## Optional development pilot (only if the lead adds it before hashing; otherwise the a-priori values above stand)
Learning rates {1e-3, 3e-4} × development blocks {0, 1} of `gate_development`, at a budget of 6,400 gate decisions,
read on `e1_development` only (`runner.py pilot …`; it refuses `e1_evaluation`). The selection rule, fixed in
advance: keep 1e-3 unless 3e-4 has the higher mean dev-suite weighted J at the final pilot checkpoint over both
blocks. Cost: about 6-7 h per run at the measured load, i.e. about 7 h on 4 workers. A smaller learning check
(lr 1e-3, development blocks 2-3, 10 updates, `--quick`) was run during the build. Its outcome is in README_BUILD.md,
and it changed no setting in this draft.

## Compute (measured 2026-10-25 under load average 16-18 on 16 threads; README_BUILD.md has the details)
- Training episode: 20.7 s on average (target-present 28.3 s, no-human 5.1 s).
- Rollout: 3,860 s per 1,000 gate decisions with a near-random gate, at 5.4 gate rows per episode. A rule-like gate
  gives 6.2 rows per training episode.
- Update: 0.05 s.
- Evaluation episode: 14.7 s.
- One run (12,800 gate decisions, 2,065-2,370 episodes, plus 280 read-out episodes) therefore takes about 12.8 h at
  this load (range 12.5-15 h). The quick learning check measured 3.3 s per gate decision. At a 12-worker load with nothing else running it should take about 10-12 h, roughly 25% faster; that figure
  is inferred, not measured.
- The 12 runs form one wave on 12 workers. *Decision for the lead before hashing:* if E1 has to share the machine
  with another campaign, the budget becomes 10,240 gate decisions (40 updates, grid 0/2,560/5,120/7,680/10,240) so
  that a run stays at or under 12 h.

## Reporting commitment
Whatever the outcome, the P1 reading, its interval, TOST p and the descriptive panel are reported in the manuscript as
a post-hoc check, next to the systems conclusion. The report also gives this plan's SHA-256 and the precheck result.
Scripts, records and checkpoints go to the repository.
