# Post-hoc E1: a learned correction trigger inside the tuned rule's executive

Final plan, written 2026-10-25 before any main run and after all results of record and of the other post-hoc
extensions were known. It is derived from `PLAN_DRAFT.md` (SHA-256 55cc414b…, kept unchanged for the record) with
every fix the lead decided after the adversarial review `REVIEW.md` (SHA-256 578a8427…). README_BUILD.md, section
"Fixes after review (2026-10-25)", lists each change by review ID. The plan binds when the lead writes `PLAN.sha256`
(and `PLAN.utc`). Until then `runner.py launch` refuses. E1 is not part of any frozen family and is labelled post hoc
in the manuscript.

## Question
Reviewers asked whether learning can earn a narrow role inside a handcrafted executive rather than replace it end to
end. Here a learned policy takes over exactly one decision of the tuned transparent rule (UncertaintyFix, sigma 1.5 m):
the correction trigger "relocalise now, or report now?", asked just before a report. Everything else is the rule's own
code.

**On the 80 E1 evaluation episodes (8 core cells × r0-9, 40 of them target-present), does this learned trigger match,
beat or fall short of the tuned rule?**

P1 is conditional on these 80 episodes. All 12 blocks are read on the same episodes. The no-human half is identical to
the rule by construction, and the rule's J is a constant. So the block-to-block variation that P1 tests comes only
from the training seed and contains no episode-sampling variance. A reading is stated for new episodes of these cells
only if the pre-declared episode-level interval (below) agrees with it.

## What is learned and what is not (fixed now)
- **Policy.** `gate_core.GatePolicy` subclasses `gnss_policies.UncertaintyFix` and is built with
  `e1_baselines.p3_rule()['parameters']` (sigma 1.5 m). `gnss_policies.py` is not edited: its SHA-256 is re-checked
  by `p3_rule()` in every process. The idea_dq code tree stays at `ffe9dfef…`, and the precheck, the launch and every
  run check this. Only `should_fix()` is replaced, and the rule's own `act()` calls it at its usual place:
  - The base action is the REPORT of the lowest-id rule-ready open track (at least 4 detections from at least 2
    credited viewpoints), and the per-report fix allowance is left (at most one attempt per report).
  - Gate action 1 (FIX) gives the rule's RELOCALIZE to its cheapest reachable anchor, preferring available ones.
    Gate action 0 gives the rule's REPORT.
  - Search, inspection, evidence bookkeeping, report choice, merge radius and RETURN are unchanged.
  - When no anchor is reachable, the rule reports anyway. The gate is then not asked and nothing is recorded; such
    events are counted.
  - When the executive rejects a RELOCALIZE (none in the 120 stored E1 rule episodes), the same decision is resampled
    with FIX masked, as `RLMissionPolicy` resamples a rejected decision. The final mask is [1, 0] and the log-prob is
    0. Only in this never-observed case does a rule-imitating gate differ from the rule: the rule would re-propose the
    fix until the rejection cap is reached.
- **Observation.** 24 public features, each scaled by a declared constant. T is the cell's mission budget and R its
  return reserve.
  - Georeference: RMS error/3 m (the rule's only input); min(time since fix, T)/T; fixes attempted and fixes accepted
    (each capped at 10 and divided by 10); last fix accepted; no fix yet.
  - Time: elapsed/T, remaining/T, return estimate/T, slack (remaining − return estimate − R)/T, T/2700 s.
  - Anchors: estimated duration of the cheapest reachable anchor/100 s, whether it is available, and the share of
    anchors that are reachable and available.
  - Reports so far (capped at 20)/20.
  - The candidate track: detections (cap 20)/20, credited viewpoints (cap 8)/8, max score, √(cov_xx + cov_yy)/10 m,
    age/T, inspections done/3.
  - The number of other rule-ready open tracks (cap 5)/5, the number of open tracks (cap 10)/10, and the fraction of
    the search plan still pending.

  The rule's own recommendation (rms ≥ 1.5) is *not* a feature. The gate gets the raw inputs only, so it is not a
  residual gate. The gate therefore has 24 features and the rule 1, and every reading below says so where it matters.
- **Model.** `GateActorCritic`, float64, no recurrence.
  - Actor: Linear(24,64)-Tanh-Linear(64,64)-Tanh-Linear(64,2), 5,890 parameters. The last layer is scaled by 0.01,
    so the initial gate is close to a fair coin.
  - Critic: a separate Linear(24,64)-Tanh-Linear(64,64)-Tanh-Linear(64,1), 5,825 parameters.
  - Training starts from scratch: the model is initialised under `torch.manual_seed(SeedSequence([20261025, 1,
    block, 0]))`, with no warm start.
- **SMDP segments** (`gate_core.build_gate_segments`; unit-tested). There is one row per executed gate decision.
  - Extent: segment k runs from gate decision k to gate decision k+1. The last segment runs to the end of the
    episode's last record.
  - Reward: the sum of the `reward-gnss-timely/1` events (`gnss_rewards.place_gnss_rewards`) on the segment's records.
    These are the report increments and the terminal correction, with γ_ref = 1.
  - Identity: kept + pre-gate + invalid-segment reward = J − dropped, asserted on every episode (|residual| ≤ 1e-9).
  - Reward before the first gate was caused by no gate decision. It is excluded from training and counted. In the
    precheck it is 0 on all 80 episodes.
  - Duration: the next gate time minus this gate time. The last segment's duration runs to the end of the last valid
    record. Durations must exceed `smdp.MIN_DURATION_S`, which `RolloutBuffer.add` enforces. The minimum measured over
    the 80 precheck episodes is 1.36 s.
  - Terminal flag: the last segment of a terminated episode is terminal.
  - Infrastructure exits: a segment holding an invalid record is an invalid row (no reward, not terminal). So is the
    last segment of an episode that ended in an infrastructure exit without an invalid record. The valid row before
    it becomes a GAE cut, bootstrapped from its stored value.
  - Episodes with no gate decision (every no-human episode under the rule) give no row.
- **PPO** (`smdp_ppo.compute_advantages` and `ppo_update`, unchanged). Every value is fixed a priori and is **not
  tuned** on any suite:
  - optimiser: Adam with a constant learning rate of **1e-3**, 10 epochs, minibatches of 64 transitions;
  - segments: max_segment_length = 1, because the model is feedforward, so minibatches are shuffled transitions;
  - losses: clip 0.2, value coefficient 0.5, **entropy coefficient 0.01**, max grad norm 0.5 per group;
  - advantage: normalised per update (`ppo_update`), with a std floor of 0.0056 J (0.1 report-equivalent); γ_ref = 1,
    τ = 60 s, λ_ref = 0.95 in time-aware mode;
  - rollout size: 256 gate decisions per update (whole episodes, so 256 to about 264);
  - budget: **12,800 gate decisions** (50 updates), with checkpoints and curve read-outs at **0, 3,200, 6,400, 9,600
    and 12,800** decisions. This is about 2,065 training episodes at the rule's 6.2 gate decisions per training
    episode (7.75 per target-present episode × 0.8).

  *Why these values.*
  - Epochs and minibatch size follow common PPO defaults. The learning rate is raised from the common 3e-4 to 1e-3
    because the budget allows only about 50 updates.
  - The entropy coefficient of 0.01 is the common PPO default (Schulman et al. 2017, Atari setting). Here it is applied
    to normalised advantages. An entropy bonus is the standard guard against a premature collapse to one action.
  - The 0.01 was chosen over the draft's 0 after the builder's quick development check. That check used development
    seeds and e1_development only, and lr 1e-3 with entropy 0. It ran gate_development blocks 2 and 3 for 10 updates
    each, about 2,590 gate decisions per block, with read-outs on e1_development r0: 8 episodes, 4 of them
    target-present.
    - Block 2 went to always-fix. Under argmax its gate fixed on every probe state. Its P(fix) on 32 probe states
      from 4 rule trajectories was 0.72 below 1.5 m and 0.87 above, and its rollout fix rate rose from 0.54 to 0.84.
    - Block 3 became RMS-dependent in the rule's direction: P(fix) 0.40 / 0.78, argmax agreeing with the rule on 78%
      of probe states, rollout fix rate 0.47 → 0.67.
    - On that 8-episode subset (rule 0.2145), the weighted J after 10 updates was 0.1999 (d −0.0147) for block 2 and
      0.1954 (d −0.0191) for block 3.
    - Across both blocks, approx_kl per update was 0.002-0.017, the clip fraction 0.02-0.26, and the critic's
      explained variance rose from 0.07 to 0.90.
  - That choice used development data only and is disclosed here with these numbers. No other setting was changed,
    and no development pilot is run.
- **Seeds.** Two new namespaces follow P3's rule (start = 20260921·1000 + index·100,000) and lie after
  dq_e2_evaluation (index 18):
  - `gate_training`, index 19: [20262821000, 20262921000);
  - `gate_development`, index 20: [20262921000, 20263021000).

  `CellRuntime` refuses any overlap with the frozen, E1 or DQ ranges. No other folder that was checked uses index 19
  or above. idea_y3 is not opened.
  - Salt: (20261025, 1) for the main runs and (20261025, 2) for development. The number 20261025 also seeds three
    unrelated generators, cited by symbol because idea_dq_posthoc_long2 is still being edited:
    - `RNG = np.random.default_rng(20261025)` in idea_dq_posthoc_b1t/b1t_analysis.py;
    - the bootstrap seed `BOOT_SEED` (`BOOT_SEED, BOOT_N = 20261025, 20_000`) in idea_dq_posthoc_long2/e4_analysis.py;
    - the `torch.Generator().manual_seed(20261025)` in `init_checks` of idea_dq_posthoc_long2/runner.py.

    These are different generators, not `SeedSequence([20261025, 1|2, …])`, so nothing collides.
  - Training episode k of block b has seed = gate_training.start + 5,000·b + k. Its cell is drawn with the frozen
    weights (0.8 over the 4 positive core cells, 0.2 over the 4 no-human ones) from SeedSequence([20261025, 1, b, 3,
    k]).
  - Action sampling uses SeedSequence([…, b, 1]) and the minibatch order SeedSequence([…, b, 2]), as in p4_train.
  - The stride of 5,000 is about 2.5 times the expected episodes per run. If it is exhausted, the run is an
    algorithmic failure, as in p4.
- **Read-outs.**
  - **Primary read-out: argmax.** The study protocol, which P4/E1 followed (`p4_train.evaluate`), makes the sampled
    read-out primary for its own reason (idea_dq/docs/study/gnss_qrl_v1/STATISTICAL_PROTOCOL_EN.md, §4.4): "Sampled
    policy is the primary inference mode, reflecting the trained stochastic policy; argmax is a separate secondary
    deployment mode." E1-gate departs from it on purpose. The comparator rule is deterministic, so the deterministic
    gate is the like-for-like policy. The sampled read-out is kept as a descriptive (below).
  - Learning curve: argmax on the development suite `e1_development` (40 episodes; the rule's stored rows are paired)
    at 0, 3,200, 6,400, 9,600 and 12,800 decisions. It is descriptive only: no checkpoint selection and no early
    stopping.
  - Final: the **final checkpoint** (no selection) is read once, by argmax, on `e1_evaluation`: 80 episodes (8 core
    cells × r0-9). Each episode is paired with the stored rule row of the same episode in
    `idea_e1/…/e1_campaign_v2/baselines/baselines.json` (SHA-256 00202039…). The pairing is checked on cell_id, seed
    and crn_sha256.
  - **Sampled read-out (descriptive only).** The final checkpoint, sampled, is also read on the 40 target-present
    e1_evaluation episodes. Episode (cell, r) uses the action stream SeedSequence([20261025, 1, 4, 5, cell_index, r]),
    the same for every block.
    - The no-human episodes are not run. The gate is never asked there: there were 0 gate calls in all 40 stored rule
      episodes, and the trajectory before the first gate call does not depend on the gate.
    - Cost: about 15 min per run at the measured load (40 × 22.6 s), and about 20-25 min under the heavier load
      projected below.
    - It has no test and no reading. If it fails, the error is recorded, and the primary read-out is kept.
- **What was read before this plan.**
  - The only e1_evaluation data read were the stored rule's own 80 rows: in the precheck's reproduction (c), and
    through the figures derived from that reproduction and quoted here. These are 7.75 gate decisions per
    target-present episode and 0 per no-human episode, the minimum segment duration of 1.36 s, 9 no-anchor skips, 0
    resamples and the episode wall times.
  - The reviewer computed the rule's SD of J on the target-present episodes from baselines.json: 0.052.
  - No gate model has been read on e1_evaluation, and none of these figures selected any setting.
  - The quick check and the smokes read e1_development only.

## Runs (12, all new)
- Blocks 0-11 of `gate_training`, one run each. Training, the final argmax read-out and the sampled read-out run in
  the same task. A completed run, whatever its curve, enters as it is.
- **Failures.**
  - An *algorithmic failure* (non-finite value, schedule exhausted) has no final checkpoint. It is recorded and never
    retried, and that block contributes no D_b.
  - Any other exception is an *infrastructure failure*. That block is retried **once** from scratch with the same
    seeds: runs are seed-deterministic, so the retry is outcome-neutral. Both attempts are kept (the first in
    `records/attempts/`, its run directory in `interrupted/`), and the final record lists the earlier one.
  - A run killed by a machine crash leaves no record. On relaunch its directory is moved to `interrupted/` and it
    restarts from the beginning with the same seeds. A block whose first attempt already failed restarts at attempt 2,
    so it never gets a second retry.
  - Infrastructure episodes inside a run stay in as scored and are counted.
- **Records** (`records/gate__b<k>.json`) are written to a temporary file and `os.replace`d. A record that does not
  parse, or is not a final record of its block, counts as no record and is moved to `interrupted/`.
- **One launch at a time.** `launch` holds an exclusive `flock` on `.launch.lock`, and its forked workers inherit it.
  A relaunch is therefore refused while the parent or any worker of an earlier launch is alive.
- **Refusals.**
  - `launch` refuses unless all of these hold:
    - PLAN.md matches `PLAN.sha256`, and every script has the SHA-256 listed below;
    - the idea_dq tree is `ffe9dfef…`;
    - every block's config equals the PLAN constants;
    - `precheck.json` passed for this plan and these scripts.
  - Each run re-checks the tree and its config, and refuses (no record) on a difference.
- No re-run with changed settings will be reported as E1. Any follow-up needs its own, separately hashed plan.

## Precheck before launch (`runner.py precheck`; launch refuses without it, for this plan's SHA-256 and these scripts)
(a) Hashes and constants:
- the idea_dq live tree equals ffe9dfef… before and after;
- `p3_rule()` passes (the gnss_policies.py hash);
- the stored baseline file matches its SHA-256;
- `config(0)`, written into precheck.json as `config_block0`, equals the PLAN constants (`config_check.PLAN_CONSTANTS`);
- the runner's module constants (blocks, stride, namespaces, budget, grid, features, parameter counts) equal the plan;
- every script has the SHA-256 that PLAN.md lists.

(b) The unit tests of `test_gate.py` (14) and `test_analysis.py` (19) pass.
- `test_gate.py` covers the segment builder and the gate bookkeeping:
  - the reward sum equals J − dropped, and the pre-gate reward is 0;
  - durations, the terminal flag, invalid rows and cuts;
  - an invalid record in mid-episode is refused;
  - abandoned decisions and a merged forced RETURN;
  - a re-asked FIX is resampled with FIX masked;
  - checkpoint round trip, and the model running through `ppo_update`.
- `test_analysis.py` covers:
  - the episode-level interval, the scope rule and the reading texts;
  - the insufficient-blocks reading and the failure counts;
  - the 80-key set equality;
  - binding records to the plan and its constants;
  - `config_check`;
  - atomic records, the parse check, the lock, the one-retry rule and the refused dropped commands.

(c) A rule-imitating gate (FIX iff rms ≥ 1.5 m), run through this exact gate code path, reproduces **all 80 stored
e1_evaluation rule rows exactly**.
- It must match every stored field except wall time: J (bitwise), fixes, decisions, crn_sha256, TP/FP, return,
  termination and elapsed time.
- On every one of these episodes, the segment identity must hold, the pre-gate reward must be 0, and the FIX rows must
  equal the fixes attempted.

(d) A training smoke on development seeds (block 19 of gate_development, about 30 episodes, 3 updates of ≥64 rows,
entropy 0.01). It must:
- log every event kind, including the sampled final read-out;
- keep the identity on every episode, and draw its seeds in range;
- reproduce its first argmax read-out episode exactly from the reloaded final checkpoint, and its first sampled
  read-out episode with a fresh generator on the same stream;
- record the per-update series.

Timings are recorded.

After hashing, the lead runs `config_check.py --precheck precheck.json`. It prints a fresh `config(0)` and asserts:
- it equals the precheck's `config_block0` and the PLAN constants;
- the precheck passed;
- the precheck's plan SHA-256 equals `PLAN.sha256`;
- its script hashes equal the table below.

The lead records that comparison.

## Outcome and analysis (fixed now; `gate_analysis.py`)
The analysis refuses unless all of these hold:
- PLAN.md is hashed, and every script has its listed SHA-256;
- every record carries this plan's SHA-256 and a config equal to the PLAN constants;
- every record carries these script hashes, and the idea_dq tree `ffe9dfef…` at its start and end.

It waits for all 12 records (`--provisional` excepted, and labelled as such).

- **P1 (primary).** Per block b, D_b = cell-weighted J(gate, final, argmax) − cell-weighted J(rule) over the 80
  paired e1_evaluation episodes.
  - The weights are the frozen cell weights (0.2 per positive cell, 0.05 per no-human cell), applied through
    `p4_train.aggregate_evaluation`. This is the J scale on which the rule's 0.2056 of record and the margin were
    defined: dq_b2's TOST_DELTA of 0.0056 was applied to AUCs of the frozen-weighted J.
  - The 80 rows of each block must be exactly the 80 stored keys (set equality, no duplicates). Each must match its
    stored rule row on crn_sha256 and on the rule's J.
  - Across the completed blocks (12 planned), the analysis computes:
    - the mean, the 95% paired t interval, the two-sided paired t p-value and dz;
    - **TOST at ±0.0056 J** (0.1 report-equivalent per episode, dq_b2.TOST_DELTA): two one-sided t tests at 0.05,
      which equals a 90% interval inside the margin;
    - Holm over the primary family. The family has one member, so Holm is the identity.
  - The rule submits no report in any of the 40 no-human e1_evaluation episodes. The gate is therefore never asked
    there, so those episodes are identical to the rule's (checked per block), and D_b comes from the positive cells.
- **Episode-level interval (pre-declared; review B1).**
  - For each of the 40 target-present episodes, d_e = the mean over the completed blocks of (J_gate − J_rule).
  - The cell-weighted paired t interval of Σ_e w_e d_e uses w_e = cell weight / episodes in the cell (0.2/10 = 0.02
    each, the weights of the weighted J). It is computed as the one-sample t interval of u_e = n·w_e·d_e, with
    df = n − 1 = 39.
  - With equal weights this is 0.8 × the ordinary paired t interval of the d_e. Its point estimate equals the P1 mean
    exactly, which the analysis asserts.
  - Both its 95% interval and its TOST (90% interval at ±0.0056) are reported.
  - If, against the design, a no-human row differs from the rule's, the interval uses all 80 episodes with their
    weights (0.02 / 0.005), and the departure is reported.
- **Descriptive.**
  - D_b as the unweighted mean over the 80 episodes, with its interval; per-cell mean differences.
  - Per positive episode, gate vs rule: fixes, false reports and timely true reports. Also the return rate,
    infrastructure episodes and gate decisions per episode.
  - The gate's fix rate by RMS bin ([0,1), [1,1.5), [1.5,2), ≥2 m), against the rule's threshold.
  - The e1_development learning curve.
  - Per-update training statistics, from each record's `update_series`:
    - approx_kl: smdp_ppo's, i.e. dq_b2's, pre-step minibatch statistic. It is the mean over the update's minibatch
      steps of the k3 estimate ((r − 1) − log r) of KL(rollout policy ‖ current policy), each taken before its
      optimiser step. It is not D_KL after the update.
    - P(fix) after each update on that update's gate states, below and above 1.5 m.
    - The rollout fix rate, clip fraction and explained variance.
  - Episodes, pre-gate and dropped reward.
  - The sampled read-out: per block, the weighted difference over the 40 target-present episodes, and its mean and
    interval across blocks. There is no test and no reading.
- **S1 is dropped.** It would have needed additional runs: the full 320-row stress-map precheck (about 50 min) and
  12 × 320 stress episodes (about 16 worker-hours). `precheck-stress` and `evaluate-stress` refuse.
- **A0.** The Freeze A replay ceiling (+0.40 report-equivalents, about 0.022 J) re-scores never_fix trajectories as if
  georeferencing were perfect. It is not an upper bound for a gate that changes timing, it is not called headroom, and
  it is quoted at most as a scale.

## Reading rules (fixed now; P1)
Definitions:
- "Rejected" means that the two-sided paired t test is rejected at 0.05 after Holm.
- "Equivalent" means TOST p < 0.05.
- Every reading begins "On the 80 E1 evaluation episodes (40 target-present)". These are the exact sentences
  `gate_analysis.reading_text` writes, with the numbers filled in.

The readings:
- **Equivalent and not rejected → "equivalent."** "…, the learned correction trigger inside the rule's executive (PPO
  with settings fixed a priori and not tuned: learning rate 1e-3, entropy coefficient 0.01, 10 epochs, minibatches of
  64 transitions, clip 0.2, 256 gate decisions per update, 12,800 gate decisions, trained from scratch) matches the
  tuned rule within 0.1 report-equivalent per episode (±0.0056 J): mean, 90% interval, TOST p. Learning recovers the
  tuned rule's performance on these episodes within the margin; it does not improve on it by more than the margin
  (±0.0056 J)."
- **Equivalent and rejected → "equivalent, with a detectable difference smaller than the margin."** "…, the learned
  trigger matches the tuned rule within ±0.0056 J (TOST p), with a detectable gain [or loss] smaller than the margin:
  mean, 95% interval."
- **Rejected with the interval above 0, not equivalent → "better."** "…, the learned trigger exceeds the tuned rule:
  mean, 95% interval. The gate sees 24 features and the rule 1 (the RMS error), so this shows only that a
  richer-information trigger beat the one-feature rule on these episodes; it does not show that the rule's threshold
  is suboptimal within its own one-feature class, nor that the gain comes from learning rather than from the extra
  information."
- **Rejected with the interval below 0, not equivalent → "worse."** "…, the learned trigger falls short of the tuned
  rule: mean, 95% interval. The PPO settings were fixed a priori and not tuned (learning rate 1e-3, entropy
  coefficient 0.01, 10 epochs, minibatches of 64 transitions, clip 0.2, 256 gate decisions per update, 12,800 gate
  decisions, trained from scratch), whereas the rule's sigma was tuned on a grid; this shows that PPO at these
  settings and this budget did not recover the tuned trigger, not that no learned trigger could."
- **Neither → "not resolved."** "…, neither a difference (two-sided paired t, Holm, 0.05) nor equivalence within
  ±0.0056 J (TOST, 0.05) was shown: mean, 95% interval. This is not evidence of equality."
- **Fewer than 8 completed blocks → "not resolved (insufficient completed blocks)."** "…, P1 is not resolved: k of 12
  blocks completed, fewer than the 8 required." The reading is written even with 0 or 1 completed block, without
  statistics.

**Scope (review B1).** A reading is stated beyond these episodes, i.e. also for new episodes of these 8 cells, only if
the episode-level interval agrees with it:
- "better": its 95% interval lies above 0;
- "worse": its 95% interval lies below 0;
- "equivalent": its 90% interval lies inside ±0.0056 (episode-level TOST p < 0.05);
- a detectable gain or loss within the margin: both of the above.

Then the text adds: "The episode-level interval agrees (…), so the reading is also stated for new episodes of these 8
cells; nothing is claimed beyond these cells and this one tuned comparator." Otherwise it adds: "The episode-level
interval does not agree (…), so the reading is conditional on these episodes only." A "not resolved" reading states
nothing beyond the episodes.

**Failures beside the reading.** Every reading ends: "Blocks: k of 12 completed; a ended in an algorithmic failure and
i in an infrastructure failure after one retry." A provisional read-out also adds the number of blocks without a
record.

In every case the comparator is one tuned rule and the result is post hoc. The learned role is deliberately narrow:
7.75 gate decisions per target-present episode (measured under the rule) and none in no-human episodes. A0 is not an
upper bound.

## Compute
Measured on 2026-10-25 under a load average of 16-18 on 16 threads (8 cores); README_BUILD.md has the details.
- Training cost is 3.3 s of wall time per gate decision (quick check), so 12,800 decisions take about 11.7 h.
- Read-outs:
  - the curve: 5 × 40 e1_development episodes, about 0.8 h;
  - the final: 80 e1_evaluation episodes, about 0.3 h;
  - the sampled read-out: 40 episodes, about 0.25 h.
- One run therefore takes about 13 h at that load. Near idle, the same episodes ran about 2× faster (stored E1 rule
  rows: 11.2-11.6 s vs 22.6 s per target-present episode). That gives roughly 6.5 h × s, where the slowdown s is
  about N/9 for N > 9 runnable CPU-bound processes, and 1 below. This is a rough model fitted to two load levels. The
  first post-review precheck (load about 5; its outputs are now in `precheck_prev_20261025T125244Z/`) agrees with it:
  its smoke measured 1.73 s per gate decision in rollout and 11.6 s per sampled read-out episode, and the rule
  reproduction measured 11.3 s per target-present episode.
- There is no resume: a crash restarts a run from zero.

| E1 workers | other load | per run | waves | E1 makespan |
|---|---|---|---|---|
| 6 | none (≈3 background) | ≈6.5-8 h | 2 | ≈13-16 h |
| 10 | none | ≈9-10 h (wave 2: ≈6.5 h) | 2 | ≈16-17 h |
| 6 | E3 (8 workers) | ≈12 h | 2 | ≈25 h |
| 6 | E4 (12 workers) | ≈15 h | 2 | ≈30 h |
| 6 | E4 + E3 (20 workers) | ≈20-21 h | 2 | ≈40-42 h |
| 10 | E4 + E3 | ≈24 h (wave 2: ≈18 h) | 2 | ≈42 h, and E4/E3 slow down most |
| 4 | E4 + E3 | ≈19-20 h | 3 | ≈58-60 h |

Any worker count from 6 to 11 needs two waves for 12 runs. Six workers therefore give the same number of waves as ten
with less contention, both for E1 and for the other campaigns. Memory is about 0.54-0.6 GB per worker: 6 workers need
about 3.6 GB and 10 about 6 GB.

## Binding: scripts, data and constants
The runs and the analysis execute exactly these scripts of this folder (SHA-256 after the last edit;
`config_check.py` and `gate_analysis.py` refuse on any difference):

| file | sha256 |
|---|---|
| `runner.py` | `6199562538131ed985b05915c468d91b041d0c1c5b1e2b2144d969b37d28a6a2` |
| `gate_core.py` | `3d052478dc088624de904634a7f1775f42a1efd6007fd8db8561c4224668cc1e` |
| `test_gate.py` | `08b83f19afebe7ef2c6bb856a88ccea6f51667f7c13e07f371849946ee60ab61` |
| `test_analysis.py` | `d680570e7fabf2cfef2fadf56584c835c56200e37bafcc9a38202bde7ea8c1aa` |
| `gate_analysis.py` | `77ee337fb649791a0d404f56cc15f41a78456665b101508b90b8b8fd16bd92ea` |
| `config_check.py` | `12efd83f44465aed85d494221ff746ad8bc947aa819241c28517952a4a319d1e` |

Other bound values:
- The idea_dq code tree: `ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952`, unchanged and never
  written.
- The stored rule rows: `baselines.json`, SHA-256 `00202039dda01d814146db77e85590996abda974b4e813bbb17c10d081572fd7`.
  Their weighted J on e1_evaluation is 0.20564463190957744, re-derived by `gate_analysis.py --selftest`.
- The PLAN constants are `config_check.PLAN_CONSTANTS`. Each main-run config must equal them exactly:
  - budget 12,800 and grid [0, 3,200, 6,400, 9,600, 12,800];
  - 256 decisions per update;
  - lr 1e-3, entropy 0.01, clip 0.2, value 0.5, grad norm 0.5, 10 epochs, minibatch 64, advantage floor 0.0056,
    max segment length 1;
  - γ 1, τ 60 s, λ 0.95 time-aware;
  - argmax primary, with the sampled read-out descriptive;
  - curve on e1_development and final on e1_evaluation;
  - salt (20261025, 1), stride 5,000 and blocks 0-11;
  - from scratch, the rule at sigma 1.5 m.

## Reporting commitment
Whatever the outcome, the manuscript reports, as a post-hoc check next to the systems conclusion:
- the P1 reading with its interval and TOST p;
- the episode-level interval and the scope it allows;
- the block tally;
- the descriptive panel.

The report also gives this plan's SHA-256 and the precheck result. Scripts, records and checkpoints go to the
repository. No re-run with changed settings will be reported as E1.
