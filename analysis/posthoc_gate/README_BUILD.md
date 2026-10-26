# E1 build record: a learned correction trigger inside the tuned rule's executive

Built 2026-10-25 by a Claude Code subagent. No main run was started. `PLAN_DRAFT.md` is a draft; it is not hashed,
and `runner.py launch` refuses until `PLAN.md` and a matching `PLAN.sha256` exist and a precheck has passed for that
hash and this code.

## Files
| file | what it is |
|---|---|
| `gate_core.py` | `GatePolicy` (a subclass of `gnss_policies.UncertaintyFix` in which only `should_fix` is replaced), `GateBook` (decision bookkeeping, rejected-FIX resampling), `RuleGate` / `NetworkGate`, `GateActorCritic` (MLP actor and critic, no GRU), `build_gate_segments` (SMDP rows), `check_segment_identity`, checkpoint I/O |
| `runner.py` | studies and namespaces, config, `run_gate_episode`, `evaluate_suite`, `train_gate` (adapted from `p4_train.train_arm`), and subcommands `test`, `precheck`, `smoke`, `launch`, `precheck-stress`, `evaluate-stress`, `pilot` |
| `test_gate.py` | 14 unit tests on synthetic records, run standalone or under pytest |
| `gate_analysis.py` | P1/S1 analysis as fixed in the plan: paired t, TOST at ±0.0056, Holm, readings. `--selftest`, `--dryrun`, `--provisional` |
| `PLAN_DRAFT.md` | the design for review: question, runs, precheck, outcome, reading rules, compute, reporting |
| `precheck_draft.json`, `precheck_draft.log`, `smoke_draft/` | the draft precheck I ran (plan_sha256 = null) |
| `precheck_stress_draft_1percell.json` | partial stress-map reproduction (16 of 320 rows) |
| `pilot_quick/`, `pilot_quick.log` | the quick learning check on development seeds (see below) |

Nothing outside this directory and the scratchpad was written. Every Python process ran with `-B` and
`sys.dont_write_bytecode`. The idea_dq live tree hash was `ffe9dfef2b24…f7745952` before and after the precheck, and
again at the end. `gnss_policies.py` was not touched: `e1_baselines.p3_rule()` passes, with policy_source_sha256
a120c2d2….

## What I ran, and what passed
Every command was run as `nice -n 10 /home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python -B …`, with at most 2
concurrent processes of my own. Another campaign (b2f, 10 workers) and other agents' prechecks were running at the
same time: load average 16-18 on 16 threads.

1. **`test_gate.py`: 14 of 14 passed.**
   - Segment rewards: the reward sum equals J − dropped; pre-gate reward is 0; a non-zero pre-gate reward is counted
     and excluded.
   - Durations and the terminal flag are correct.
   - Invalid rows: an invalid last record gives an invalid row and a GAE cut, bootstrapped from that row's value
     (checked through `compute_advantages`). An infrastructure exit without an invalid record gives an invalid last
     row. An invalid record in the middle of an episode is refused.
   - Abandoned or unbound decisions are skipped. A forced RETURN is merged into the last segment. An episode with no
     gate decision gives no rows.
   - A re-asked FIX is resampled in place with FIX masked: final mask [1,0], log-prob exactly 0, one row, accepted by
     `RolloutBuffer`. A new observation abandons the old decision.
   - Rule-gate threshold.
   - Model: parameter count, near-fair-coin initialisation, a real `ppo_update` step.
   - Checkpoint round trip, and no overwrite.
2. **`runner.py precheck --workers 2 --tag _draft`: passed** (`precheck_draft.json`; 1,713 s wall).
   - (a) Hashes: tree before and after = ffe9dfef…; `p3_rule()` OK; baselines.json SHA-256 00202039… OK.
   - (b) Unit tests: 14 of 14.
   - (c) **The rule-imitating gate (rms ≥ 1.5) through the gate code path reproduced all 80 stored e1_evaluation rule
     rows exactly.**
     - Every stored field matched: J (repr), fixes_attempted, decisions, crn_sha256, seed, tp_timely, tp_spatial,
       fp_spatial, duplicates, return_ok, termination_reason, rejections, elapsed_s.
     - On all 80 episodes the segment identity residual is 0, the pre-gate reward is 0, and FIX rows equal fixes
       attempted.
     - 310 gate rows: 7.75 per target-present episode, 0 per no-human episode.
     - Minimum segment duration 1.356 s, above MIN_DURATION_S = 1e-6. The scout's "15.1 s" was based on 3 episodes.
     - 0 resamples. 9 "no anchor reachable" skips; there the rule reports without a fix, the gate is not asked, and
       the rows still match.
   - (d) Smoke (development study, gate_development block 19, 3 updates of ≥64 rows): passed.
     - 37 training episodes, 199 gate decisions.
     - All event kinds were logged (arm_start 1, episode 37, update 3, checkpoint 3, evaluation 2, final_evaluation 1,
       arm_end 1).
     - Identity residual max 0.0, seeds inside the development range, 0 infrastructure episodes.
     - The reloaded final checkpoint re-ran its first read-out episode identically in every field.
     - Max RSS 0.54 GB.
3. **`runner.py precheck-stress --workers 2 --per-cell 1`: 16 of 16 stored stress rule rows reproduced**, one per
   cell, including the out-of-training drifts 0 and 8 m (146 s). The full 320-row reproduction, which
   `evaluate-stress` requires, was not run.
4. **`gate_analysis.py --selftest`: passed.**
   - The stored rule J of record, 0.20564463190957744, is re-derived exactly through `p4_train.aggregate_evaluation`.
   - The t p-value and TOST p equal scipy's `ttest_1samp`, and TOST agrees with the 90% interval.
   - Holm and the reading table were checked.
5. **`gate_analysis.py --dryrun`** (12 synthetic records built from the stored rule rows, written to the scratchpad):
   the whole read-out ran. No-human rows were identical to the rule, and the pairing checks passed.
6. **`runner.py launch` and `evaluate-stress` without a hashed plan both refuse**, and no directory is created.
7. **Quick learning check** (`runner.py pilot --lrs 1e-3 --blocks 2 3 --budget 2560 --workers 2 --quick`):
   development seeds and e1_development r0 only; see the section below.

## Timings measured (load average 16-18 on 16 threads, nice 10)
| quantity | value |
|---|---|
| rule episode through the gate path (precheck, 2 workers), target-present | 22.6 s mean |
| rule episode, no-human | 4.1 s mean |
| 80-episode e1_evaluation pass, 2 workers | 556 s wall |
| training episode (smoke, near-random gate), all cells | 20.7 s mean (target-present 28.3 s, no-human 5.1 s) |
| rollout per 1,000 gate decisions (smoke, 5.4 rows per episode) | 3,860 s |
| PPO update (64 rows, 10 epochs) | 0.05 s |
| read-out episode (argmax, e1_development) | 14.7 s mean |
| 16 stress episodes, 2 workers | 146 s wall |
| reference: the stored E1 rule rows (4 workers, near idle) | target-present 11.2-11.6 s, no-human 2.2-3.0 s |

Projection for one main run: 12,800 gate decisions is 2,065 training episodes at the rule's 6.2 rows per episode, or
2,370 at the smoke's 5.4. At the 3.3 s per gate decision measured in the quick learning check, plus 280 read-out
episodes at about 15 s each, that is **about 12.8 h at today's measured load (range 12.5-15 h)**. With E1 alone on 12
workers it should be about 10-12 h; that figure is an inference (about 25% less contention), not a measurement.
- Full campaign: one wave of 12 runs on 12 workers, about 130-170 worker-hours.
- Precheck re-run after hashing: about 30 min on 2 workers at this load (more workers make it faster: the 80
  episodes are 8 per-cell tasks).
- `precheck-stress`, full 320 rows: about 50 min on 2 workers.
- `evaluate-stress` for 12 finals: about 16 worker-hours.

## Quick learning check on development seeds
Command: `runner.py pilot --lrs 1e-3 --blocks 2 3 --budget 2560 --workers 2 --quick`. It used the a-priori settings,
gate_development blocks 2 and 3, and 10 updates (about 2,590 gate decisions and 410-430 episodes per block). Read-outs
were argmax on e1_development r0 only: 8 episodes, 4 of them target-present, so they are very noisy. e1_evaluation
was not touched. Each block took 8,628-8,761 s wall. Output is in `pilot_quick/`. A read-only probe
(`/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad/pilot_probe.py`) scored the checkpoints on the 32 gate states of 4 rule trajectories (e1_development r1).

| | block 2 | block 3 |
|---|---|---|
| dev-subset weighted J at 0 (rule 0.2145) | 0.058 (d −0.156) | 0.201 (d −0.013; argmax was always-fix) |
| after 10 updates | 0.1999 (**d −0.0147**) | 0.1954 (**d −0.0191**) |
| P(fix) on probe states, rms < 1.5 / ≥ 1.5 | 0.72 / 0.87 (argmax: always fix) | 0.40 / 0.78 (argmax agrees with the rule on 78% of states) |
| rollout fix rate, update 1 → 10 | 0.54 → 0.84 | 0.47 → 0.67 |
| critic explained variance, update 1 → 10 | 0.07 → 0.90 | 0.07 → 0.90 |
| approx_kl per update | 0.002-0.017 | 0.004-0.016 |
| clip fraction per update | 0.03-0.23 | 0.02-0.26 |

What this shows:
- PPO on the gate learns. The critic becomes informative (explained variance 0.9).
- P(fix) moves away from the rule's "never fix" failure mode. In one block it becomes RMS-dependent in the rule's
  direction. In the other it converges towards "always fix", which in these cells costs about 0.013-0.015 J relative
  to the rule.
- After 20% of the planned budget, both blocks are 0.015-0.019 J below the rule on the 8-episode dev subset. That is
  about 3× the TOST margin.
- These blocks do not say whether the remaining 80% of the budget closes the gap.
- On this evidence the most likely P1 readings are "worse" or "not resolved"; "equivalent" needs every block to
  converge close to the rule's threshold.
- Nothing in PLAN_DRAFT.md was changed because of this check. Possible a-priori changes, each based on development
  data only and each to be decided before hashing: an entropy bonus against the always-fix attractor; or giving the
  gate the rule's flag (rms ≥ 1.5) as a feature, which makes it a residual gate. The second makes the question
  easier and the claim weaker.

Throughput in this check: 3.3 s of wall time per gate decision, 6.1 gate rows per training episode, 17-22 s per
training episode, at load average 17-18.

## Open risks (honest list)
1. **Learning reaches the rule only partly by 20% of the budget.**
   - The quick check (2 development blocks, 10 updates) learned something: one block an RMS-dependent trigger, the
     other "always fix".
   - Both blocks were still 0.015-0.019 J below the rule on a tiny dev subset; the TOST margin is 0.0056.
   - The early gate can also collapse to "report now", which is never_fix, under argmax (smoke: d4 nominal J −0.44).
   - The most likely P1 readings are "worse" or "not resolved". The plan's reading rules already state what each
     one means.
2. **Budget vs wall time.** A run fits in 12 h only if E1 runs alone on 12 workers. At today's load a run takes
   about 12.8 h (range 12.5-15 h). Before hashing, either keep 12,800 and accept that, or use the 10,240 alternative written
   into the plan.
3. **Hyperparameters were set a priori** (lr 1e-3, 10 epochs, minibatch 64, entropy 0, advantage floor 0.0056, 256
   rows per update). They are not tuned. Per-update approx_kl was 0.005-0.014 in the smoke on 64-row batches; with
   256 rows (40 optimiser steps) it may be larger. It is logged, and clipping bounds it.
4. **Primary scale.**
   - I made P1 the cell-weighted difference (0.2/0.05, `aggregate_evaluation`). That is the scale of the rule's 0.2056
     and of the 0.0056 margin.
   - Read literally, the brief's "mean over the 80 paired episodes" is the unweighted mean, which here equals 0.625 ×
     the weighted one, because the no-human differences are exactly 0.
   - Both are computed; the lead should confirm which one is primary before hashing.
5. **Small effect range and unknown power.**
   - 40 of the 80 evaluation episodes are identical to the rule by construction.
   - The scout's (unchecked) P3 tuning-grid neighbours of the rule differ from it by −0.011 to −0.017 J. A learned
     gate that behaves like a nearby threshold may therefore land outside ±0.0056 ("worse").
   - The between-block SD of D_b is unknown, so the TOST power is unknown.
6. **No resume.** A machine crash restarts a 10-15 h run from zero (moved to `interrupted/`, same seeds).
7. **Stress map (secondary).**
   - Its evaluator calls `dq_freeze_a.stress_cells` and `stress_seed` directly, bypassing `dq.require_open` and its
     calendar cut-off of 2026-10-31. The post-hoc runners already do this.
   - Only 16 of the 320 stored rule rows have been reproduced. `evaluate-stress` refuses until all 320 are
     (`precheck-stress` with no `--per-cell`, after hashing).
8. **Paths that never ran on real episodes.**
   - Rejected-RELOCALIZE resampling and the infrastructure rows are covered by synthetic tests only: there were 0
     rejections and 0 infrastructure episodes in 80 + 37 + 16 episodes.
   - In the rejected-RELOCALIZE case the gate deliberately departs from the rule: FIX is masked, while the rule would
     retry until the rejection cap.
9. **Private attributes of the rule are read.** The features read `_credited`, `_inspections` and `status` from the
   rule; reading has no side effects. The candidate track is recomputed by a pure copy of the rule's report branch.
   The 80/80 reproduction shows it agrees, because `should_fix` would raise otherwise.
10. **Seed namespaces.** I found no use of indices 19 and 20 (gate_training, gate_development) in idea_dq or the
    post-hoc folders. *Corrected after review (minor 9; cited by symbol after round 2, because
    idea_dq_posthoc_long2 is still being edited and line numbers move):* the number 20261025 is used as a plain RNG
    seed in three places: `RNG = np.random.default_rng(20261025)` in `idea_dq_posthoc_b1t/b1t_analysis.py`; the
    bootstrap seed `BOOT_SEED` (`BOOT_SEED, BOOT_N = 20261025, 20_000`) in `idea_dq_posthoc_long2/e4_analysis.py`;
    and the `torch.Generator().manual_seed(20261025)` in `init_checks` of `idea_dq_posthoc_long2/runner.py`. These are
    different generators, not `SeedSequence([20261025, 1|2, …])`, so nothing collides. idea_y3 was not checked; the
    rules forbid opening it.
11. **The draft precheck does not authorise a launch.** `precheck_draft.json` has plan_sha256 null, and `runner.py`
    changed afterwards (the pilot quick mode was added). After the lead hashes PLAN.md, `runner.py precheck --workers
    N` must be run again: it writes `precheck.json` and `smoke/`, and launch checks the plan hash and the
    runner/gate_core/test hashes.
12. **Timings were taken on an oversubscribed machine.** Load was 16-18 on 8 physical cores and 16 threads, with
    other agents active, so per-episode times are about 2× the idle values.

## Fixes after review (2026-10-25)
A Claude Code subagent applied these after `REVIEW.md`, following the lead's binding decisions, item by item. No
main run was started, and no outcome of any pilot, smoke or run was read for these fixes. The quick check's numbers
quoted in PLAN.md are the ones already recorded above. `PLAN_DRAFT.md` is unchanged. The final plan is `PLAN.md`; it
is not hashed (the lead writes `PLAN.sha256` and `PLAN.utc`).

### Blocking items
- **B1 (P1 is conditional on 40 fixed episodes).**
  - The Question and every reading now state that P1 is conditional on the 80 E1 evaluation episodes (40
    target-present). Every reading sentence that `gate_analysis.reading_text` writes begins "On the 80 E1 evaluation
    episodes (40 target-present)".
  - New in `gate_analysis.py`, `episode_level()`:
    - per target-present episode, d_e = mean over the completed blocks of (J_gate − J_rule);
    - the cell-weighted paired t interval of Σ w_e d_e, with w_e = cell weight / episodes in the cell (0.02 each),
      computed as the one-sample t of u_e = n·w_e·d_e, df 39. This equals 0.8 × the paired t interval of the d_e;
    - its point estimate equals the P1 mean, which the analysis asserts;
    - the 95% interval and an episode-level TOST are reported;
    - if, against the design, a no-human row differs from the rule's, the interval uses all 80 episodes with their
      weights.
  - New reading rule, `generalises()`. A reading is stated for new episodes of these 8 cells only if the
    episode-level interval agrees with it:
    - better: 95% interval > 0;
    - worse: 95% interval < 0;
    - equivalent: 90% interval inside ±0.0056;
    - a detectable gain or loss within the margin: both.
    Otherwise the text says the reading "is conditional on these episodes only". "Not resolved" states nothing beyond
    the episodes.
  - S1 was dropped (see B3), so the episode-level interval is the only generalisation check.
  - Tests, in `test_analysis.py`:
    - the scaled-t identity against scipy, and the estimate = P1 mean;
    - the 80-episode fallback, and blocks read on different episodes refused;
    - the scope table;
    - the "common policy" case the reviewer described: blocks converge to one non-rule policy, so P1 says "worse" at
      p ≈ 0, while the episode-level 95% interval contains 0 and the reading stays conditional.
- **B2 (two readings overclaimed).** Reworded in PLAN.md and in `reading_text`:
  - "Worse" now says the PPO settings were fixed a priori and not tuned, lists them (lr 1e-3, entropy coefficient
    0.01, 10 epochs, minibatch 64, clip 0.2, 256 decisions per update, 12,800 decisions, from scratch), and says that
    the rule's sigma was tuned on a grid.
  - "Better" now says the gate has 24 features and the rule 1, so a gain shows only that a richer-information trigger
    beat the one-feature rule on these episodes. It says that neither threshold suboptimality within the rule's class
    nor learning (rather than information) is shown. "The tuned trigger is not optimal" is deleted.
  - Added in Runs and in the Reporting commitment: "No re-run with changed settings will be reported as E1. Any
    follow-up needs its own, separately hashed plan."
  - Tests check these phrases.
- **B3 (open alternatives).** Every alternative is closed.
  - Budget: 12,800 gate decisions with grid [0, 3,200, 6,400, 9,600, 12,800]. The 10,240 alternative is deleted.
  - No development pilot (compute). `runner.py pilot` now refuses.
  - PPO fixed a priori: lr 1e-3, **entropy coefficient 0.01 (was 0)**, from scratch. The config gains an
    `initialisation` field stating it. PLAN.md gives the justification, based on development data only and disclosed
    with the quick check's numbers: one block went to always-fix under entropy 0, and an entropy bonus is the
    standard guard.
  - S1 dropped, because it needs additional runs: `precheck-stress` and `evaluate-stress` now refuse, and the S1 code
    is removed from `gate_analysis.py`.
  - P1 scale: the weighted J. The "lead should confirm" hedge is deleted.
  - Binding:
    - New `config_check.py` holds `PLAN_CONSTANTS` (the design values PLAN.md states). `runner.py` refuses to import
      if its BUDGET, GRID, PPO, tree or baselines hash differ from them.
    - `launch` and every run assert each block's config against them.
    - The precheck writes `config_block0` into its JSON and checks it.
    - `config_check.py [--precheck precheck.json]` prints a fresh `config(0)` and asserts the constants, the module
      constants, the tree and the script hashes PLAN.md lists. With `--precheck`, it also asserts that config(0)
      equals the precheck's, that the precheck passed, and its plan hash and script hashes.

### Minor items
1. **Compute.** PLAN.md restates the per-run estimate as about 13 h at the measured load (6.5 h × slowdown). It adds
   a table for 4, 6 and 10 E1 workers alongside E3 and/or E4. It recommends 6 workers: from 6 to 11 workers it is
   always two waves for 12 runs, and 6 means less contention.
2. **Atomic records.**
   - `write_json_atomic` writes to a temporary file, fsyncs it, then `os.replace`s it. It is used for records,
     attempts, the precheck JSON and the smoke JSON.
   - `read_record` and `pending_blocks` parse-check existing records. An unparsable or foreign record counts as no
     record and is moved to `interrupted/<utc>/`.
3. **Single-instance lock.** `acquire_lock` takes a non-blocking exclusive `flock` on `.launch.lock`, held for the
   life of `launch`. Forked workers inherit the descriptor, so a relaunch is refused while any worker of an earlier
   launch is alive. A test checks this with a forked child.
4. **Failure policy.**
   - An infrastructure failure (any exception other than the algorithmic ones) is retried once from scratch with the
     same seeds. The first attempt's record goes to `records/attempts/` and its run directory to `interrupted/`, and
     the final record lists it.
   - Algorithmic failures are never retried.
   - After a crash, a block whose attempt 1 already failed restarts at attempt 2.
   - The launch loop is now `drive()`, and it is tested with a fake task.
5. **Tree refusal.**
   - `launch` refuses unless the idea_dq tree is `ffe9dfef…`.
   - Each run re-checks the tree and its config at start and refuses (no record) on a difference, and it records the
     tree at its end.
   - The analysis requires the tree at both start and end of every completed run.
6. **Analysis bound to the plan.**
   - `gate_analysis.main` calls `config_check.bound_plan()`. PLAN.md must match `PLAN.sha256`, and every script must
     have the SHA-256 that PLAN.md lists, including `gate_analysis.py` itself.
   - Each record must carry the plan's SHA-256 (top level and in its config), a config equal to the PLAN constants,
     and the listed script hashes.
   - `_require_launchable` no longer exempts `gate_analysis.py`.
7. **Distinct keys.** `block_contrast` requires the 80 rows to be exactly the 80 stored keys, with no duplicates.
8. **Descriptives.**
   - Each run now keeps an `update_series` in its summary: approx_kl, clip fraction, entropy, explained variance,
     rollout fix rate, and P(fix) after the update below and above 1.5 m with their counts.
   - `gate_analysis` extracts approx_kl and P(fix) per update. approx_kl is labelled as smdp_ppo's (dq_b2's) pre-step
     minibatch k3 statistic, not D_KL after the update.
9. **Salt sentence (risk 10).** Corrected in place above.
10. **What was read from e1_evaluation.** PLAN.md now states it: the stored rule's own 80 rows only (the reproduction
    and the figures derived from it), plus the reviewer's SD of 0.052. No gate model was read on e1_evaluation, and
    nothing was selected.
11. **Argmax primary.**
    - PLAN.md gives the one-line justification: the comparator rule is deterministic, so the deterministic gate is
      the like-for-like policy.
    - A sampled read-out is added as a descriptive. It runs on the 40 target-present e1_evaluation episodes, with
      stream SeedSequence([20261025, 1, 4, 5, cell, r]), at about 15 min per run at the measured load. The no-human
      episodes are not run, because the gate is never asked there.
    - It is wrapped so that its failure cannot cost the primary read-out. The smoke now checks it, including a
      reload re-run on the same stream.
12. **Development pilot.** Not run (lead decision). The entropy bonus of B3 is the a-priori guard instead.
13. **Fewer than 2 blocks.** `main` now writes "not resolved (insufficient completed blocks)", with the counts and no
    statistics, instead of exiting.
14. **Failures beside the reading.** Every reading text ends with the block tally: completed, algorithmic failures,
    infrastructure failures after one retry, and blocks without a record if provisional.

### Files
| file | before | after |
|---|---|---|
| `runner.py` | fd354bce6209a2c8… | 6199562538131ed985b05915c468d91b041d0c1c5b1e2b2144d969b37d28a6a2 |
| `gate_analysis.py` | 36ef069df3e7be5f… | 77ee337fb649791a0d404f56cc15f41a78456665b101508b90b8b8fd16bd92ea (after round 2; round 1 was 1fa69693…) |
| `test_gate.py` | ee399259131761e0… | 08b83f19afebe7ef2c6bb856a88ccea6f51667f7c13e07f371849946ee60ab61 (only the scratch directory of the checkpoint test) |
| `gate_core.py` | 3d052478dc088624… | unchanged |
| `config_check.py` | new | 12efd83f44465aed85d494221ff746ad8bc947aa819241c28517952a4a319d1e |
| `test_analysis.py` | new (19 tests) | d680570e7fabf2cfef2fadf56584c835c56200e37bafcc9a38202bde7ea8c1aa (after round 2; round 1 was fab07a47…) |
| `PLAN.md` | new | 20faccb84e4f9c79bd821af38a2190b0f0c110b2ac7c61f49e05879857e67a7c at hand-over (after round 2; round 1 was e47b923c…; the lead hashes it) |
| `README_BUILD.md` | | risk 10 corrected (by symbol after round 2); this section |

Moved, not deleted:
- `precheck_draft.json`, `precheck_draft.log` and `smoke_draft/` are now in `precheck_prev_20261025T120715Z/`.
- Copies of the four pre-fix scripts are in `precheck_prev_20261025T120715Z/code_before_review_fixes/`.
- Round 2: the first post-review `precheck_postreview.json`, `precheck_postreview.log` and `smoke_postreview/` are
  now in `precheck_prev_20261025T125244Z/`.

New outputs:
- `precheck_postreview.json`, `precheck_postreview.log` and `smoke_postreview/` (see below).
- `.launch.lock`, left by the check that `launch` refuses without a plan.

`precheck_stress_draft_1percell.json` stays where it was, as the record of the dropped S1.

### Precheck after the fixes, first run (`runner.py precheck --workers 2 --tag _postreview`: passed)
Its outputs are now in `precheck_prev_20261025T125244Z/` (moved in round 2, below; not deleted).
This precheck was run before PLAN.md existed, so its plan_sha256 is null and it cannot authorise a launch. The lead
runs the precheck of record after hashing. Its wall time was 881 s, at a load average of about 5.
- (a) Hashes and constants:
  - the tree was ffe9dfef… before and after;
  - `p3_rule()` and the baselines.json hash passed;
  - `config(0)` (written as `config_block0`) has no mismatch with `PLAN_CONSTANTS`, and the module constants have
    none either (entropy_coef 0.01, budget 12,800, grid [0, 3,200, 6,400, 9,600, 12,800], config version
    e1-gate-config/2).
- (b) Unit tests: 33 of 33 (`test_gate.py` 14, `test_analysis.py` 19). They also pass under pytest, and the
  in-memory mutation checks in the scratchpad (`fix_e1_gate/mutation_check.py`) show that the new tests fail when
  the rules they cover are broken.
- (c) **The rule-imitating gate reproduced all 80 stored e1_evaluation rule rows exactly.**
  - Segment identity held on every episode, the pre-gate reward was 0 on all, and FIX rows equal the fixes attempted.
  - 310 gate rows: 7.75 per target-present episode and 0 per no-human episode.
  - Minimum segment duration 1.356 s, 0 resamples, 9 no-anchor skips.
  - Wall time 279 s, at 11.3 s per target-present and 2.1 s per no-human episode.
- (d) Smoke passed (entropy 0.01):
  - 37 episodes, 3 updates, 199 gate decisions, seeds in the development range, identity residual 0;
  - every event kind was logged, including `final_evaluation_sampled`;
  - the argmax reload re-run and the sampled reload re-run (same stream, fresh generator) were identical;
  - the update series has 3 entries.
  - Timing: 9.3 s per training episode, 1,730 s per 1,000 gate decisions, and 46.6 s for the 4-episode sampled
    read-out. No outcome value of the smoke was read.
- Other checks:
  - `gate_analysis.py --selftest` and `--dryrun` (synthetic records) pass.
  - `config_check.py` against the written PLAN.md reports ok: the constants, the module constants, the tree, and
    the six scripts at their listed SHA-256.
  - `config_check.py --precheck precheck_postreview.json` flags only the expected plan hash, which is null in this
    precheck. `config_block0` differs from a fresh `config(0)` only in `plan_sha256`.
  - `runner.py launch` (no plan), `evaluate-stress`, `precheck-stress` and `pilot` all refuse.

### Round 2: defects found by the independent verifier
An independent verifier checked PLAN.md, this file and the scripts after the fixes above and found three defects.
All three are fixed below. No outcome value of any pilot, smoke or run was read for them. Only status, counts, hashes
and timings of the new precheck were read.

- **V1 (minor 9, risk 10): stale line numbers.**
  - PLAN.md (Seeds, Salt) and risk 10 above cited the other uses of 20261025 by line number. long2 is still being
    edited by its own agent, and two of those numbers had moved: `BOOT_SEED` is now at `e4_analysis.py:61`, not
    :33, and the `torch.Generator` seed at `runner.py:296`, not :165.
  - Both texts now cite them by symbol: `RNG` in b1t_analysis.py, `BOOT_SEED` in long2/e4_analysis.py, and the
    `torch.Generator().manual_seed(20261025)` in `init_checks` of long2/runner.py.
  - A read-only grep of the Python files of idea_dq and of every idea_dq_posthoc_* folder confirms the substance.
    These are still the only three other uses (long2's `provenance/` holds older copies of the same two lines), and
    none is a `SeedSequence([20261025, 1|2, …])`.
  - Text only.
- **V2 (minor 11): wrong reason for P4/E1's primary read-out.**
  - PLAN.md said that P4/E1 made the sampled read-out primary "because there the learned policy replaced the whole
    executive".
  - It now quotes the protocol's own reason (idea_dq/docs/study/gnss_qrl_v1/STATISTICAL_PROTOCOL_EN.md §4.4, line
    137, which `p4_train.evaluate` cites): "Sampled policy is the primary inference mode, reflecting the trained
    stochastic policy; argmax is a separate secondary deployment mode."
  - It says plainly that E1-gate departs from the protocol on purpose, and keeps the like-for-like justification:
    the comparator rule is deterministic.
  - Text only.
- **V3 (B2): the "equivalent" sentence claimed more than the test shows.**
  - That reading is TOST p < 0.05 without rejection: the 90% interval lies inside ±0.0056 J and the 95% interval
    contains 0. That does not exclude a gain up to the margin, so "it does not improve on it" claimed too much.
  - The sentence now ends: "Learning recovers the tuned rule's performance on these episodes within the margin; it
    does not improve on it by more than the margin (±0.0056 J)." It is changed in PLAN.md (Reading rules) and in
    `gate_analysis.reading_text`.
  - `test_reading_texts_are_conditional_and_qualified` now asserts the new phrase, and asserts that the old ending is
    absent.
  - An in-memory mutation that restores the old sentence makes the test fail
    (`/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad/fix_e1_gate/mutation_check_round2.py`).
  - Both files run in precheck (b). Their SHA-256 are updated in the PLAN.md table and in the files table above. The
    first post-review precheck's outputs were moved aside, and the precheck was re-run (below).
- **Consequential text change in PLAN.md (Compute).** The timings quoted from "the post-review precheck" now say "the
  first post-review precheck" and where its outputs are. No figure changed.

PLAN.md was final before the new precheck started. The precheck's `plan_sha256` therefore equals the PLAN.md
SHA-256 at hand-over.

Files changed in round 2 (SHA-256 after the last edit):

| file | round 1 | after round 2 |
|---|---|---|
| `gate_analysis.py` | 1fa69693a013d3e2… | 77ee337fb649791a0d404f56cc15f41a78456665b101508b90b8b8fd16bd92ea |
| `test_analysis.py` | fab07a47b1f9f9c7… | d680570e7fabf2cfef2fadf56584c835c56200e37bafcc9a38202bde7ea8c1aa (still 19 tests) |
| `PLAN.md` | e47b923c88682ae3… | 20faccb84e4f9c79bd821af38a2190b0f0c110b2ac7c61f49e05879857e67a7c at hand-over (not hashed; the lead writes `PLAN.sha256`) |
| `README_BUILD.md` | | risk 10, the files table, the first precheck heading, the move list, this section |

Unchanged:
- `runner.py` 6199562538131ed985b05915c468d91b041d0c1c5b1e2b2144d969b37d28a6a2
- `gate_core.py` 3d052478dc088624de904634a7f1775f42a1efd6007fd8db8561c4224668cc1e
- `test_gate.py` 08b83f19afebe7ef2c6bb856a88ccea6f51667f7c13e07f371849946ee60ab61
- `config_check.py` 12efd83f44465aed85d494221ff746ad8bc947aa819241c28517952a4a319d1e
- `PLAN_DRAFT.md` 55cc414b…

### Precheck after round 2 (`runner.py precheck --workers 2 --tag _postreview`: passed)
Its outputs are `precheck_postreview.json`, `precheck_postreview.log` and `smoke_postreview/`. Its plan_sha256 is
20faccb8… (PLAN.md at hand-over), but PLAN.md is not hashed and the precheck is tagged, so it cannot authorise a
launch. The lead runs the precheck of record, untagged, after hashing.
- Wall time 1,435 s, at a load average of about 13 (other campaigns running: b2k launched 8 workers, and long2 had
  3 pilots).
- (a) Hashes and constants:
  - the tree was ffe9dfef… before and after;
  - `p3_rule()` and the baselines.json hash passed;
  - `config_block0` and the module constants have no mismatch with the PLAN constants;
  - **`scripts_vs_plan` is empty**: all six scripts have the SHA-256 that PLAN.md lists. This check ran for the
    first time, because PLAN.md now exists.
- (b) Unit tests: 33 of 33. `test_analysis.py` also passes under pytest (19), run without writing caches.
- (c) **The rule-imitating gate reproduced all 80 stored e1_evaluation rule rows exactly (80/80 identical).**
  - Segment identity held on all 80, the pre-gate reward was 0 on all, and the FIX rows equal the fixes attempted.
  - 310 gate rows (7.75 per target-present episode, 0 per no-human episode); minimum segment duration 1.356 s; 0
    resamples; 9 no-anchor skips.
  - Wall time 471 s: 18.7 s per target-present and 3.3 s per no-human episode.
- (d) Smoke passed:
  - 37 episodes, 3 updates, 199 gate decisions; seeds in the development range; identity residual 0; 0
    infrastructure episodes;
  - every event kind was logged, including `final_evaluation_sampled`;
  - the argmax and sampled reload re-runs were identical; the update series has 3 entries.
  - Timing: 14.1 s per training episode, 2,615 s per 1,000 gate decisions in rollout, and 72.6 s for the 4-episode
    sampled read-out. No outcome value of the smoke was read.
- Other checks:
  - `gate_analysis.py --selftest` and `--dryrun` (synthetic records) pass.
  - `config_check.py` (no precheck) reports ok against the hand-over PLAN.md.
  - `config_check.py --precheck precheck_postreview.json` flags only `plan_sha256`, because PLAN.md is not hashed.
    The precheck's plan hash equals the hand-over PLAN.md's. `config_block0` equals a fresh `config(0)`, and the code
    hashes equal the PLAN.md table.
