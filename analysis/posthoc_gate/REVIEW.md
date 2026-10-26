# Independent review of the E1 gate build (idea_dq_posthoc_gate)

Reviewer: an adversarial Claude Code subagent, 2026-10-25. I edited none of the builder's files. My only process
was a single `nice -n 10 python -B` run, under 1 minute. Scratch scripts are in
`/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad/gate_review/`.

**Verdict: fix-first.** I found no code bug in the training, pairing, seed or segment path. The code does what
PLAN_DRAFT.md says. The blocking items are about what the plan claims and what the analysis tests. They need
plan-text and analysis changes before hashing; none of them needs a new run.

## What I checked myself, and what holds
- **Hashes and tests.** I re-ran `test_gate.py`: 14 of 14 pass. `p4.live_copy_tree()` is still `ffe9dfef…`.
  `gate_analysis.py --selftest` passes and re-derives the rule's J of record, 0.20564463190957744. No
  `__pycache__` appeared in this directory.
- **Draft precheck.** `precheck_draft.json` shows 80 of 80 rows identical, with no diffs and J identical by repr.
  It covers all 8 cells × r0-9. Segment identity, zero pre-gate reward and FIX rows = fixes attempted all hold.
  gate_stats: 310 calls, 165 fixes, 0 resamples, 9 no-anchor skips. The smoke passed and its reload re-run was
  identical in every field. `gate_core.py` and `test_gate.py` have the same SHA-256 as at the precheck; only
  `runner.py` changed afterwards, as the README says.
- **Namespaces and seeds.** gate_training = [20262821000, 20262921000) and gate_development =
  [20262921000, 20263021000). Neither overlaps any frozen, E1 or DQ range: the highest other end is
  20262821000. The patch `p4.BLOCK_STRIDE = 5000` is live: block 11 uses offsets 55,000-59,999, and episode
  5,000 raises `ScheduleExhausted` before the P4Error namespace check can trigger (p4_train.py:387-398). No
  sibling folder defines namespace index 19 or 20. I checked b2f, b2g, b2k, b1r, long, long2 and the hqrl
  copies; I did not open idea_y3*. Cell probabilities are 0.2/0.05, and the schedule's share of positive cells
  is 0.787 over 2,000 draws.
- **The rule hook.** `GatePolicy.should_fix` (gate_core.py:256-270) sits exactly where the rule asks
  (gnss_policies.py:125-137: base REPORT and `_fix_budget_left`). It uses the same `_cheapest_anchor` fallback.
  `candidate()` is a faithful copy of the report branch of `_evidence_action` (executive.py:423-427).
- **Rejected RELOCALIZE.** The executive calls `feedback(action, False, …)` and then `continue`s with the
  same observation object (executive.py:1014-1021). The identity-based resample in `GateBook.ask` therefore
  fires as designed. It never occurred in 80 + 37 + 16 + about 840 episodes.
- **Selection.** Every pilot and smoke read-out used `e1_development` only, under study e1gate_dev with salt
  [20261025, 2] (checked in the metrics.jsonl of `pilot_quick/*` and `smoke_draft`). `config()` refuses
  `e1_evaluation` for a development study (runner.py:140-141). The only thing read from e1_evaluation is the
  rule's own behaviour, during the rule reproduction.
- **Pairing.** The final read-out pairs each row with the stored rule row on (cell_id, seed) and checks
  crn_sha256 (runner.py:213-217). The analysis re-checks crn_sha256 and rule_J (gate_analysis.py:105-108).
  The weighted scale is the right primary: dq_b2's TOST_DELTA = 0.0056 (dq_b2.py:72) was applied to AUCs of
  the frozen-weighted J (dq_b2.py:300-318).
- **Statistics.** The TOST and t-test match scipy, Holm is trivially correct, and the reading table matches the
  plan.

## Blocking (fix in PLAN.md / gate_analysis.py before hashing)

### B1. P1 is conditional on 40 fixed episodes, but its readings are worded as general claims
- **What the test can see.** All 12 blocks are read on the same 80 e1_evaluation episodes. The no-human half is
  identical to the rule by construction, so in practice that leaves 40 positive episodes. The rule's J is a
  constant, so D_b varies only with the training seed. The one-sample t across blocks therefore contains no
  episode-sampling variance.
- **Why that matters.** Suppose most blocks converge to a common policy that is not the rule's, for example a
  slightly different threshold. D_b is then nearly constant across blocks, the SE across blocks is tiny, and
  "better", "worse" or "equivalent" is declared about these 40 episodes with p ≈ 0.
- **Scale of the omitted uncertainty.**
  - The rule's J on positive e1_evaluation episodes has SD 0.052; I computed this from baselines.json.
  - The paired SD between nearby triggers is plausibly 0.025-0.06 J. That is the scout's figure, unchecked.
  - At 0.03, the episode-sampling SE of the weighted D is about 0.8 × 0.03/√40 ≈ 0.004. That is the same size as
    the margin of 0.0056.
- **What this means for the readings.** "Matches the tuned rule within 0.1 report-equivalent" and "the tuned
  trigger is not optimal" are claims about the trigger in general. Neither is supported by this test alone.
- **Fix.**
  1. State in the Question and in every reading that P1 is conditional on the E1 evaluation episodes.
  2. Pre-declare an episode-level interval. For each positive episode, take d_e = mean over blocks of
     (J_gate − J_rule). Then either run a cell-weighted paired t over the 40 positive episodes, or a stratified
     two-level bootstrap over blocks × episodes with a fixed seed and B fixed now. Implement it in
     gate_analysis.py.
  3. Add a reading rule: "better", "worse" or "equivalent" is stated beyond these episodes only if the
     episode-level interval agrees. Otherwise the wording is "on the E1 evaluation episodes …". S1, which uses
     320 independent episodes, is then the natural generalisation check.

### B2. Two reading sentences overclaim relative to the design
- **"Worse."** It must say the PPO settings were fixed a priori and not tuned: lr 1e-3, entropy 0, from scratch,
  12,800 decisions. The rule's sigma, by contrast, was tuned on a grid. The build's own quick check already shows
  the always-fix attractor.
- **"Better."** It must say the gate has 24 features while the rule has 1. A gain then shows only that a
  richer-information trigger beats the one-feature rule on these episodes. It does not show that the tuned
  threshold is suboptimal within its own class, and it does not show that the gain comes from learning rather
  than from information. Delete or qualify "the tuned trigger is not optimal".
- **Add a sentence.** No re-run with changed settings will be reported as E1. Any follow-up needs its own,
  separately hashed plan.

### B3. The draft leaves alternatives open that a hashed plan must close
Before hashing, the lead has to decide each of these, and the runner.py constants must then match the text:
- the budget and grid: 12,800 with [0, 3,200, …, 12,800], or 10,240 with [0, 2,560, …, 10,240]. `BUDGET` and
  `GRID` are at runner.py:73-74.
- the optional development pilot: run or not. If it is run, its rule must be applied before hashing.
- S1: kept or dropped.
- the P1 scale. Weighted is correct (see above). Delete the "lead should confirm" hedge.
- whether the rule flag or an entropy bonus is added.

Nothing in the code binds the plan text to these constants. After hashing, compare `config(0)` in the new
precheck.json with the plan, and record that comparison.

## Minor issues and recommendations
1. **Compute is likely optimistic.**
   - b2f is still running: 26 of 36 records, 10 workers. Pilots from b2k and long2 are also running.
   - Adding 12 E1 workers gives 22-26 runnable processes on 8 cores / 16 threads.
   - The 3.3 s per gate decision was measured at load average about 17, with only 2 of the builder's processes.
   - So expect about 1.3-1.6× longer: roughly 16-20 h per run, with no resume.
   - The plan says the 10,240 alternative keeps a run "at or under 12 h". That holds only on a near-idle machine.
   - Recommendation: launch when the total stays at or below about 14 processes, and restate the estimate.
2. **Non-atomic records.** `records/*.json` is written with `write_text` (runner.py:614). A crash mid-write
   leaves a truncated file, which `launch()` treats as done (runner.py:604). Write to a temporary file and use
   `os.replace`, and have launch parse-check existing records.
3. **No lock or PID guard.** If the parent dies while its workers survive, a relaunch moves a live run directory
   to `interrupted/` (runner.py:606-609). The surviving worker then writes by path into the new run directory:
   it appends to metrics.jsonl, and an `'xb'` checkpoint name can collide, giving FileExistsError, then
   "failure", never retried. Add a lock file, or refuse when a `runner.py launch` process is alive.
4. **Failure policy differs from P4.** "Any other failure is recorded and never retried" departs from P4's rule,
   where an infrastructure failure is retried with the same seeds. Runs are seed-deterministic, so one retry is
   outcome-neutral. Consider retrying once and reporting both attempts.
5. **No tree refusal at launch.** `launch()` and `run_one` log `live_copy_tree()` but do not refuse when it
   differs from `ffe9dfef…` (runner.py:598-600, 557-573). Add the refusal.
6. **The analysis is not bound.**
   - `_require_launchable` excludes gate_analysis.py from the hash check (runner.py:588).
   - gate_analysis.py does not check its own hash, the records' `plan_sha256`, or the records' config (budget,
     lr) against PLAN.md (gate_analysis.py:153-170).
   - Recommendation: refuse records whose `plan_sha256` differs from the hash of PLAN.md, and check the
     analysis script's hash against the one recorded in precheck.json.
7. **No distinct-key check.** `block_contrast` checks that there are 80 rows and that each row is paired, but not
   that the 80 rows are 80 distinct stored keys. Add a set-equality check against `stored`.
8. **Descriptives the analysis does not produce.**
   - The plan lists approx_kl per update and P(fix) above and below 1.5 m per update. gate_analysis.py does not
     extract them; they exist only in `runs/*/metrics.jsonl`. Either add the extraction or drop them from the
     plan.
   - Label approx_kl as B2's statistic: the mean over minibatch steps of the KL before each step. It is not
     D_KL after the update.
9. **The README's salt claim is wrong (risk 10).** 20261025 is used as an RNG seed in
   `idea_dq_posthoc_b1t/b1t_analysis.py:17`, `idea_dq_posthoc_long2/e4_analysis.py:33` and
   `idea_dq_posthoc_long2/runner.py:165`. These are different generators, not `SeedSequence([20261025, 1|2, …])`,
   so nothing collides, but correct the sentence.
10. **Say exactly what was read from e1_evaluation.** The figures 7.75 and 6.2 gate decisions per episode, and
    the minimum duration of 1.36 s, come from the rule's rows on e1_evaluation. They are not gate outcomes and
    selected no setting, but the plan should say "only the stored rule's behaviour on e1_evaluation was read".
11. **Argmax only.** The final read-out is argmax only, which is reasonable against a deterministic rule. P4/E1
    made the sampled read-out primary (p4_train.evaluate docstring), so give a one-line justification, or add a
    sampled read-out as a descriptive (80 episodes, about 20 min per run).
12. **Consider running the development pilot.**
    - In the quick check, both development blocks were 0.015-0.019 J below the rule at 20% of the budget, and one
      block went to always-fix.
    - Before hashing, consider the optional development pilot with one entropy arm added, for example 0.01, and
      the selection rule fixed now.
    - That is about 7 h on 4 workers, on development data only. It is not required for integrity, but without it
      a "worse" reading is easy to dismiss as untuned PPO.
13. **No reading for fewer than 2 blocks.** `gate_analysis.main` exits with fewer than 2 completed blocks without
    writing the pre-declared "not resolved (insufficient completed blocks)" (gate_analysis.py:161). Write the
    reading anyway.
14. **Report failures beside the reading.** Blocks that end in an algorithmic failure contribute no D_b. That is
    declared, as in P4, but state that the number of failed blocks is reported beside the reading.

## The five questions, in short
1. **Does the code do what the plan says?** Yes, as far as I checked: pairing, seeds, grid, suites, a single lr,
   segment rewards and the identity. There is no freeze, decay or KL logic in E1.
2. **Does any selection read forbidden outcome data?** No. Only the rule's own e1_evaluation rows were read,
   for the reproduction. No gate outcome on e1_evaluation exists yet.
3. **Are the contrasts, tests, margins and reading rules complete and honest?** They are complete for the
   block-level test. They are not yet honest about the inference scope (B1) or two of the readings (B2), and the
   alternatives are still open (B3).
4. **Could a crash or resume corrupt the pairing?** It cannot mispair: pairing is by key and CRN hash. A
   truncated record or an orphaned worker can lose or kill a block (minor 2 and 3).
5. **What would make a reviewer reject the result?** Mainly B1 and B2: block-level equivalence on 40 shared
   episodes presented as general, and untuned PPO compared with a tuned rule. Also the wall-time risk under the
   current load.
