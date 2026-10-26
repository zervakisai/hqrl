# B1R: adversarial review of the build (2026-10-25)

Reviewer: an independent subagent. I read PLAN_DRAFT.md, runner.py, b1r_analysis.py, test_b1r.py and README_BUILD.md.
I also read the code they reuse: the B1 tree a66f8979 (`dq_b1.py`, `dq_runtime.py`, `dq_batches.py`,
`p4_train.py` parts) and the extension C and E runners.

I edited none of the builder's files. This file is the only one I wrote in the folder. Scratch scripts are in
/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad/b1r_review/.

**Verdict: FIX FIRST.** The machinery is sound and faithful to B1, B1C and B1E. Three things in the plan must be
fixed before it is hashed (B1-B3 below). One code fix should land before the precheck of record (m1). None of
them needs a new run.

## What I checked independently (all read-only, nice 10, one process)
- **Reused runners.** Their hashes match REUSED: a712ee54f556... and 909d6f54346d....
- **The redirect is real.** `save_rows` builds `HERE / 'rows'` at call time (b1c runner.py:333). `load_rows` and
  `unit` read the module global `B1C` at call time (b1e runner.py:45, 138, 219). So `configure_stage` really
  redirects both, and nothing is written into the B1C or B1E folders.
- **Fresh indices take no index-limited path.** Every function the runner calls for i >= 100 avoids `b1.slot()`
  and any stored file keyed by index:
  - `build_unit`, `rows_of`, `twin_of`, `certify_unit`;
  - `tp.certify_rows`, `c_ext.unit_arms`, `c_ext.save_rows`;
  - `e_ext.unit` (it reads only the staged rows and the staged C record).
  The only index-limited code, `c_ext.reproduction` and `mission`, is not called.
- **The episode functions are faithful.** The copied `run_episode`, `collect_set` and `trajectory_parity` match
  B1's (dq_b1.py, `run_unit_episode`, `collect_set`, `trajectory_parity_b1`) in four ways:
  - the seed rule and the `require_open` gate;
  - the episode order (offset outer, cell inner);
  - the infrastructure/invalid rule;
  - the P4 logic.
  The only difference is the `label.replace('#','_')`, which has no effect: the labels are 'unit' and 'twin'.
- **No global RNG coupling.** Skipping the oracle set O does not shift K or H, because each episode has its own
  generator. The K and H prefixes in the quick check, and B1C's own regeneration, confirm this.
- **Seeds are disjoint.**
  - Episode seeds are [start+25,600, start+34,816). B1 uses [0, 18,432) and B1N uses dq_fit [50,000, 77,648).
  - A grep over every ChatGPT folder (idea_y3* and παπερ3 excluded, never opened) finds no other code that draws
    from dq_fit, dq_calibration or dq_heldout, apart from B1, B1N and copies of them.
  - B1G uses indices 0..35 only. D-6 used E1 checkpoints, not `stream(50, i)`, so the fresh inits are not
    development units.
  - Action-stream seeds are only 32-bit (`p4.stream` -> `generate_state(1)`). I therefore checked them directly:
    0 collisions between the 27,648 fresh seeds and B1's 79,487 distinct seeds, and none with the 36 fresh init
    seeds (b1r_review/action_collide.py).
- **The analysis on the stored original units reproduces the published values** (b1r_review/orig_check.py):
  - R1 −5.698 [−8.21, −3.19]. It is identical with B1's own instrument values.
  - R2 −4.315 [−5.75, −2.88].
  - R3 two-sided −0.599 [−2.21, +1.02], p = 0.456. R3 non-inferiority p = 0.0038.
  - S2 +4.71.
  - On those units the internal consistency check passes for all 36 (max relative difference 0.0). All 36 units
    have 20/20 ok CTRIG draws and feasible optimisers at both kappa.
- **pytest.** `test_b1r.py -k "not launch"`: 11 passed. I skipped the launch tests because they fork a 2-worker
  pool.
- **No stray writes.** Nothing under idea_dq, the B1 tree, the B1C/B1E/B1N/B1X/B1G/B1T folders or the B1 campaign
  has an mtime today.
- **Hashes.** runner.py, b1r_analysis.py and test_b1r.py match the hashes in README_BUILD.md, and
  precheck_quick.json (run 2) was produced by the current runner.py (4852a037...).

## Blocking (fix in PLAN_DRAFT.md before hashing)

**B1. R3's margin needs its scale disclosed, and the claim worded to match.**
On the original units at kappa = 0.01, the mean arm gains (x 1e-5) are TAN 4.38, OPT-TAN 3.99, OPT-CIRC 3.39 and
HQ-Fisher −1.31.
- delta = 2.85 is about 71% of the comparator's (OPT-TAN's) mean held-out gain.
- So "non-inferior" here means only that OPT-CIRC keeps about 29% or more of OPT-TAN's gain. A reviewer will
  read "non-inferior" as much stronger than that.
- On top of this, the margin was chosen knowing the original units would pass (p = 0.0038; this is disclosed).

The fix is the lead's choice, made before the hash:
- (a) Keep delta = 2.85e-5. State in the plan that it is about 71% of OPT-TAN's mean gain on the original units.
  Restrict the confirmatory wording, in the plan and in the analysis `readings()`, to "the reachability gap is less
  than half of B1's natural-step shortfall". Drop "non-inferior" / "replicated at margin" (b1r_analysis.py:233)
  unless it carries that qualifier.
- (b) Anchor delta to the comparator instead, e.g. 50% of OPT-TAN's gain, about 2.0e-5. If the true mean is −0.60,
  power falls (noncentral t, computed) to:
  - 0.33 at alpha/3 and 0.53 at 0.05 with 36 units;
  - 0.44 and 0.64 with 48 units.
  The builder's figures for delta = 2.85 check out: 0.73 / 0.87 with 36 units, 0.86 / 0.94 with 48. So (b) is
  realistic only with many more units.
- (c) Demote R3 to secondary and keep TOST as descriptive.

Whatever the choice, the plan should also add OPT-CIRC − OPT-TAN on the fit-set objective L_F as a descriptive
value. Reachability concerns the optimiser's own objective, while G_H also mixes in overfitting: on the original
units OPT-TAN − TAN = −0.39 on H.

**B2. Reading rules are missing for adverse and mixed outcomes** (PLAN_DRAFT.md:135-159). The task requires a rule
for every outcome. Add the following:
- **R1 or R2 significant in the opposite direction:** the two-sided 95% CI lies entirely above 0. Say that this
  *contradicts* B1/B1C, not merely "not replicated". The manuscript wording must then remove the path claim, not
  just qualify it.
- **R3 rejected, but its two-sided difference is significantly negative:** "non-inferior at delta, yet OPT-CIRC
  gains less". This is a real possibility with sd about 4.8. Today the rule only covers "R3 not rejected and
  negative".
- **R2 rejected without R1:** the plan's R2 text ("also falls short ..., replicated; the generic reading ... is
  confirmed") presupposes R1. The analysis already distinguishes this case (b1r_analysis.py:250-256), but the plan
  is the binding document, so put the conditional wording there. Also say that the "generic" reading is an
  existence claim: S1 not rejected is not equivalence.
- **The 30-unit incompleteness statement** (PLAN_DRAFT.md:123-124) is in the plan but not in the analysis. Add a
  flag to `main()` so the output carries it.

**B3. There is no prespecified procedure for systematic failures or a failed precheck of record**
(PLAN_DRAFT.md:113-117, 172-181).
- **No episode has yet run at a fresh index.** The precheck and the smoke test use unit 0, where the fresh rule and
  B1's rule coincide. I found no index-dependent defect by reading the code, but the first fresh unit is the first
  real test of that path.
- **A systematic failure would void the replication.** Under the current rules ("no unit is replaced; only
  crash-interrupted units are rerun"), a deterministic code fault in that path would turn all 36 units into
  permanent failures. The plan allows no remedy, and the 2026-10-31T00:00Z cut-off leaves little room for a
  deviation.
- **The same gap after a failed precheck.** If the precheck of record fails, `precheck()` refuses to overwrite
  and the plan does not say what happens next.

Add, before the hash:
- **A canary.** Fresh unit 100 runs first. Only its status, certificate and timing are read; no arm value is
  read. Then the other 35 are launched.
- **A systematic-failure rule.** If two or more units fail with the same exception, the launch stops. The cause is
  diagnosed without reading any arm value. The fix and the new runner hash go into PLAN_LOG as a disclosed
  deviation, and every unit is rerun from scratch. This is safe because the pipeline is deterministic, so a rerun
  cannot select values.
- **Environmental failures.** Say whether a unit lost to MemoryError, OSError or disk-full may be rerun. Same
  determinism argument. The present rule throws such units away.
- **Failed precheck.** If the precheck of record fails, there is no launch. After diagnosis it reruns into a new
  name (e.g. precheck_2.json), logged in PLAN_LOG.

## Minor (fix before the precheck of record where marked; otherwise recommended)
- **m1 (fix before precheck).** `runner.compare` treats NaN against a number as a match (runner.py:349-357). The
  relative difference is NaN, and `NaN > tol` is False. I verified this: `compare({'G_H': 1.0}, {'G_H': nan})` and
  the reverse both return ok. Today unit 0's stored C/E records contain no NaN, but a NaN produced by the
  regenerated pipeline where B1C stored a number would pass the precheck silently. Fix: a mismatch whenever
  `isnan(a) != isnan(b)`.
- **m2. The certificate runs after the arms** (runner.py:294-316). B1 certified before computing instruments
  (dq_b1.py:448-459). As written:
  - the P4 re-flight, which calls `require_open`, happens about 15 min after the data, so a unit can fail at the
    cut-off after its data is complete;
  - a certificate failure wastes the arm time;
  - the order differs from B1's fail-closed order.
  Move the certificate (P1-P7 with P3 and P4) to right after the data. This changes no value.
- **m3. The launch does not check that runner.py is the file that passed the precheck.** precheck.json stores
  `runner_sha256`, but `require_ready` (runner.py:556-576) does not compare it, nor the B1 tree's
  `live_copy_tree()`. The unit records also carry no runner hash (runner.py:265-268). Add both: a refusal on a
  mismatch, and `runner_sha256` in each record.
- **m4. A failed serialisation leaves an empty record.** The record is opened with 'x' *before* `json.dumps` runs
  (runner.py:609-610). If serialisation fails, an empty file is left behind and the relaunch skips the unit as
  done. Serialise first, then write a temporary file and rename it.
- **m5. A killed worker hangs the launch.** `multiprocessing.Pool.imap_unordered` hangs if a worker is SIGKILLed
  (for example by the OOM killer). A short wall-clock watchdog note in the run instructions is enough: no record
  for more than 2x the expected unit time means investigate.
- **m6. The plan should disclose that B1R draws under B1's approval.** B1R draws 27,648 new seeds from the
  B1-gated namespaces under B1's Codex approval (`dq_runtime.GATED` / `require_open`), which approved B1's seed
  manifest (slots 0..71). The precedent is B1N's extra fit seeds. One sentence in the plan.
- **m7. E's internal reproduction check skips missing gains.** When either side lacks G_H, the arm is skipped and
  `reproduction_max_rel_diff` stays 0 (b1e runner.py:152-158). So "consistent" does not detect E failing to
  calibrate where C succeeded. This is harmless in practice (same function, same rows). If kept, add to the
  exclusion rule that E's HQ-Fisher and TAN statuses must equal C's.
- **m8. `rows_of` diagnostics are discarded** (runner.py:286; B1 kept them as `row_info`, e.g.
  baseline_fallback_rows). Record them.
- **m9. Hard-coded unit set.** FRESH = 100..135 is hard-coded in both runner.py and b1r_analysis.py. If the lead
  picks 48 units or another form for R3, both files and `seeds` must change before the hash and re-run.
- **m10. Launch workers.** Plan step 4 says `--workers 10`, while B2F runs 10 workers. Values do not depend on the
  worker count: one torch thread per worker, deterministic. The plan can say so and leave the count to the
  operator. 4-6 workers beside B2F, or 10 after it.
- **m11. The run-1 change cannot be checked.** The run-1 runner (sha 5b7fc101...) was not archived, so the claim
  that only the quick criterion changed between runs 1 and 2 cannot be checked against code. Both runs' JSON are
  kept, which is honest. In future, archive the runner with each development run.
- **m12. The analysis wording is inconsistent.** It calls an R3 rejection "replicated", but B1E's R3 was a null
  two-sided result. Use "shown".
- **m13. Limitation to state.** The replication uses the same instrument code as B1C/B1E, so it cannot detect an
  implementation error they share. Together with the single scene (8 frozen cells, already disclosed), this
  belongs in the limitations.

## Answers to the five questions
1. **Does the code match the plan?** Yes, for:
   - units, seeds and action streams;
   - the sets F, K and H;
   - the arms: C with 20 draws and no CTRIG-16; E with 5 and 4 starts;
   - the certificate, the precheck checks (a)-(g), the tests, Holm, the exclusions and the plan-hash refusal.
   The exceptions are the 30-unit flag (B2) and the certificate order (m2). This study has no training, lr, freeze,
   decay, KL controller or segment rewards: kappa calibration is B1's, unchanged.
2. **Does any selection read data it must not?** No. No fresh-unit episode or record exists. The margin and the
   directions were chosen from the published original units, and this is disclosed. No hyperparameter is
   selected.
3. **Are the contrasts and rules fixed, complete and honest?** The contrasts, tests and Holm family are fixed.
   Before the hash, the plan needs the margin's scale and wording (B1), adverse and mixed reading rules (B2) and a
   failure procedure (B3).
4. **Would a crash or resume corrupt pairing?** No. Pairing is within a unit, and the pipeline is deterministic.
   Resume skips records, moves partial staging aside and reruns from scratch. Residual risks are m4 and m5.
5. **What could make a reviewer reject the result?** Mainly R3's loose margin chosen after seeing data (B1). Then
   the single scene and shared code (m13), and governance gaps that would show if failures occur (B3).
