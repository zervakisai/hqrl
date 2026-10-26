# B1R build notes: what was run, what passed, timings, open risks

Scouting and design for a prespecified confirmatory replication of three B1 local contrasts on 36 fresh two-layer
units (see PLAN_DRAFT.md). **No fresh unit has been built into episodes, and no main run or precheck of record has
run.** Everything below used B1 unit shipped 0 (values already published) or static checks.

## Files
- `PLAN_DRAFT.md`: the draft plan. It is not hashed; the lead reviews it, renames it PLAN.md and hashes it.
- `runner.py`: the subcommands `seeds`, `precheck [--quick]` and `launch --workers N`. It imports, unchanged, the B1
  tree of record (`idea_dq_b1_tree_a66f8979`, a66f8979), extension C's runner (sha256 a712ee54f556...) and extension
  E's runner (909d6f54346d...). It refuses to start if either runner's hash differs.
- `b1r_analysis.py`: families R and S and the descriptive values, exactly as in the draft plan. It refuses partial
  data unless given `--provisional`, and refuses records run under another PLAN.md.
- `test_b1r.py`: 15 unit tests (statistics, analysis logic on synthetic records, seed rule, comparison helper,
  resumable launch with a fake task).
- `precheck_quick.json` / `.log` and `precheck_quick/`: the development check, run 2 (final). It is not the precheck
  of record.
- `dev_runs/quick_run1/`: run 1 of the development check, kept as run.
- `dev_runs/smoke_run_unit.json` and `dev_runs/smoke_run_unit/`: the smoke test of the full unit path (item 7 below).
- Hashes at hand-off (record them in PLAN_LOG.md before the launch):
  - runner.py 4852a0375c79f1790f60f73213ecaf7b9714a0ed716be14e169c6d1d7f419a7e
  - b1r_analysis.py bb47234606cb87f779bc2bd7e5043e07661f20b55ca1b9cb17c607f296e00de1
  - test_b1r.py e0e66ccf09b128dd6c705758c93f1c36e25984d1fcf6d9762fe66bfc852b544c

## Where B1 and the extensions come from (located)
- B1 code: `idea_dq_b1_tree_a66f8979/scripts/gnss_qrl_runtime/dq_b1.py`. It is the tree of record, extracted
  unchanged from `output/study/gnss_qrl_v1/dq/source_lock_a66f8979`; the live `idea_dq` tree is ffe9dfef and is not
  B1's. The B1 unit records are in `idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units/`.
- The extensions in `idea_dq_posthoc_b1{x,c,g,e,n,t}/` (runner.py, *_analysis.py, PLAN.md, PLAN_LOG.md, records/).
  - B1C regenerated F, K and H of all 72 units and saved compact rows (`b1c/rows/`, 196 MB).
  - B1E and B1N work on those saved rows.
  - B1G and B1T reuse B1's episode seeds, as common random numbers.
  - B1X and B1C reproduce B1's G_H to 3e-9 or better.

## How units and seeds are generated, and why the fresh units never overlap B1's 72
B1 has 72 units: 36 two-layer and 36 three-layer, with the three-layer units sharing the two-layer trunks.
- **B1 unit i** (`dq_b1.py` lines 100-118, 143-156, 243-251):
  - initialisation `torch.manual_seed(Study('dq').stream(50, i))`, i = 0..35;
  - slot = i for two-layer and 36 + i for three-layer;
  - episode seed = namespace.start + (slot x 8 + cell) x per_cell + offset, with per_cell 32 in dq_fit, dq_oracle,
    dq_calibration and dq_heldout (100,000 seeds each), and 10 in dq_fresh_eval;
  - action stream `Study('dq').stream(SET_TAG[set], 1000 + slot, cell, offset)`.
  - So B1 occupies [start, start + 18,432) of each of the four set namespaces, and stream units 1000..1071.
- **Other users of the namespaces:**
  - B1N: dq_fit [start + 50,000, start + 77,648), stream units 1000..1035 at offsets 1000+;
  - B1C mission: dq_fresh_eval [start + 50,000, start + 61,520), offsets 1000+;
  - D-6: only dq_development, stream units below 100;
  - B2, LONG and the other post-hoc training runs: dq_e2_* namespaces and `stream(block, 0)` initialisations.
- **Fresh unit i = 100..135:** B1's own rules with slot = i, with `dq_b1.build_unit` unchanged. This gives:
  - seeds in [start + 25,600, start + 34,816) of dq_fit, dq_calibration and dq_heldout;
  - stream units 1100..1135;
  - new CTRIG draws and OPT random starts, because extension C's and extension E's SHA-256 tags include the index.
- `runner.py seeds` (5 s, no episode) **passed**:
  - 0 overlaps with B1's formula for both circuits, with all 73,728 seeds stored in B1's 72 unit records, and with
    B1N's range;
  - for i = 0..35 this runner's rule is identical to `dq_b1.episode_seed` / `stream_unit`;
  - the 36 initialisation seeds are distinct from B1's and from `stream(block, 0)` for blocks 0..199;
  - the 36 circuit-weight hashes are distinct from B1's;
  - B1 unit 0's weights are rebuilt exactly.
- Why `dq_b1.collect_set` is not called directly: `dq_b1.slot()` refuses an index of 36 or more. `runner.py` copies
  B1's three small episode functions (`run_unit_episode`, `collect_set`, `trajectory_parity_b1`). It changes only the
  slot and stream-unit rule, and keeps B1's namespace gate `dq_runtime.require_open` on every episode. Everything else
  is called unchanged:
  - `build_unit`, `rows_of`, `twin_of` and `certify_unit`;
  - extension C's `save_rows` and `unit_arms`, and extension E's `unit`.
  Two module globals of the imported runners are redirected to this folder's `stage/`: `b1c_runner.HERE`, where
  save_rows writes, and `b1e_runner.B1C`, where E reads rows and the staged C record. The precheck verifies that this
  is value-preserving.

## What was run and what passed
All runs used nice -n 10, at most 2 concurrent processes, and `python -B` with `sys.dont_write_bytecode`.
1. Tree hashes (read-only): live `idea_dq` = ffe9dfef2b24... (509 files), unchanged at the end of the session; B1
   tree = a66f89793664... (506 files).
2. `runner.py seeds`: passed (above).
3. `runner.py precheck --quick`, run 1 (`dev_runs/quick_run1/`). B1 unit shipped 0, 2 processes:
   - Part B: arms through this runner on B1C's saved rows of unit 0. **Exact reproduction**:
     - 1,084 extension C values (HQ-Fisher, TAN, QTAY, and all 20 CTRIG draws with ladder and CQR; statuses,
       alphas, KLs, G_H, diagnostics) against B1C's record: max relative difference 0.0;
     - 199 extension E values (OPT-CIRC and OPT-TAN end points, iterations, messages, G_H and L_F; HQ-Fisher,
       HQ-plain, TAN, TAN-c; first crossing) against B1E's record: max relative difference 0.0.
   - Part A: offsets 0-1 of F, K and H (48 episodes) through this runner's episode code:
     - seeds, J and decision counts **identical** to B1's stored episode records;
     - mask, action and episode ids identical to B1C's saved rows;
     - certificate passed: P1 1.2e-14, P2 1.1e-14, P3 1.3e-15, P4 51 decisions, max |dlogp| 6.7e-16.
   - Run 1 was **marked failed** by its own bitwise rule on the prefix angles: 8 of 2,876 angle values in F and in H
     differed by at most 1.9e-16 (0 in K). Cause: `dq_batches.replay` runs every row of a set through the GRU as one
     batch, and a 16-episode batch takes a different BLAS blocking path than the 256-episode batch B1C replayed. This
     is a defect of the subset comparison, not of the seed or data path.
   - Only the quick-mode criterion was changed afterwards: prefix angles must agree within 1e-15, and the bitwise
     flag and the max difference are still reported. The precheck of record replays full sets, and stays bitwise.
4. `runner.py precheck --quick`, run 2 (final, `precheck_quick.json`): **passed**.
   - Seeds and B1 tree checks passed.
   - Part A: episodes identical; mask, action and episode identical; prefix angles max |diff| 1.9e-16 (F), 0 (K) and
     1.1e-16 (H), within 1e-15; certificate passed.
   - Part B: 1,084 extension C values and 199 extension E values reproduced with max relative difference 0.0.
5. `pytest test_b1r.py`: **15 passed**. The first run had 4 failures:
   - one wrong expectation in my own Holm test (fixed);
   - t_test divided by zero on a zero-variance synthetic contrast; it now returns p = None with a note.
6. Analysis code on the real record formats: `b1r_analysis.unit_values` and `contrasts`, applied to the stored
   B1C + B1E records of the 36 original two-layer units, reproduce the published values:
   - R1 = -5.70 [-8.21, -3.19];
   - R2 = -4.31 [-5.75, -2.88];
   - R3 two-sided = -0.60 [-2.21, +1.02], p = 0.456;
   - S2 = +4.71.
   On those original units the R3 non-inferiority test at delta = 2.85e-5 would give p = 0.0038 (one-sided 95% lower
   bound -1.94e-5). This is disclosed in the plan.

7. Smoke test of the full `run_unit` path with arms (`dev_runs/smoke_run_unit.json`): B1 unit 0, offsets 0-1 only.
   - Status complete; rows saved; extension E read the staged rows and C record, and its recomputed HQ-Fisher / TAN
     equal C's exactly (0.0 at both kappa).
   - All arms ok at kappa = 0.01: 20 of 20 CTRIG draws, and both optimisers feasible. The certificate passed and the
     record serialises.
   - Timings: 2.35 s per episode at load 16-18; the arms are faster on 48-episode rows.
   - The arm values on this subset are meaningless and were not inspected.

Not run: the precheck of record (`runner.py precheck`, which regenerates all 768 episodes of unit 0; about
40-60 min, above the 15-min budget for this task), any fresh unit, and the launch.

## Timings
Measured under the current load: the B2F launch at 10 workers, nice 5, plus other agents' prechecks; load average
12-18 on 16 threads.
- Episodes, run 1: 48 episodes, 2,218 decisions in 95.3 s, i.e. **1.99 s per episode and 43.0 s per 1,000
  decisions** (46 decisions per episode). Rows, replay and save_rows took 0.2 s for 48 episodes. The certificate took
  0.8 s, including the P4 re-flight. In run 2 (load 17-18): 119.4 s, i.e. **2.49 s per episode and 53.8 s per
  1,000 decisions**.
- Extension C arms (20 CTRIG draws x {CTRIG, x0.1, x10, CQR} x 2 kappa, plus HQ-Fisher, TAN and QTAY; no CTRIG-16):
  **122 s** in run 1 and 133 s in run 2.
- Extension E arms (SLSQP OPT-CIRC 5 starts and OPT-TAN 4 starts, at both kappa; first-crossing grid of 4,000 KL
  evaluations per kappa; HQ-plain, TAN-c): **749 s** in run 1 and 782 s in run 2.
- From the extensions' stored records, two-layer units:
  - B1 campaign (8 workers): 1.18 s per episode, 26 s per 1,000 decisions, about 300 s per 256-episode set;
  - B1C (about 17 concurrent): data for 3 sets 2,042 s (median), instruments 302 s including CTRIG-16;
  - B1C precheck (2 processes): data 1,031 s;
  - B1E (about 12 concurrent): 856 s per unit (median); SLSQP median 176 iterations (OPT-CIRC) and 169 (OPT-TAN) at
    kappa = 0.01.

## Cost estimate for 36 fresh units
Per unit there are 768 episodes (F, K and H; O is not needed), extension C's arms, extension E's arms and the
certificate.
- At the current load, single process: 768 x 1.99 + 122 + 749 + 5 ≈ **2,400 s** at load 12-15, and
  768 x 2.49 + 133 + 782 ≈ **2,830 s** at load 17. That is 40-47 min, 0.67-0.79 CPU-h per unit, and 24-28 CPU-h for
  36 units.
- **10 workers on an otherwise idle machine** (after B2F): about 1.3 s per episode below saturation, so about
  1,000 + 90 + 550 ≈ 1,650 s per unit. 36 units take 4 waves, about **1.8-2.0 h wall**.
- **10 workers alongside B2F's 10** (about 20-22 CPU-bound processes on 8 cores / 16 threads, past the saturation of
  about 12 measured in the compute map): about 3,000-3,500 s per unit, so **3.5-4.5 h wall**, and B2F slows by a
  similar factor. Recommendation: launch after B2F, or with 4-6 workers beside it.
- Disk: about 2.7 MB of staged rows per unit (about 100 MB) and about 1.5 MB per record. RAM: well under 1 GB per
  process (B1C ran the same pipeline).
- The precheck of record adds one unit's cost: about 30 min idle, 40-60 min under the current load, 1 process.

## Open risks and decisions for the lead
1. **DQ calendar cut-off 2026-10-31T00:00Z** (`dq_runtime.CALENDAR_CUTOFF_UTC`). `require_open` refuses the B1
   namespaces after it, and this runner keeps that gate on every episode, as B1, B1C and B1N do. So the precheck of
   record and all 36 units must finish data collection before then. A unit running at the cut-off fails, and the
   launch refuses with fewer than 8 h left. After the cut-off nobody can regenerate these episodes with this code;
   the staged rows let the arms be recomputed without episodes. Bypassing the gate would be a deviation from B1's
   governance. It is not implemented; that would be the lead's explicit call, documented in PLAN_LOG.
2. **Machine load:** see the cost estimate. Other agents (b2k, long2) were also running prechecks during the
   measurement.
3. **R3's form and margin** (lead's decision before hashing):
   - The draft makes R3 a non-inferiority test at delta = 2.85e-5, half of B1's HQ-Fisher - TAN mean.
   - The margin was chosen knowing B1E's interval: its one-sided 95% lower bound, -1.94e-5, would pass this test.
     This is disclosed in the plan.
   - Alternatives: a two-sided difference test, which is weak for a null-predicted contrast and cannot confirm
     "not distinguishable"; or TOST, slightly less powerful. Both are reported as secondary anyway.
   - R3's power is the weakest: 0.73 at alpha/3 (0.87 at 0.05) if the true mean is B1E's -0.60e-5. 48 units would
     raise it to about 0.86 (0.94) at +33% cost; the runner then needs FRESH = 100..147.
4. **E2 (OPT-CIRC - HQ-Fisher)** carries half of B1E's reading. It sits in secondary family S, not in R. The lead may
   move it into R (Holm over four).
5. **Scope:** two-layer only, kappa = 0.01 primary. No mission evaluation (B1C family C), no SPAN, RND-k or FIT-k,
   no CTRIG-16. All of these would add cost. The oracle set O is not collected, so B1's own FIT-k instrument cannot
   be recomputed later.
6. **Redirected module globals** (`b1c_runner.HERE`, `b1e_runner.B1C`): value-preserving per the quick check, part B.
   `configure_stage` runs inside every worker. A future edit to either reused runner is refused by its hash.
7. **Partial-batch floating point:** only the quick check is affected (above). The precheck of record must show
   bitwise equality of the full rows. If the BLAS library or thread count changed since B1C, that check would fail
   loudly rather than silently.
8. **The precheck of record has not run.** The data path is verified on 48 of 768 episodes, and the arm stack on
   B1C's saved rows. The full chain on regenerated full sets is verified only by `runner.py precheck`, which the lead
   runs after hashing.
9. **Single scene:** the fresh units share B1's 8 frozen core cells. The replication covers new policies and new
   episodes, not new scenes; the single-scene limitation stands.
10. **Hashing order:** `runner.py` reads PLAN.md at import. The precheck must run after the hash so that precheck.json
    records the plan's SHA-256, and the launch checks that. Record the hashes of runner.py and b1r_analysis.py in
    PLAN_LOG.md before the launch.

## Integrity
- No file was written outside this folder and the scratchpad. A `find -newer` check over idea_dq (incl. output),
  the B1 tree and the six extension folders found nothing.
- `idea_dq` stayed at ffe9dfef.
- No forbidden folder (idea_y3*, Desktop/παπερ3, REPORT_c12b.md) was opened.
- Nothing was pushed or published.
- No fresh-unit data exists. Only published values of B1 unit 0 were regenerated and compared.

## Fixes after review (2026-10-25)
The fixes implement the lead's binding decisions on REVIEW.md, item by item.

The final plan is **PLAN.md**: complete, self-contained, derived from PLAN_DRAFT.md. PLAN_DRAFT.md is unchanged
(sha256 7f4bc182...).

What was not done, by rule:
- PLAN.sha256 and PLAN.utc were not written;
- no fresh unit was run;
- the canary and the launch were not run;
- the precheck of record was not run (the lead runs it after hashing);
- no main-run outcome exists or was read.

### Blocking items
**B1, option (c): R3 is secondary.**
- *Plan.* Family R is now R1 and R2, with Holm over two. R3 moves into the secondary family S (S1, S2 and R3, Holm over
  three). This is my reading of "secondary": it is the plan's existing secondary family, and it keeps multiplicity
  controlled. Every R3 report carries the disclosed scale:
  - delta = 2.85e-5 is about 71% of OPT-TAN's mean held-out gain on the original units (3.99e-5);
  - so passing means OPT-CIRC keeps at least about 29% of that gain;
  - the margin was chosen knowing that the original units pass (p = 0.0038).
  R3 is never worded as "non-inferior" or "replicated" in a headline.
- *Descriptive values added:* OPT-CIRC - OPT-TAN on the fit-set objective L_F, OPT-TAN - TAN on L_F, the arms' mean L_F,
  the L_F difference in the side-by-side with the original units, and delta's share of the original and of the fresh
  mean OPT-TAN gain. The power section is recomputed at the new Holm levels from the draft's stated sds (no data
  read).
- *b1r_analysis.py:* `family_R` covers two tests and `family_S` three. The R3 key is renamed
  `R3_OPT-CIRC_minus_OPT-TAN_margin`. The constant `R3_SCALE` goes with every R3 reading. New contrasts:
  `OPT-CIRC_minus_OPT-TAN_L_F`, `OPT-TAN_minus_TAN_L_F` and `arm_means_L_F`.
- *Check on the original units (real record format; booleans and counts only):* the recomputed R1, R2, R3 two-sided
  and S2 means equal the published values, the margin p is 0.0038, delta / OPT-TAN's mean gain is 0.71, and the L_F
  contrasts have n = 36. The script is scratchpad/fix_b1r/orig_format_check.py.

**B2: reading rules for adverse and mixed outcomes** (PLAN.md "Reading rules"; `b1r_analysis.readings()`).
1. *R1 or R2 in the opposite direction* (two-sided 95% CI entirely above 0). This is stated as contradicting B1 (R1)
   or B1C (R2), and the path claim is removed rather than qualified. The analysis flags this as
   `family_R.opposite_direction`.
2. *R3 passing its margin while its two-sided difference is negative at 0.05* is read as "within the margin, yet
   worse". The analysis reports `family_S.R3_two_sided_sign_at_0.05`. With R1 also rejected, the joint R1/R3
   reading has its own branch (round 2, below).
3. *R2 rejected without R1* uses conditional wording: "shown for the matched chart only, conditional on R1; the
   generic reading is not confirmed". Every R2 reading notes that S1 not rejected is not equivalence, and that the
   generic reading is an existence claim.
4. *Incomplete flag.* `main()` writes `incomplete` and `incomplete_statement` when R1 or R2 has fewer than 30 units,
   and the readings carry the statement.

**B3: failure procedures** (PLAN.md "Failure rules"; runner.py).
1. *Canary.* `runner.py launch --canary` runs fresh unit 100 alone. The full launch refuses until the canary has a
   record. A failed canary blocks the full launch until it is logged (`--canary-failure-logged`).
   `runner.py status` prints only counts, statuses, failure classes and errors, certificates, timings and running
   units; a test checks that no arm value appears in it.
2. *Systematic failure.* When two records share an exception type, the launch stops: the launcher terminates its own
   pool. A relaunch then refuses, unless `--failures-diagnosed` is given after a PLAN_LOG entry.
   `runner.py supersede --deviation-logged` moves records/, stage/, interrupted/, failed_environmental/ and
   records_conflict/ to superseded/<UTC>/ (nothing is deleted). The canary is then required again. The analysis
   refuses records from more than one runner, or from a runner whose hash is in neither PLAN.md nor PLAN_LOG.md. So
   after a fix, every unit must be rerun.
3. *Environmental failures.* Records now carry `failure_class`: environmental (MemoryError, OSError), parity_refused,
   episode_failure, refused or algorithmic_failure, as in B1's statuses. They also carry `error_type` and
   `error_where`. `launch --rerun-environmental` moves an environmental failure to failed_environmental/<UTC>/ and
   reruns the unit from scratch, once per unit.
4. *Failed precheck of record.* `precheck` writes precheck.json, then precheck_2.json, precheck_3.json and so on, and
   never overwrites. `require_ready` reads only the latest one: it refuses if that one failed, or if its folder exists
   without its JSON (running or interrupted).

### Minor items
- **m1.** `compare()` now reports a mismatch whenever exactly one side is NaN. Test added.
- **m2.** `run_unit()` now follows B1's fail-closed order:
  - F, K, H;
  - static certificate;
  - rows;
  - P3 and P4;
  - staged rows;
  - arms.
  No value changes. Every arm uses explicit seeded generators. I checked by grep that the reused C and E code draws no
  global RNG; B1 itself ran this order. The quick precheck part A passed with the new order (below).
- **m3.** `require_ready()` checks:
  - that the latest precheck's `runner_sha256` equals the running runner's;
  - its reused-runner hashes;
  - the B1 tree both in the precheck and live (`p4.live_copy_tree()`);
  - that runner.py did not change after import;
  - that the runner hash is written in PLAN.md or PLAN_LOG.md.
  Every unit record carries `runner_sha256`. `check_full` requires it, as criterion (h).
- **m4.** `write_record()` serialises first, writes a temporary file and calls `os.replace`. An unserialisable record
  becomes a failure record that says so. An existing record is never overwritten: a second one goes to
  records_conflict/. A lock (launch.lock) refuses a second concurrent launch; a stale lock is moved to interrupted/.
- **m5.** The watchdog note is in PLAN.md, run instructions step 6. `run_unit` writes a start marker,
  stage/started/<unit>.json (pid, UTC). `runner.py status` lists the units started without a record, with their age
  and whether their pid is alive. It checks /proc and never signals a process.
- **m6.** Disclosure sentence in PLAN.md (Units, "Approval"): 27,648 new seeds drawn under B1's approval, with B1N as
  the precedent.
- **m7.** `consistency()` also requires E's HQ-Fisher and TAN statuses to equal C's at both kappa. It lists the
  reasons for each excluded unit. Test added. All 36 original units pass it.
- **m8.** `row_info` (the `rows_of` diagnostics) is kept in every record. The precheck of record reports it against
  B1's stored row_info. This is not a pass criterion, because it has not been tested on full sets.
- **m9.** FRESH = 100..135 and n = 36 are kept. A test checks that runner.FRESH equals b1r_analysis.FRESH.
- **m10.** PLAN.md states that values do not depend on the worker count. The CLI default is now 5 workers
  (`--workers`), with advice of 4-6 beside other campaigns.
- **m11.** Noted in PLAN.md ("Development history"). From now on every precheck archives its runner
  (<precheck>/runner_snapshot.py.txt), and every launch does too (launch_snapshots/<UTC>_runner.py.txt).
- **m12.** R3 readings say "shown", never "replicated", and never "non-inferior". A test checks every R3 branch.
- **m13.** PLAN.md has a Limitations section: shared instrument code, single scene, R3's local optimiser and post-hoc
  margin, and the shared approval.

### Prechecks and tests re-run
- **Quick precheck, run 3** (development check, not the precheck of record). runner.py was edited, so it was re-run.
  The run-2 outputs (precheck_quick.json, precheck_quick.log, precheck_quick/) were first moved, not deleted, to
  `precheck_prev_2026-10-25T121501Z/`. Run 3 (`precheck_quick.json`, sha 51bed69a...) ran with runner 4c3bf71f76ae,
  2 processes, nice 10, load 3-8. It **passed**:
  - seeds and B1 tree (a66f8979) checks passed;
  - *Part A* (the new certificate order; offsets 0-1, 48 episodes): seeds, J and decisions identical to B1's
    records; mask, action and episode identical to B1C's rows; prefix angles within 1e-15 (max 1.9e-16 F, 0 K,
    1.1e-16 H, as in run 2); certificate passed (P1 1.2e-14, P2 1.1e-14, P3 1.3e-15, P4 51 decisions,
    6.7e-16);
  - *Part B*: 1,084 extension C values and 199 extension E values, max relative difference 0.0;
  - timing: 1.1 s per episode, C arms 79 s, E arms 483 s.
  The run archives its runner in `precheck_quick/runner_snapshot.py.txt`.
- **Smoke test of the edited launch path** (`dev_runs/smoke_launch_2026-10-25T122451Z.json`, sha 669065e0...).
  `runner.launch` ran the real `run_unit` with arms on B1 unit shipped 0, offsets 0-1, 1 worker. The results:
  - status complete;
  - `runner_sha256` and `row_info` in the record;
  - certificate passed before the arms;
  - extension E complete and internally consistent;
  - the record was written atomically (no temporary file left), the lock was released, the runner was archived,
    and the launches.jsonl entry is complete;
  - wall time 140 s.
  A dry run of the precheck-of-record comparison (`check_full`) on that record raised no exception and reported
  not passed, as expected for 48 of 768 episodes. Only status, certificate, timing and pass flags were read; no arm
  value was printed or inspected. The script is scratchpad/fix_b1r/smoke_launch.py.
- `runner.py status` and `runner.py launch --canary` were run from the command line. Status printed counts only, and
  the launch refused ("PLAN.md is not hashed yet") before creating anything.
- **pytest** `test_b1r.py`: **35 passed** (it was 15). New tests:
  - Holm over two;
  - FRESH agreement;
  - L_F and the R3 scale in the readings, and no "non-inferior" / "replicated" R3 wording;
  - the opposite-direction, within-margin-yet-worse, R2-without-R1 and incomplete readings;
  - the m7 status exclusion;
  - refusals of mixed or unrecorded runner hashes and of an unrecorded analysis hash;
  - NaN comparison and failure classes;
  - require_ready: runner, tree, latest precheck, precheck_2 naming, cut-off, and the runner hash in PLAN.md or
    PLAN_LOG.md;
  - canary first and alone, and a failed canary that blocks;
  - stop on two failures of one type, supersede, and the canary again;
  - diagnosed failures; environmental rerun once; unserialisable record; a second launch refused; a stale lock;
  - status shows no arm value.
  The launch tests now use 1 worker.
- **Integrity.** No file outside this folder and the scratchpad was written. A `find -newermt` over idea_dq, the B1
  tree and the B1C/E/N/X/G/T folders is empty. `idea_dq` is still ffe9dfef2b24... (509 files, recomputed
  read-only). No `__pycache__` is left in this folder. Nothing under idea_y3* or Desktop/παπερ3 was opened. No
  fresh-unit episode exists.

### Round 2: the verifier's defect (B2 rule 2 with R1, the joint R1/R3 reading)
**Defect.** Take the case where R1 is rejected and R3 passes its margin while its two-sided difference is negative at
0.05. The two readings then contradicted each other:
- R3's reading said "within the margin, yet worse": optimised circuit steps gain less than the best linear tangent
  step found.
- The joint R1/R3 reading still said the result "supports, at that margin only, locating the shortfall in the
  natural step's path rather than in what circuit policies can reach".

A reachability gap (smaller than delta) has been shown in that case, so "rather than" overclaimed. The defect was
confirmed with `readings()` on the synthetic case (R1 rejected, R3 rejected, sign -1). Before the fix, it returned
exactly that pair.

**Fix.**
- *b1r_analysis.py, `readings()`.* A new first branch of `joint_R1_R3` applies when R1 and R3 are rejected and
  `two_sided_sign(p['R3_two_sided_difference']) < 0`. It reads: "the B1 shortfall replicates (R1); secondary:
  optimised circuit steps come within delta of the best linear tangent step found (local optimiser) but gain
  significantly less, so a reachability gap smaller than delta is shown; at that margin the shortfall lies mainly,
  not wholly, in the natural step's path". This is the verifier's wording, plus "(local optimiser)" to match the
  other branch. The old "Both" wording, with "rather than", now applies only when the two-sided difference is not
  negative at 0.05. The other joint branches (R1 only, R3 only, neither) are unchanged.
- *PLAN.md, "R1 with the secondary R3".* The rule "Both" is split the same way:
  - "Both, within the margin yet worse", which states that "rather than" is not used;
  - "Both, otherwise".
  The script table in PLAN.md carries the new hashes of b1r_analysis.py and test_b1r.py.
- *test_b1r.py.*
  - `test_within_margin_yet_worse` (R1 is rejected there) now asserts that R1 is rejected, and that the joint
    reading does not contain "rather than" and does contain "gain significantly less", "reachability gap smaller than
    delta is shown" and "mainly, not wholly".
  - The new `test_joint_r1_r3_branches` calls `readings()` directly for every R3 two-sided outcome (negative,
    positive or unresolved at 0.05) with R1 and R3 rejected. It checks that only the negative case takes the new
    branch, and that the others keep "rather than in what circuit policies can reach". It also checks that the R1-only
    and R3-only branches do not depend on the sign.

**Checks.**
- pytest `test_b1r.py`: **36 passed** (35 before, plus the new test).
- The analysis finds its new hash in PLAN.md (`recorded(ANALYSIS_SHA256, HERE)` is True; `main()` was not run, because
  no record exists).
- runner.py is unchanged (4c3bf71f76ae...) and does not import b1r_analysis.py. b1r_analysis.py is not on any
  precheck path, so no precheck was re-run. The quick precheck run 3 remains the current development check.
- No main-run or pilot outcome was read. No `__pycache__` is left in this folder.

### Round 3: the verifier's two defects
This round supersedes three earlier items above: the round-2 joint R1/R3 wording ("mainly, not wholly, in the natural
step's path"; "rather than ..."), the R3 wording "reachability gap", and B3 item 3's classification by exception type
only. Those sections are kept as history.

**Defect 1: the R3 and joint R1/R3 readings claimed more than any prespecified test shows** (PLAN.md R3 and joint
rules; `b1r_analysis.readings()`, R3 and joint branches; the "reachability gap" wording).

The verifier is right on both counts:
- *(a) "mainly".* delta = 2.85e-5 is half of B1's original shortfall (5.70e-5), not of B1R's own. R1 can be
  Holm-rejected with a fresh shortfall well below 5.70e-5 (the 80%-power minimum detectable effect is 3.6e-5, as the
  plan's power section states). A gap just under delta would then be most of the fresh shortfall. So "mainly, not
  wholly, in the natural step's path" and "rather than in what circuit policies can reach" did not follow from R1 and
  R3.
- *(b) reachability.* The plan defines reachability on L_F. I recomputed the original-unit L_F difference with
  `b1r_analysis` on the stored B1E records (scratchpad/fix_b1r/orig_lf_check.py): OPT-CIRC - OPT-TAN on L_F is
  -7.98 [-9.34, -6.62] x 1e-5 (n = 36), 2.80 delta. On G_H it is -0.60 [-2.21, +1.02]. These values match the
  verifier's. Calling the G_H difference "the reachability gap" contradicted the plan's own definition.

I also found a further point. "Mainly in the natural step's path" treats the shortfall as path plus gap. In fact
HQ-Fisher - TAN = (HQ-Fisher - OPT-CIRC) + (OPT-CIRC - OPT-TAN) + (OPT-TAN - TAN) on H. So even with the -2 delta
condition, only "the shortfall lies mainly outside the gap between the best optimised steps found" follows. The fix
uses that wording, and the plan names the remainder without apportioning it.

**Fix.**
- *b1r_analysis.py.*
  - **R3 readings.** They are now worded in held-out gain: "the held-out gap between the best optimised circuit step
    and the best linear tangent step found (OPT-CIRC - OPT-TAN on G_H; local optimiser)". Each reading carries, in the
    same sentence:
    - delta's anchor (`DELTA_ANCHOR`: fixed at half of B1's original shortfall, 5.70e-5, not rescaled);
    - the L_F difference with its 95% interval (`lf_clause`; "not available (n = ...)" when it cannot be computed).
    No R3 reading says "reachability gap" any more. The not-rejected, negative-at-0.05 case no longer says "secondary
    evidence of a reachability limit". It now reads: evidence that the best optimised circuit step found gains less on
    H, and "it is not read as a reachability limit, because reachability concerns L_F".
  - **Joint R1/R3 reading** (R1 and R3 both rejected). It follows a fixed, deterministic rule, `at_most_half`: B1R's
    own R1 mean must be at or below -2 delta (`JOINT_R1_LIMIT` = -5.70e-5).
    - *At or below the limit:* "delta is at most half of the fresh shortfall's point estimate (R1 mean ..., at or
      below -2 delta = -5.70 x 1e-5), so on held-out gain this gap accounts for less than half of the fresh
      shortfall, which lies mainly[, not wholly,] outside the gap between the best optimised steps found; this is a
      held-out reading: reachability concerns L_F and is not tested". "Not wholly" appears only when R3's two-sided
      difference is negative at 0.05.
    - *Above the limit:* "... so how the fresh shortfall splits between the natural step's path and what circuit
      policies can reach is not settled".
    - "Rather than" is gone from every branch. The R3-only branch also carries the L_F clause.
  - **Outputs.** `main()` writes the rule's inputs as `joint_R1_R3_rule`. `R3_SCALE` now states delta's anchor, that R3
    tests held-out gain and not reachability, and the original-unit L_F difference (-7.98e-5 [-9.34, -6.62], about
    2.8 delta).
- *PLAN.md.*
  - **R3's scale.** It now anchors delta explicitly to B1's original shortfall, not rescaled.
  - **New paragraph "R3 tests held-out gain, not reachability".** It gives the wording rule and both original-unit
    differences.
  - **Descriptive section.** The L_F bullet gives the original-unit L_F value and says that the L_F difference is
    reported in the same sentence as every R3 and joint reading.
  - **Reading rules.** The R3 and "R1 with the secondary R3" rules are rewritten as above. The joint rule gives its
    reason (the anchor, and the MDE of 3.6e-5) and names the remainder of the shortfall
    (HQ-Fisher - OPT-CIRC plus OPT-TAN - TAN), which it does not apportion.
  - **Design choices.** The L_F value is disclosed as known before the plan. The plan states that the original units
    do not support "path rather than reachability", and that the -2 delta limit was fixed before any fresh unit
    existed.
  - **Limitations.** R3 tests held-out gain only.
- *test_b1r.py.*
  - `test_within_margin_yet_worse`: the synthetic fresh shortfall is now about -7e-5, so the rule applies. The test
    asserts the rule's inputs and "mainly, not wholly," together with the L_F clause, and asserts the absence of
    "rather than" and "reachability gap".
  - `test_joint_r1_r3_branches` (rewritten) runs four R1 means (-8.0, exactly -5.70, -5.69, -3.6) against four R3
    two-sided outcomes. It checks:
    - "mainly" only at or below the limit, and "not wholly" only when the difference is negative at 0.05;
    - "not settled" above the limit;
    - never "rather than" or "reachability gap";
    - the L_F clause and delta's anchor, in one sentence;
    - that the R1-only and R3-only branches do not depend on the sign or the mean;
    - that a missing R1 mean never takes the "mainly" branch.
  - New `test_r3_readings_are_held_out_and_carry_l_f`: every R3 branch, and the "not available" L_F case.
  - New `test_disclosed_original_unit_values`: the values disclosed in PLAN.md and `R3_SCALE` equal those recomputed
    from the stored original-unit records (R1 = -5.70 = -2.00 delta; L_F -7.98 [-9.34, -6.62], n = 36, -2.8 delta).

**Defect 2: environmental failures were recognised only by exception type** (runner.py `failure_class`, `run_unit`;
PLAN.md failure rule 3). I confirmed the verifier's cases with the venv's torch 2.14
(scratchpad/fix_b1r/probe_errors.py):
- torch.save to /dev/full raises RuntimeError: "basic_ios::clear: iostream error" for a small object, "[enforce fail
  at inline_container.cc:672] . unexpected pos 704 vs 598" for a large one;
- an impossible CPU allocation raises RuntimeError "... DefaultCPUAllocator: can't allocate memory ... Error code 12
  (Cannot allocate memory)";
- extension C's `save_rows` turns any exception into `{'saved': False, 'reason': 'Type: message'}`, and extension E's
  `unit()` records `'Type: message'`. The runner re-raises both as RuntimeError, previously classed as
  'algorithmic_failure'.

So `--rerun-environmental` could not rerun these units. Two such failures triggered rule 2, which had no branch for
them.

**Fix (in code, plus a plan fallback).**
- *runner.py.*
  - `environmental_match(exc)`, used by `failure_class`, classes a failure as environmental in either of two cases:
    - by type (MemoryError, OSError and their subclasses, as before);
    - by its text `'Type: message'`. The text check matches the names of every builtin exception class that
      `ENVIRONMENTAL` covers (`ENVIRONMENTAL_NAMES`, whole words), so a re-reported text is classed as the exception
      itself would be. It also matches `ENVIRONMENTAL_TEXT`: "can't allocate memory", "cannot allocate memory", "out of
      memory", "no space left on device", "disk quota exceeded", "file write failed", "PytorchStreamWriter failed",
      "unexpected pos N vs N" and "iostream error", in any case.
  - Not matched: "rebuilt features differ", a ValueError or KeyError re-reported by E, "unexpected position",
    "MemoryErrors", torch.load's "PytorchStreamReader failed" (a read, not a write), and B1's infrastructure-episode
    BatchError (still 'episode_failure').
  - Every failure record now also carries `environmental_match` (for example "text 'unexpected pos 704 vs 598'" or "type
    MemoryError"). `status` and the precheck show it.
- *runner.py, rule 2.* The verifier noted that no branch of rule 2 fitted environmental failures. Two environmental
  failures still stop the launch. The refusal and stop messages now name the environmental branch.
- *runner.py, the scope of `--failures-diagnosed`.* This is a related hole that I closed. The flag used to acknowledge
  a failure type for the whole launch. After two out-of-memory RuntimeErrors were diagnosed, a later systematic code
  fault raising RuntimeError would then never have stopped the launch. The flag now acknowledges the failed units that
  the diagnosis read, per type, and `launches.jsonl` records them as `failures_acknowledged_units`. A later failure of
  the same type is undiagnosed and stops the launch again, which is what rule 2 says.
- *PLAN.md.*
  - **Rule 3.** It now defines an environmental failure as memory exhaustion or a failed write, wherever it was
    raised. It names the type and text classification, and the record field `environmental_match`.
  - **Fallback in rule 3** (the verifier's plan-only option, kept as a backstop for a pattern the code does not know).
    The lead classifies the failure from status, error and traceback only, and logs it in PLAN_LOG. The record is
    moved by hand to failed_environmental/<UTC>/, and the next launch reruns the unit once from scratch as
    interrupted. `rerun_once_already` then sees the moved record, so a second environmental failure stays final.
  - **Rule 2.** New branch: environmental failures with no code fault continue with `--failures-diagnosed
    --rerun-environmental`. The plan also states the narrower scope of `--failures-diagnosed`.
  - **Rule 1.** An environmental canary failure is rerun with `launch --canary --rerun-environmental`.
- *test_b1r.py.* Four new tests:
  - `test_environmental_by_error_text` covers the verifier's three cases as the runner re-raises them, other
    disk-full, quota and E texts, and the negatives listed above;
  - `test_environmental_real_torch_failures` produces the real torch errors (impossible allocation; torch.save to
    /dev/full, small and large) and checks their class, raw and wrapped as `run_unit` raises them;
  - `test_environmental_runtime_errors_stop_then_rerun_once`: two out-of-memory RuntimeErrors stop the launch, a
    plain relaunch refuses, and `--failures-diagnosed --rerun-environmental` reruns both once;
  - `test_a_diagnosis_covers_only_the_units_it_read`: a new BatchError after two diagnosed ones stops the launch.

### Round 3, checks
- **pytest** `test_b1r.py`: **42 passed** (36 before, plus 6 new; 2 rewritten). Against the pre-round-3 runner.py and
  b1r_analysis.py (copies in scratchpad/fix_b1r/round3_orig/), the same test file fails 8 tests: the 2 rewritten and
  the 6 new ones. So the tests detect both defects.
- **Quick precheck, run 4** (development check, not the precheck of record). runner.py was edited, so it was re-run.
  The run-3 outputs (precheck_quick.json sha 51bed69a..., precheck_quick.log, precheck_quick/) were first moved, not
  deleted, to `precheck_prev_2026-10-25T130827Z/`. Run 4 (`precheck_quick.json`, sha 8e170904...) ran with runner
  90ac63deff4b, 2 processes, nice 10, load 12-14, from 13:08 to 13:21 UTC. It **passed**:
  - seeds and B1 tree (a66f8979) checks passed;
  - *Part A* (offsets 0-1, 48 episodes): seeds, J and decisions identical to B1's records; mask, action and episode
    identical to B1C's rows; prefix angles within 1e-15 (max 1.9e-16 F, 0 K, 1.1e-16 H, as in runs 2 and 3); the
    certificate passed (P1 1.2e-14, P2 1.1e-14, P3 1.3e-15, P4 51 decisions, 6.7e-16);
  - *Part B*: 1,084 extension C values and 199 extension E values, max relative difference 0.0, nothing missing;
  - timing at load 12-14: 2.1 s per episode, C arms 99 s, E arms 665 s.
  The run's plan_sha256 is the round-2 PLAN.md (a1df8938...), because the plan was edited after the run started. This
  is a development check, so it does not matter: the precheck of record runs under the hashed plan.
- **End-to-end check of case (i)** (scratchpad/fix_b1r/smoke_env_failure.py; `dev_runs/smoke_env_2026-10-25T132156Z.json`).
  `run_unit` ran on B1 unit shipped 0 (offsets 0-1, with arms) with the staged rows file made a symlink to /dev/full.
  The result:
  - status failure, failure_class **environmental**, `environmental_match` "text 'unexpected pos 3072 vs 2984'";
  - error "RuntimeError: rows not saved: {'saved': False, 'reason': 'RuntimeError: [enforce fail at
    inline_container.cc:672] . unexpected pos 3072 vs 2984'}", raised at runner.py:363;
  - the certificate had passed, no arm was computed, and the run took 83 s.
  The symlink was removed afterwards; /dev/full itself was not touched. Only status, failure fields, certificate and
  timing were read.
- **Smoke test of the launch path under the new runner** (scratchpad/fix_b1r/smoke_launch.py, unchanged from round 1;
  `dev_runs/smoke_launch_2026-10-25T132330Z.json`, sha 06808fc1...). `runner.launch` ran the real `run_unit` with
  arms on B1 unit shipped 0 (offsets 0-1, 1 worker). The results:
  - status complete, and `runner_sha256` 90ac63de... in the record;
  - certificate passed (the values of runs 3 and 4);
  - extension E complete and internally consistent;
  - no temporary file left, the lock released, the runner archived;
  - `launches.jsonl` carries the new `failures_acknowledged_units` ({});
  - wall time 249 s at load 12.
  The `check_full` dry run raised nothing and reported not passed, as expected for 48 of 768 episodes. Only status,
  certificate, timing and pass flags were read.
- **Integrity.** No file outside this folder and the scratchpad was written. A `find -newermt "2026-10-25 14:50"` over idea_dq
  (including output), the B1 tree and the B1C/E/N/X/G/T folders is empty. Recomputed read-only with code_version's
  algorithm (scratchpad/fix_b1r/tree_hash_readonly.py, run with `-I`):
  - `idea_dq` is still ffe9dfef2b24... (509 files);
  - the B1 tree is still a66f8979... (506 files).
  No `__pycache__` is left in this folder. records/ and stage/ do not exist, so no fresh unit was run. Nothing under
  idea_y3* or Desktop/παπερ3 was opened. No main-run or pilot outcome was read. The only values read were these:
  - the stored original-unit values (B1, B1C and B1E records: R1 and the L_F and G_H differences), which are
    published and whose rules were fixed long ago;
  - the reproduction differences, statuses and timings of B1 unit 0 (published).

### Files changed, with their sha256 after the last edit
After round 3, the table holds the current hashes. Earlier hashes are given in brackets.

| file | sha256 | change |
|---|---|---|
| PLAN.md | 070394b94e405018783d709746a9d9c969eba86c369d1f1602aedf16736f6eb2 | new: the final plan (not hashed into PLAN.sha256; the lead does that). Round 2: joint R1/R3 rule split. Round 3: R3 in held-out terms, delta's anchor, the L_F difference (table row, scale paragraph, descriptive bullet, design choices, limitation), the joint rule with the -2 delta limit, failure rules 1-3 (environmental by type or text, fallback, rule-2 branch, scope of --failures-diagnosed), run step 7, script table (round 2: a1df8938...; round 1: 9dfdcfa5...) |
| runner.py | 90ac63deff4b6445c5dda0d02eb09627c0f4487337bfecc298eab4021032e62e | B3, m1-m5, m8, m10, m11 (was 4852a037...); round 3: `environmental_match`, text classification, per-unit acknowledgement, messages (round 1-2: 4c3bf71f...) |
| b1r_analysis.py | ef5c05152cfe078b0d11c0a0b321ff3aaca1433b9f5311c47b86c08d7a91dc97 | B1, B2, m7, m12, hash checks (was bb472346...); round 3: R3 and joint readings, `at_most_half`, `lf_clause`, `joint_R1_R3_rule`, `R3_SCALE` (round 2: e40e3fb2...; round 1: 94eb0fc5...) |
| test_b1r.py | 64910192246cb51fd9505f7b0964e2f511a421852f7920c6c268a0742dbce693 | 42 tests (was e0e66ccf...; round 1: 55701ece..., 35 tests; round 2: 443c8149..., 36 tests) |
| README_BUILD.md | (this file) | this section appended; the earlier sections are unchanged |
| PLAN_DRAFT.md | 7f4bc18219254f29ebd6239572e7acf80913d637ce6d3b20dcbd4ff3144cd538 | unchanged, kept for the record |

The "Hashes at hand-off" listed in "Files" above are superseded by this table and by PLAN.md's table.

New outputs:
- `precheck_quick.json` and `precheck_quick.log`, and `precheck_quick/` (run 4, round 3; run 3 before it);
- `precheck_prev_2026-10-25T121501Z/` (run 2, moved);
- `precheck_prev_2026-10-25T130827Z/` (run 3, moved in round 3);
- `dev_runs/smoke_launch_2026-10-25T122451Z/` and its .json (round 1);
- round 3: `dev_runs/smoke_env_2026-10-25T132156Z/` (sha of its .json 4ba86cca...) and
  `dev_runs/smoke_launch_2026-10-25T132330Z/` (06808fc1...), each with its .json.

### For the lead
1. **R3's family.** R3 is in family S, so S is Holm over three. If you prefer R3 as a stand-alone secondary test (with
   S kept at Holm over two), change PLAN.md ("Secondary analysis") and `b1r_analysis.main` before hashing. Then
   recompute the hashes.
2. **R2 contradicted.** B2 rule 1 says "the path claim is removed". For R2, I read this as removing the matched-chart
   (generic) form of the path claim. Removing the circuit claim as well would contradict a replicated R1. PLAN.md
   names the claim removed in each case.
3. **Next steps.** Hash PLAN.md. Run the precheck of record (768 episodes of unit 0, 30-60 min, 1 process). Then the
   canary, then the other 35 with 4-6 workers, starting before 2026-10-30T00:00Z (PLAN.md, run instructions).
4. **Round 3, joint R1/R3 rule.** I made "mainly" conditional on B1R's own R1 mean being at or below -2 delta (the
   verifier's deterministic rule). I worded it as "mainly outside the gap between the best optimised steps found", not
   as "mainly in the natural step's path", because the rest of the shortfall also contains OPT-TAN - TAN. If you
   prefer the verifier's other option, which drops "mainly" in every case, delete the `at_most_half` branch in
   `readings()` and the first joint bullet in PLAN.md before hashing. Then recompute the hashes.
5. **Round 3, environmental classification.** It is done in code, by type or text, and the plan keeps the manual
   fallback. A misclassified deterministic fault costs only one rerun: the rerun is from scratch and deterministic, and
   a second environmental failure is final. Two failures of one type still stop the launch under rule 2.
