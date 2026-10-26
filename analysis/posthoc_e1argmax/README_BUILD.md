# E1-ARGMAX build record: the argmax read-out of the 48 E1 final policies

Built on 2026-10-26. This record covers the build, the tests and the prechecks only. **No main run was launched,
PLAN.md does not exist, and no argmax read-out of record exists.**

## Files
| file | what it is |
|---|---|
| `runner.py` | `test`, `precheck [--workers N] [--tag _x]` and `launch --workers N`. Every read-out calls `p4_train.evaluate` unchanged. |
| `analysis.py` | The read-out of record, run once: `--selftest` and `--dryrun` (synthetic records in a scratch directory). |
| `test_runner.py` | 18 unit and integration tests: binding, row comparison by repr, records, the 48-policy table. |
| `PLAN_DRAFT.md` | The draft plan, **not hashed**, for the lead's review. It includes the decisions to take before hashing. |
| `PLAN_LOG.md` | The log of plan events. |
| `precheck_build.json`, `.log`, `precheck_build/` | The first build precheck (earlier `runner.py`): passed. |
| `precheck_build2.json`, `.log`, `precheck_build2/` | The build precheck with the final scripts: **passed**. `sample/` holds the 48 re-run sampled records, `probe/` the 8 argmax-probe records (times and digests only), and the test logs are alongside. |

Final script SHA-256:
- `runner.py`: `b64156af705b581eea00401a1292190baf6197802faa977302dd79bfa8171231`;
- `analysis.py`: `4faf8d85289e6caf4cc147fa6ac1fa45f8effe4071d5b25cc92d307827d138ee`;
- `test_runner.py`: `e1cf141f0c3d14477ccdeef93a9adc0ec4a227471a467f5bb6c99d9fc145d98d`.

## What was found before building
- **The 48 policies.** They are the E1 evaluation jobs in `e1_campaign_v2/evaluation/jobs.jsonl`, re-derived through
  E1's own proof code (`e1_study.proven_stage`). The selection is `tuning/selection.json`, re-derived by
  `e1_study.verify_selection_record`:
  - quantum (HQ), lr3e-3_ent0;
  - bottleneck (C-matched), lr3e-3_ent0;
  - mlp (C-strong), lr3e-3_ent0;
  - surrogate625 (Surrogate625), lr1e-3_ent0.

  The blocks are 10–21. All 48 runs completed at the first attempt.
- **The final policy.** It is `checkpoints/final_update…_decisions…pt`. Its SHA-256 matches the run summary in all 48
  runs, and it is tensor-identical to `update…pt`, the checkpoint written at the 30,000 grid point. Its sampled
  read-out (evaluation event `points: [30000]`, point index 10) is Table 5's: the mean over blocks reproduces 0.0243,
  0.0374, 0.0520 and 0.0449, and the E1 report's credited-report and false-report panel to 1e-12.
- **The code and interpreter.**
  - The idea_e1 code tree is now `4a1f2e8e…`, the same tree as in the E1 freeze and in all 48 run manifests.
  - The idea_dq tree is `ffe9dfef…`. Every file it shares with idea_e1 is byte-identical; idea_dq only adds 22 dq_*
    files.
  - E1 ran with `idea_e1/.venv/bin/python` (`run_logs/e1_chain.sh`): CPython 3.12.13, packages from
    `idea/.venv` via `e1_paths.pth`, and torch 2.14.0+cpu, as in the run manifests. The runner requires this
    interpreter, `-B` and no CUDA.
- **The rule.** `baselines.json` has SHA-256 `00202039…`. The P3 rule's weighted J on e1_evaluation re-derives as
  `0.20564463190957744` (repr). Its timely credited reports are 2.3885 per episode, and its false reports 0.075.

## What I ran, and what passed
1. `runner.py test`: test_runner.py passed 18 of 18, and `analysis.py --selftest` passed. The self-test checks the
   t interval and p against scipy and Holm against a worked example. It re-derives the rule's J of record, the rule's
   credited reports, Table 5's sampled means and the E1 report panel.
2. `analysis.py --dryrun` on synthetic records passed. It covers the full output, the refusal of a missing record,
   the refusal of a record of another plan, and the refusal of a second read-out (it runs once).
3. **Launch machinery**, in a scratch copy (`launchtest` in the session scratchpad). That copy had the mode patched to
   `sample`, 2 policies, and a fake plan and precheck, so no argmax outcome was produced.
   - The launch refused without `-B`, and refused while another process held `.launch.lock`.
   - It wrote 2 records atomically. These sample-mode records also equal the stored rows exactly.
   - A relaunch found nothing pending.
   - An unparsable record was moved to `records/aside/` and re-run.
   - In the real directory, `launch`, the untagged `precheck` and `analysis.py` all refuse ("PLAN.md does not
     exist").
4. **Build precheck 1** (`--tag _build`; 8 workers; 909 s): passed. 3,840 of 3,840 episodes identical.
5. **Build precheck 2, with the final scripts** (`--tag _build2`; 8 workers; nice 10; 907 s; load about 2, rising
   to 10–12): **passed**.
   - (a) Both trees were at their hashes at the start and at the end. The approved E1 freeze `c80eb591…`, campaign
     `76bc358d…`, baselines, selection and all 48 jobs re-proved. The policy table digest is `edd1633e…`. No module
     loaded from outside idea_e1 (the venv packages are allowed).
   - (b) The tests passed.
   - (c) **The sampled reproduction: 48 of 48 policies and 3,840 of 3,840 episodes identical in every stored field by
     repr** (cell_id, seed, J, tp_timely, tp_spatial, fp_spatial, return_ok, fixes_attempted, decisions,
     termination_reason). There were 0 differing fields, and per-cell and weighted J were identical for all 48. The
     path (tree, checkpoint, evaluation stream, label, environment) therefore reproduces E1 exactly. The only change
     for the main run is `modes: ['argmax']`.
   - (d) **The argmax probe** (block 10 of each family, 8 episodes, 2 processes) gave equal digests in both
     processes for all four families, and the same digests as build precheck 1. That makes four processes and two
     contexts (cells pre-loaded or not) with identical argmax rows. The sampled 8-episode subsets, run after the
     argmax episodes in the same process, were identical to the stored rows. Only times and digests were kept.
   - (f) The maximum RSS was 0.53 GiB per worker. MemAvailable was 11.0 GiB at the start and 10.8 GiB at the end.
     `free -m` is recorded in the JSON.
   - (g) Nothing under idea_e1 or idea_dq was created or modified. `find -newer` against a marker made before any
     build step also finds nothing, and no `__pycache__` exists anywhere in this directory.

## Timings (precheck_build2, 8 workers, nice 10, load 10–12 on 16 threads)
| family | sampled read-out per policy | argmax/sampled per episode | E1's own final read-out (12 workers, training load) |
|---|---|---|---|
| HQ | 133 s | 1.03 | 124 s |
| C-matched | 130 s | 1.19 | 135 s |
| C-strong | 172 s | 1.17 | 163 s |
| Surrogate625 | 127 s | 0.48 | 134 s |

- All 48 sampled read-outs took 861 s of makespan. Per worker, CPU time equals wall time (1 torch thread).
- **The main run is estimated at about 14 min with 8 workers, 16 min with 7 and 19 min with 6.** The sum of task
  times is about 6,650 s, scheduled longest first. Even 2× is under 40 min.
- The precheck of record takes about 15 min.
- Build precheck 1 measured larger argmax/sampled ratios (1.2–1.8) because its argmax episodes paid each cell's bank
  loading. Build precheck 2 pre-loads the cells (5.6 s per process), so its ratios are the clean ones.

## How to finish (lead)
1. Review `PLAN_DRAFT.md` and take the decisions in its last section. Write `PLAN.md` from it, deleting that section,
   and keep the binding table in step with the scripts. Then:
   `sha256sum PLAN.md > PLAN.sha256; date -u +%FT%TZ > PLAN.utc`. Log it in PLAN_LOG.md.
2. The precheck of record:
   `cd ~/Documents/ChatGPT/idea_dq_posthoc_e1argmax && CUDA_VISIBLE_DEVICES='' nice -n 10 ~/Documents/ChatGPT/idea_e1/.venv/bin/python -B runner.py precheck --workers 8 > precheck.log 2>&1`
   This takes about 15 min and writes `precheck.json`, which must say `"passed": true`.
3. Launch (resumable):
   `CUDA_VISIBLE_DEVICES='' nice -n 10 ~/Documents/ChatGPT/idea_e1/.venv/bin/python -B runner.py launch --workers 8 > launch.log 2>&1`
   This takes about 14 min.
4. Once all 48 records exist, run the analysis once:
   `~/Documents/ChatGPT/idea_e1/.venv/bin/python -B analysis.py | tee analysis_of_record.out`
   It writes `argmax_results.json`. Log the run in PLAN_LOG.md.

## Open risks (honest list)
- **Post hoc.** E1's protocol (section 4.4) made the sampled policy the primary read-out. Argmax is a mode E1's own
  code provides, but no E1 analysis froze it. Table 5's sampled values stay of record, and argmax is reported beside
  them.
- **The conditioning.** All 12 blocks are read on the same 80 episodes, and the rule's J is a constant on them. The
  block-level intervals therefore carry training-seed variation only. The episode-level interval is descriptive and
  is used only for the scope sentence.
- **R-B's 3× threshold is at its boundary under the sampled read-out**: r = 3.01, with a bootstrap of 2.41 to 3.49
  (data of record). See decision 1 in the draft.
- **The best family in R-B is chosen on the argmax results.** The bootstrap re-chooses it in every resample. This
  adds the selection variability, and the max-of-means bias makes the ratio interval conservative for the rule.
- **The idea_dq tree.** Launch refuses if the idea_dq tree changes, even though nothing imports it. This follows the
  instruction that it must stay unchanged; if some other job edits idea_dq, the launch stops.
- Nothing was found that would invalidate the study: the read-out path reproduces E1 exactly, and argmax was
  reproducible across processes.
