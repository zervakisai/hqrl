# Post-hoc E1-ARGMAX: the argmax read-out of the 48 E1 final policies next to Table 5

Final plan, written 2026-10-26 after the build prechecks (`precheck_build.json`, `precheck_build2.json`; tagged, they cannot authorise a launch) and before any argmax read-out of record; edited from `PLAN_DRAFT.md` by the lead.

The study is **post hoc**: it was planned after the referees' comments on Table 5, and after every E1 result of record
was known. It is **evaluation only**. There is no training, no tuning and no selection, and nothing that E1 ran is
changed. No re-run with changed settings will be reported as this study; any follow-up needs its own, separately
hashed plan.

## Question
Referees object to the systems comparison in Table 5, where the tuned rule's weighted J is 0.206 and every learned E1
family's is at most 0.052. The learned policies were read out by *sampling* their action distributions (the E1 config
has `"evaluation": {"suite": "e1_evaluation", "modes": ["sample"]}`), but the rule acts deterministically.

**Read out deterministically (argmax), on the same 80 e1_evaluation episodes, do the 48 E1 final policies still fall
short of the rule? And does the rule still earn several times the best family's timely credited reports?**

## What is evaluated (fixed now)
- **Policies: 48.** These are the four E1 families, each at its E1-selected candidate, times the E1 evaluation blocks
  10–21 (N = 12). The selection is `tuning/selection.json`, re-derived from the tuning evidence by
  `e1_study.verify_selection_record`:

  | family | role | candidate |
  |---|---|---|
  | `mlp` | C-strong | lr3e-3_ent0 |
  | `bottleneck` | C-matched | lr3e-3_ent0 |
  | `surrogate625` | Surrogate625 | lr1e-3_ent0 |
  | `quantum` | HQ | lr3e-3_ent0 |

  All 48 E1 evaluation runs completed. None has a retry and none ended in an algorithmic failure.
- **Checkpoint.** Each run's `final_update…_decisions…pt` is used: its run summary marks it `final: true`, and its
  SHA-256 is checked against the summary on every load. It is tensor-for-tensor identical to the checkpoint that run
  wrote at the 30,000-decision grid point. The trainer saves before it evaluates, and no update falls between the two,
  so this is the state whose sampled read-out at point index 10 is Table 5's "final" row. (The policy decisions per
  run are 30,002–30,481, under the E1 overshoot convention.) The 48 jobs, both checkpoint hashes, the point index and
  the stored sampled weighted J are bound by the **policy table digest
  `edd1633ec79dc9a2fe3a5af89afcc7aa787375cf272b1baa94dcb03240502fcd`** (`runner.table_digest`). Launch and the
  analysis refuse on any other digest.
- **Episodes: 80**, the E1 `e1_evaluation` suite: the 8 core cells × realisations 0–9. Each episode's seed comes from
  `p3_design.seed_for` in the E1 `e1_evaluation` namespace, as frozen in the E1 seed manifest. These are the same 80
  episodes as Table 5's sampled read-out and the rule's rows of record (`baselines.json`, `p3_rule/e1_evaluation`), in
  the same order. Common random numbers per episode are as in E1.
- **Read-out.** `p4_train.evaluate` is called **unchanged**, with `{"suite": "e1_evaluation", "modes": ["argmax"]}`,
  point index 10, `gamma_ref` and `reference_duration_s` from the job's config, and the label `e1-<family>-b<block>-eval10`.
  That is exactly the call the E1 trainer would have made at 30,000 decisions had its config listed `argmax`, a mode
  `p4_train.EVALUATION_MODES = ('sample', 'argmax')` provides.
  - In argmax mode, `RLMissionPolicy` runs with `deterministic=True`, no generator and `compute_values=False`.
  - Each decision is the argmax of the masked logits, and ties go to the lowest index.
  - A rejected action is masked for that observation, and the argmax is taken again.
  - The quantum head is analytic, as in E1. The dtype is float64, and torch uses 1 thread (the config's
    `torch_threads`).
  - Everything else is E1's own code path, unchanged: the encoder, masks, RETURN rule, executive, cells, frozen cell
    weights (0.2 for target-present cells, 0.05 for no-human cells), score and aggregation
    (`p4_train.aggregate_evaluation`).
- **Code and interpreter.**
  - The idea_e1 code tree is imported unchanged. Its live hash is
    `4a1f2e8e6a76cbc3ba286397139da9bb888c28f2e8a4bf84adb9f2494e46ddc5`, the tree named by the approved E1 freeze
    `c80eb591…` and by all 48 E1 run manifests. `p4_train.require_approved_freeze` re-verifies the freeze in every
    process.
  - The interpreter is the one E1 ran with: `idea_e1/.venv/bin/python` (CPython 3.12.13, torch 2.14.0+cpu, packages
    from `idea/.venv`), started with `-B`.
  - Every worker checks that the key modules (`p4_train`, `fast_mission`, `run_p3_cells`, `aerial_qrl`, …) load from
    idea_e1, and that nothing loads from idea_dq or `idea/src`.
  - The idea_dq tree (`ffe9dfef…`) is recorded at the start and end of every process and never written. Its files
    shared with idea_e1 are byte-identical.
- **CPU only.** The machine freezes when NVIDIA BAR1 is exhausted, so no process may touch the GPU.
  - `CUDA_VISIBLE_DEVICES=''` is set before torch is imported, and the torch build is `+cpu` anyway.
  - Each worker uses 1 torch thread, and the processes run at nice ≥ 10.
  - Each policy is evaluated in a fresh interpreter, as each E1 job was (a spawn pool with `max_tasks_per_child=1`).

## Runs (48 read-outs, all new)
- There is one record per policy: `records/<family>__<candidate>__b<block>.json`. Each record is written to a
  temporary file and then moved into place with `os.replace`.
- Each record carries the 80 episode rows: cell_id, seed, J, tp_timely, tp_spatial, fp_spatial, return_ok,
  fixes_attempted, decisions and termination_reason.
- It also carries the plan's SHA-256, the script hashes and both code-tree hashes at start and at end, the checkpoint's
  SHA-256 and identity checks, the environment, the module origins, wall and CPU time, and max RSS.
- **Deterministic.** The argmax read-out uses no random stream, so a repeat gives the same rows (the precheck's
  probe (d) checks this across processes). An infrastructure failure, meaning an exception or a dead worker, is
  therefore retried **once**, which is outcome-neutral. The failed attempt is kept in `records/attempts/`.
- After two failed attempts, a policy is not run again. The study is then reported as incomplete, naming that policy.
  Nothing is substituted, and the analysis refuses a partial read-out.
- A record that does not parse, or is not a completed argmax record of this plan, is moved to `records/aside/` and
  re-run.
- **Refusals.**
  - `launch` refuses unless all of these hold:
    - PLAN.md matches PLAN.sha256, and every script has the SHA-256 listed below;
    - both code trees are at their hashes;
    - the policy table re-proves with the digest above;
    - the environment checks pass (E1 interpreter, -B, no CUDA, torch 2.14.0+cpu, nice ≥ 10);
    - `precheck.json` (untagged) passed for this plan and these scripts;
    - MemAvailable ≥ workers × 0.8 GiB + 2 GiB.
  - Each worker re-checks the trees, the scripts, the plan SHA-256, the environment and the module origins, and
    refuses (no record) on a difference.
  - One launch runs at a time (an flock on `.launch.lock`), and a relaunch resumes from the records that exist.

## Precheck before launch (`runner.py precheck`; launch refuses without it, for this plan and these scripts)
(a) **Static checks.**
- Both trees are at their hashes, before and after.
- The approved E1 freeze, the campaign (`76bc358d…`), the baselines, the selection and all 48 evaluation jobs re-prove
  through E1's own proof code (`e1_study.load_campaign`, `require_baselines`, `verify_selection_record` and
  `proven_stage`). That code checks the configurations, run manifests, training rows, evaluation grid and suite, and
  every stored J against its episode rows.
- `baselines.json` matches its SHA-256 (`00202039…`). The rule's weighted J re-derives to `0.20564463190957744` by
  repr, and the rule's timely credited reports to 2.3885 per episode.
- The policy table and its digest are built (and checked against this plan once it is hashed).

(b) The unit tests pass: `test_runner.py` (18) and `analysis.py --selftest`.
- The self-test checks the statistics against scipy.
- It re-derives the rule's J of record, and Table 5's sampled family means from the stored rows: 0.0243 (HQ), 0.0374
  (C-matched), 0.0520 (C-strong) and 0.0449 (Surrogate625).
- It also checks the E1 report's credited-report and false-report panel.

(c) **The validity check.** The stored *sampled* read-out of **all 48** final policies is re-run through this runner's
own read-out path: the same function, the same checkpoint loading, and `p4_train.evaluate` with `modes: ['sample']`.
- **Every stored field of every one of the 3,840 episodes must be equal by repr**, J included.
- Per-cell and weighted J must also be equal by repr.
- A single difference fails the precheck. The path is then not used: its cause (tree, checkpoint, stream,
  environment) is found first.

(d) **Outcome-blind argmax probe.** For each family, block 10's final policy runs argmax on the 8 realisation-0
episodes, twice, in two separate processes.
- Only the wall time per episode and one SHA-256 of the 8 rows are kept. The rows are never written or printed.
- The two digests must agree. The main record of that policy must later reproduce the same digest: the analysis
  refuses otherwise.
- Each process then re-runs the same 8 episodes sampled. These must equal the stored rows, and they give the
  argmax/sampled time ratio.

(e) Timings and the estimate of the main run's wall time.

(f) Memory: MemAvailable, free -m, and the maximum RSS of a worker. The launch rule of 0.8 GiB per worker must cover
the measured maximum.

(g) No file or directory under idea_e1 or idea_dq was created or modified during the precheck.

**Build precheck with the final scripts (`precheck_build2.json`, 2026-10-26 12:37Z, 8 workers, nice 10): passed.**
A tagged precheck cannot authorise a launch.
- (a) Both trees were at their hashes before and after. The freeze, campaign, selection and 48 jobs re-proved. The
  rule's J re-derived as 0.20564463190957744.
- (b) All tests passed.
- (c) **48 of 48 policies and 3,840 of 3,840 episodes were identical in every stored field, by repr, with 0 differing
  fields. Per-cell and weighted J were identical for all 48.**
- (d) The argmax digests agreed between the two processes for all four families. They also equal the digests of the
  first build precheck (`precheck_build.json`), where the cells were not pre-loaded. That makes four processes in two
  contexts with the same argmax rows. The sampled 8-episode subsets were identical in every process.
- (f) The maximum RSS was 0.53 GiB per worker.
- (g) No file under either tree was touched.

The first build precheck (`precheck_build.json`, an earlier `runner.py` without the timing warm-up) also passed, with
3,840 of 3,840 episodes identical.

## Outcome and analysis (fixed now; `analysis.py`, run once after all 48 records exist)
Every J is frozen-cell-weighted (`p4_train.aggregate_evaluation`). For family f and block b = 10, …, 21:
- `J_arg(f,b)` is the argmax read-out of the final policy on the 80 episodes.
- `J_smp(f,b)` is the stored E1 sampled read-out of the same state, at 30,000 decisions (Table 5's). It comes from the
  episode rows that `e1_study.proven_stage` proves and that precheck (c) reproduced exactly.
- `J_rule = 0.20564463190957744` is the rule on the same 80 episodes. It is a constant.

The analysis reports:
- **Per family:**
  - the mean over the 12 blocks of J_arg, with its 95% t interval (df 11), beside the same for J_smp;
  - the pooled timely credited reports per episode (`e1_report.credited`: tp_timely if the episode returned, else 0)
    and the pooled false reports (fp_spatial), with return rate, fixes, decisions and termination reasons. These are
    pooled over the 960 episodes as in Table 5, which equals the mean of the 12 block means.
- **D_as(f,b) = J_arg − J_smp**, paired by block: the mean, the 95% t interval, and how many blocks went up.
  **Descriptive.** The same is reported for credited and false reports.
- **D_rule(f,b) = J_arg − J_rule:** the mean, the 95% t interval and the two-sided one-sample t p, with **Holm over the
  4 family contrasts** (α = 0.05). This is R-A.
- **Episode-level interval (descriptive, as in the E1 gate study).** Per episode, d_e = mean over blocks of
  J_arg(f,b,e) − J_rule(e). The interval is the one-sample t interval of u_e = 80 · w_e · d_e, with w_e = the cell
  weight / 10 and df 79. Its point estimate equals mean D_rule exactly.
- **R-B:**
  - r = (the rule's pooled timely credited reports, 2.3885 per episode, recomputed from the rule rows) / (the best
    family's pooled argmax timely credited reports). The best family is the highest point estimate; a tie goes to the
    family order HQ, C-matched, C-strong, Surrogate625.
  - Its interval comes from a **paired block bootstrap**: 10,000 resamples with seed 20261026. The same resampled
    blocks are used for every family, since blocks share their training schedule across families, and the best family
    is re-chosen in every resample.
  - The bootstrap gives the 2.5% and 97.5% quantiles of best/rule, which are inverted to an interval of r. A best
    family with 0 credited reports gives r = ∞.
  - For reference: r for every family, and the sampled r (3.01 against C-strong in Table 5).

**Conditioning.** The block-level intervals condition on the 80 episodes. Every block is read on the same episodes,
and the rule's J is a constant on them, so the intervals contain training-seed variation only, not episode-sampling
variation. `argmax_results.json` and every reading sentence say so.

## Reading rules (fixed now)
These are the sentences `analysis.reading_text` writes, with the numbers filled in.
- **R-A (per family; Holm over the 4 contrasts learned(argmax) − rule).**
  - **"The systems gap holds under the argmax read-out"** if the Holm-adjusted two-sided p ≤ 0.05 and the mean D_rule
    < 0. This implies that the unadjusted 95% interval of D_rule lies below 0, and both are reported.
  - **"Reversed"** if the Holm-adjusted p ≤ 0.05 and the mean > 0.
  - **"Not resolved"** otherwise. This is not evidence of equality.
  - Scope: "holds" or "reversed" is also stated for new episodes of these 8 cells only if the episode-level 95%
    interval agrees, i.e. lies below 0 for "holds" or above 0 for "reversed". Otherwise the sentence says that the
    reading holds for these 80 episodes only. Nothing is claimed beyond these cells and this one tuned comparator.
- **R-B ("the rule earns at least k × the best family's timely credited reports", k = 2 and k = 3).**
  - **"Holds"** if the point ratio r ≥ k and the bootstrap lower bound ≥ k.
  - **"Fails"** if r < k and the bootstrap upper bound < k.
  - **"Not resolved"** otherwise.
- **D (argmax − sampled) is descriptive.** No test is made and no reading is drawn from it.
- **What is not done.**
  - Table 5's sampled values stay of record. The argmax read-out is reported next to them, never in their place, and
    a family's better read-out is never picked as its headline.
  - The rule is not re-run, since it is deterministic and its 80 rows are of record. No episode, cell, weight or score
    changes.

## Compute
Measured in `precheck_build2.json` with 8 workers at nice 10. The load average was about 2 at the start and 10–12
during the run, on 16 threads.
- **Sampled read-out**, per policy (80 episodes, fresh process), mean: HQ 133 s, C-matched 130 s, C-strong 172 s and
  Surrogate625 127 s. All 48 took 861 s of makespan.
- **Argmax/sampled wall per episode** (probe, 8 episodes × 2 processes, cells pre-loaded): HQ 1.03, C-matched 1.19,
  C-strong 1.17 and Surrogate625 0.48.
- **Estimated main run** (48 × 80 argmax episodes; sum of task times about 6,650 s, scheduled longest first):
  - **8 workers about 14 min**, 7 workers about 16 min, 6 workers about 19 min, at this load.
  - Allow 2× for a busier machine or longer argmax episodes in other blocks: under 40 min in any case.
- **The precheck of record** (after hashing) takes about 15 min with 8 workers: the 48 sampled reproductions plus
  the probe.
- **Memory.** The maximum RSS is 0.53 GiB per worker, so 8 workers need about 4.3 GiB. MemAvailable was 10.8–11.0
  GiB. The launch requires workers × 0.8 + 2 GiB, which is 8.4 GiB for 8 workers.
- **CPU only.** CPU time equals wall time per worker (single-threaded torch), and nothing touches the GPU.

## Decisions fixed by the lead (2026-10-26, before hashing)
1. **R-B.** Both thresholds are computed as drafted. The manuscript states the k = 2 reading as the claim, and the
   k = 3 reading descriptively, always with the ratio and its bootstrap interval ("about three times" only with the
   interval). This is fixed before any argmax read-out exists; under the stored sampled read-out the k = 3 reading is
   already "not resolved" (r = 3.01, 2.41 to 3.49).
2. **R-A** as drafted (Holm over the four families), with the episode-level scope rule as drafted.
3. **Workers: 8.** The memory rule (workers × 0.8 GiB + 2 GiB) stays.
4. **Manuscript text for every outcome** of R-A and R-B was written before this plan was hashed: package commit
   8a48b82 (`tools/panel_values.py`, `tools/analysis_text.py` of the EAAI package). Every post-result wording change
   is logged in PLAN_LOG.md.
5. **Disclosure.** The build prechecks ran 8 argmax episodes per family (block 10, realisation 0) four times; only wall
   times and SHA-256 digests were kept, nothing was printed or read, and the main records must reproduce the digests.

## Binding: scripts, data and constants
The runs and the analysis execute exactly these scripts of this folder (SHA-256 after the last edit). `runner.py launch`
and `analysis.py` refuse on any difference:

| file | sha256 |
|---|---|
| `runner.py` | `b64156af705b581eea00401a1292190baf6197802faa977302dd79bfa8171231` |
| `analysis.py` | `4faf8d85289e6caf4cc147fa6ac1fa45f8effe4071d5b25cc92d307827d138ee` |
| `test_runner.py` | `e1cf141f0c3d14477ccdeef93a9adc0ec4a227471a467f5bb6c99d9fc145d98d` |

(If the lead edits a script before hashing, this table must be updated, and the precheck of record re-checks it.)

Other bound values (the constants of `runner.py`, between `DESIGN` and `END DESIGN`):
- the idea_e1 code tree `4a1f2e8e6a76cbc3ba286397139da9bb888c28f2e8a4bf84adb9f2494e46ddc5` (imported, never written);
- the idea_dq code tree `ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952` (never written);
- the approved E1 freeze `c80eb591ed5952de638c2bef4553608b8f25aefe1170f200936f882275467965`;
- the E1 campaign `76bc358d824567c0f413e946bccb5c6ab0fb1d84ef26d79f09459dd557f4dd56`;
- `baselines.json` `00202039dda01d814146db77e85590996abda974b4e813bbb17c10d081572fd7`, whose rule weighted J on
  e1_evaluation is `0.20564463190957744`;
- the policy table digest `edd1633ec79dc9a2fe3a5af89afcc7aa787375cf272b1baa94dcb03240502fcd`;
- the read-out: `argmax`, suite `e1_evaluation`, point index 10 (30,000 decisions), blocks 10–21, the candidates above;
- the probe: block 10, realisation 0 of each cell;
- the analysis: α = 0.05, Holm over 4, a bootstrap of 10,000 resamples with seed 20261026, and thresholds 2 and 3.

## Reporting commitment
Whatever the outcome, the manuscript reports the following next to Table 5, as a post-hoc check:
- per family, the argmax weighted J with its interval;
- argmax − sampled;
- the R-A reading with its Holm-adjusted p and its scope;
- the pooled timely credited and false reports;
- the R-B ratio with its bootstrap interval and both readings;
- the sentence on conditioning on the 80 episodes;
- this plan's SHA-256 and the precheck result.

The records and scripts go to the repository. Every post-result wording change is logged in `PLAN_LOG.md`.
