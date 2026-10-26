# Post-hoc additions of 2026-10-25/26 for the public repository (local staging)

Staged on 2026-10-26 from the study folders `idea_dq_posthoc_{b2f,b2k,long2,gate,b1r,e1argmax}` and three re-analysis tools of the submission package. Nothing here has been committed, pushed or uploaded; merging it into the repository needs the authors' approval. The layout is that of `hqrl_additions_hardening_2026_10_24` (the earlier staging, unchanged).

## Layout

- `analysis/posthoc_<key>/`: the plan (`PLAN.md`, `PLAN.sha256`, `PLAN.utc`), the draft plan, the plan log (`PLAN_LOG.md`), the plan review (`REVIEW.md`) and build record (`README_BUILD.md`) that the plans cite, the scripts (runner, analysis, tests), launch logs, precheck summaries, and the result file of record.
- `output/study/gnss_qrl_v1/dq/<name>/`: per-unit or per-block records (`records/`), per-run training and evaluation logs (`runs/<job>/run/[<head>/]metrics.jsonl`), pilot records and logs, and a copy of the result file.
- `SHA256SUMS.additions`: SHA-256 of every staged file (sha256sum format, paths relative to this folder; `sha256sum -c SHA256SUMS.additions`). It does not list itself or this README, which was written after it.
- The output folder names were chosen at staging, following the earlier staging's pattern; no manuscript text refers to them.

## Folders

| analysis folder | output folder | study | plan label | manuscript / supplement | units | files, size |
|---|---|---|---|---|---|---|
| `posthoc_b2f` | `b2_frozen_angles_posthoc` | B2F: HQ re-trained on the 36 B2 blocks with its 16 circuit angles excluded from optimisation | PL11 | Section 4.7; Supplementary Material S5 | 36 runs (records/F_quantum__b0-b35, runs/.../metrics.jsonl) | 15 + 73 files, 35.7 MB |
| `posthoc_b2k` | `b2_kl_matched_posthoc` | E3 (B2K): early training at matched per-update KL (B2 protocol) | PL12 (draft), PL13 | Sections 4.6 and 4.7; Supplementary Material S6 | 72 main runs (36 T1_quantum, 36 T2_surrogate625) and 18 calibration-pilot runs on development seeds (pilot/) | 36 + 181 files, 84.1 MB |
| `posthoc_long2` | `b2_extended_budget_retuned_posthoc` | E4 (LONG2): C-strong and C-matched re-tuned, with the rule's evidence counts as inputs, on blocks 4-9 of the extended-budget check (300,000 decisions) | PL14 | Section 4.2.1; Supplementary Material S7 | 12 runs (6 mlp = C-strong, 6 bottleneck = C-matched) and 8 entropy-pilot runs (pilot/) | 36 + 41 files, 115.5 MB |
| `posthoc_gate` | `e1_learned_gate_posthoc` | E1-gate ("Post-hoc E1" in its plan): a learned correction trigger inside the tuned rule's executive, 12 blocks, evaluated on the 80 E1 evaluation episodes | PL15 | Section 4.2.1; Supplementary Material S8 | 12 blocks (records/gate__b0-b11, runs/.../metrics.jsonl) and 2 development blocks of the quick learning check (pilot_quick/) | 34 + 29 files, 35.0 MB |
| `posthoc_b1r` | `b1_fresh_units_posthoc` | B1R: prespecified replication of B1's local path contrasts on 36 fresh two-layer units | PL16 | Section 4.4; Supplementary Material S9 (Table S4) | 36 units (records/shipped_100-135) | 30 + 37 files, 36.9 MB |
| `posthoc_e1argmax` | `e1_argmax_readout_posthoc` | E1-ARGMAX: argmax read-out of the 48 E1 final policies on the 80 E1 evaluation episodes (evaluation only) | PL17 | Sections 3.4 and 4.2.1 (with Table 5); Supplementary Material S10 | 48 policy records (4 families x 12 blocks) | 25 + 49 files, 1.9 MB |
| `posthoc_reanalysis` | none | Re-analyses of stored records: `tools/reanalysis_panel.py`, `tools/gate_descriptives.py`, `tools/b1r_side_by_side.py` and their outputs in `work/` (paths relative as in the package's `reviewer_audit/`) | none (no new runs) | S10 and Sections 4.2.1, 4.8 (reanalysis_panel); S8 and Section 4.2.1 descriptive panel (gate_descriptives); S9 Table S4 (b1r_side_by_side) | none | 6 files, 0.1 MB |

### `posthoc_b2f` (PL11)

- Plan: `PLAN.md`, SHA-256 `a5e573b365ae694375e9f04eb36ed8b7e102120110e60484811d972ede422cd6`, hashed 2026-10-25T07:21:39Z (`PLAN.utc`).
- b2f_analysis.py: F1, F2, TOST, Holm and readings of the plan; b2f_results.json is the read-out of record (PLAN_LOG.md, 2026-10-25T11:44:01Z).
- b2f_decomposition.py/.json and b2f_identity_check.py/.json are POST HOC, written after the result and not in PLAN.md (PLAN_LOG.md, 2026-10-25T11:48:51Z and 11:50:08Z).
- PLAN.sha256 holds the digest only (no file name), so it is checked by comparing it with sha256(PLAN.md), not with `sha256sum -c`.

### `posthoc_b2k` (PL13)

- Plan: `PLAN.md`, SHA-256 `3f8ba848a631246de45e25ee3871fd7d681ed979c642acceb14a9842370e552c`, hashed 2026-10-25T12:48:13Z (`PLAN.utc`).
- Two plan identities: the draft snapshot taken at the pilot launch (pilot/PLAN_DRAFT_at_pilot_launch_2026-10-25T075118Z.md, PL12) and the final plan hashed after the pilot and before the 72 main runs (PLAN.md, PL13).
- precheck_prev_20261025T120553Z/runner.py.96bb8b08 is the runner that produced the pilot and selection.json; PLAN.md pins it by SHA-256 and pilot/launches.jsonl logs the same runner_sha256.
- The superseded prechecks (precheck_attempt1*, precheck_prev_20261025T120553Z/) are kept as summaries only.

### `posthoc_long2` (PL14)

- Plan: `PLAN.md`, SHA-256 `03e93f1a267e6b5b52225c851020ffeb29e41c26ee473add32c73a9a0754060e`, hashed 2026-10-25T13:23:42Z (`PLAN.utc`).
- provenance/ holds the byte copies the plan refers to (runner at the pilot launch, analysis and test drafts, the build record before fixes); each file name carries its SHA-256 prefix.
- The plan sends a restarts ledger and kept copies to the repository; no run was restarted (PLAN_LOG.md, 2026-10-25T23:35:06Z; e4_results.json restart_prefix_comparisons is empty for all 12 runs), so neither exists.
- tests/ holds the unit tests and their pytest outputs.

### `posthoc_gate` (PL15)

- Plan: `PLAN.md`, SHA-256 `20faccb84e4f9c79bd821af38a2190b0f0c110b2ac7c61f49e05879857e67a7c`, hashed 2026-10-25T13:33:59Z (`PLAN.utc`).
- pilot_quick/ is the development check on which the entropy coefficient 0.01 was chosen; PLAN.md discloses it with its numbers, so it is staged.
- PLAN.md commits "Scripts, records and checkpoints" to the repository; checkpoints were excluded by the staging rule (see below): 12 final checkpoints, 3.6 MB; all 72 main-run checkpoints, 19.2 MB.
- gate_analysis_of_record.out and config_check_of_record.out are the outputs of record; precheck.json is the precheck of record (PLAN_LOG.md, 2026-10-25T13:55:22Z).

### `posthoc_b1r` (PL16)

- Plan: `PLAN.md`, SHA-256 `2ed516a317256c1448fc34538d6ab79cd139d1e803a5c5cc92010fa33037d831`, hashed 2026-10-25T13:44:21Z (`PLAN.utc`).
- launch_snapshots/ and precheck/runner_snapshot.py.txt are the runner copies archived at each launch and at the precheck of record (PLAN.md, "Development history (m11)").
- dev_runs/quick_run1/precheck_quick.json/.log are staged because PLAN.md (m11) keeps the JSON of development quick-check runs 1 and 2 (run 2: precheck_prev_2026-10-25T121501Z/).
- analysis_run.log is the log of the single analysis run of record (PLAN_LOG.md, 2026-10-26T05:19:16Z).

### `posthoc_e1argmax` (PL17)

- Plan: `PLAN.md`, SHA-256 `51ba24e388f9aecd3924e047fe86e9d937a5ff168720c0ef60e7006e904bfda0`, hashed 2026-10-26T12:56:24Z (`PLAN.utc`).
- precheck_build.json and precheck_build2.json are the tagged build prechecks that PLAN.md cites; precheck.json is the precheck of record.
- Every record in output/.../records/ matches the SHA-256 listed in argmax_results.json (records_sha256, 48/48).

### `posthoc_reanalysis` (no plan)

- Post-hoc, read-only re-analyses of stored records; no run, no test or reading of record changes. `work/reanalysis_panel.json` records the SHA-256 of the script that wrote it (`script_sha256`, equal to the staged script); `work/gate_descriptives.json` records the SHA-256 of the gate_results.json it read (`source_sha256`).
- `b1r_side_by_side.py` and `reanalysis_panel.py` import `b1r_analysis.py` from the B1R study folder by absolute path; they were not run for this staging.

## Plan labels (PL1-PL17, PL1-A)

The double-anonymised manuscript and supplement name each plan by a label; the label stands for the plan whose SHA-256 begins with the key in the package's `tools/plan_labels.json`. Full identities, recomputed from the staged files:

| label | SHA-256 | file | study |
|---|---|---|---|
| PL1 | `58471290aea646c6914f38f10ec4d28c1fb839b3d31bc706d85992c855de1ef2` | `analysis/posthoc_long/PLAN.md` | LONG: ten times the training budget for the classical heads (earlier staging) |
| PL1-A | `51fcbd4e4a06beccda8aefaa5d14bc1fc560938d504b56a308ad23bb82c7d8e1` | `analysis/posthoc_long/PLAN_ADDENDUM.md` | addendum of LONG and LONG-HQ (earlier staging; identical bytes in posthoc_longhq/) |
| PL2 | `91f7fde44339f5bd094619faba00a24f3ee0a95105dc887e7cef1d8a0046d8ef` | `analysis/posthoc_longhq/PLAN.md` | LONG-HQ: HQ under the same extended training (earlier staging) |
| PL3 | `c1643c27ebfba342a8e2a83b431d4a69d4e989f8a49d1bfafa77c97e7fb8e3f1` | `analysis/posthoc_b1c/PLAN.md` | B1C: matched-tangent classical charts and the mission effect of local updates (earlier staging) |
| PL4 | `42835046e437078f88dfc94341c4faa495ac15b5cfe56e48bc1ee020b9851487` | `analysis/posthoc_b1g/PLAN.md` | B1G: path comparison at generic circuit angles and on C-matched (earlier staging) |
| PL5 | `418c0947322a79b429f2e9b0f5155d72a61ab2625e26c428f2f4719af0a4d3de` | `analysis/posthoc_b1e/PLAN.md` | B1E: optimised circuit steps against the best linear tangent step (earlier staging) |
| PL6 | `70997e8f88fb1df0f9d9724fda5f45e07dfa00acc896ca7802a0a1d469448552` | `analysis/posthoc_b1n/PLAN.md` | B1N: fit-set size (earlier staging) |
| PL7 | `725050f05128944a1121e856d338f5e0051352b568ac7d19143580560bb21d53` | `analysis/posthoc_b1t/PLAN.md` | B1T: identical-policy local comparison at trained policies (earlier staging) |
| PL8 | `9717ddf3bbb3a94e1421e650ee6358a3076318e324e7e86ffa2285477f4f0063` | `analysis/posthoc_cmlr/PLAN.md` | CMLR: C-matched beyond the edge of its learning-rate grid (earlier staging) |
| PL9 | `a5f99655ae89d5b448ae951efb3b11c555fd982828ef9dd515c35ff77fbeb960` | `analysis/posthoc_b2g/PLAN.md` | B2G: HQ trained from generic circuit angles (earlier staging) |
| PL10 | `52fee107572cd4be89d8fc5dd46904ea67fba25f9780d6ee54bbaa0e0b4d3be6` | `analysis/posthoc_sensl/PLAN.md` | SENSL: descriptive re-scoring at false-report weights 0.5 and 2.0 (earlier staging) |
| PL11 | `a5e573b365ae694375e9f04eb36ed8b7e102120110e60484811d972ede422cd6` | `analysis/posthoc_b2f/PLAN.md` | B2F, this folder |
| PL12 | `663283ddf929a35573e884d6036feb57a31331435b949117a77fab523560a9bf` | `analysis/posthoc_b2k/pilot/PLAN_DRAFT_at_pilot_launch_2026-10-25T075118Z.md` | E3/B2K draft-plan snapshot at the pilot launch, this folder |
| PL13 | `3f8ba848a631246de45e25ee3871fd7d681ed979c642acceb14a9842370e552c` | `analysis/posthoc_b2k/PLAN.md` | E3/B2K final plan, this folder |
| PL14 | `03e93f1a267e6b5b52225c851020ffeb29e41c26ee473add32c73a9a0754060e` | `analysis/posthoc_long2/PLAN.md` | E4/LONG2, this folder |
| PL15 | `20faccb84e4f9c79bd821af38a2190b0f0c110b2ac7c61f49e05879857e67a7c` | `analysis/posthoc_gate/PLAN.md` | E1-gate, this folder |
| PL16 | `2ed516a317256c1448fc34538d6ab79cd139d1e803a5c5cc92010fa33037d831` | `analysis/posthoc_b1r/PLAN.md` | B1R, this folder |
| PL17 | `51ba24e388f9aecd3924e047fe86e9d937a5ff168720c0ef60e7006e904bfda0` | `analysis/posthoc_e1argmax/PLAN.md` | E1-ARGMAX, this folder |

## What was staged and what was not

Staged: every top-level file of each study folder except process lock files; `records/`; the `metrics.jsonl` of every main and pilot run; the pilots (`pilot/`, `pilot_quick/`), `provenance/`, `tests/` and `launch_snapshots/`; from dry-run folders (prechecks, smokes, superseded prechecks) only their summary files (`.json`, `.log`, `.out` directly inside) and archived code copies; the three re-analysis tools and their outputs.

| excluded | files | size |
|---|---|---|
| model checkpoint or row cache (*.pt) | 1698 | 1970.5 MB |
| dry-run output (precheck/smoke run: training metrics, sample/probe/stage records) | 207 | 6.6 MB |
| intermediate working state of the B1R runner (stage/): staged C-arm records identical to the 'arms' of records/ (36/36), arm records (identical to records/) and start markers; the row caches (*.pt) are counted with the checkpoints | 72 | 1.3 MB |
| scratch: development smoke/quick-check runs (dev_runs/) | 17 | 0.5 MB |
| __pycache__ / bytecode | 1 | 0.0 MB |
| process lock file (pid/argv of a finished launcher; no record content) | 4 | 0.0 MB |
| total | 1999 | 1978.9 MB |

No source file, checkpoints included, exceeds 20 MB, so the 20 MB rule excluded nothing; the largest staged file is 8.5 MB (`runs/mlp__b4/run/mlp/metrics.jsonl` of E4). Empty folders (`records/attempts/` of the gate and E1-ARGMAX studies: no failed attempt) are not staged.

## Before publication

- Absolute paths: `/home/<user>` occurs 1430 times in 129 staged files (scripts, logs and records), and the session scratch path `<session scratch folder>` 14 times in 11 files. They were not rewritten, because the files are hashed of record. The user name in these paths identifies an author, while the manuscript withholds the repository link for double-anonymised review. The scripts read stored records outside this folder by absolute path (for example the B2 runs, the B1/B1C/B1E records and the E1 baselines), so re-running them elsewhere needs those trees and a path change.
- Secrets and tokens: 0 matches; e-mail addresses: 0.
- Model checkpoints are not included, except those of the E1-gate study (its plan sends them to the repository; 72 files, 19 MB, listed in `SHA256SUMS`).


## Notes
- The lines of `SHA256SUMS.additions` are merged into the repository's `SHA256SUMS`.
- The staged files keep absolute paths of the authors' workstation in logs and records (their hashes are of record, so
  they are not rewritten). The repository is named only to the editor, not to the reviewers.
