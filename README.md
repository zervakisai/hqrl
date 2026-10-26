# hqrl — Same Policy, Different Learning

Code and summary results for the study *Same Policy, Different Learning: Evaluating a Hybrid Quantum Policy Head
against Classical Controls in a Simulated Unmanned Aerial Vehicle Flood Search-and-Report Benchmark*.

A four-qubit hybrid quantum policy head (HQ) computes a finite Fourier series of its input angles, so each
circuit can be transplanted into an exact classical Fourier twin that starts from the **same policy**. From that
shared starting point, after matching the measured mean forward KL divergence of the update on a calibration set,
the study compares how much held-out improvement an update realised through the circuit achieves against updates
taken in the twin's coefficient space.

## Studies

| Study | What it does | Module |
|---|---|---|
| Transplant | Exact HQ → Fourier-twin transplant with a fail-closed parity certificate | `scripts/gnss_qrl_runtime/dq_transplant.py` |
| Instruments | Exact local quantities (gradients, Fisher, natural steps, KL calibration) in float64 | `dq_instruments.py`, `dq_circuits.py` |
| D-6 | Two-unit instrument pilot that fixed the B1 settings before any validation unit existed | `dq_d6.py` |
| B1 | Identical-policy local comparison on 72 fresh units: HQ-Fisher, SPAN, RND-k, TAN and controls | `dq_b1.py`, `dq_b1_analysis.py` |
| B2 | From-scratch training replication (HQ, C-matched, exact S625 twin; 36 paired blocks, 30,000 decisions) | `dq_b2.py`, `dq_b2_analysis.py` |
| Freeze A | Stress evaluation on the 12 E1 evaluation blocks | `dq_freeze_a.py`, `dq_freeze_a_analysis.py` |
| Post hoc | Descriptive analyses (finite-step path difference, arm ladder, pair-preserving bootstrap) | `analysis/posthoc/` |
| Post hoc v3 | Pre-submission checks: B2 at matched learning rates (72 new runs, plan fixed by hash before the runs), damping limits of B1, sign-flip permutation tests, held-out KL transfer, stress-test block intervals, KL efficiency | `analysis/posthoc_v3/` |
| Post hoc HQLR | B2's HQ re-trained at learning rate 1e-2, beyond its selection grid (36 new runs; own hashed plan) | `analysis/posthoc_hqlr/` |
| Post hoc B1X | B1 extended with re-linearised circuit steps and a damping ladder on the 72 B1 units, run with the B1 code tree (own hashed plan) | `analysis/posthoc_b1x/` |
| Post hoc CMLR | B2's C-matched at learning rate 1e-2 (own hashed plan) | `analysis/posthoc_cmlr/` |
| Post hoc B1C | Matched classical trigonometric charts (CTRIG) for the local path comparison | `analysis/posthoc_b1c/` |
| Post hoc B1G | Local comparison at generic circuit angles | `analysis/posthoc_b1g/` |
| Post hoc B1E | Optimised circuit and tangent steps (OPT-CIRC, OPT-TAN) | `analysis/posthoc_b1e/` |
| Post hoc B1N | Local comparison with a four times larger fit set | `analysis/posthoc_b1n/` |
| Post hoc B1T | Local comparison at the trained B2 policies | `analysis/posthoc_b1t/` |
| Post hoc LONG, LONG-HQ | Ten times the training budget for C-strong, C-matched and HQ | `analysis/posthoc_long/`, `analysis/posthoc_longhq/` |
| Post hoc B2G | Early training from generic circuit angles | `analysis/posthoc_b2g/` |
| Post hoc SENSL | Alternative false-report weights (re-analysis of stored stress records) | `analysis/posthoc_sensl/` |
| Post hoc B2F | HQ with frozen circuit angles | `analysis/posthoc_b2f/` |
| Post hoc B2K | KL-matched early training (learning rates calibrated to mean per-update KL) | `analysis/posthoc_b2k/` |
| Post hoc E4 (long2) | C-strong and C-matched re-tuned with the rule's evidence counts, 300,000 decisions | `analysis/posthoc_long2/` |
| Post hoc gate | A learned correction trigger inside the rule's executive | `analysis/posthoc_gate/` |
| B1R | Prespecified replication of the local path contrasts on 36 fresh units | `analysis/posthoc_b1r/` |
| Post hoc E1 argmax | Argmax read-out of the 48 E1 final policies (evaluation only) | `analysis/posthoc_e1argmax/` |
| Re-analyses | Interval re-analyses of stored records and descriptive summaries | `analysis/posthoc_reanalysis/` |

Every confirmatory study was frozen before it was run: a freeze names the code tree, seeds, namespaces and
analysis, and the runners refuse to start when the live tree differs from the frozen one.

## Layout

```
src/aerial_qrl/              environment, mission executive, georeferencing, PPO and actor heads
scripts/gnss_qrl_runtime/    study runtime (dq_*), the P3/P4/E1 machinery it reuses
tests/gnss_qrl_dq/           tests of the study code
analysis/posthoc*/           post-hoc scripts, analysis plans and their outputs
output/study/gnss_qrl_v1/     records and reports of record (see below)
output/mission/scenes/       Gazebo scene of the case study (see below)
```

## Data and results

All files below are the original outputs of record, unchanged. `SHA256SUMS` lists their SHA-256 identities
(the identities given in Appendix D of the paper); verify with `sha256sum -c SHA256SUMS`.

| Path under `output/study/gnss_qrl_v1/` | Content |
|---|---|
| `dq/development/d6/analysis/d6_results.json` | D-6 instrument pilot result |
| `dq/freeze_a_campaign_v1/a0/a0.json`, `a1/a1.json` | Freeze A stress evaluations A0 and A1 |
| `dq/freeze_a_analysis_v1/freeze_a_report.json` | Freeze A report |
| `dq/b1_campaign_v1/units/` | B1 per-unit records (72 units: 36 shipped, 36 held-out circuits) |
| `dq/b1_analysis_v1/b1_report.json` | B1 report: primary family (M-H1, A-H1), gate, secondary contrasts |
| `dq/b2_campaign_v3/runs/*/run/*/metrics.jsonl` | B2 per-run records: training episodes, PPO updates and checkpoint evaluations (108 runs) |
| `dq/b2_analysis_v1/b2_report.json` | B2 report |
| `e1/e1_report_v1/report.json` | E1 report (earlier exploratory comparison of learned heads) |
| `p3/p3_gate_verdict_v1/gate_verdict.json` | P3 gate verdict (earlier planner study) |

Post-hoc descriptive outputs are in `analysis/posthoc/` and `analysis/posthoc_v3/`. The matched-learning-rate runs are
in `output/study/gnss_qrl_v1/dq/b2_matched_lr_posthoc/` (`runs/*/run/*/metrics.jsonl`, `records/`, `lr_results.json`);
their analysis plan, `analysis/posthoc_v3/PLAN.md`, was written and hashed before any of them ran. The same holds for the
HQ-at-1e-2 runs (`dq/b2_hq_lr1e-2_posthoc/`, plan `analysis/posthoc_hqlr/PLAN.md`) and the B1 extension
(`dq/b1_relinearised_posthoc/`, plan `analysis/posthoc_b1x/PLAN.md`). Changes after hashing are logged next to each plan
(`PLAN_ADDENDUM.md`, `PLAN_LOG.md`). Model checkpoints are not included; `hqlr_analysis.py` reads them only for the
descriptive angle-displacement values, which are stored in the result JSON files. `SHA256SUMS` also lists every
post-hoc file and every scene file.

The post-hoc studies added before submission (2026-10-23 to 2026-10-26) are described in
`analysis/POSTHOC_ADDITIONS_2026-10-26.md`, which also maps the plan labels used in the Supplementary Material
(PL1–PL17, PL1-A) to the full SHA-256 of each plan. Each plan was hashed before its runs; model checkpoints are not
included. `SHA256SUMS` lists every added file.

## Simulation scene

`output/mission/scenes/flood_sar_village_v1r6_s0_build1_norigs/` is the Gazebo scene of the flood case study,
unchanged: the world files (`worlds/all_victims.sdf`, `worlds/absent.sdf`), the scenario and evaluator files
(`public/`, `private/`), the ground model with its CC0 ambientCG textures, the three derived house models, the
asset-attribution file `ATTRIBUTION.md` and the scene manifest `manifest.json` (SHA-256
`b6647dad1c5f0c0a0db1d182d261abef85f01af319c2bdc763e77610097ca8de`, the scene identity recorded in every part of
observation bank s0).

The scene uses 26 Gazebo Fuel models (six under CC BY 4.0, 20 under CC0 1.0) without copying them: the world files
include them through absolute `file://` URIs into a local Fuel cache, so they have to be pointed at a local download
of the same model versions, whose versions, URLs and per-file SHA-256 values are in `manifest.json` (`assets`).
`ATTRIBUTION.md` is kept as generated. Its title carries the generator's template name (`flood_sar_segment_v1`), and
its "Derived model" section lists only `house_1_pbr`; `house_2_pbr` and `house_3_pbr` are derived in the same way
(`manifest.json`, `derived_models`).

Observation bank s0, the views and detector outputs recorded in this scene (about 54 GB), is too large for the
repository and is available from the authors on reasonable request.

## Installation

Python 3.12.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-lock.txt
```

## Tests

```bash
python -m pytest tests/gnss_qrl_dq
```

The circuit, instrument, transplant and analysis tests run on synthetic data. Tests that need the simulator's
benchmark configuration and observation bank are skipped (`tests/conftest.py`).
