# hqrl — Same Policy, Different Learning

Code and summary results for the study *Same Policy, Different Learning: Diagnosing a Quantum
Reinforcement-Learning Update Deficit in UAV Search and Rescue Reporting*, with a simulated flood case study.

A four-qubit hybrid quantum policy head (HQ) computes a finite Fourier series of its input angles, so each
circuit can be transplanted into an exact classical Fourier twin that starts from the **same policy**. From that
shared starting point, after matching the measured mean forward KL divergence of the update on a calibration set,
the study compares how much held-out improvement an update realised through the circuit achieves against updates
taken in the twin's coefficient space.

Author: Konstantinos Zervakis

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

Every confirmatory study was frozen before it was run: a freeze names the code tree, seeds, namespaces and
analysis, and the runners refuse to start when the live tree differs from the frozen one.

## Layout

```
src/aerial_qrl/              environment, mission executive, georeferencing, PPO and actor heads
scripts/gnss_qrl_runtime/    study runtime (dq_*), the P3/P4/E1 machinery it reuses
tests/gnss_qrl_dq/           tests of the study code
analysis/posthoc/            post-hoc descriptive scripts and their outputs
output/study/gnss_qrl_v1/     records and reports of record (see below)
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

Post-hoc descriptive outputs are in `analysis/posthoc/`.

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
