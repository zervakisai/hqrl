# Execution log (not part of the plan; PLAN.md is unchanged after hashing)
- 2026-10-24T18:32:17Z plan hashed (a5f99655ae89); no run existed.
- 2026-10-24T18:34:31Z runner.py written after the hash (1b04a439edf7); b2g_analysis.py written before any run (acf88206d90a); its --selftest on stored B2 runs reproduces B-H1 (−0.0087, p = 0.0017). 08_SYNTHESIS_RULES.md Addendum C written before any run.
- 2026-10-24T18:35:22Z precheck passed (precheck.json: only actor.weights differ from the stored block-0 initial checkpoint, angles within (−π, π), mean |θ| 1.46 vs 0.094 rad; with the replacement disabled the code path reproduced stored HQ block 0 exactly for 3,000 decisions: 2 evaluation, 6 update, 77 episode events). Launched 36 runs, 12 workers, nice 19.
- 2026-10-24T18:39:32Z manuscript templates for every G1 reading committed in the package (commit aadd927: results_edits.b2g_edits, synthesis_phrases q9_*, texts.py branches, limitation (c)) while records/ held 0 files; each branch built from synthetic values in scratch (abstract 249-250 words).

- 2026-10-24T20:44:21Z — Before any result was read (24/36 runs finished; analysis not run; no metric opened): dry run of the full
  manuscript build with synthetic result files for each of the three G1 readings (scratch only). All three build; one
  abstract phrase moved so that "generic angles" qualifies only the C-matched contrast (08_SYNTHESIS_RULES.md Addendum
  C.1). No change to PLAN.md, the runner or the analysis script.

- 2026-10-24T21:26:25Z — All 36 runs completed (none failed, none interrupted, none retried; launcher exited). PLAN.md
  SHA-256 re-verified (a5f99655ae89…). `b2g_analysis.py --selftest` reproduced B-H1 (−0.0087, p = 0.0017); then
  `b2g_analysis.py` (non-provisional; stdout in analysis_stdout.txt) wrote b2g_results.json (SHA-256 4b7ee75eb03c…).
  G1 HQ(generic) − C-matched AUC −0.0117 [−0.0164, −0.0070], p = 1.3 × 10⁻⁵, dz −0.85, Holm rejected →
  reading "below zero: the deficit persists from generic angles". G2 HQ(generic) − HQ(small-angle) −0.0030
  [−0.0061, +0.0001], p = 0.057, Holm not rejected. Reported in the manuscript by the templates committed before the
  runs (aadd927) as amended before any result by Addendum C.1 (da7a6ef).
- Independent re-derivation (reviewer_audit/tools/independent_b2g_check.py; does not import this folder's scripts):
  AUC rebuilt from the per-episode evaluation events for all 108 runs (max |diff| 7 × 10⁻¹⁸), G1/G2 and Holm recomputed
  with scipy.stats.ttest_rel (identical), and for all 36 blocks the new initial checkpoint equals the stored B2 HQ one in
  every parameter except actor.weights, which equal the planned U(−π, π) draw; arm_start seeds and model identical.
