# Execution log (not part of the plan; PLAN.md is unchanged after hashing)
- 2026-10-23T11:58:57Z plan hashed (no run of this check existed). runner.py derived from the HQ-at-1e-2 runner (arm bottleneck, candidate lr1e-2_ent0); its precheck now re-runs the stored B2 C-matched of block 0 at 3e-3.
- 2026-10-23T12:02:00Z precheck passed (2 evaluations, 6 updates, 67 episodes identical to the stored B2 run); launched with 5 workers (runner daa4c51d5a79).
- 2026-10-23T12:44:13Z cmlr_analysis.py written before any result of this extension was read (fee4910d9b77).
- 2026-10-24T00:02:14Z all 36 runs completed; cmlr_analysis.py (fee4910d9b77, unchanged) run without --provisional -> cmlr_results.json (55a1eaa85a5e). Reading as computed: Z1 below zero (HQ 1e-2 − C-matched 1e-2 AUC −0.0090 [−0.0147, −0.0034], p = 0.0027).
