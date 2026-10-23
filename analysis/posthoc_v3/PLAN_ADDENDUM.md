# Addendum to PLAN.md: interruption of the launch by machine crashes

Written 2026-10-22 ~16:05 EEST, before any run below was restarted. PLAN.md (SHA-256 b8ec9f9f…) is unchanged.

## What happened
The launch of 2026-10-22T12:16:34Z (12 workers) was stopped at 15:49 EEST when the machine had to be restarted after
an NVIDIA display-driver lock-up ("RC watchdog: GPU is probably locked", kernel log of boot -1). The runs are CPU-only;
the lock-up is unrelated to them (the same lock-up had already restarted the machine at 15:03, before the launch).
At that point 3 of the 72 runs had completed (records/L1_surrogate625__b0, L1_surrogate625__b2, L2_quantum__b0);
12 were in progress with no record; 57 had not started. No run had failed.

## Rule for the remaining runs (fixed now)
- Completed records are kept as they are. A run with a record of any status is never run again; a failure stays a
  failure, as PLAN.md says.
- A run directory without a record was killed by the crash. It is moved, unchanged, to `interrupted/<UTC time>/` and
  the run is started again from the beginning, never resumed from a checkpoint. All seeds, streams and schedules are
  fixed by `dq_b2.STUDY_B2`, so the restarted run is the same computation the uninterrupted run would have been (the
  pre-check showed identical episodes and evaluations, with update metrics equal to 3.5e-14 relative difference).
- The rule is applied to every interrupted run without looking at any partial result, so it cannot select on outcome.
- `relaunch.py` applies this rule; it imports `runner.run_one` unchanged and refuses to start if the B2 code tree or
  PLAN.md differ from launch.json. If the machine crashes again, the same rule and script are used again, and each
  relaunch is logged in `relaunches.jsonl`.
- The analysis of PLAN.md is unchanged. The manuscript states that the launch was interrupted by a machine crash and
  that the interrupted runs were restarted from the beginning under this addendum.
