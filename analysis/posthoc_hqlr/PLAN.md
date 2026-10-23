# Post-hoc check: HQ beyond the edge of its learning-rate grid (B2)

Written 2026-10-22 before any run below existed, after the B2 results of record were known. Not part of any frozen
confirmatory family; labelled post hoc in the manuscript.

## Question
B2's HQ and C-matched both use 3e-3, the upper edge of the grid {3e-4, 1e-3, 3e-3} from which E1 selected them. A
reviewer can read B-H1 (HQ learned less than C-matched) as HQ being under-trained at the edge of its grid, since the
circuit angles moved little in E1. This check trains HQ at the next half-decade, 1e-2.

## Runs (36, all new, none retried; a run interrupted by a machine crash is restarted from the beginning)
Same study object as B2 (`dq_b2.STUDY_B2`: salt, schedules, action and minibatch streams, initialisation, the 80
common-random-number evaluation episodes at the same 11 points). Only HQ's Adam learning rate differs: 1e-2, blocks
0-35, each paired with the stored B2 runs of the same block. The B2 code tree is imported unchanged; the runner adds a
candidate 'lr1e-2_ent0' to the study's candidate table in its own process, because E1's validator admits only frozen
candidates.

## Outcome and analysis (fixed now)
Outcome: B2's AUC from each run's summary.
- Q1: AUC(HQ, 1e-2; new) - AUC(C-matched, 3e-3; stored), paired over 36 blocks: mean, 95% paired t interval,
  two-sided paired t p-value, mean/sd. Single post-hoc test at 0.05.
- Descriptive: AUC(HQ 1e-2) - AUC(HQ 3e-3; stored); final-checkpoint weighted J of each; mean absolute displacement of
  the circuit angles between the first and the last checkpoint, for HQ at 3e-3 (stored) and at 1e-2.
Reading: Q1 interval below zero -> the B-H1 deficit persists when HQ uses the next learning rate. Q1 not resolved ->
the B-H1 deficit depends on HQ's learning-rate selection. Q1 interval above zero -> HQ at 1e-2 exceeds C-matched at its
selected rate, so B-H1 reflects HQ's selection at the grid edge. In the last two cases C-matched is also trained at
1e-2 on the same blocks under this plan and HQ - C-matched at 1e-2 is reported descriptively.

## Reporting commitment
Whatever the outcome, Q1 and the descriptive values are reported in the manuscript next to B-H1 with this plan's
SHA-256; scripts and records go to the repository.
