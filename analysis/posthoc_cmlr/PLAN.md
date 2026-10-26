# Post-hoc check: C-matched beyond the edge of its learning-rate grid (B2), symmetric to the HQ check

Written 2026-10-23 before any run below existed, after the B2 results of record and the HQ-at-1e-2 check
(`idea_dq_posthoc_hqlr`, plan SHA-256 a9e67510...) were known. Not part of any frozen confirmatory family; every result
is labelled post hoc in the manuscript.

## Question
B2's HQ and C-matched both use 3e-3, the upper edge of the grid {3e-4, 1e-3, 3e-3} from which E1 selected them. The
HQ-at-1e-2 check re-trained only HQ at the next half-decade: HQ (1e-2) - C-matched (3e-3) was -0.0055
[-0.0105, -0.0005]. A reviewer can object that only one arm was given the extra learning rate. This check trains
C-matched at 1e-2 on the same blocks, so that both arms have been run at {3e-3, 1e-2}.

## Runs (36, all new, none retried; a run interrupted by a machine crash is restarted from the beginning)
Same study object as B2 (`dq_b2.STUDY_B2`: salt, schedules, action and minibatch streams, initialisation, the 80
common-random-number evaluation episodes at the same 11 points). Only C-matched's (`bottleneck`) Adam learning rate
differs: 1e-2, blocks 0-35. The B2 code tree is imported unchanged; the runner adds the candidate 'lr1e-2_ent0' to the
study's candidate table in its own process, exactly as the HQ-at-1e-2 runner did.

Pre-check before the launch (no outcome of the new runs is computed): C-matched of block 0 at its stored 3e-3, re-run
for the first 3,000 nominal decisions, must reproduce the stored B2 episodes, updates and evaluations exactly.

## Outcome and analysis (fixed now)
Outcome: B2's AUC from each run's summary (trapezoidal learning-curve area over the nominal grid / budget).
- Z1 (single post-hoc test at 0.05): AUC(HQ, 1e-2; stored, HQ-at-1e-2 check) - AUC(C-matched, 1e-2; new), paired over
  the 36 blocks: mean, 95% paired t interval, two-sided paired t p-value, mean/sd, share of blocks below zero.
- Descriptive (intervals, no tests): AUC(C-matched 1e-2) - AUC(C-matched 3e-3; stored B2); final-checkpoint weighted J
  of C-matched at 1e-2; mean per-update approximate KL of C-matched at 1e-2 and 3e-3 (from the update records).

## Reading rules (fixed now)
- Z1 interval below zero: HQ learned less than C-matched also when both used 1e-2; with B-H1 (both at 3e-3) the
  deficit is present at both learning rates at which both arms were run.
- Z1 not resolved: at 1e-2 the HQ - C-matched difference is not resolved; this is not evidence of equality, and the
  manuscript then states that B-H1 holds at the selected learning rate and is not resolved at 1e-2.
- Z1 interval above zero: HQ at 1e-2 exceeded C-matched at 1e-2; B-H1 is then reported as learning-rate dependent.
Descriptively, if C-matched's own AUC rises from 3e-3 to 1e-2, its selection was also at a grid edge, which is stated.

## Reporting commitment
Whatever the outcome, Z1 and the descriptive values are reported in the manuscript next to the HQ-at-1e-2 check
(Section 4.7, Table 11) with this plan's SHA-256; scripts and records go to the repository.
