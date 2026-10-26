# Post-hoc check: HQ under the same extended training as the classical heads (companion to LONG)

Written 2026-10-23 before any run below existed, after all results of record were known, after the plan of the
classical long-training check (idea_dq_posthoc_long/PLAN.md, SHA-256 58471290…) was hashed and before any of its runs
started. Not part of any frozen family; labelled post hoc.

## Question
B2 found that HQ learned less than C-matched over 30,000 decisions. Does that deficit persist, shrink or reverse when
both are trained for longer at equal environment-decision budgets?

## Runs (6, all new, none retried; a run killed by a machine crash is restarted from the beginning)
HQ (`quantum`) at its E1-selected configuration (learning rate 3e-3, entropy 0, the same treatment as C-matched and
C-strong in LONG), on B2 blocks 4-9, 300,000 nominal decisions, checkpoints every 30,000, the B2 study object and the
widened stride (10,000 seeds per block) exactly as in LONG. Within a block, HQ and LONG's C-matched and C-strong runs
therefore share the initialisation seed, the training schedule (cells and environment seeds), the action and minibatch
streams and the 80 evaluation episodes at every checkpoint: the comparison is paired at equal decision budgets.
Wall-clock and compute are not equal (exact simulation makes HQ slower); both are reported.

## Outcome and analysis (fixed now; n = 6 blocks; descriptive interval estimates)
Let T be the largest checkpoint that every HQ run and every paired C-matched run of LONG has reached by the cut-off
below (T = 300,000 if all complete).
- H1: weighted J of HQ minus C-matched at T, paired over blocks: mean, 95% t interval, two-sided p (reported, n = 6).
- H2: the change of that gap between 30,000 decisions and T, paired over blocks (does the deficit persist with
  training?).
- Descriptive: AUC of each arm over [0, T] divided by T; learning curves; HQ − C-strong at T; wall time per 30,000
  decisions; circuit-angle displacement between the first and last checkpoints.
Reading rules (fixed now): H1 interval below zero → the HQ deficit persists at T decisions; H1 interval containing zero
→ not resolved at n = 6 (not evidence of equality); H1 interval above zero → HQ exceeds C-matched at T. H2 is reported
as the direction and size of the change, with its interval. The manuscript states T explicitly; a T below 300,000 is
reported as a truncated comparison, never as a completed 300,000-decision experiment.
Stopping rule: compute stops at 2026-10-24T03:00Z (06:00 local); records and checkpoints written by then are kept; no
run is extended or shortened otherwise. Runs execute at the lowest CPU priority so that the other extensions are not
delayed.

## Reporting commitment
Reported in Section 4.7 and limitation (h) with this plan's SHA-256, whatever the outcome, including if T < 300,000.
