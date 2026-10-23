# Post-hoc matched-learning-rate check for B2 (HQ vs exact S625 twin)

Written 2026-10-22, before any run below existed, after the B2 results of record were known.
This is NOT part of any frozen confirmatory family. It answers one anticipated reviewer question (EAAI audit item 59):
B2 compared HQ at learning rate 3e-3 with its exact twin at 1e-3, so the HQ-twin difference (B-H2) may be a
learning-rate effect.

## Runs (72, all new, none retried)
Same study object as B2 (`dq_b2.STUDY_B2`: salt, training schedule, action and minibatch streams, model
initialisation, 80 common-random-number evaluation episodes at the same 11 nominal points). Only the Adam learning
rate of one arm differs from B2.
- L1 arm: S625-twin at 3e-3, blocks 0-35 -> paired with B2's stored HQ runs (3e-3), same blocks.
- L2 arm: HQ at 1e-3, blocks 0-35 -> paired with B2's stored S625-twin runs (1e-3), same blocks.
The B2 code tree in `idea_dq` is imported unchanged; the runner here only sets the learning rate.

## Determinism pre-check (before the 72 runs)
Re-run B2's stored S625-twin (1e-3) of block 0 for the first 3,000 nominal decisions. Its evaluation at point 0 and
at point 3,000 and its first update metrics must equal the stored B2 records exactly. If not, the pairing with
stored runs is not exact and this will be stated.

## Outcome and analysis (fixed now)
Outcome: B2's AUC (trapezoidal learning-curve area over the nominal grid / 30,000), taken from each run's summary.
- L1: AUC(HQ, 3e-3; stored) - AUC(twin, 3e-3; new), paired over 36 blocks.
- L2: AUC(HQ, 1e-3; new) - AUC(twin, 1e-3; stored), paired over 36 blocks.
Report for each: mean, 95% paired t interval, two-sided paired t p-value, standardised effect (mean/sd), and the
Holm decision over the two (familywise 0.05), labelled post hoc. Descriptive only: within-arm learning-rate
effects AUC(HQ 3e-3) - AUC(HQ 1e-3) and AUC(twin 3e-3) - AUC(twin 1e-3); final-checkpoint weighted J.
Any failed run is reported as failed and the block dropped from that contrast only.

## Reporting commitment
Whatever the direction or size, L1 and L2 are reported in the manuscript as a post-hoc sensitivity check,
next to B-H2, with this plan's SHA-256.
