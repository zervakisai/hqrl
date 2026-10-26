# Post-hoc descriptive re-scoring: false-report weight 0.5 and 2.0 (the P3 declared sensitivities)

Written 2026-10-23 before any re-scored value was computed, after all results of record were known. Descriptive only:
no test, no reading rule that changes a conclusion of record. No new episode is run.

## Why
The P3 protocol declared false-report weights of 0.5 and 2.0 as descriptive sensitivities of Eq. (1) (`p3_freeze.py`,
CONVENTIONS: "0.5 and 2.0 are the declared descriptive sensitivities"; "choosing among them after seeing results is
forbidden"). Its final stage was never run because the planner gate failed, so the manuscript reports only weight 1.
This re-scoring reports both declared weights beside weight 1; weight 1 remains the score of record.

## Score
For every stored episode: J_λ = (1[return] · Σ timely credit − λ · FP_spatial) / 18, from the episode's stored
tp_timely, fp_spatial and return_ok. Check: J_1 must equal the stored J of every episode (to 1e-12), or the script stops.

## Outputs (all descriptive: means and individual 95% t intervals over the statistical unit of each study)
1. Freeze A stress evaluation (48 E1 finals, the tuned rule, the uniformly random policy; 16 cells × 20 realisations):
   cell-weighted J_λ per drift level (weights 0.4 target-present / 0.1 target-absent per budget, as in Freeze A), per
   family (mean over 12 blocks with a 95% t interval), the rule and random; the rule minus the best learned family.
2. B2 (108 stored runs): AUC_λ from the evaluation episodes on the nominal grid; B-H1 and B-H2 at λ = 0.5, 1, 2 (paired
   over 36 blocks; mean, 95% t interval, p reported as descriptive).
Reporting: a sentence in Section 4.8 and one in Section 4.6 or Table 11, whatever the values, labelled post hoc and
descriptive; limitation (b) states that the other declared final-only sensitivities were not run.
