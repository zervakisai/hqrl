# Post-hoc check: ten times the training budget for the classical heads (undertraining assessment)

Written 2026-10-23 before any run below existed, after all results of record were known and after two independent
reviews (reviewer_audit/work/review_B.md CR2, review_C.md C1) objected that the learned controllers were under-trained
(30,000 decisions, about 570 episodes, curves still rising). Not part of any frozen family; labelled post hoc.

## Question
Does the systems conclusion (no learned controller approached the tuned rule) depend on the 30,000-decision budget?
Does the learning of the classical heads continue beyond it?

## Runs (12, all new, none retried; a run killed by a machine crash is restarted from the beginning)
C-strong (`mlp`) and C-matched (`bottleneck`), each at its E1-selected configuration (learning rate 3e-3, entropy 0),
on B2 blocks 4-9, for 300,000 nominal decisions with checkpoints every 30,000 (11 points). The B2 study object is
reused (salt, initialisation, action and minibatch streams, the 80 common-random-number evaluation episodes at every
checkpoint). Only the training-schedule stride changes: B2 gives each block 1,000 training-episode seeds
(seed = start + 1,000 block + episode), which a 300,000-decision run would exhaust, so these runs use
seed = start + 10,000 block + episode in the same namespace. For blocks 4-9 this range, [start + 40,000,
start + 100,000), contains no seed used by B2 or by the earlier post-hoc runs (all below start + 36,000). The runs
therefore share their initialisation with the stored B2 runs of the same block but meet different training episodes
from the first one. HQ is not run (exact circuit simulation makes it 2-3 times slower; the budget did not allow it).

## Outcome and analysis (fixed now; descriptive, n = 6 blocks per arm)
- Weighted J at each checkpoint (B2's aggregation) and its mean over blocks with a 95% t interval.
- L1: final J (300k) minus J at 30k, per arm, paired over blocks: does learning continue?
- L2: final J (300k) against the tuned rule's weighted J on the E1 evaluation episodes (0.206) and on the Freeze A
  core cells: the fraction of the 30k gap to the rule that is closed, per arm (mean over blocks, bootstrap interval).
- Learning-curve slope over the last three checkpoints (240k-300k): is the curve still rising?
- Components at the final checkpoint: timely-credited reports, false reports, correction attempts per episode.
- For reference, the same blocks' stored B2 values at 30k.
Reading rules (fixed now): if every arm's final J stays below half the rule's J, the systems conclusion is reported
as holding at ten times the budget for these classical heads and configurations; if an arm's final J reaches more than
half the rule's, the application conclusion is restated as budget-specific. In both cases the result is reported.
Stopping rule: if the runs have not all completed by 2026-10-24T03:00Z, the read-out is at the largest checkpoint that
every run has reached, stated as such; no run is extended or shortened after its start.

## Reporting commitment
Reported in Section 4.2.1 or 4.6 and in limitation (h) with this plan's SHA-256, whatever the outcome; scripts and
records go to the repository.
