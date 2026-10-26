# Post-hoc check: HQ trained from generic circuit angles (B2)

Written 2026-10-24 before any run below existed, after all results of record and of the 2026-10-23 post-hoc extensions
were known, in response to the final adversarial read (reviewer_audit/work/stress_test_2026_10_24.md, objection M5).
Not part of any frozen family; labelled post hoc in the manuscript.

## Question
B2 initialised HQ's 16 circuit angles as 0.1 × N(0, 1). Near θ = 0 the RZ tangent directions vanish, and the trained
angles stayed near their initial values (median mean |θ| 0.071 rad at the B2 final checkpoints). B1G showed that the
local path difference persists at generic angles, but no training run started from them. Does HQ's early-training
deficit relative to C-matched (B-H1) persist when HQ starts from generic circuit angles?

## Runs (36, all new, none retried; a run interrupted by a machine crash is restarted from the beginning)
HQ (`quantum`) on B2 blocks 0-35 with the B2 study object unchanged: salt, initialisation stream, training schedule,
action and minibatch streams, the 80 common-random-number evaluation episodes at the same 11 nominal points (0, 3,000,
…, 30,000 decisions), learning rate 3 × 10⁻³ and entropy coefficient 0 (HQ's E1 selection), 30,000 nominal decisions.
The only change: after the model is built (`p4_train.build_model`, which seeds the global torch generator from the
block's initialisation stream and builds every component), its 16 circuit angles (`actor.weights`) are replaced by
independent draws from U(−π, π) taken from a separate `torch.Generator` seeded with
`numpy.random.SeedSequence([20261024, block]).generate_state(1)[0]`. The global generator is not touched, so the
encoder, GRU, critic, projection and readout are initialised exactly as in the stored B2 HQ run of the same block. Each
new run is therefore paired with the stored HQ run of its block (differs only in the 16 angles) and with the stored
C-matched run (as in B-H1).

Precheck before launch: (a) with the angle replacement disabled, the runner's code path re-runs the stored B2 HQ run of
block 0 for its first 3,000 decisions and reproduces its evaluation, update and episode events exactly; (b) with the
replacement enabled, the model built for block 0 equals the stored B2 HQ block-0 initial checkpoint in every parameter
except `actor.weights`, whose 16 values lie in (−π, π). The launch waits for both.

## Outcome and analysis (fixed now)
Outcome: B2's AUC (learning-curve area over the nominal budget) from each run's summary, as for B-H1.
- G1: AUC(HQ, generic angles; new) − AUC(C-matched; stored), paired over the 36 blocks: mean, 95% paired t interval,
  two-sided paired t p-value, dz.
- G2: AUC(HQ, generic angles; new) − AUC(HQ, small-angle; stored), paired over the 36 blocks: same statistics.
- Holm adjustment over {G1, G2} at familywise 0.05.
- Descriptive: weighted J at checkpoint 0 and at the final checkpoint of both HQ variants and of C-matched; mean |θ| at
  the first and final checkpoints and mean |Δθ| between them; the same blocks' B-H1 for scale.
Reading rules (fixed now):
- G1 interval below zero and Holm-rejected → the early-training deficit relative to C-matched persists when HQ starts
  from generic circuit angles; the small-angle initialisation does not account for it.
- G1 not rejected → not resolved from generic angles (not evidence of equality); the deficit is reported as shown from
  the small-angle initialisation only.
- G1 interval above zero and Holm-rejected → from generic angles HQ learned more than C-matched; B-H1 is specific to
  the small-angle initialisation, and the manuscript's training statements are restated accordingly.
- G2 is reported with its interval and Holm decision as the effect of the initialisation on HQ's learning-curve area.
No time cut-off; the read-out waits for all 36 runs.

## Reporting commitment
Whatever the outcome, G1, G2 and the descriptive values are reported in Section 4.7 next to B-H1 and in limitation (c),
with this plan's SHA-256, and the abstract follows 08_SYNTHESIS_RULES.md Addendum C (written before any run of this
check). Scripts and records go to the repository.
