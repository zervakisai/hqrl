# Post-hoc check: HQ trained with frozen circuit angles (B2)

Written 2026-10-25 before any run below existed. At that point all results of record, all post-hoc extensions up to
B2G, and two independent cold reads of the manuscript were known (reviewer_audit/work/cold_reads_2026_10_25.md).
This check is not part of any frozen family and is labelled post hoc in the manuscript.

## Question
B1 diagnoses updates of the 16 circuit angles only. In B2 the angles barely moved (median mean |Δθ| 0.052 rad at
3 × 10⁻³), and in the stress study resetting the trained angles changed weighted J by −0.0008 [−0.0021, +0.0004]. Two
questions follow.
1. Does training the angles contribute to HQ's early learning at all?
2. Does HQ's early-training deficit relative to C-matched (B-H1) remain when the angles are not trained?

With its angles frozen, the hybrid head is an exactly classical model: a fixed trigonometric feature map, which the
exact twin represents, followed by the trainable classical projection and readout. A read-only check on synthetic
batches found its gradients equal to those of the twin with its Fourier layer frozen to 1.3 × 10⁻¹⁵.

## Runs (36, all new, none retried; a run interrupted by a machine crash is restarted from the beginning)
HQ (`quantum`) runs on B2 blocks 0–35, with the B2 study object unchanged:
- salt, initialisation stream, training schedule, action and minibatch streams;
- the 80 common-random-number evaluation episodes at the same 11 nominal points (0, 3,000, …, 30,000 decisions);
- learning rate 3 × 10⁻³ and entropy coefficient 0 (HQ's E1 selection);
- 30,000 nominal decisions.

The only change: after the model is built (`p4_train.build_model`), `actor.weights` (the 16 circuit angles) is set to
`requires_grad = False`. Adam is created afterwards inside `train_arm` and skips parameters without a gradient, and the
gradient-norm clip ignores them, so the angles keep their initial values throughout. The four final-layer RZ angles
already have zero gradient, so freezing 16 is the same as freezing the 12 effective angles.

Each new run is paired with two stored B2 runs of its block:
- the stored HQ run: same initialisation, same angles, same streams; it differs only in whether the angles are
  trained;
- the stored C-matched run (as in B-H1).

Precheck before launch:
- (a) With freezing disabled, the runner's code path re-runs the stored B2 HQ run of block 0 for its first 3,000
  decisions and reproduces its evaluation, update and episode events exactly.
- (b) With freezing enabled, the model built for block 0 equals the stored B2 HQ block-0 initial checkpoint in every
  tensor.
- (c) A 3,000-decision run of block 0 with freezing enabled keeps `actor.weights` equal (torch.equal) to the initial
  angles in every saved checkpoint.

The launch waits for all three.

## Outcome and analysis (fixed now)
The outcome is B2's AUC (learning-curve area over the nominal budget) from each run's summary, as for B-H1.

Contrasts, each paired over the 36 blocks (mean, 95% paired t interval, two-sided paired t p-value, dz):
- F1: AUC(HQ, frozen angles; new) − AUC(HQ, trained angles; stored B2).
- F2: AUC(HQ, frozen angles; new) − AUC(C-matched; stored B2).

Tests:
- Holm adjustment over the two t tests {F1, F2} at familywise 0.05.
- Equivalence for F1: the TOST procedure of B2 (margin ±0.0056 AUC, i.e. ±0.1 RE for Nref = 18; 90% interval inside
  the margin).

Descriptive values:
- weighted J at checkpoint 0 and at the final checkpoint for HQ (frozen), HQ (trained) and C-matched;
- the share of blocks with F1 < 0 and with F2 < 0;
- max |θ_final − θ_initial| over all frozen runs (must be 0).

Reading rules (fixed now):
- F1, three cases:
  - TOST establishes equivalence → training the circuit angles made no difference to early learning within ±0.1 RE:
    what HQ learned in B2 is carried by its classical layers around a fixed circuit feature map.
  - Holm-rejected with the interval below zero → training the angles helped HQ's early learning.
  - Holm-rejected with the interval above zero → HQ learned more with its angles frozen: training the angles hurt early
    learning. This is consistent with B1's shortfall of circuit-realised steps but does not establish it as the cause.
  - Otherwise → not resolved (not evidence of equality).
- F2, three cases:
  - Holm-rejected with the interval below zero → the deficit relative to C-matched remains when the angles are not
    trained, so it does not come from optimising the angles. The B1 diagnosis of angle updates then cannot explain it.
  - Holm-rejected with the interval above zero → with frozen angles HQ learned more than C-matched; B-H1 then depends on
    angle training.
  - Otherwise → not resolved.

No time cut-off: the read-out waits for all 36 runs.

## Reporting commitment
Whatever the outcome, F1 (with its TOST), F2 and the descriptive values are reported in Section 4.7, and a sentence
relating them to B1 goes in the B1/B2 paragraph of 4.7 and in 5.1, with this plan's SHA-256. The manuscript text for
each case is written in tools/results_edits_week.py before the analysis is run. Scripts and records go to the
repository.
