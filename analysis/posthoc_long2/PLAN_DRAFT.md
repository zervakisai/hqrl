# Post-hoc check E4: best-effort, information-matched, re-tuned PPO for the classical heads (300,000 decisions)

DRAFT for review (not hashed; the lead hashes PLAN.md after review). Written 2026-10-25 before any E4 pilot or main
run existed, after all results of record, B2, LONG (`idea_dq_posthoc_long`) and LONG-HQ were known. Not part of any
frozen family; labelled post hoc in the manuscript. Code: `runner.py`, `e4lib.py`, `rule_reference.py`,
`e4_analysis.py`, `tests/test_e4.py` in this folder; the idea_dq tree (live hash ffe9dfef…) is imported unchanged.
Provenance (not part of the design): the precheck passed (precheck.json) and the development pilot was launched
2026-10-25T08:12:07Z under this draft as it then stood (byte copy: pilot/PLAN_DRAFT_at_pilot_launch_2026-10-25T081207Z.md).
The pilot section below must not change once any pilot result has been read; if it is changed before that, the pilot
is discarded and re-run.

## Question
LONG found that C-strong and C-matched, trained for 300,000 decisions at their E1 configuration (Adam at constant
learning rate 3e-3, entropy coefficient 0), ended below half the tuned rule's J in every block. Two objections remain
(Read 2, problem 2): (i) the learned observation lacks evidence the rule uses (the rule reports only tracks with ≥ 4
detections from ≥ 2 credited distinct viewpoints and stops inspecting a track after 3 inspections; the learned
observation has neither count), and (ii) the configuration was not re-tuned for 300,000 decisions (C-strong collapsed
to a near-deterministic policy in 5 of 6 LONG blocks; median last-update entropy 0.26 nats). Does the systems conclusion
survive a best-effort classical PPO that has the rule's evidence counts and a standard long-run configuration
(entropy bonus, learning rate decayed to 0)?

## What changes relative to LONG (and nothing else)
1. Observation `obs-encoding/4-gnss+ev16/1` (270 = 254 + 16). Features 0-253 are the frozen `obs-encoding/4-gnss`,
   byte for byte. Appended, for each of the 8 track slots of the frozen encoder (unreported tracks, its order):
   `min(credited distinct viewpoints, 4) / 4` and `min(executed inspections, 3) / 3` of that track (0 for an empty
   slot). Both counts are kept by the rule's own code (`executive.EvidencePolicy._update_evidence` on every `act`,
   `EvidencePolicy._evidence_feedback` on every `feedback`, as `GnssRulePolicy` does), from public quantities only
   (public detection counts, the vehicle's local position, executed-inspection feedback). `RLMissionPolicy` inherits
   this bookkeeping but never updated it; the subclass `InfoRLMissionPolicy` only updates it and appends it. Action
   space, masks, RETURN rule and rejection handling are unchanged.
   Not added (stated as a limitation): the rule's per-view novelty filter, sticky abandonment and the per-report
   fix-attempt counter; and nothing changes the interface asymmetries (free RELOCALIZE, masked voluntary RETURN).
2. Model. LONG's model is built exactly as in LONG (same initialisation stream; bitwise equal to LONG's stored initial
   checkpoint of the same block and head), then its first linear layer receives 16 zero input columns. At
   initialisation the E4 policy therefore is LONG's policy (logits equal to < 1e-12; 0 on real observations), and the
   first rollout (the point-0 evaluation and the episodes of update 1) is identical to LONG's (precheck c).
3. PPO. Entropy coefficient e_h > 0, chosen per head by the development pilot below from {0.003, 0.01}. Adam learning
   rate decays linearly from lr0 = 3e-3 (each head's E1 selection) to 0 over the 300,000-decision budget:
   lr_u = lr0 × (1 − D_u / 300,000), D_u = policy decisions consumed by the updates before update u, set in every
   parameter group before update u and logged with it (update 1 runs at lr0, as in LONG).
lr0 is not piloted: E1's 30,000-decision tuning selected 3e-3 over 1e-3 for both classical heads (C-strong AUC 0.0380
vs 0.0373, C-matched 0.0164 vs 0.0133), linear decay already halves the mean step, and a 2 × 2 × 2 lr0 × entropy pilot
does not fit the compute cap of this week.
Everything else is LONG's: the B2 study object (salt (20260923, 2), initialisation, action and minibatch streams),
dq_e2_training with 10,000 seeds per block on blocks 4-9 (so each E4 run meets exactly the training cells and
disturbance seeds of LONG's run of the same block and head, in the same order), 512 transitions per update, clip 0.2,
value coefficient 0.5, gradient-norm clip 0.5 per group, 4 epochs, minibatches of ≤ 256 transitions, segments of ≤ 32,
advantage std floor 0.05, γ = 1 with the time-aware λ (0.95, 60 s), float64, one thread, and the evaluation: sampled
read-out on the 80 common-random-number episodes of dq_e2_evaluation (8 core cells × realisations 0-9) at LONG's 11
points (0, 30,000, …, 300,000; each at the first update boundary at or after it), with the evaluation action stream
keyed by the point index as in LONG, so every E4 evaluation is paired with LONG's at the same point.
The two changes (information and re-tuning) are confounded by design: E4 is one best-effort arm, and n = 6 blocks
cannot separate their contributions. HQ is not run (the question concerns the classical baselines).

## Development pilot (development seeds only; this section is fixed before the pilot starts)
Runs (8): {C-strong, C-matched} × entropy {0.003, 0.01} × development blocks {6, 7}, 100,000 decisions each, under
the main configuration (observation and model as above, lr0 3e-3 decaying over the 300,000 horizon: the pilot is the
first third of the main schedule). Study `e4_dev`, salt (20260923, 41); training seeds dq_development.start +
10,000 × block + episode, i.e. [start + 60,000, start + 80,000); development evaluation suite: the 8 core cells ×
dq_development realisations 600-609 (seeds start + 1,000 × seed_cell_index + r, all below start + 7,000), sampled
read-out at 0, 50,000, 80,000 and 100,000 decisions. No dq_e2_training or dq_e2_evaluation seed is used, and no
pilot step reads anything from dq_e2_evaluation.
Selection rule, per head (applied by `runner.py pilot-select`, written to pilot/selection.json before any main run):
- Score S(e) = mean weighted development J at 80,000 and 100,000 decisions over both development blocks.
- Eligible: completed in both blocks (no algorithmic failure) and last-update mean policy entropy ≥ 0.30 nats in both
  blocks (LONG's collapse level). If no value is eligible, the entropy guard is lifted (failures stay ineligible); if
  both values failed, e = 0.01.
- One eligible value: it is chosen. Two: the larger S, except that |S(0.01) − S(0.003)| < 0.0056 J (0.1
  report-equivalent per episode, dq_b2.TOST_DELTA) chooses 0.003 (the smaller entropy).
The pilot's runs and all S values are reported whatever is selected; they are development material, not outcomes.

## Main runs (12, all new, none retried)
C-strong (`mlp`) and C-matched (`bottleneck`) × B2 blocks 4-9, 300,000 nominal decisions, entropy e_h from the pilot,
lr0 3e-3 decaying linearly to 0. A run killed by a machine crash is moved to interrupted/ and restarted from the
beginning (identical seeds); a failure is recorded and never retried.
Precheck before any pilot or main run (`runner.py precheck`, precheck.json; the launch refuses unless it passed for the
exact runner.py and e4lib.py being launched):
(a) with every change switched off, the runner's code path (all wrappers in pass-through) re-runs stored LONG block 4
    of both heads for 3,000 decisions and reproduces its point-0 evaluation, update, episode and arm_start events
    exactly;
(b) LONG's initial model is rebuilt bitwise; the widened model equals it in every tensor except the 16 zero columns,
    its logits/values equal the base model's to < 1e-12, and the global torch generator is left where the base build
    leaves it (plus the unit tests: the encoder on real observations reproduces features 0-253 of the frozen encoder at
    every decision and the 16 features equal an independent re-implementation of the rule's bookkeeping);
(c) a 3,000-decision run of each head with every change on (entropy 0.01) completes; its point-0 evaluation and the
    episodes of update 1 equal LONG's; its schedule is LONG's; every update logs lr_u exactly as defined and the
    entropy coefficient; the evidence features are populated; the final checkpoint holds the 270-input model and the
    last lr.

## Paired rule reference (computed before this plan was hashed; not an outcome of E4)
The tuned rule (`uncertainty_fix`, σ 1.5 m, `e1_baselines.p3_rule()`, run by `e1_baselines.run_policy_episode`) on
exactly the 80 dq_e2_evaluation episodes of every B2, LONG, LONG-HQ and E4 evaluation (`rule_reference.py`,
rule_reference.json, SHA-256 5b8b8d71a445…): J_rule = its cell-weighted J (p4_train.aggregate_evaluation) = 0.21133,
J_rule / 2 = 0.10566. The rule draws no action stream, so this single set of 80 rows pairs with every evaluation point
of every learned run (same cell, seed and CRN group). Checks: (cell_id, seed) of every
stored B2/LONG/LONG-HQ evaluation event equal the reference's, in order; two episodes re-flown identically; the same
code path reproduces two stored E1 rule rows exactly (J, crn_sha256, decisions).

## Outcome and analysis (fixed now; `e4_analysis.py`)
Outcome: weighted J (sampled read-out, B2/LONG aggregation) at each checkpoint; primary at 300,000 decisions.
Primary, per head h ∈ {C-strong, C-matched}, n = 6 blocks, paired by block with the stored LONG run:
- P1_h = J300k(E4) − J300k(LONG): mean, 95% paired t interval, two-sided paired t p-value, dz; Holm over
  {P1_C-strong, P1_C-matched} at familywise 0.05. No equivalence test.
- P2_h = share of LONG's gap to the paired rule reference closed: Σ_b (J_E4 − J_LONG) / Σ_b (J_rule − J_LONG), with a
  95% percentile bootstrap interval over blocks (20,000 resamples, seed 20261025); per-block shares descriptively.
- P3 (half-rule check): for each head, the blocks with J300k(E4) ≥ J_rule / 2, the maximum block J300k and the mean;
  LONG's threshold 0.103 is reported alongside.
Descriptive (no test, not in the Holm family): mean J at every checkpoint (E4 and LONG) and E4 − LONG at each; AUC
over 0-300,000 (LONG's trapezoid) E4 − LONG; J at 30,000 E4 − LONG; slope over the last three checkpoints; J300k(E4) −
J_rule with its t interval and J300k(E4) / J_rule; components at 300,000 (timely-credited reports, false reports,
correction attempts, decisions per episode) for E4, LONG and the rule; collapse indicators (blocks with ≤ 0.012
timely-credited reports per episode at 300,000, LONG's forensic cut; blocks with last-update entropy < 0.30 nats) for
E4 and LONG; the highest single-block J at any checkpoint; the evidence-feature diagnostics (decisions with a nonzero
viewpoint / inspection feature); the pilot table and selection.

Reading rules (fixed now):
- P1, per head: Holm-rejected with the interval above 0 → "the best-effort configuration (the rule's evidence counts,
  entropy bonus, decayed learning rate) ended higher than LONG at 300,000 decisions", reported with P2; Holm-rejected
  below 0 → "it ended lower than LONG"; otherwise → "not resolved at n = 6 (not evidence of equality)".
- Systems conclusion (P3): every block of both heads below J_rule / 2 → "the systems conclusion holds for
  best-effort, information-matched, re-tuned classical PPO at 300,000 decisions" (scope: two classical heads, one
  scene, these 16 evidence features, this budget). Some block ≥ J_rule / 2 but both head means below → the conclusion is
  restated as "no learned controller approached the rule on average", naming the block(s) that reached half the rule,
  and the application conclusion is called budget- and configuration-specific. A head's mean ≥ J_rule / 2 → the
  systems conclusion is withdrawn for best-effort classical PPO at 300,000 decisions and restated as "under the E1/B2
  configuration the learned controllers stayed far below the rule; with the rule's evidence counts and re-tuning,
  C-x reached [share] of the rule". In every case, a head whose 95% interval for J300k(E4) − J_rule reaches 0 is
  reported as "not shown to be below the rule".
- Collapse is not a failure (P4 protocol section 6): a collapsed run is completed and enters as it is. A run with an
  algorithmic failure (FloatingPointError or ScheduleExhausted) has no 300,000 value: P1 and P2 use the blocks where it
  has one (n stated), a sensitivity analysis carries the failed run's last evaluated J forward, and the failure count
  qualifies the reading; with 3 or more failures in a head, P1 is not computed for it and the head is reported as
  "failed in k of 6 blocks".
- Stopping rule: the read-out waits for all 12 runs. If any run has not finished by 2026-10-30T12:00Z, the read-out
  (`e4_analysis.py --deadline-readout`) is at the largest checkpoint every unfailed E4 run has reached, paired with
  LONG at the same checkpoint and stated as such; no run is extended, shortened or restarted for this reason.

## Reporting commitment
Whatever the outcome, P1-P3, the descriptive values above, the pilot table, the precheck and the rule reference are
reported in Section 4.2.1 (next to LONG) and in limitation (h), with this plan's SHA-256, the configuration caveat
(one best-effort configuration, information and re-tuning confounded, n = 6) and the evidence features not added.
Scripts, records and the pilot go to the repository.
