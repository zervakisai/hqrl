# Execution log (not part of the plan; PLAN.md is unchanged)
- 2026-10-22T13:06:46Z precheck attempt 1 refused at the B1 namespace gate before any episode: the live idea_dq tree
  (ffe9dfef) is not the B1 tree of record (a66f8979). runner.py now imports the B1 tree extracted unchanged from
  output/study/gnss_qrl_v1/dq/source_lock_a66f8979/source_lock.tar.gz (code_version.py reproduces a66f8979); only
  the IDEA_DQ path changed. No arm value existed at that point.
3b6cf9c6ee18e75e0e843dcf7a2d8a98a0aff64f295f18144b9f4c9a16f7251c  runner.py
- 2026-10-23 02:41 EEST: all 72 unit records complete. The first final run of b1x_analysis.py (02:45) showed that its
  generated manuscript text said the units "reproduced the stored gains" although five units (shipped 13, 14, 15;
  heldout 13, 22) failed the reproduction check of PLAN.md and were excluded, as the plan requires. Only the
  generated wording was then changed: it now states the exclusions and their cause (HQ-Fisher at kappa = 0.05 only,
  step scale up to 6e8; at kappa <= 0.01 all 72 units reproduce within 3e-9). It also adds descriptive values that
  the plan lists or that follow arithmetically from the planned contrasts: the m = 4 ladder step, the median cosine
  and sub-step count of HQ-Fisher-16 and TAN-16, and the share X1 / (X1 - X2). It also adds a sensitivity analysis
  that is not in the plan: X1 and X2 with the excluded units reinstated (same reading). The exclusion rule, X1, X2,
  Holm, the reading rules and the verdict ('partial') are unchanged.
  (also: in the damping sentence, 'in the N retained two-layer units' and the delta=1e-6 note were added at the same time)
- 2026-10-23 between 02:50 and 03:00 EEST, after an independent read of the generated text: wording only. "16 sub-steps" was wrong
  (m sets the sub-step KL, kappa/m^2; the count is a median of 32): the arms are now described as iterated sub-steps
  with the count reported. The text is split into three paragraphs and now gives the three-layer p-values, the Holm
  decisions, the sub-step range, the m = 4 comparator, TAN-16 - TAN, a percentile bootstrap interval for the
  descriptive share (20,000 unit resamples, seed 20261023) and the shares at kappa = 0.001. It scopes the damping
  reading to kappa = 0.01 (as in the plan) and lists unresolved damping cells outside it. It also adds the caveat
  that the experiment does not separate step size from the curvature of the circuit's reachable set. No test,
  exclusion, reading rule or verdict changed.
