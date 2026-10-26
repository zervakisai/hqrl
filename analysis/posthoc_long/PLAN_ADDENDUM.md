# Addendum to PLAN.md (written and hashed before any run of this check started)

2026-10-23, about 16:35 local: the authors removed the time limit ("let it take as long as needed, as long as it is
scientifically rigorous"). The stopping rule of PLAN.md is replaced: every run continues to its planned 300,000
decisions; there is no time cut-off and no read-out at a truncated checkpoint. A run interrupted by a machine crash is
restarted from the beginning (PLAN.md). Everything else in PLAN.md is unchanged. The runs execute at the lowest CPU
priority (nice 19), so that the other extensions finish first; this changes no seed, configuration or outcome.
