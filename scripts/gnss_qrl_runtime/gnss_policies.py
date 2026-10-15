"""Rule and planner comparators for task flood-sar-village-gnss/1 (IMPLEMENTATION_PLAN_EL.md WP3).

None of these sees evaluator truth. Every one of them decides from the PUBLIC observation only: local tracks,
the filter's own covariance summary and fix age, public anchor geometry with its availability and route cost,
remaining time and the return reserve. The short-horizon planner is a planner, never an "oracle": it has no
privileged information and its score is not an upper bound on anything.

The comparators, and why each one is here
    never_fix              the floor: search and report, never relocalise. Shows what drift alone costs.
    always_fix_before_report   the trivial rule that the 18 Sep pricing showed solves the easy variant. It is
                           the rule to beat, not a strawman: `STATISTICAL_PROTOCOL_EN.md` section 5.3 requires
                           the reference to be the strongest tuned eligible simple rule.
    periodic_fix           relocalise whenever the last accepted fix is older than `stale_after_s`.
    uncertainty_fix        relocalise whenever the published RMS error exceeds `sigma_threshold_m`. This is the
                           belief-driven rule, and the one a learned policy has to beat to be worth anything.
    end_batch_at_home      search, hold every candidate, then RELOCALIZE(home), submit the whole batch, RETURN.
                           Mandatory: the model contract permits delayed batch submission, so a comparator has
                           to represent it rather than the study pretending it is not a strategy.
    short_horizon_planner  a one-step-lookahead utility rule over the same public inputs: it compares the
                           expected timely credit of reporting now against reporting after the cheapest
                           reachable fix, and charges the fix its full public route and dwell estimate.

All of them share the executive's own feasibility rules: an action they propose that the executive rejects is
retried in the next decision, exactly as for every other policy.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from aerial_qrl.mission.contracts import Action, ActionType                     # noqa: E402
from aerial_qrl.mission.executive import CoverageBaselinePolicy                 # noqa: E402
from aerial_qrl.training.encoding import TOLERANCE_SCALE_M                      # noqa: E402

#: Every comparator's name, in the order the runner reports them.
COMPARATORS = ('never_fix', 'always_fix_before_report', 'periodic_fix', 'uncertainty_fix', 'end_batch_at_home',
               'short_horizon_planner')

#: The behaviour of these policies, versioned. review 10 found that `end_batch_at_home` was not the
#: strategy the protocol preregistered - it paid a fix before nearly every report instead of one home fix for
#: the whole batch - which made a mandatory comparator weaker than declared. A candidate hash names a name and
#: its parameters; it says nothing about what the code does with them. This constant, and the source hash the
#: comparator protocol records beside it, is what binds a frozen candidate to a behaviour. Bump it whenever a
#: comparator's decisions change, and old runs will no longer be comparable with new ones by construction.
#:
#: /2  the end-batch phase machine replaces the rule that fixed before nearly every report (review 10).
#: /3  the holder asks for outstanding inspection work with the report branch suppressed. The pilot executor
#:     smoke caught /2 releasing a batch of ONE on the real bank, because `_evidence_action` returns a REPORT
#:     as soon as any single track ripens and that masked every remaining inspection.
POLICY_BEHAVIOUR_VERSION = 'gnss-comparator-policies/3'

#: The end-batch phases, in order. Declared here because the frozen protocol quotes them.
BATCH_SEARCHING, BATCH_FIXING, BATCH_RELEASING = 'searching', 'fixing', 'releasing'
#: The single frozen fallback when the one home fix cannot happen or does not succeed: release the batch
#: unfixed and return. The alternative - retrying the anchor - is what the livelock guard exists to prevent,
#: and a strategy that can spend the whole budget retrying is not the declared strategy.
BATCH_FALLBACK_RULE = 'release_unfixed_and_return/1'


class GnssRulePolicy(CoverageBaselinePolicy):
    """Coverage search with a declared relocalisation rule on top.

    Everything except the relocalisation decision is the validated coverage baseline: the same search plan, the
    same inspection rule, and the same evidence rule deciding WHICH track is ready to report and WHEN. A
    comparator here therefore differs from the baseline in exactly one respect - whether it pays for a fix
    before submitting - which is what makes the comparison about relocalisation rather than about search.

    Why a fix is only ever considered just before a report: only a submitted coordinate is georeferenced, so a
    fix taken early and followed by ten minutes of searching has decayed by the time it would have mattered.
    `end_batch_at_home` overrides this structure on purpose, because delaying the whole batch is the other
    strategy the model contract admits.

    Livelock guard, and why it is public: a failed fix does not reset `time_since_fix_s`, so a rule phrased as
    "fix before reporting" would retry forever at an unavailable anchor and never report. Each policy therefore
    spends at most `max_fix_attempts_per_report` attempts between consecutive reports, counted from the public
    `fixes_attempted`. It is a declared rule parameter, not a tuned threshold.
    """
    name = 'gnss_rule'

    def __init__(self, briefing, obstacle_map, config, reserve, home_anchor_id='A_home',
                 max_fix_attempts_per_report=1):
        super().__init__(briefing, obstacle_map, config, reserve)
        self.home_anchor_id = home_anchor_id
        self.max_fix_attempts_per_report = int(max_fix_attempts_per_report)
        self._fixes_at_last_report = 0
        self._seen_fixes_attempted = 0
        self.parameters = {'max_fix_attempts_per_report': self.max_fix_attempts_per_report}

    # -- public helpers, all reading the observation only ---------------------------------------------
    @staticmethod
    def _anchors(observation, available_only=False):
        return [a for a in observation.anchors if a.reachable and (a.available or not available_only)]

    def _cheapest_anchor(self, observation, available_only=True):
        anchors = self._anchors(observation, available_only)
        if not anchors and available_only:
            anchors = self._anchors(observation, False)
        return min(anchors, key=lambda a: (a.estimated_duration_s, a.anchor_id)) if anchors else None

    @staticmethod
    def _relocalize(anchor):
        return Action(ActionType.RELOCALIZE, anchor_id=anchor.anchor_id)

    def _search_or_return(self):
        waypoint = self._pending()
        return Action(ActionType.RETURN) if waypoint is None else Action(ActionType.SEARCH_SECTOR, sector_id=waypoint.sector_id)

    def _fix_budget_left(self, observation):
        return observation.georeference.fixes_attempted - self._fixes_at_last_report < self.max_fix_attempts_per_report

    def should_fix(self, observation):
        """Rule hook: True when this policy wants a fix before the report it is about to submit."""
        return False

    def _require_gnss(self, observation):
        if observation.georeference is None:
            raise ValueError('this policy requires a flood-sar-village-gnss/1 observation')
        self._seen_fixes_attempted = observation.georeference.fixes_attempted

    def act(self, observation):
        self._require_gnss(observation)
        self._update_evidence(observation)
        base = self._evidence_action(observation)          # the baseline's REPORT / INSPECT_VIEW / None
        if base is not None and base.action_type == ActionType.REPORT:
            if self._fix_budget_left(observation) and self.should_fix(observation):
                anchor = self._cheapest_anchor(observation)
                if anchor is not None:
                    return self._relocalize(anchor)
            return base
        if base is not None:
            return base                                    # inspection, unchanged from the baseline
        return self._search_or_return()

    def feedback(self, action, executed, reason=None, goal=None):
        if action.action_type == ActionType.REPORT and executed:
            self._fixes_at_last_report = self._seen_fixes_attempted
        super().feedback(action, executed, reason, goal)


class NeverFix(GnssRulePolicy):
    name = 'never_fix'


class AlwaysFixBeforeReport(GnssRulePolicy):
    """Relocalise immediately before every report: the R2 rule the 18 Sep pricing found hard to beat.

    Two guards keep it a fair reference rather than a strawman or a livelock. It skips the fix when the
    published error is already numerically negligible (with no drift a fix can only cost time), and it takes at
    most one attempt per report, so an unavailable anchor costs one visit rather than the whole budget. Right
    after an accepted fix `time_since_fix_s` is zero, which is the signal to submit.
    """
    name = 'always_fix_before_report'

    def should_fix(self, observation):
        geo = observation.georeference
        return geo.time_since_fix_s > 0. and geo.rms_horizontal_error_m > 1e-9


class PeriodicFix(GnssRulePolicy):
    name = 'periodic_fix'

    def __init__(self, *args, stale_after_s=300., **kwargs):
        super().__init__(*args, **kwargs)
        self.stale_after_s = float(stale_after_s)
        self.parameters = dict(self.parameters, stale_after_s=self.stale_after_s)

    def should_fix(self, observation):
        return observation.georeference.time_since_fix_s >= self.stale_after_s


class UncertaintyFix(GnssRulePolicy):
    """Relocalise when the filter's own expected error approaches the reporting tolerance."""
    name = 'uncertainty_fix'

    def __init__(self, *args, sigma_threshold_m=0.5 * TOLERANCE_SCALE_M, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_threshold_m = float(sigma_threshold_m)
        self.parameters = dict(self.parameters, sigma_threshold_m=self.sigma_threshold_m)

    def should_fix(self, observation):
        return observation.georeference.rms_horizontal_error_m >= self.sigma_threshold_m


class EndBatchAtHome(GnssRulePolicy):
    """Search, hold every candidate, pay ONE home fix, release the whole batch, return. A phase machine.

    This is the comparator the model contract singles out: because v1 has one common bias rather than an
    independent error history per observation, delaying the whole batch until after a late fix is a real
    strategy. It is mandatory, and `STATISTICAL_PROTOCOL_EN.md` section 5.3 names it, so it has to be the
    strategy that was preregistered rather than something weaker wearing its name.

    The defect this replaces (review 10)
        The previous version re-decided the fix question on every decision. After each 1 s report dwell
        `time_since_fix_s` was positive again, and the inherited `feedback` reset the per-report fix allowance
        on every executed report, so it relocalised before nearly every report. The accepted P2 trace
        `end_batch_at_home__core__r0` shows it plainly: decisions 125-145 alternate RELOCALIZE, REPORT, with
        **six fix attempts for six reports**, and inspections in between. That is a different and more
        expensive strategy than the declared one, and a mandatory comparator made artificially weak is a way of
        making anything compared against it look better.

    The phases, and the single transition out of each
        `searching`   evidence work and coverage, holding every reportable track. Inspection is asked for
                      with the report branch suppressed, so a ripe candidate cannot mask the inspections the
                      rest of the batch still needs. Leaves when the search plan is exhausted AND no
                      inspection is left.
        `fixing`      exactly one RELOCALIZE(home) has been ISSUED. Leaves on the next decision whatever
                      happened to it - executed, rejected or refused - so there is never a second attempt.
        `releasing`   reports only, consecutively, each paying the common dwell. No search, no inspection and
                      no fix can be chosen here; anything that is not a report is a RETURN.

    The frozen fallback (`BATCH_FALLBACK_RULE`)
        If the home anchor is unreachable, or a fix would buy nothing, the batch is released unfixed. If the
        one attempt is issued and fails, the attempt stays paid and the batch is released anyway. Retrying is
        never the fallback: a strategy that can spend its whole budget at an unavailable anchor is not this
        strategy, and the livelock guard exists for exactly that reason.

    The risk this strategy really carries, stated rather than engineered away
        Holding the batch means the reserve can force the return before a single report is released, and the
        whole batch is then lost. That is the cost of the strategy, not a defect in it, so no early-release
        threshold is added here - one would be a tuned parameter that the frozen grid does not contain.

    `should_fix` is not used: this policy owns its whole decision, and the inherited per-report allowance no
    longer governs anything, because the phase machine permits exactly one attempt in the whole episode.
    """
    name = 'end_batch_at_home'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = BATCH_SEARCHING
        self.fix_attempt = None                            # set once, when the single attempt is issued
        self.fallback = None                               # set when the batch is released without a fix
        self.parameters = dict(self.parameters, fallback_rule=BATCH_FALLBACK_RULE,
                               behaviour_version=POLICY_BEHAVIOUR_VERSION)

    @property
    def batch_state(self):
        """What the phase machine did, for the audit and for the tests. Public, and derived from nothing else."""
        return {'phase': self.phase, 'fix_attempt': self.fix_attempt, 'fallback': self.fallback,
                'fallback_rule': BATCH_FALLBACK_RULE, 'behaviour_version': POLICY_BEHAVIOUR_VERSION}

    def _worth_fixing(self, observation):
        geo = observation.georeference
        return geo.time_since_fix_s > 0. and geo.rms_horizontal_error_m > 1e-9

    def _record_fix_outcome(self, observation):
        """Read what the ONE attempt actually bought, from the public summary.

        `feedback(executed=True)` only says the macro ran: a relocalisation flown to an anchor that was
        unavailable on arrival executes and accepts nothing. `last_fix_accepted` is the public field that
        distinguishes them, so the fallback is decided by the filter's own bookkeeping rather than by whether
        the vehicle moved.
        """
        if self.fix_attempt is None:
            return
        geo = observation.georeference
        if self.fix_attempt['accepted'] is None:
            self.fix_attempt = dict(self.fix_attempt, accepted=geo.last_fix_accepted,
                                    fixes_accepted=int(geo.fixes_accepted))
        if self.fix_attempt['accepted'] is not True and self.fallback is None:
            self.fallback = 'single_fix_attempt_failed'

    def _release(self, base, fallback=None):
        self.phase = BATCH_RELEASING
        if fallback is not None and self.fallback is None:
            self.fallback = fallback
        if base is not None and base.action_type == ActionType.REPORT:
            return base
        return Action(ActionType.RETURN)

    def act(self, observation):
        self._require_gnss(observation)
        self._update_evidence(observation)
        base = self._evidence_action(observation)

        if self.phase == BATCH_RELEASING:
            return self._release(base)                     # reports, then RETURN; nothing else is reachable

        if self.phase == BATCH_FIXING:
            self._record_fix_outcome(observation)
            return self._release(base)                     # the one attempt is spent, whatever it bought

        # The evidence work comes first, and it is asked for WITHOUT the report branch. `_evidence_action`
        # returns a REPORT as soon as any one track is ready, so a holder that consulted only `base` would
        # stop inspecting the moment its first candidate ripened and would then release a batch of one. That
        # is what the first version of this phase machine did, and the pilot smoke caught it: 1-2 reports
        # where every other rule made 6-8.
        inspection = self._evidence_action(observation, allow_report=False)
        if inspection is not None:
            return inspection
        search = self._search_or_return()
        if search.action_type != ActionType.RETURN:
            return search                                  # still searching: every ready candidate is held

        if base is None:
            return self._release(None, 'nothing_held')     # nothing to submit: land
        home = next((a for a in observation.anchors if a.anchor_id == self.home_anchor_id and a.reachable), None)
        if home is None:
            return self._release(base, 'home_anchor_unreachable')
        if not self._worth_fixing(observation):
            return self._release(base, 'fix_would_buy_nothing')
        self.phase = BATCH_FIXING
        self.fix_attempt = {'anchor_id': home.anchor_id, 'issued': True, 'executed': None, 'accepted': None,
                            'reason': None, 'elapsed_s': float(observation.elapsed_s)}
        return self._relocalize(home)                      # the one paid attempt for the whole batch

    def feedback(self, action, executed, reason=None, goal=None):
        if (action.action_type == ActionType.RELOCALIZE and self.fix_attempt is not None
                and self.fix_attempt['executed'] is None):
            self.fix_attempt = dict(self.fix_attempt, executed=bool(executed), reason=reason)
            if not executed:                               # rejected before it ever flew: nothing to read back
                self.fix_attempt = dict(self.fix_attempt, accepted=False)
                if self.fallback is None:
                    self.fallback = 'single_fix_attempt_failed'
        super().feedback(action, executed, reason, goal)


class ShortHorizonPlanner(GnssRulePolicy):
    """One-step utility comparison on public quantities only.

    For the next reportable track it compares two options in units of the score's timely credit:

      report now        p_ok(sigma_now) * w(t_now)
      fix then report   p_ok(sigma_after) * w(t_now + fix duration + report dwell)

    `p_ok(sigma)` is the probability that a two-dimensional zero-mean error of RMS `sigma` lands inside the
    matching tolerance, i.e. the Rayleigh CDF 1 - exp(-tol^2 / sigma^2) with sigma the PER-AXIS value. That is
    an assumption about the filter's own belief, not knowledge of the realised error, and it is declared here
    rather than tuned. `sigma_after` is the posterior the filter would hold after one measurement, computed
    from the public covariance and the declared measurement noise.

    This is a planner, not an oracle: it has no truth, and its score bounds nothing.
    """
    name = 'short_horizon_planner'

    def __init__(self, *args, measurement_sigma_m=0.5, report_dwell_s=6., tolerance_m=TOLERANCE_SCALE_M, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement_sigma_m = float(measurement_sigma_m)
        self.report_dwell_s = float(report_dwell_s)
        self.tolerance_m = float(tolerance_m)
        self.parameters = dict(self.parameters, measurement_sigma_m=self.measurement_sigma_m,
                               report_dwell_s=self.report_dwell_s, tolerance_m=self.tolerance_m)

    def _p_ok(self, rms_total_m):
        """Rayleigh CDF at the tolerance for a 2-D zero-mean error whose total RMS is rms_total_m."""
        if rms_total_m <= 1e-12:
            return 1.
        per_axis_variance = rms_total_m ** 2 / 2.
        return 1. - math.exp(-self.tolerance_m ** 2 / (2. * per_axis_variance))

    def _posterior_rms_m(self, rms_total_m):
        """Total RMS after one measurement: per axis p -> p R / (p + R), with R = measurement_sigma^2."""
        prior = rms_total_m ** 2 / 2.
        r = self.measurement_sigma_m ** 2
        return math.sqrt(2. * (prior * r / (prior + r))) if prior + r > 0 else 0.

    def act(self, observation):
        self._require_gnss(observation)
        self._update_evidence(observation)
        base = self._evidence_action(observation)
        if base is None or base.action_type != ActionType.REPORT:
            return base if base is not None else self._search_or_return()
        budget = float(self.briefing.budgets.mission_time_s)
        elapsed = observation.elapsed_s
        rms = observation.georeference.rms_horizontal_error_m

        def weight(at_s):
            return max(0., 1. - at_s / budget)

        now = self._p_ok(rms) * weight(elapsed + self.report_dwell_s)
        anchor = self._cheapest_anchor(observation)
        if anchor is not None and self._fix_budget_left(observation):
            after = self._p_ok(self._posterior_rms_m(rms)) * weight(elapsed + anchor.estimated_duration_s + self.report_dwell_s)
            if after > now:
                return self._relocalize(anchor)
        return base


POLICY_CLASSES = {'never_fix': NeverFix, 'always_fix_before_report': AlwaysFixBeforeReport, 'periodic_fix': PeriodicFix,
                  'uncertainty_fix': UncertaintyFix, 'end_batch_at_home': EndBatchAtHome,
                  'short_horizon_planner': ShortHorizonPlanner}
assert tuple(POLICY_CLASSES) == COMPARATORS


def build_policy(name, briefing, obstacle_map, config, reserve, **parameters):
    """Factory used by the runner; unknown names and unknown parameters fail loudly."""
    if name not in POLICY_CLASSES:
        raise ValueError(f'policy: expected one of {COMPARATORS}, got {name!r}')
    return POLICY_CLASSES[name](briefing, obstacle_map, config, reserve, **parameters)


__all__ = ['COMPARATORS', 'POLICY_CLASSES', 'AlwaysFixBeforeReport', 'EndBatchAtHome', 'GnssRulePolicy', 'NeverFix',
           'PeriodicFix', 'ShortHorizonPlanner', 'UncertaintyFix', 'build_policy']
