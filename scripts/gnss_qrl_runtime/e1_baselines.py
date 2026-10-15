"""E1 reference policies on the E1 suites: the P3 rule (operational comparator) and the random policy (diagnostic).

`E1_FAIR_COMPARISON_PROPOSAL_EN.md` revision 2, section 5, and review 27 correction 1: both are frozen before
any learned run and evaluated first, on exactly the E1 development and evaluation episodes, once each. Neither is
tuned on E1, and neither ever selects, ranks or removes a learned candidate: their J, FP and return are context in
every outcome panel.

    P3 rule   `uncertainty_fix`, sigma 1.5 m: the P3-selected reference, taken by candidate hash from the frozen
              comparator protocol and checked against the P3 reference selection; run through the same cell and
              episode construction (`run_p3_cells.prepare_cell` / `cell_episode_parts`) as every P3 and P4 episode.
    random    the learned arms' own policy (`RLMissionPolicy`, the same encoder, action space, masks, RETURN rule and
              rejection handling) with an actor whose logits are exactly zero, so every decision is a uniform draw
              among the valid actions, from a declared E1-salted stream per episode.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch                                                                                     # noqa: E402

import fast_mission                                                                              # noqa: E402
import p4_train as p4                                                                            # noqa: E402
import run_p3_cells as runner                                                                    # noqa: E402
from aerial_qrl.training.encoding import LAYOUT_V4_GNSS                                          # noqa: E402
from aerial_qrl.training.reward_gnss import check_reward_identity, episode_rewards                # noqa: E402
from aerial_qrl.training.rl_policy import RLMissionPolicy                                        # noqa: E402
from aerial_qrl.training.smdp_ppo import RecurrentActorCritic                                    # noqa: E402

VERSION = 'gnss-e1-baselines/1'
STUDY = p4.STUDIES['e1']
C = 'output/study/gnss_qrl_v1/frozen/'
COMPARATOR_PROTOCOL = C + 'comparator_protocol_v4/comparator_protocol.json'
REFERENCE_SELECTION = C + 'p3_reference_selection_v1/reference_selection.json'
RULE_SHA256 = '09e437b9c3db35ebb7bd07b065484d4258f3f102fb7e2eeae88b361604835b4d'
RANDOM_STREAM = 5                                   # a new stream id; 0-4 are the trainer's
SUITE_INDEX = {'e1_development': 0, 'e1_evaluation': 1}
RANDOM_STREAM_RULE = (f'SeedSequence([{p4.E1_SALT}, {RANDOM_STREAM}, suite index (e1_development 0, e1_evaluation 1), '
                      'cell_index, realisation])')
POLICIES = ('p3_rule', 'random')
EPISODE_FIELDS = ('cell_id', 'seed', 'J', 'tp_timely', 'tp_spatial', 'fp_spatial', 'return_ok', 'fixes_attempted',
                  'decisions', 'termination_reason')


class BaselineError(ValueError):
    pass


def random_stream(suite, cell, realization):
    return STUDY.stream(RANDOM_STREAM, SUITE_INDEX[suite], cell['cell_index'], realization)


def p3_rule():
    """The P3 reference rule, by hash, from the frozen comparator protocol; refused if anything disagrees."""
    protocol = json.loads((ROOT / COMPARATOR_PROTOCOL).read_text())['comparator_protocol']
    selection = json.loads((ROOT / REFERENCE_SELECTION).read_text())['reference_selection']
    found = [c for c in protocol['candidates'] if c['candidate_sha256'] == RULE_SHA256]
    if len(found) != 1 or selection['reference_sha256'] != RULE_SHA256 or \
            (found[0]['name'], found[0]['parameters']) != (selection['reference_name'],
                                                           selection['reference_parameters']):
        raise BaselineError('the P3 reference rule is not the frozen, selected candidate')
    if runner.sha256_file(ROOT / 'scripts/gnss_qrl_runtime/gnss_policies.py') != protocol['policy_source_sha256']:
        raise BaselineError('gnss_policies.py is not the policy source the comparator protocol froze')
    rule = found[0]
    return {'name': rule['name'], 'parameters': dict(rule['parameters']), 'candidate_sha256': rule['candidate_sha256'],
            'comparator_protocol_sha256': protocol['protocol_sha256'],
            'policy_source_sha256': protocol['policy_source_sha256']}


def zero_actor_model():
    """A RecurrentActorCritic whose actor logits are exactly 0 for every input: masked, that is uniform."""
    torch.manual_seed(STUDY.stream(RANDOM_STREAM, 99))          # the trunk is irrelevant but deterministic
    model = RecurrentActorCritic(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, 'mlp', hidden=64,
                                 critic_gradient='stop', memory='gru', actor_head='linear').to(dtype=torch.float64)
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
    model.eval()
    return model


def run_policy_episode(runtime, cell, namespace, seed, policy, episode_id, rule=None, generator=None, model=None):
    """One episode of a reference policy on the prepared cell, scored exactly as a learned episode is."""
    static = runtime.static(cell)
    backend_config, group, service = runner.cell_episode_parts(static, cell, namespace, seed)

    def factory(briefing, obstacle_map, mission_config, reserve):
        if policy == 'p3_rule':
            return runner.build_policy(rule['name'], briefing, obstacle_map, mission_config, reserve,
                                       **rule['parameters'])
        return RLMissionPolicy(briefing, obstacle_map, mission_config, reserve, model, generator=generator,
                               deterministic=False, compute_values=False, obs_encoding=LAYOUT_V4_GNSS.obs_version,
                               return_mask_rule=p4.RETURN_MASK_RULE, anchor_ids=runtime.anchor_ids())
    started = time.perf_counter()
    result = fast_mission.run_fast_episode(static['context'], static['banks'], runtime.time_model, episode_id,
                                           backend_config=backend_config,
                                           executive_kwargs={'georeference': service, 'anchor_set': static['anchors']},
                                           policy_factory=factory, reports_for_matching=service.global_report_positions)
    outcome = result['outcome']
    return_ok = outcome.termination_reason == 'mission_complete' and outcome.returned and outcome.landed
    breakdown = episode_rewards(service.global_report_positions(), [s.sim_time_s for s in service.submissions],
                                static['context']['evaluator'].targets, static['score'], return_ok)
    check_reward_identity(breakdown)
    final = breakdown.final
    return {'cell_id': cell['cell_id'], 'seed': seed, 'crn_sha256': group['crn_sha256'],
            'termination_reason': outcome.termination_reason, 'return_ok': return_ok, 'J': breakdown.score,
            'tp_timely': final.tp_timely, 'tp_spatial': final.tp_spatial, 'fp_spatial': final.fp_spatial,
            'duplicates': final.duplicates, 'fixes_attempted': len(service.measurements),
            'decisions': outcome.decisions, 'rejections': outcome.rejections, 'elapsed_s': outcome.elapsed_s,
            'wall_s': time.perf_counter() - started}


def evaluate_policy(runtime, policy, suite):
    """Every episode of an E1 suite, once, for one reference policy; with the trainer's own aggregation."""
    import e1_seed_manifest
    rule = p3_rule() if policy == 'p3_rule' else None
    model = zero_actor_model() if policy == 'random' else None
    namespace, _ = STUDY.suites[suite]
    episodes = []
    for cell, r, seed in e1_seed_manifest.suite_episodes(runtime, suite):
        generator = (torch.Generator().manual_seed(random_stream(suite, cell, r)) if policy == 'random' else None)
        episodes.append(run_policy_episode(runtime, cell, namespace, seed, policy,
                                           f'e1-{policy}-{suite}-c{cell["cell_index"]}-r{r}', rule=rule,
                                           generator=generator, model=model))
    per_cell, weighted = p4.aggregate_evaluation(runtime, episodes)
    return {'policy': policy, 'suite': suite, 'rule': rule, 'per_cell_J': per_cell, 'weighted_J': weighted,
            'episodes': episodes}


def _job(args):
    policy, suite = args
    import e1_seed_manifest
    return evaluate_policy(e1_seed_manifest.e1_runtime(), policy, suite)


def run_stage(out_dir, workers=4):
    """The baseline stage: both policies on both suites, written once to `baselines.json`."""
    from concurrent.futures import ProcessPoolExecutor
    out_dir = Path(out_dir)
    path = out_dir / 'baselines.json'
    if path.exists():
        raise SystemExit(f'{path} exists; never overwritten')
    tasks = [(policy, suite) for policy in POLICIES for suite in STUDY.suites]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_job, tasks))
    payload = {'version': VERSION, 'rule': p3_rule(), 'random_stream_rule': RANDOM_STREAM_RULE,
               'results': {f'{r["policy"]}/{r["suite"]}': r for r in results}}
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=float) + '\n')
    return path


def verify_baselines(path, runtime, seed_manifest):
    """Prove the baseline stage is complete and re-derivable; returns {policy/suite: result}. Refuses otherwise."""
    payload = json.loads(Path(path).read_text())
    if payload.get('version') != VERSION or payload.get('rule') != p3_rule():
        raise BaselineError('the baselines are not this version or not the frozen P3 rule')
    expected = {f'{p}/{s}' for p in POLICIES for s in STUDY.suites}
    if set(payload['results']) != expected:
        raise BaselineError(f'the baseline stage is incomplete: {sorted(payload["results"])}')
    for key, result in payload['results'].items():
        policy, suite = key.split('/')
        frozen = [(e['cell_id'], e['seed']) for e in seed_manifest['suites'][suite]['episodes']]
        if [(e['cell_id'], e['seed']) for e in result['episodes']] != frozen:
            raise BaselineError(f'{key}: the episodes are not exactly the frozen {suite} suite, once each, in order')
        per_cell, weighted = p4.aggregate_evaluation(runtime, result['episodes'])
        if p4.canonical_sha256([per_cell, weighted]) != p4.canonical_sha256([result['per_cell_J'],
                                                                             result['weighted_J']]):
            raise BaselineError(f'{key}: the stored J is not its episodes\' J')
    return payload['results']


def summary(result):
    """The unweighted means of the outcome panel over a baseline's episodes, plus its weighted J."""
    episodes = result['episodes']
    mean = lambda k: sum(float(e[k]) for e in episodes) / len(episodes)
    return {'weighted_J': result['weighted_J'], 'return_rate': mean('return_ok'), 'fp_spatial': mean('fp_spatial'),
            'tp_timely': mean('tp_timely'), 'tp_spatial': mean('tp_spatial'),
            'fixes_attempted': mean('fixes_attempted'), 'episodes': len(episodes)}
