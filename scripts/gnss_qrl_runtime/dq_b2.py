"""Freeze B2: from-scratch training replication of HQ, C-matched and the exact S625 twin (review 38 addendum).

    .venv/bin/python scripts/gnss_qrl_runtime/dq_b2.py smoke     --out <new dir>        # development cost smoke
    .venv/bin/python scripts/gnss_qrl_runtime/dq_b2.py precision --out <new dir>        # frozen precision calculation
    .venv/bin/python scripts/gnss_qrl_runtime/dq_b2.py manifest  --out <new dir>
    .venv/bin/python scripts/gnss_qrl_runtime/dq_b2.py freeze    --out <new dir> --manifest <abs path>
    .venv/bin/python scripts/gnss_qrl_runtime/dq_b2.py launch    --campaign <abs dir> --workers 12   # REFUSES without approval

The trainer is E1's own `p4_train.train_arm`, byte-identical to the E1 tree. B2 changes nothing in it. At import,
this module:
- registers two studies in `p4_train.STUDIES`: `dq_b2` (the campaign) and `dq_b2_smoke` (development only);
- replaces `p4_train.build_model` in THIS process with `dq_build_model`. That function returns E1's own model for
  HQ and C-matched. For the S625 twin it builds the block's untrained HQ with the same seed, transplants it exactly
  (dq_transplant.make_twin) and certifies the static parity (P1, P2, P5, P6, P7). Any failure is a refusal.

Design (planning values; the frozen manifest records them):
    arms          HQ (quantum, lr 3e-3), C-matched (bottleneck, lr 3e-3), S625-twin (surrogate625 twin, lr 1e-3);
                  entropy 0. These are E1's selected candidates, declared in advance; nothing is tuned in B2.
    blocks        0-35 of dq_e2_training, whole triplets per block, submitted in ascending block order
    schedule      E1's rule under the dq_b2 salt: cell ~ frozen weights from SeedSequence([salt, block, 3, episode]),
                  seed = dq_e2_training.start + block * 1000 + episode; the same for every arm of a block (paired)
    evaluation    E1's: sampled read-out at 11 points 0..30,000 decisions, 8 core cells x 10 realisations of
                  dq_e2_evaluation, CRN stream SeedSequence([salt, 4, point, cell, r]), the same for every arm
    twin parity   at deployment: static P1, P2, P5, P6, P7 at build; P4 = the block's first training episode re-flown by
                  HQ and twin with fresh copies of the block's action stream (identical actions, observations,
                  masks and termination; |dlog pi| <= 1e-9); recorded per S625-twin run, fail closed
    KL logging    per update, PPO's approx_kl (smdp_ppo.ppo_update metrics), already in metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_circuits as qc                                                                          # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402

# ------------------------------------------------------------------ FROZEN DESIGN ----------------------------------
TOOL = 'gnss-dq-b2/1'
B2_SALT = (dq.DQ_SALT, 2)
SMOKE_SALT = (dq.DQ_SALT, 3)
ARMS = {'quantum': {'learning_rate': 3e-3, 'entropy_coef': 0.0, 'role': 'HQ', 'candidate': 'lr3e-3_ent0'},
        'bottleneck': {'learning_rate': 3e-3, 'entropy_coef': 0.0, 'role': 'C-matched', 'candidate': 'lr3e-3_ent0'},
        'surrogate625': {'learning_rate': 1e-3, 'entropy_coef': 0.0, 'role': 'S625-twin', 'candidate': 'lr1e-3_ent0'}}
CANDIDATES = {'lr3e-3_ent0': {'learning_rate': 3e-3, 'entropy_coef': 0.0},
              'lr1e-3_ent0': {'learning_rate': 1e-3, 'entropy_coef': 0.0}}
CAMPAIGN = ROOT / 'output' / 'study' / 'gnss_qrl_v1' / 'dq' / 'b2_campaign_v3'
CAMPAIGN_README = 'output/study/gnss_qrl_v1/dq/b2_campaign_v3/README.md'
TEST_BUDGET, TEST_GRID = 1, [0]
#: development runs (smoke, tests) name this label instead of an approved freeze; it can never be an approval hash
DEV_FREEZE_LABEL = 'd' * 64
BLOCKS = tuple(range(36))
BUDGET = 30_000
GRID = list(range(0, BUDGET + 1, 3000))
EVAL_REALIZATIONS = tuple(range(10))
SMOKE_BLOCK, SMOKE_BUDGET, SMOKE_GRID, SMOKE_REALIZATIONS = 90, 6000, [0, 3000, 6000], (950, 951)
E1_REPORT = ROOT / 'output/study/gnss_qrl_v1/e1/e1_report_v1/report.json'
E1_REPORT_SHA256 = 'a8d4859bd569f3c7eb2b820af1ee6f1a7ebedf699ae81d282b4c924a5843527a'
TOST_DELTA = 0.0056                    # 0.1 report-equivalent per episode (plan section 3.4), fixed before B2 data
WORKERS = 12
# E1's frozen PPO settings (config.json of every E1 evaluation run), except the arm's learning rate and entropy.
PPO_BASE = {'clip_range': 0.2, 'value_coef': 0.5, 'max_grad_norm': 0.5, 'epochs': 4, 'minibatch_transitions': 256,
            'advantage_std_floor': 0.05, 'max_segment_length': 32}
# ------------------------------------------------------------------ END FROZEN DESIGN -------------------------------


class B2Error(dq.DQError):
    """A B2 refusal."""


def _study(name, salt, training_namespace, suite, realizations, blocks, approval, roles):
    return p4.Study(name=name, tool=f'{TOOL}:{name}', salt=salt, training_namespace=training_namespace,
                    extra_namespaces=dq.DQ_NAMESPACES, blocks={name: blocks},
                    suites={suite: (training_namespace.replace('training', 'evaluation') if name == 'dq_b2'
                                    else 'dq_development', realizations)},
                    roles={role: (name, suite, budget, grid) for role, (budget, grid) in roles.items()},
                    candidates=CANDIDATES, evaluation_suites=(suite,),
                    approval_path=approval['path'], approval_version=approval['version'],
                    approval_key=approval['key'],
                    non_claim=('A DQ B2 run records the training of one arm of one block under the frozen B2 protocol; '
                               'it supports no claim beyond the frozen B2 analysis.'))


STUDY_B2 = _study('dq_b2', B2_SALT, 'dq_e2_training', 'dq_b2_evaluation', EVAL_REALIZATIONS, BLOCKS, dq.APPROVALS['b2'],
                  {'dq_b2': (BUDGET, GRID)})
STUDY_SMOKE = _study('dq_b2_smoke', SMOKE_SALT, 'dq_development', 'dq_b2_smoke_evaluation', SMOKE_REALIZATIONS,
                     (SMOKE_BLOCK,), dq.APPROVALS['b2'],
                     {'dq_b2_smoke': (SMOKE_BUDGET, SMOKE_GRID), 'dq_b2_test': (TEST_BUDGET, TEST_GRID)})
p4.STUDIES.setdefault('dq_b2', STUDY_B2)
p4.STUDIES.setdefault('dq_b2_smoke', STUDY_SMOKE)
if p4.STUDIES['dq_b2'] is not STUDY_B2 or p4.STUDIES['dq_b2_smoke'] is not STUDY_SMOKE:
    raise B2Error('another dq_b2 study is already registered in this process')


def runtime(study):
    c = 'output/study/gnss_qrl_v1/frozen/'
    return p4.CellRuntime(ROOT / (c + 'p3_freeze_v1/p3_frozen_configuration.json'),
                          ROOT / (c + 'p3_execution_manifest_v1/p3_execution_manifest.json'),
                          ROOT / 'output/training/time_model_village_v1/20260917T074923Z/time_model.json', study=study)


# -- the S625 twin at deployment -----------------------------------------------------------------------------------------

_ORIGINAL_BUILD = p4.build_model
TWIN_RECORDS = {}


def dq_build_model(arm, config, block):
    """E1's build for HQ and C-matched; for surrogate625 in a DQ B2 study, the exact twin of the block's HQ."""
    if arm != 'surrogate625' or config.get('study') not in ('dq_b2', 'dq_b2_smoke'):
        return _ORIGINAL_BUILD(arm, config, block)
    hq = _ORIGINAL_BUILD('quantum', config, block)
    twin, C = tp.make_twin(hq)                                   # refuses anything but a float64 shipped HQ
    support, _ = tp.support(qc.SHIPPED)
    TWIN_RECORDS[(config['study'], block)] = {'static': tp.certify_static(hq, twin, C, support)}
    twin.train()
    return twin


p4.build_model = dq_build_model


def twin_trajectory_parity(study, rt, config, block):
    """P4 at deployment: the block's first training episode, flown by the block's HQ and by its twin, each with a
    fresh copy of the block's action stream (the training generators themselves are not touched)."""
    cell, seed = rt.schedule(block, 0)
    runs = []
    for arm in ('quantum', 'surrogate625'):
        model = dq_build_model(arm, config, block).eval()
        generator = torch.Generator().manual_seed(study.stream(block, p4.ACTION_STREAM))
        runs.append(p4.run_learned_episode(rt, cell, study.training_namespace, seed, model, generator, False,
                                           f'{study.name}-parity-{arm}-b{block}', 0, config['gamma_ref'],
                                           config['reference_duration_s']))
    a, b = (r['policy'].decisions for r in runs)
    worst = 0.
    if len(a) != len(b):
        raise tp.TransplantParityError(f'B2 P4: {len(a)} vs {len(b)} decisions')
    for x, y in zip(a, b):
        if x.action_index != y.action_index or not np.array_equal(x.obs, y.obs) or \
                not np.array_equal(x.public_mask, y.public_mask):
            raise tp.TransplantParityError(f'B2 P4: decision {x.index} differs')
        if x.log_prob is not None:
            worst = max(worst, abs(x.log_prob - y.log_prob))
    if runs[0]['record']['J'] != runs[1]['record']['J'] or not worst <= tp.TOL['P4']:
        raise tp.TransplantParityError(f'B2 P4: outcome or log-prob differs ({worst:.3e})')
    return {'cell_id': cell['cell_id'], 'seed': seed, 'decisions': len(a), 'max_abs_dlogprob': worst}


# -- configurations and one run --------------------------------------------------------------------------------------------

def config_for(arm, block, study, budget=BUDGET, grid=None, freeze_sha256=None, role=None):
    suite = study.evaluation_suites[0]
    role = role or study.name
    return {'hidden': 64, 'memory': 'gru', 'critic_gradient': 'stop', 'dtype': 'float64', 'torch_threads': 1,
            'note': f'{study.name}: {ARMS[arm]["role"]} block {block}', 'max_updates': None,
            'config_version': p4.CONFIG_VERSION,
            'freeze': 'output/study/gnss_qrl_v1/frozen/p3_freeze_v1/p3_frozen_configuration.json',
            'execution_manifest': 'output/study/gnss_qrl_v1/frozen/p3_execution_manifest_v1/p3_execution_manifest.json',
            'time_model': 'output/training/time_model_village_v1/20260917T074923Z/time_model.json',
            'critic': 'mlp', 'transitions_per_update': 512, 'gamma_ref': 1.0, 'reference_duration_s': 60.0,
            'lambda_ref': 0.95, 'lambda_mode': 'time', 'study': study.name, 'role': role, 'arms': [arm],
            'candidate': ARMS[arm]['candidate'], 'block': block,
            'total_policy_decisions': budget, 'checkpoint_decisions': list(grid or GRID),
            'ppo': dict(PPO_BASE, learning_rate=ARMS[arm]['learning_rate'], entropy_coef=ARMS[arm]['entropy_coef']),
            'evaluation': {'suite': suite, 'modes': ['sample']}, 'protocol_freeze_sha256': freeze_sha256}


def validate(arm, config, study, approved_sha256):
    """E1's own validator, then B2's bindings: arm <-> candidate, study/role, namespace, freeze, and the exact frozen
    configuration (the reconstructed config must be identical). Any difference is a refusal before any output."""
    config = p4.validate_config(config)
    if config['arms'] != [arm] or arm not in ARMS:
        raise B2Error(f'arms: exactly [{arm}] (one arm per run)')
    if config['candidate'] != ARMS[arm]['candidate']:
        raise B2Error(f'{arm} runs candidate {ARMS[arm]["candidate"]}, not {config["candidate"]}')
    if config['study'] != study.name or config['role'] not in study.roles:
        raise B2Error(f'study/role: {study.name} with one of {sorted(study.roles)}')
    if study.name == 'dq_b2' and study.training_namespace != 'dq_e2_training':
        raise B2Error('the B2 study trains in dq_e2_training only')
    if config['protocol_freeze_sha256'] != approved_sha256:
        raise B2Error('the configuration names another freeze than the approved one')
    budget, grid = study.roles[config['role']][2], study.roles[config['role']][3]
    expected = p4.validate_config(config_for(arm, config['block'], study, budget, grid, approved_sha256,
                                             role=config['role']))
    if dq.canonical_sha256(config) != dq.canonical_sha256(expected):
        raise B2Error('the configuration differs from the frozen B2 configuration')
    return config


def run_one(task):
    """One (arm, block) run; returns its record (failures recorded, never retried)."""
    study_name, arm, block, out_dir, role, approved_sha256, test_root = task
    study = p4.STUDIES[study_name]
    torch.set_num_threads(1)
    record = {'study': study_name, 'arm': arm, 'role': ARMS[arm]['role'], 'block': block, 'status': 'started'}
    started = time.perf_counter()
    try:
        if study_name == 'dq_b2':
            _, approved, _ = p4.require_approved_freeze(approved_sha256, study=STUDY_B2)
            for ns in ('dq_e2_training', 'dq_e2_evaluation'):
                dq.require_open(ns)
        else:
            dq.require_open('dq_development')
            if approved_sha256 != DEV_FREEZE_LABEL:
                raise B2Error('development runs carry the development label, never a freeze hash')
            approved = approved_sha256
        budget, grid = study.roles[role][2], study.roles[role][3]
        config = validate(arm, config_for(arm, block, study, budget, grid, approved, role=role), study, approved)
        rt = runtime(study)
        run_dir = Path(out_dir) / f'{arm}__b{block}'
        dq.vet_output(run_dir, test_root=test_root)
        record.update(config=config, config_sha256=dq.canonical_sha256(config), code_tree_sha256=p4.live_copy_tree(),
                      protocol_freeze_sha256=approved)
        if arm == 'surrogate625':
            record['twin_parity'] = {'P4': twin_trajectory_parity(study, rt, config, block)}
        summary = p4.train_arm(arm, config, rt, run_dir / 'run' / arm)
        if arm == 'surrogate625':
            record['twin_parity'].update(TWIN_RECORDS[(study_name, block)])
        record.update(status=summary['status'], summary=summary)
    except tp.TransplantParityError as exc:
        record.update(status='parity_refused', error=str(exc))
    except (dq.DQError, p4.P4Error, p4.FreezeBindingError) as exc:
        record.update(status='refused', error=str(exc))
    except Exception as exc:                                                      # infrastructure: recorded
        import traceback
        record.update(status='infrastructure_failure', error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc()[-4000:])
    record['wall_s'] = time.perf_counter() - started
    return record


def tasks(study_name, out_dir, role, approved_sha256, blocks=None, test_root=None):
    blocks = blocks if blocks is not None else STUDY_B2.blocks['dq_b2']
    return [(study_name, arm, b, str(out_dir), role, approved_sha256, test_root)
            for b in blocks for arm in ARMS]                   # whole triplets, ascending block order


def _pool_run(task_list, workers, write):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for record in pool.imap(run_one, task_list):
            write(record)


# -- smoke, precision, manifest, freeze, launch -----------------------------------------------------------------------------

def smoke(out):
    """Development cost smoke: one triplet in dq_development at 6,000 decisions, 2 evaluation realisations."""
    dq.require_open('dq_development')
    dq.claim_output(out, directory=True)
    records = []
    _pool_run(tasks('dq_b2_smoke', out, 'dq_b2_smoke', DEV_FREEZE_LABEL, blocks=(SMOKE_BLOCK,)), 3, records.append)
    cost = {}
    for r in records:
        s = r.get('summary') or {}
        cpu = s.get('cpu', {})
        decisions = s.get('policy_decisions') or 0
        cost[r['arm']] = {'status': r['status'], 'policy_decisions': decisions, 'cpu': cpu,
                          'train_cpu_s_per_decision': (cpu.get('rollout_s', 0) + cpu.get('update_s', 0)) / decisions
                          if decisions else None,
                          'eval_cpu_s_per_point_per_episode': cpu.get('evaluation_s', 0) / (len(SMOKE_GRID) * 16)
                          if cpu else None,
                          'twin_parity': r.get('twin_parity'), 'error': r.get('error')}
    projected = {}
    for arm, c in cost.items():
        if c['train_cpu_s_per_decision'] is not None:
            projected[arm] = (c['train_cpu_s_per_decision'] * BUDGET
                              + c['eval_cpu_s_per_point_per_episode'] * len(GRID) * 8 * len(EVAL_REALIZATIONS)) / 3600
    result = {'statement': dq.DEVELOPMENT_STATEMENT, 'cost': cost, 'projected_cpu_h_per_run': projected,
              'projected_cpu_h_total': sum(projected.values()) * len(BLOCKS) if len(projected) == 3 else None,
              'records': records}
    dq.write_json(Path(out) / 'smoke.json', json.loads(json.dumps(result, default=float)))
    return result


def e1_paired():
    """Per-block paired AUC differences of E1 (development material for the precision calculation only)."""
    if dq.sha256_file(E1_REPORT) != E1_REPORT_SHA256:
        raise B2Error('the E1 report of record changed')
    r = json.loads(E1_REPORT.read_text())
    auc = {(j['family'], j['block']): j['auc'] for j in r['jobs'].values()}
    blocks = sorted({b for _, b in auc})
    return {'quantum_minus_bottleneck': [auc[('quantum', b)] - auc[('bottleneck', b)] for b in blocks],
            'quantum_minus_surrogate625_rand': [auc[('quantum', b)] - auc[('surrogate625', b)] for b in blocks]}


def precision(n=len(BLOCKS), alpha_holm=0.025):
    """The frozen precision calculation (paired t, normal-approximation power) from E1's paired SDs."""
    from scipy import stats
    out = {'n': n, 'source': 'E1 report of record a8d4859b; S625 in E1 was randomly initialised, not a twin'}
    for name, d in e1_paired().items():
        sd, mean = float(np.std(d, ddof=1)), float(np.mean(d))
        se = sd / math.sqrt(n)
        q = stats.t.ppf(0.975, n - 1)
        power = lambda effect, a: float(stats.norm.sf(stats.norm.ppf(1 - a / 2) - abs(effect) / se))     # noqa: E731
        rec = {'e1_mean': mean, 'e1_sd': sd, 'half_width_95': q * se,
               'power_at_e1_mean_alpha_0.05': power(mean, 0.05), 'power_at_e1_mean_holm_0.025': power(mean, alpha_holm),
               'power_at_half_e1_mean_holm_0.025': power(mean / 2, alpha_holm)}
        if name == 'quantum_minus_bottleneck':
            q90 = stats.t.ppf(0.95, n - 1)
            # probability that the 90% CI lies inside +/- delta when the true difference is 0 or E1's mean
            for tag, true in (('if_zero', 0.0), ('if_e1_mean', mean)):
                lo_ok = stats.norm.cdf((TOST_DELTA - q90 * se - true) / se)
                hi_ok = stats.norm.cdf((TOST_DELTA - q90 * se + true) / se)
                rec[f'tost_pass_probability_{tag}'] = float(max(0., lo_ok + hi_ok - 1))
        out[name] = rec
    return out


FROZEN_FILES = (CAMPAIGN_README, 'scripts/gnss_qrl_runtime/dq_b2.py', 'scripts/gnss_qrl_runtime/dq_b2_analysis.py',
                'scripts/gnss_qrl_runtime/p4_train.py', 'scripts/gnss_qrl_runtime/dq_transplant.py',
                'scripts/gnss_qrl_runtime/dq_circuits.py', 'scripts/gnss_qrl_runtime/dq_runtime.py',
                'src/aerial_qrl/training/smdp_ppo.py', 'src/aerial_qrl/training/actor_heads.py',
                'tests/gnss_qrl_dq/test_dq_b2.py', 'docs/study/gnss_qrl_v1/frozen/DQ_B2_PROTOCOL_EN.md')


def manifest():
    rt = runtime(STUDY_B2)
    schedule = {b: [[k, *(lambda cs: (cs[0]['cell_id'], cs[1]))(rt.schedule(b, k))] for k in range(5)] for b in BLOCKS}
    return {'manifest_version': 'gnss-dq-b2-manifest/1', 'study': 'dq_b2', 'salt': list(B2_SALT),
            'namespaces': {k: dq.DQ_NAMESPACES[k] for k in ('dq_e2_training', 'dq_e2_evaluation')},
            'arms': ARMS, 'candidates': CANDIDATES, 'blocks': list(BLOCKS), 'budget': BUDGET, 'grid': GRID,
            'campaign': str(CAMPAIGN.relative_to(ROOT)),
            'config_rule': 'each run config = dq_b2.config_for(arm, block, STUDY_B2, freeze_sha256=<approved>), exactly',
            'evaluation_realizations': list(EVAL_REALIZATIONS), 'ppo_base': PPO_BASE,
            'runs': [f'{arm}__b{b}' for b in BLOCKS for arm in ARMS], 'wave_order': 'whole triplets, ascending block',
            'schedule_first_5': {str(b): s for b, s in schedule.items()},
            'streams': {'model_init': 'SeedSequence([salt..., block, 0])', 'action': 'SeedSequence([salt..., block, 1])',
                        'minibatch': 'SeedSequence([salt..., block, 2])',
                        'evaluation': 'SeedSequence([salt..., 4, point, cell_index, r])'},
            'twin': 'S625-twin = exact transplant of the block\'s untrained HQ (same init seed); P1-P7 at deployment',
            'tost_delta': TOST_DELTA, 'precision': precision()}


def write_manifest(out):
    dq.claim_output(out, directory=True)
    m = manifest()
    m['manifest_sha256'] = dq.canonical_sha256(m)
    path = Path(out) / 'b2_manifest.json'
    dq.write_json(path, {'utc': dq.utc(), 'seed_manifest': m})
    return path, m['manifest_sha256']


def write_freeze(out, manifest_path):
    dq.watch_trees()
    manifest_path = Path(manifest_path).resolve()
    dq.claim_output(out, directory=True)
    m = json.loads(manifest_path.read_text())['seed_manifest']
    record = {'tool': TOOL, 'utc': dq.utc(), 'protocol': 'docs/study/gnss_qrl_v1/frozen/DQ_B2_PROTOCOL_EN.md',
              'files': {f: dq.sha256_file(ROOT / f) for f in FROZEN_FILES},
              'seed_manifest': str(manifest_path.relative_to(ROOT.resolve())), 'seed_manifest_sha256': m['manifest_sha256'],
              'copy_code_tree_sha256': dq.dq_tree()[0], 'statement': 'Freeze B2 package; nothing has run under it'}
    path = Path(out) / 'freeze.json'
    dq.write_json(path, record)
    return path, dq.sha256_file(path)


def launch(campaign, workers, expected_sha256):
    """Run all 108 runs under the APPROVED freeze, into the exact frozen campaign, once. No retry, no resume."""
    freeze_path, approved, record = p4.require_approved_freeze(expected_sha256, study=STUDY_B2)
    for ns in ('dq_e2_training', 'dq_e2_evaluation'):
        dq.require_open(ns)
    campaign = Path(campaign).resolve()
    if campaign != CAMPAIGN.resolve():
        raise B2Error(f'the approved campaign is {CAMPAIGN}, not {campaign}')
    if sorted(p.name for p in campaign.iterdir()) != ['README.md']:
        raise B2Error(f'{campaign} is not pristine (only README.md allowed): a second launch or resume is refused')
    if dq.sha256_file(campaign / 'README.md') != record['files'][CAMPAIGN_README]:
        raise B2Error('the campaign README is not the frozen one')
    runs, records_dir = campaign / 'runs', campaign / 'records'
    dq.claim_output(runs, directory=True)
    dq.claim_output(records_dir, directory=True)
    dq.write_json(campaign / 'launch.json', {'utc': dq.utc(), 'freeze': str(freeze_path), 'freeze_sha256': approved,
                                             'code_tree_sha256': p4.live_copy_tree(), 'workers': workers})
    dq.append_tree_watch('b2 launch')

    def write(rec):
        dq.write_json(records_dir / f'{rec["arm"]}__b{rec["block"]}.json', json.loads(json.dumps(rec, default=float)))
    _pool_run(tasks('dq_b2', runs, 'dq_b2', approved), workers, write)
    dq.append_tree_watch('b2 done')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('smoke', 'precision', 'manifest'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True, type=Path)
    fr = sub.add_parser('freeze')
    fr.add_argument('--out', required=True, type=Path)
    fr.add_argument('--manifest', required=True, type=Path)
    la = sub.add_parser('launch')
    la.add_argument('--campaign', required=True, type=Path)
    la.add_argument('--workers', type=int, default=WORKERS)
    la.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'smoke':
        print(json.dumps(smoke(args.out.resolve())['projected_cpu_h_per_run'], default=float))
    elif args.command == 'precision':
        dq.claim_output(args.out.resolve(), directory=True)
        dq.write_json(args.out.resolve() / 'precision.json', precision())
    elif args.command == 'manifest':
        print(json.dumps(dict(zip(('path', 'manifest_sha256'), map(str, write_manifest(args.out.resolve()))))))
    elif args.command == 'freeze':
        print(json.dumps(dict(zip(('path', 'sha256'), map(str, write_freeze(args.out.resolve(), args.manifest))))))
    else:
        launch(args.campaign.resolve(), args.workers, args.freeze_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
