"""E1 gate: the design constants PLAN.md states, asserted against runner.config(block) (review B3).

    python config_check.py                          # prints config(0); asserts the PLAN constants, the runner's
                                                    # module constants, the idea_dq tree and the scripts listed in
                                                    # PLAN.md (exit 1 on any mismatch)
    python config_check.py --precheck precheck.json # also: the precheck's config_block0 equals a fresh config(0),
                                                    # its plan_sha256 equals PLAN.sha256, it passed, and its
                                                    # code_sha256 equals the scripts listed in PLAN.md

runner.py (launch and every run) and gate_analysis.py (every record) import the same constants from here, so the
plan text, the runs and the analysis are bound to one set of values. This module imports nothing of runner at
import time.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PLAN, PLAN_HASH_FILE = HERE / 'PLAN.md', HERE / 'PLAN.sha256'

#: every script in this folder that the precheck, the runs or the analysis execute; PLAN.md lists each SHA-256
SCRIPTS = ('runner.py', 'gate_core.py', 'test_gate.py', 'test_analysis.py', 'gate_analysis.py', 'config_check.py')
EXPECTED_TREE = 'ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952'
BASELINES_SHA256 = '00202039dda01d814146db77e85590996abda974b4e813bbb17c10d081572fd7'

FEATURES = (
    'rms_horizontal_error_m/3', 'min(time_since_fix_s,T)/T', 'min(fixes_attempted,10)/10',
    'min(fixes_accepted,10)/10', 'last_fix_accepted', 'no_fix_yet', 'elapsed_s/T', 'remaining_time_s/T',
    'return_estimate_s/T', '(remaining-return_estimate-R)/T', 'T/2700', 'cheapest_anchor.estimated_duration_s/100',
    'cheapest_anchor.available', 'share_of_anchors_reachable_and_available', 'min(reports,20)/20',
    'candidate.detections(cap20)/20', 'candidate.credited_viewpoints(cap8)/8', 'candidate.max_score',
    'candidate.sqrt(cov_xx+cov_yy)/10', 'candidate.age_s/T', 'candidate.inspections/max_inspections_per_track',
    'other_rule_ready_open_tracks(cap5)/5', 'open_tracks(cap10)/10', 'plan_pending_fraction')

#: config(block) of a main run must equal these values exactly (PLAN.md, 'What is learned', 'PPO', 'Read-outs')
PLAN_CONSTANTS = {
    'version': 'e1-gate-config/2', 'tool': 'e1-gate-train/1', 'study': 'e1gate', 'salt': [20261025, 1],
    'training_namespace': 'gate_training',
    'training_seed_rule': 'gate_training.start + block * 5000 + episode',
    'initialisation': 'from_scratch: torch.manual_seed(SeedSequence([20261025, 1, block, 0])), no warm start',
    'model': {'kind': 'gate-mlp/1', 'hidden': 64, 'features': list(FEATURES)},
    'budget_gate_decisions': 12_800, 'grid': [0, 3_200, 6_400, 9_600, 12_800], 'transitions_per_update': 256,
    'ppo': {'learning_rate': 1e-3, 'clip_range': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'max_grad_norm': 0.5,
            'epochs': 10, 'minibatch_transitions': 64, 'advantage_std_floor': 0.0056, 'max_segment_length': 1},
    'gamma_ref': 1.0, 'reference_duration_s': 60.0, 'lambda_ref': 0.95, 'lambda_mode': 'time',
    'curve_suite': 'e1_development', 'final_suite': 'e1_evaluation', 'readout': 'argmax',
    'sampled_readout': {'role': 'descriptive', 'suite': 'final_suite', 'cells': 'target-present',
                        'eval_stream': 4, 'point_index': 5,
                        'stream_rule': 'SeedSequence([20261025, 1, 4, 5, cell_index, realization])'},
    'curve_realizations': None, 'final_realizations': None,
}
RULE_PARAMETERS = {'sigma_threshold_m': 1.5}
RULE_NAME = 'uncertainty_fix'
CONFIG_KEYS = set(PLAN_CONSTANTS) | {'block', 'rule', 'plan_sha256', 'code_sha256'}
BLOCKS = tuple(range(12))
STRIDE = 5_000
NAMESPACES = {'gate_training': (20262821000, 20262921000), 'gate_development': (20262921000, 20263021000)}
MODEL_PARAMETERS = {'actor': 5_890, 'critic': 5_825, 'total': 11_715}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same(a, b):
    """Exact equality with type discipline (1e-3 == 0.001 is fine; 0 vs 0.0 for a float knob is not)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def config_mismatches(cfg, block):
    """[(key, found, planned)] for every difference between a main-run config and PLAN.md; [] when it matches."""
    out = []
    for key, planned in PLAN_CONSTANTS.items():
        if key not in cfg or not _same(cfg[key], planned):
            out.append((key, cfg.get(key, '<missing>'), planned))
    if cfg.get('block') != block or isinstance(cfg.get('block'), bool):
        out.append(('block', cfg.get('block'), block))
    rule = cfg.get('rule') or {}
    if rule.get('name') != RULE_NAME or not _same(rule.get('parameters'), RULE_PARAMETERS):
        out.append(('rule', rule, {'name': RULE_NAME, 'parameters': RULE_PARAMETERS}))
    extra = set(cfg) - CONFIG_KEYS
    if extra:
        out.append(('unplanned keys', sorted(extra), []))
    return out


def assert_config(cfg, block):
    bad = config_mismatches(cfg, block)
    if bad:
        raise SystemExit(f'config(block {block}) differs from PLAN.md: {bad}')


def listed_scripts(plan_text):
    """{file: sha256} from PLAN.md's table rows of the form | `name.py` | `<64 hex>` |."""
    rows = re.findall(r'^\|\s*`([A-Za-z0-9_.]+\.py)`\s*\|\s*`([0-9a-f]{64})`\s*\|', plan_text, flags=re.M)
    out = {}
    for name, digest in rows:
        if name in out and out[name] != digest:
            raise SystemExit(f'PLAN.md lists {name} twice with different hashes')
        out[name] = digest
    return out


def script_mismatches(listed, here=HERE):
    """[(file, current, listed)] for scripts that differ from PLAN.md's listing, or are missing from it."""
    bad = []
    if set(listed) != set(SCRIPTS):
        bad.append(('listed set', sorted(listed), sorted(SCRIPTS)))
    for name in SCRIPTS:
        current = sha256(here / name) if (here / name).exists() else None
        if listed.get(name) != current:
            bad.append((name, current, listed.get(name)))
    return bad


def bound_plan(here=HERE):
    """The hashed plan, or SystemExit: PLAN.md exists, PLAN.sha256 names its SHA-256, and every script listed in
    PLAN.md has exactly the listed SHA-256 now. Returns {'sha256': ..., 'scripts': {file: sha256}}."""
    plan, hash_file = here / 'PLAN.md', here / 'PLAN.sha256'
    if not plan.exists():
        raise SystemExit('PLAN.md does not exist')
    digest = sha256(plan)
    if not hash_file.exists() or hash_file.read_text().split()[0] != digest:
        raise SystemExit('PLAN.md is not hashed (PLAN.sha256 missing or different)')
    listed = listed_scripts(plan.read_text())
    bad = script_mismatches(listed, here)
    if bad:
        raise SystemExit(f'scripts differ from the SHA-256 listed in PLAN.md: {bad}')
    return {'sha256': digest, 'scripts': listed}


def module_mismatches(runner, p4):
    """Module-level constants of runner.py / the patched p4_train that PLAN.md states."""
    bad = []
    checks = {'BLOCKS': (tuple(runner.BLOCKS), BLOCKS), 'STRIDE': (runner.STRIDE, STRIDE),
              'p4.BLOCK_STRIDE': (p4.BLOCK_STRIDE, STRIDE), 'EXPECTED_TREE': (runner.EXPECTED_TREE, EXPECTED_TREE),
              'BASELINES_SHA256': (runner.BASELINES_SHA256, BASELINES_SHA256),
              'BUDGET': (runner.BUDGET, PLAN_CONSTANTS['budget_gate_decisions']),
              'GRID': (list(runner.GRID), PLAN_CONSTANTS['grid']),
              'gate_core.FEATURES': (list(runner.gc.FEATURES), list(FEATURES))}
    for name, (lo, hi) in NAMESPACES.items():
        space = runner.GATE_NAMESPACES[name]
        checks[f'namespace {name}'] = ((space['start'], space['end']), (lo, hi))
    for key, (found, planned) in checks.items():
        if not _same(found, planned):
            bad.append((key, found, planned))
    state = runner.torch.get_rng_state()                 # build_model seeds the global torch RNG: restore it
    try:
        counts = runner.build_model(runner.STUDY_MAIN, 0).parameter_counts()
    finally:
        runner.torch.set_rng_state(state)
    if counts != MODEL_PARAMETERS:
        bad.append(('model parameters', counts, MODEL_PARAMETERS))
    return bad


def main(argv):
    sys.path.insert(0, str(HERE))
    import runner                                                     # noqa: E402  (imports idea_dq, unchanged)
    import p4_train as p4                                             # noqa: E402
    cfg = runner.config(0)
    print(json.dumps(cfg, indent=1, default=str))
    problems = {'config(0)': config_mismatches(cfg, 0), 'module constants': module_mismatches(runner, p4)}
    for b in BLOCKS:
        bad = config_mismatches(runner.config(b), b)
        if bad:
            problems[f'config({b})'] = bad
    tree = p4.live_copy_tree()
    if tree != EXPECTED_TREE:
        problems['idea_dq tree'] = [(tree, EXPECTED_TREE)]
    listed = listed_scripts(PLAN.read_text()) if PLAN.exists() else {}
    problems['scripts vs PLAN.md'] = script_mismatches(listed) if PLAN.exists() else [('PLAN.md', 'missing', '')]
    hashed = PLAN.exists() and PLAN_HASH_FILE.exists() and PLAN_HASH_FILE.read_text().split()[0] == sha256(PLAN)
    status = {'plan_sha256': sha256(PLAN) if PLAN.exists() else None, 'plan_hashed': hashed}
    if hashed and cfg['plan_sha256'] != status['plan_sha256']:
        problems['config plan_sha256'] = [(cfg['plan_sha256'], status['plan_sha256'])]
    if listed and any(cfg['code_sha256'].get(k) != v for k, v in listed.items()):
        problems['config code_sha256 vs PLAN.md'] = [(cfg['code_sha256'], listed)]
    if '--precheck' in argv:
        path = Path(argv[argv.index('--precheck') + 1])
        pre = json.loads(path.read_text())
        fresh = json.loads(json.dumps(cfg, default=str))
        rows = []
        if pre.get('config_block0') != fresh:
            rows.append(('config_block0', 'differs from a fresh config(0)'))
        if not pre.get('passed'):
            rows.append(('passed', pre.get('passed')))
        if not hashed or pre.get('plan_sha256') != status['plan_sha256']:
            rows.append(('plan_sha256', pre.get('plan_sha256'), status['plan_sha256'], 'hashed' if hashed else
                         'PLAN.md is not hashed'))
        if listed and pre.get('code_sha256') != listed:
            rows.append(('code_sha256', pre.get('code_sha256'), listed))
        problems[f'precheck {path.name}'] = rows
        status['precheck'] = str(path)
    problems = {k: v for k, v in problems.items() if v}
    status['ok'] = not problems
    status['problems'] = problems
    print(json.dumps(status, indent=1, default=str))
    return 0 if status['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
