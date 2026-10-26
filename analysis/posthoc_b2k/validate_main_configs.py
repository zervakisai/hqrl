"""Review M6: re-validates the 72 main configurations at the SELECTED learning rates (selection.json), from a separate
script. No training, no run, no outcome read; writes one JSON file into this folder and never overwrites one.

    python -B validate_main_configs.py      # -> main_configs_validation_<placeholder | plan-xxxxxxxx>.json

The precheck's static check (c) validated the same 72 configurations at the pilot's middle candidates (7e-4, 1.4e-3);
`launch` builds them at the selected rates without re-validating. Here, for every block and arm:
  - runner.config_main(arm, block, lr from selection.json, plan hash) passes E1's validator p4_train.validate_config;
  - it differs from dq_b2.config_for(arm, block, STUDY_B2) only in ppo.learning_rate, candidate, note and
    protocol_freeze_sha256;
  - ppo.learning_rate equals float(lr_str) exactly, entropy_coef is 0, candidate is e3_lr<lr_str>_ent0, and block,
    arms, budget, grid and evaluation suite are B2's.
Before PLAN.sha256 exists the plan hash is a placeholder ('f' * 64); after the lead hashes PLAN.md it is PLAN.sha256's
content (optional re-run; the protocol_freeze_sha256 value is a declared difference either way).
"""
import hashlib
import json
from pathlib import Path

import runner
from runner import dq_b2, p4

HERE = Path(__file__).resolve().parent
ALLOWED = {'ppo.learning_rate', 'candidate', 'note', 'protocol_freeze_sha256'}


def differing_keys(new, old):
    diffs = set()
    for k in set(new) | set(old):
        if k == 'ppo':
            diffs |= {f'ppo.{q}' for q in set(new['ppo']) | set(old['ppo']) if new['ppo'].get(q) != old['ppo'].get(q)}
        elif new.get(k) != old.get(k):
            diffs.add(k)
    return diffs


def main():
    sel = json.loads((HERE / 'selection.json').read_text())
    if not sel.get('ready'):
        raise SystemExit('selection.json is not ready')
    plan_file = HERE / 'PLAN.sha256'
    plan = plan_file.read_text().split()[0] if plan_file.exists() else None
    label = f'plan-{plan[:8]}' if plan else 'placeholder'
    out_path = HERE / f'main_configs_validation_{label}.json'
    if out_path.exists():
        raise SystemExit(f'{out_path.name} exists (never overwritten)')
    rows, all_ok = [], True
    for b in dq_b2.BLOCKS:
        for tag, arm in runner.SETS.items():
            lr_str = sel['arms'][arm]['lr_str']
            lr = sel['arms'][arm]['lr']
            new = runner.config_main(arm, b, lr, plan=plan or 'f' * 64)
            old = dq_b2.config_for(arm, b, dq_b2.STUDY_B2, freeze_sha256='0' * 64)
            diffs = differing_keys(new, old)
            try:
                p4.validate_config(new)
                validator = 'passed'
            except Exception as exc:                                     # recorded, makes the check fail
                validator = f'{type(exc).__name__}: {exc}'
            checks = {
                'validator_passed': validator == 'passed',
                'only_declared_keys_differ': diffs <= ALLOWED and 'ppo.learning_rate' in diffs,
                'lr_equals_lr_str': float(lr_str) == lr == new['ppo']['learning_rate'],
                'entropy_zero': new['ppo']['entropy_coef'] == 0.0,
                'candidate_name': new['candidate'] == f'e3_lr{lr_str}_ent0',
                'b2_block_arm_budget_grid_evaluation': (new['block'] == b and new['arms'] == [arm] and
                                                        new['total_policy_decisions'] == dq_b2.BUDGET and
                                                        new['checkpoint_decisions'] == list(dq_b2.GRID) and
                                                        new['evaluation'] == old['evaluation'] and
                                                        new['study'] == 'dq_b2' and new['role'] == 'dq_b2')}
            ok = all(checks.values())
            all_ok = all_ok and ok
            rows.append({'run': f'{tag}_{arm}__b{b}', 'lr_str': lr_str, 'differing_keys': sorted(diffs),
                         'validator': validator, 'checks': checks, 'ok': ok,
                         'config_sha256': hashlib.sha256(json.dumps(new, sort_keys=True, default=float).encode())
                         .hexdigest()})
    out = {'utc': p4.utc(), 'runner_sha256': runner.RUNNER_SHA256,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'plan_hash_used': plan or 'placeholder ' + 'f' * 64, 'selection': sel['arms'],
           'b2_code_tree_live_copy': p4.live_copy_tree(), 'configs': len(rows), 'all_ok': all_ok, 'rows': rows}
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('utc', 'runner_sha256', 'plan_hash_used', 'configs', 'all_ok',
                                          'b2_code_tree_live_copy')}, indent=1))
    print('differing keys (union):', sorted(set().union(*[set(r['differing_keys']) for r in rows])))
    if not all_ok:
        raise SystemExit('validation FAILED: ' + ', '.join(r['run'] for r in rows if not r['ok']))


if __name__ == '__main__':
    main()
