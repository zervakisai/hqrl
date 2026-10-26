"""Paired rule reference for B2 / LONG / E4 (see PLAN_DRAFT.md): the tuned rule on exactly the 80 evaluation episodes
that every B2, LONG, LONG-HQ and E4 checkpoint evaluation flies (namespace dq_e2_evaluation, 8 core cells x
realisations 0-9, seeds p4.evaluation_seed). Imports the idea_dq tree unchanged; one process; writes rule_reference.json
once (never overwritten).

    python -B rule_reference.py run       # 80 rule episodes + determinism re-run + stored-E1-row reproduction + pairing
    python -B rule_reference.py verify    # re-checks the written file against every stored B2/LONG/LONG-HQ evaluation

The rule is `uncertainty_fix`, sigma 1.5 m, taken by hash through e1_baselines.p3_rule() (which refuses if
gnss_policies.py or the comparator protocol moved), run by e1_baselines.run_policy_episode: the same cell and episode
construction (run_p3_cells.prepare_cell / cell_episode_parts) as every learned episode. The rule draws no action
stream, so its 80 rows pair with every evaluation point of every learned run (same cell, seed and CRN group).
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e4lib                                                                                      # noqa: E402,F401
import dq_b2                                                                                      # noqa: E402
import e1_baselines as e1b                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402

OUT = HERE / 'rule_reference.json'
E1_BASELINES = Path('/home/konstantinos/Documents/ChatGPT/idea_e1/output/study/gnss_qrl_v1/e1/e1_campaign_v2/'
                    'baselines/baselines.json')
LONGHQ_DIR = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_longhq')
ROW_KEYS = ('cell_id', 'seed', 'crn_sha256', 'J', 'tp_timely', 'tp_spatial', 'fp_spatial', 'return_ok',
            'fixes_attempted', 'decisions', 'termination_reason', 'rejections', 'duplicates', 'elapsed_s')


def episodes_in_order(rt):
    namespace, realizations = dq_b2.STUDY_B2.suites[dq_b2.STUDY_B2.evaluation_suites[0]]
    assert namespace == 'dq_e2_evaluation' and tuple(realizations) == tuple(range(10)), (namespace, realizations)
    return namespace, [(cell, r, p4.evaluation_seed(rt, namespace, cell, r)) for cell in rt.cells for r in realizations]


def stored_evaluation_files():
    files = {'B2': sorted(glob.glob(str(e4lib.B2_RUNS / '*__b*' / 'run' / '*' / 'metrics.jsonl'))),
             'LONG': sorted(glob.glob(str(e4lib.LONG_DIR / 'runs' / '*__b*' / 'run' / '*' / 'metrics.jsonl'))),
             'LONG-HQ': sorted(glob.glob(str(LONGHQ_DIR / 'runs' / '*__b*' / 'run' / '*' / 'metrics.jsonl')))}
    return files


def pairing_check(pairs):
    """Every stored evaluation event of B2, LONG and LONG-HQ must list exactly these (cell_id, seed) in this order."""
    out = {}
    for name, files in stored_evaluation_files().items():
        events = mismatched = 0
        for f in files:
            for line in Path(f).read_text().splitlines():
                if '"kind": "evaluation"' not in line:
                    continue
                e = json.loads(line)
                if e.get('kind') != 'evaluation':
                    continue
                events += 1
                got = [(x['cell_id'], x['seed']) for x in e['results']['sample']['episodes']]
                mismatched += got != pairs
        out[name] = {'runs': len(files), 'evaluation_events': events, 'mismatched_events': mismatched}
    return out


def run():
    if OUT.exists():
        raise SystemExit(f'{OUT} exists; never overwritten')
    import torch
    torch.set_num_threads(1)
    tree = p4.live_copy_tree()
    if tree != e4lib.EXPECTED_TREE:
        raise SystemExit(f'idea_dq live tree {tree[:12]} is not {e4lib.EXPECTED_TREE[:12]}')
    t0 = time.perf_counter()
    rule = e1b.p3_rule()
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    namespace, plan = episodes_in_order(rt)
    rows = []
    for cell, r, seed in plan:
        row = e1b.run_policy_episode(rt, cell, namespace, seed, 'p3_rule',
                                     f'e4-rule-reference-c{cell["cell_index"]}-r{r}', rule=rule)
        rows.append(dict({k: row[k] for k in ROW_KEYS}, cell_index=cell['cell_index'], realization=r,
                         wall_s=row['wall_s']))
        print(cell['cell_id'], r, seed, round(row['J'], 4), round(row['wall_s'], 1), flush=True)
    per_cell, weighted = p4.aggregate_evaluation(rt, rows)
    # determinism: the first target-present and the first no-human episode flown again
    again = []
    for i in (0, 10):
        cell, r, seed = plan[i]
        row = e1b.run_policy_episode(rt, cell, namespace, seed, 'p3_rule', f'e4-rule-reference-again-{i}', rule=rule)
        again.append({'index': i, 'identical': all(row[k] == rows[i][k] for k in ROW_KEYS if k != 'elapsed_s')
                      and row['elapsed_s'] == rows[i]['elapsed_s']})
    # the same code path reproduces two stored E1 rule rows (e1_evaluation suite of the E1 campaign)
    stored = json.loads(E1_BASELINES.read_text())['results']['p3_rule/e1_evaluation']['episodes']
    rt_e1 = p4.CellRuntime(dq_b2.ROOT / 'output/study/gnss_qrl_v1/claude/p3_freeze_v1/p3_frozen_configuration.json',
                           dq_b2.ROOT / 'output/study/gnss_qrl_v1/claude/p3_execution_manifest_v1/p3_execution_manifest.json',
                           dq_b2.ROOT / 'output/training/time_model_village_v1/20260917T074923Z/time_model.json',
                           study=p4.STUDIES['e1'])
    cells = {c['cell_id']: c for c in rt_e1.cells}
    e1_check = []
    for i in (0, 10):
        s = stored[i]
        row = e1b.run_policy_episode(rt_e1, cells[s['cell_id']], 'e1_evaluation', s['seed'], 'p3_rule',
                                     f'e4-rule-e1-row-{i}', rule=rule)
        keys = [k for k in ('cell_id', 'seed', 'crn_sha256', 'J', 'tp_timely', 'fp_spatial', 'fixes_attempted',
                            'decisions') if k in s]
        e1_check.append({'index': i, 'cell_id': s['cell_id'], 'keys': keys,
                         'identical': all(row[k] == s[k] for k in keys)})
    pairs = [(row['cell_id'], row['seed']) for row in rows]
    pairing = pairing_check(pairs)
    passed = (all(a['identical'] for a in again) and all(c['identical'] for c in e1_check)
              and all(v['mismatched_events'] == 0 and v['evaluation_events'] > 0 for v in pairing.values()))
    payload = {
        'what': ('tuned rule (uncertainty_fix, sigma 1.5 m) on the 80 dq_e2_evaluation episodes of every B2/LONG/'
                 'LONG-HQ/E4 checkpoint evaluation; rows in the trainer evaluation order (cell order x realisation)'),
        'rule': rule, 'namespace': namespace, 'realizations': list(range(10)),
        'seed_rule': 'p4_train.evaluation_seed = p3_design.seed_for(dq_e2_evaluation, seed_cell_index, realisation)',
        'weighted_J': weighted, 'per_cell_J': per_cell,
        'cell_weights': {c['cell_id']: float(w) for c, w in zip(rt.cells, rt.probabilities)},
        'half_weighted_J': weighted / 2, 'episodes': rows,
        'checks': {'determinism_rerun': again, 'stored_e1_rule_rows_reproduced': e1_check,
                   'pairing_cell_id_seed_vs_stored_evaluations': pairing, 'passed': passed},
        'provenance': {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'idea_dq_tree': tree,
                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'e4lib_sha256': hashlib.sha256((HERE / 'e4lib.py').read_bytes()).hexdigest(),
                       'wall_s': time.perf_counter() - t0,
                       'mean_episode_wall_s': sum(r['wall_s'] for r in rows) / len(rows)}}
    with open(OUT, 'x') as handle:
        handle.write(json.dumps(payload, indent=1, default=float) + '\n')
    print(json.dumps({'weighted_J': weighted, 'per_cell_J': per_cell, 'checks': payload['checks'],
                      'wall_s': payload['provenance']['wall_s']}, indent=1, default=float))


def verify():
    payload = json.loads(OUT.read_text())
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    _, plan = episodes_in_order(rt)
    pairs = [(c['cell_id'], s) for c, _, s in plan]
    per_cell, weighted = p4.aggregate_evaluation(rt, payload['episodes'])
    ok = ([(r['cell_id'], r['seed']) for r in payload['episodes']] == pairs and weighted == payload['weighted_J']
          and per_cell == payload['per_cell_J'])
    pairing = pairing_check(pairs)
    print(json.dumps({'rows_and_aggregate_ok': ok, 'pairing': pairing, 'weighted_J': weighted}, indent=1))
    return ok and all(v['mismatched_events'] == 0 for v in pairing.values())


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'run':
        run()
    elif cmd == 'verify':
        raise SystemExit(0 if verify() else 1)
    else:
        raise SystemExit(__doc__)
