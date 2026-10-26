"""POST-RESULT descriptive read-out of LONG and LONG-HQ (not part of either PLAN.md; written after long_results.json was read).

Why: the planned read-out showed C-strong's mean J falling from 30,000 to 300,000 decisions. This script lists, per run,
what the policies did at every checkpoint, so the manuscript can describe the decline without a test: timely-credited
reports, weighted J, and the mean policy entropy logged by the PPO updates. No interval, test or selection is computed;
every per-block value is written out, so no conclusion rests on the 0.01 cut used for the counts below.

    python long_descriptive_post.py      # writes long_descriptive_post.json
"""
import json
from pathlib import Path

import numpy as np

from long_analysis import HERE, HQ_DIR, BLOCKS, GRID, curve

ARMS = (('mlp', 'C-strong', HERE), ('bottleneck', 'C-matched', HERE), ('quantum', 'HQ', HQ_DIR))
NO_TIMELY = 0.01          # timely-credited reports per episode at or below which a checkpoint is listed as 'none'


def entropies(path):
    """(policy decisions so far, mean policy entropy in nats) for every PPO update of a run."""
    out = []
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        if e.get('kind') == 'update':
            out.append((e['cumulative'].get('policy_decisions'), e['metrics']['entropy']))
    return out


def main():
    res = {'note': 'post-result descriptive read-out, not part of either plan; no test or interval',
           'no_timely_cut': NO_TIMELY, 'arms': {}}
    for arm, role, root in ARMS:
        blocks = {}
        for b in BLOCKS:
            f = root / 'runs' / f'{arm}__b{b}' / 'run' / arm / 'metrics.jsonl'
            c, done = curve(f)
            assert done and max(c) == 300000, (arm, b)
            J = [c[p][0] for p in GRID]
            tp = [c[p][1] for p in GRID]
            later_none = [p for i, p in enumerate(GRID) if i > 0 and all(t <= NO_TIMELY for t in tp[i:])]
            ent = entropies(f)
            blocks[b] = {'J': J, 'timely': tp, 'peak_J': max(J), 'peak_at': GRID[int(np.argmax(J))],
                         'final_J': J[-1], 'final_timely': tp[-1],
                         'no_timely_from': later_none[0] if later_none else None,
                         'entropy_first_update': ent[0][1], 'entropy_last_update': ent[-1][1]}
        fam_tp = [float(np.mean([blocks[b]['timely'][i] for b in BLOCKS])) for i in range(len(GRID))]
        fam_J = [float(np.mean([blocks[b]['J'][i] for b in BLOCKS])) for i in range(len(GRID))]
        res['arms'][role] = {
            'blocks': blocks,
            'n_final_no_timely': sum(blocks[b]['final_timely'] <= NO_TIMELY for b in BLOCKS),
            'no_timely_from': {b: blocks[b]['no_timely_from'] for b in BLOCKS},
            'median_entropy_last_update': float(np.median([blocks[b]['entropy_last_update'] for b in BLOCKS])),
            'median_entropy_first_update': float(np.median([blocks[b]['entropy_first_update'] for b in BLOCKS])),
            'family_mean_timely_max': max(fam_tp), 'family_mean_timely_max_at': GRID[int(np.argmax(fam_tp))],
            'family_mean_J_max': max(fam_J), 'family_mean_J_max_at': GRID[int(np.argmax(fam_J))]}
    best = max(((v['blocks'][b]['peak_J'], role, b, v['blocks'][b]['peak_at']) for role, v in res['arms'].items()
                for b in BLOCKS))
    role, b = best[1], best[2]
    i = GRID.index(best[3])
    res['highest_single_block_any_checkpoint'] = {'J': best[0], 'role': role, 'block': b, 'decisions': best[3],
                                                  'timely': res['arms'][role]['blocks'][b]['timely'][i],
                                                  'final_J': res['arms'][role]['blocks'][b]['final_J']}
    (HERE / 'long_descriptive_post.json').write_text(json.dumps(res, indent=1))
    for role, v in res['arms'].items():
        print(role, 'final no-timely', v['n_final_no_timely'], v['no_timely_from'], 'entropy first/last (median)',
              round(v['median_entropy_first_update'], 2), round(v['median_entropy_last_update'], 2),
              'family timely max', round(v['family_mean_timely_max'], 3), '@', v['family_mean_timely_max_at'],
              'family J max', round(v['family_mean_J_max'], 4), '@', v['family_mean_J_max_at'])
        print('   final timely', [round(v['blocks'][b]['final_timely'], 2) for b in BLOCKS],
              'peaks', [(round(v['blocks'][b]['peak_J'], 3), v['blocks'][b]['peak_at']) for b in BLOCKS])
    print('highest single block', res['highest_single_block_any_checkpoint'])


if __name__ == '__main__':
    main()
