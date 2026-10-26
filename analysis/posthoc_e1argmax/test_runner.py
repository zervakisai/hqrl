"""Unit tests of runner.py (run by `runner.py test` and by the precheck). Read-only outside a temporary directory."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner                                                                                     # noqa: E402


def _row(**over):
    row = {'cell_id': 'd2__nominal_2700s__positive', 'seed': 123, 'J': 0.1234567890123, 'tp_timely': 0.75,
           'tp_spatial': 1, 'fp_spatial': 0, 'return_ok': True, 'fixes_attempted': 3, 'decisions': 40,
           'termination_reason': 'mission_complete'}
    row.update(over)
    return json.loads(json.dumps(row))


def _plan_dir(tmp, scripts=None, digest='ab' * 32, plan_hash=None):
    tmp = Path(tmp)
    for name in runner.SCRIPTS:
        (tmp / name).write_text(f'# {name}\n')
    hashes = {n: hashlib.sha256((tmp / n).read_bytes()).hexdigest() for n in runner.SCRIPTS}
    hashes.update(scripts or {})
    rows = '\n'.join(f'| `{n}` | `{h}` |' for n, h in hashes.items())
    (tmp / 'PLAN.md').write_text(f'# plan\n\n| file | sha256 |\n|---|---|\n{rows}\n\nThe policy table digest is '
                                 f'`{digest}`.\n')
    real = hashlib.sha256((tmp / 'PLAN.md').read_bytes()).hexdigest()
    (tmp / 'PLAN.sha256').write_text(f'{plan_hash or real}  PLAN.md\n')
    return real


class Binding(unittest.TestCase):
    def test_bound_plan_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = _plan_dir(tmp)
            plan = runner.bound_plan(Path(tmp))
            self.assertEqual(plan['sha256'], real)
            self.assertEqual(plan['policy_table_digest'], 'ab' * 32)
            self.assertEqual(set(plan['scripts']), set(runner.SCRIPTS))

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            _plan_dir(tmp)
            (Path(tmp) / 'analysis.py').write_text('# changed\n')
            with self.assertRaises(SystemExit):
                runner.bound_plan(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            _plan_dir(tmp, plan_hash='0' * 64)
            with self.assertRaises(SystemExit):
                runner.bound_plan(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            _plan_dir(tmp)
            (Path(tmp) / 'PLAN.sha256').unlink()
            with self.assertRaises(SystemExit):
                runner.bound_plan(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            _plan_dir(tmp)
            (Path(tmp) / 'PLAN.md').unlink()
            with self.assertRaises(SystemExit):
                runner.bound_plan(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:                 # a script missing from the table
            _plan_dir(tmp)
            text = (Path(tmp) / 'PLAN.md').read_text().replace('| `test_runner.py`', '| test_runner.py')
            (Path(tmp) / 'PLAN.md').write_text(text)
            (Path(tmp) / 'PLAN.sha256').write_text(hashlib.sha256(text.encode()).hexdigest() + '  PLAN.md\n')
            with self.assertRaises(SystemExit):
                runner.bound_plan(Path(tmp))

    def test_listed_scripts_parse(self):
        text = '| `runner.py` | `' + 'a' * 64 + '` |\n| `x` | `short` |\n'
        self.assertEqual(runner.listed_scripts(text), {'runner.py': 'a' * 64})


class Rows(unittest.TestCase):
    def test_identical(self):
        stored = [_row(seed=s) for s in range(3)]
        self.assertEqual(runner.compare_rows([dict(r) for r in stored], stored), [])

    def test_one_ulp(self):
        stored = [_row()]
        new = [dict(stored[0], J=math.nextafter(stored[0]['J'], 1.0))]
        diffs = runner.compare_rows(new, stored)
        self.assertEqual([d['field'] for d in diffs], ['J'])

    def test_type_and_fields_and_count(self):
        stored = [_row()]
        self.assertTrue(runner.compare_rows([dict(stored[0], return_ok=1)], stored))      # bool vs int
        self.assertTrue(runner.compare_rows([dict(stored[0], tp_spatial=1.0)], stored))   # int vs float
        short = dict(stored[0])
        del short['decisions']
        self.assertTrue(runner.compare_rows([short], stored))
        self.assertTrue(runner.compare_rows([], stored))

    def test_numpy_float_is_compared_as_json_stores_it(self):
        import numpy as np
        stored = [_row(J=0.1)]
        self.assertEqual(runner.compare_rows([dict(stored[0], J=np.float64(0.1))], stored), [])

    def test_digest(self):
        rows = [_row(seed=1), _row(seed=2)]
        self.assertEqual(runner.rows_digest(rows), runner.rows_digest([dict(r) for r in rows]))
        self.assertNotEqual(runner.rows_digest(rows), runner.rows_digest(rows[::-1]))
        self.assertNotEqual(runner.rows_digest(rows), runner.rows_digest([rows[0], _row(seed=2, J=0.2)]))

    def test_credited(self):
        self.assertEqual(runner.e1_report_credited(_row(tp_timely=1.5)), 1.5)
        self.assertEqual(runner.e1_report_credited(_row(tp_timely=1.5, return_ok=False)), 0.)


class Records(unittest.TestCase):
    def _entry_and_record(self):
        stored = [_row(seed=s) for s in range(80)]
        entry = {'job': 'mlp__lr3e-3_ent0__b14', 'family': 'mlp', 'block': 14,
                 'final_checkpoint': {'sha256': 'c' * 64}, 'stored': {'episodes': stored}}
        prov = {'idea_e1_tree': runner.EXPECTED_E1_TREE, 'idea_dq_tree': runner.EXPECTED_DQ_TREE,
                'scripts': {'runner.py': 'r'}}
        rec = {'version': runner.VERSION, 'kind': 'read_out', 'status': 'completed', 'mode': 'argmax',
               'job': entry['job'], 'family': 'mlp', 'block': 14, 'plan_sha256': 'p', 'start': prov, 'end': prov,
               'checkpoint': {'sha256': 'c' * 64}, 'episodes': [dict(r, J=0.5) for r in stored]}
        return entry, rec

    def test_valid(self):
        entry, rec = self._entry_and_record()
        self.assertEqual(runner.check_record(rec, entry, 'p', {'runner.py': 'r'}), [])

    def test_invalid(self):
        entry, rec = self._entry_and_record()
        for key, value in (('plan_sha256', 'q'), ('mode', 'sample'), ('status', 'infrastructure_failure'),
                           ('block', 15), ('checkpoint', {'sha256': 'd' * 64})):
            self.assertTrue(runner.check_record(dict(rec, **{key: value}), entry, 'p', {'runner.py': 'r'}), key)
        self.assertTrue(runner.check_record(dict(rec, end=dict(rec['end'], idea_e1_tree='x')), entry, 'p',
                                            {'runner.py': 'r'}))
        self.assertTrue(runner.check_record(dict(rec, start=dict(rec['start'], scripts={})), entry, 'p',
                                            {'runner.py': 'r'}))
        self.assertTrue(runner.check_record(dict(rec, episodes=rec['episodes'][::-1]), entry, 'p', {'runner.py': 'r'}))
        self.assertTrue(runner.check_record(dict(rec, episodes=rec['episodes'][:79]), entry, 'p', {'runner.py': 'r'}))
        self.assertTrue(runner.check_record('x', entry, 'p', {}))

    def test_read_record_unparsable(self):
        entry, rec = self._entry_and_record()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'r.json'
            path.write_text('{not json')
            self.assertIsNone(runner.read_record(path, entry, 'p', {'runner.py': 'r'}))
            runner.write_json_atomic(path, rec)
            self.assertIsNotNone(runner.read_record(path, entry, 'p', {'runner.py': 'r'}))
            self.assertEqual(sorted(os.listdir(tmp)), ['r.json'])                 # no temporary file left


class Helpers(unittest.TestCase):
    def test_lpt(self):
        self.assertEqual(runner.lpt_makespan([3, 3, 2, 2, 2], 2), 7)          # LPT, not the optimum 6
        self.assertEqual(runner.lpt_makespan([5], 4), 5)

    def test_light(self):
        self.assertNotIn('stored', runner.light({'job': 'j', 'stored': {'episodes': []}}))

    def test_files_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            since = time.time() - 1.0
            (Path(tmp) / 'a').write_text('x')
            self.assertEqual([Path(p).name for p in runner.files_touched_since([Path(tmp)], since)], ['a'])
            self.assertEqual(runner.files_touched_since([Path(tmp)], time.time() + 60), [])


class PolicyTable(unittest.TestCase):
    """Integration (read-only): the 48 policies re-prove through the frozen E1 code, deterministically."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = runner.e1_context()
        cls.table = runner.policy_table(cls.ctx)

    def test_48(self):
        t = self.table
        self.assertEqual(len(t), 48)
        self.assertEqual({(e['family'], e['candidate']) for e in t}, set(runner.EXPECTED_CANDIDATES.items()))
        self.assertEqual(sorted({e['block'] for e in t}), list(runner.BLOCKS))
        self.assertTrue(all(e['point_index'] == 10 for e in t))
        self.assertTrue(all(Path(e['final_checkpoint']['path']).name.startswith('final_') for e in t))
        self.assertTrue(all(len(e['stored']['episodes']) == 80 for e in t))

    def test_digest_stable(self):
        self.assertEqual(runner.table_digest(self.table), runner.table_digest(runner.policy_table(self.ctx)))

    def test_stored_order_is_the_rule_order(self):
        rule = self.ctx['baselines']['p3_rule/e1_evaluation']['episodes']
        keys = [(e['cell_id'], e['seed']) for e in rule]
        self.assertTrue(all([(x['cell_id'], x['seed']) for x in e['stored']['episodes']] == keys for e in self.table))


if __name__ == '__main__':
    unittest.main(verbosity=2)
