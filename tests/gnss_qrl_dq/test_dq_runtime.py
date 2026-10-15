"""DQ isolation: namespaces, the approval gate, the write guard and the E1 modules DQ reuses (DQ protocol section 2)."""
from __future__ import annotations

import datetime
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_runtime as dq                                                                          # noqa: E402
import p4_train as p4                                                                            # noqa: E402

E1_ROOT = ROOT.parent / 'idea_e1'


def test_dq_namespaces_follow_p3_rule_after_e1_and_overlap_nothing():
    starts = sorted((s['start'], s['end'], n) for n, s in dq.DQ_NAMESPACES.items())
    assert starts[0][0] == p4.E1_NAMESPACES['e1_evaluation']['end']       # directly after E1's last namespace
    for (s0, e0, _), (s1, e1, _) in zip(starts, starts[1:]):
        assert e0 == s1 and e0 - s0 == dq.NAMESPACE_WIDTH
    runtime = dq.cell_runtime()                    # CellRuntime refuses any overlap with P3/P4/E1 frozen ranges
    frozen = dict(runtime.execution['seed_namespaces'], **p4.E1_NAMESPACES)
    for name, space in dq.DQ_NAMESPACES.items():
        assert runtime.namespaces[name] == space
        for other_name, other in frozen.items():
            assert not (space['start'] < other['end'] and other['start'] < space['end']), (name, other_name)


def test_dq_salt_differs_from_every_earlier_study_and_streams_differ():
    assert dq.DQ_STUDY.salt == (dq.DQ_SALT,) and dq.DQ_SALT != p4.E1_SALT
    assert p4.STUDIES['p4'].salt == ()
    e1 = p4.STUDIES['e1']
    for unit in range(1000):
        assert dq.DQ_STUDY.stream(unit, p4.INIT_STREAM) != e1.stream(unit, p4.INIT_STREAM)
        assert dq.DQ_STUDY.stream(unit, p4.INIT_STREAM) != p4.STUDIES['p4'].stream(unit, p4.INIT_STREAM)


def test_only_dq_development_is_open_and_every_other_namespace_refuses():
    assert dq.require_open('dq_development') == 'dq_development'
    # Phase (24 Sep 2026): Freeze A and B1 were approved and have run under tree a66f8979; any later tree must not
    # reopen them, so their approvals refuse against the live tree. B2 has no approval yet.
    for name in dq.DQ_NAMESPACES:
        if name == 'dq_development':
            continue
        stage = dq.GATED[name]
        if stage in ('a', 'b1') and dq.tree_of(dq.ROOT)[0] != 'a66f89793664b63bdf6e1ece843667972d30d142dc0fd5b8c2f684b7fbeecb56':
            assert dq.APPROVALS[stage]['path'].exists()
            with pytest.raises(dq.DQError, match='live code tree'):
                dq.require_open(name)
            continue
        if stage == 'b2':
            assert not dq.APPROVALS[stage]['path'].exists()
        with pytest.raises(dq.DQError, match='closed'):
            dq.require_open(name)
    for foreign in ('e1_evaluation', 'final_evaluation', 'main_training', 'gate'):
        with pytest.raises(dq.DQError, match='not a DQ namespace'):
            dq.require_open(foreign)


def test_calendar_cutoff_closes_even_the_development_namespace():
    after = datetime.datetime.fromisoformat(dq.CALENDAR_CUTOFF_UTC) + datetime.timedelta(seconds=1)
    with pytest.raises(dq.DQError, match='cut-off'):
        dq.require_open('dq_development', now=after)


def test_write_guard_refuses_every_symlinked_frozen_target(tmp_path):
    # every symlink under output/ leads into a frozen checkout; the guard must refuse writing through each
    links = [p for p in (ROOT / 'output').rglob('*') if p.is_symlink()] + \
            [p for p in (ROOT / 'output').iterdir() if p.is_symlink()]
    assert links, 'the DQ output layout must contain the read-only symlinks'
    for link in set(links):
        with pytest.raises(dq.DQError, match='outside'):
            dq.claim_output(link / 'dq_probe_never_written.json')
        assert not (link / 'dq_probe_never_written.json').exists()


def test_write_guard_refuses_a_symlink_planted_inside_the_dq_root(tmp_path):
    root = tmp_path / 'dqroot'
    root.mkdir()
    outside = tmp_path / 'frozen'
    outside.mkdir()
    (root / 'escape').symlink_to(outside)
    with pytest.raises(dq.DQError, match='outside'):
        dq.claim_output(root / 'escape' / 'x.json', test_root=root)
    assert not (outside / 'x.json').exists()


def test_write_guard_never_overwrites(tmp_path):
    path = dq.write_json(tmp_path / 'a' / 'x.json', {'a': 1}, test_root=tmp_path)
    assert path.read_text().strip().startswith('{')
    with pytest.raises(dq.DQError, match='exists'):
        dq.write_json(tmp_path / 'a' / 'x.json', {'a': 2}, test_root=tmp_path)
    assert '"a": 1' in path.read_text()


def test_write_guard_refuses_other_repository_paths(tmp_path):
    for target in (ROOT / 'docs' / 'x.json', ROOT / 'output' / 'x.json', ROOT / 'output' / 'study' / 'gnss_qrl_v1' / 'x.json',
                   dq.OUTPUT_ROOT):
        with pytest.raises(dq.DQError):
            dq.claim_output(target)


@pytest.mark.parametrize('relative', dq.E1_REUSED_MODULES)
def test_reused_e1_modules_are_byte_identical_to_the_approved_e1_tree(relative):
    assert dq.sha256_file(ROOT / relative) == dq.sha256_file(E1_ROOT / relative)


def test_watched_trees_are_their_approved_trees():
    record = dq.watch_trees()
    assert set(record) == {'idea', 'idea_e1', 'idea_p4', 'idea_e1t'}
    assert all(r['unchanged'] for r in record.values())
