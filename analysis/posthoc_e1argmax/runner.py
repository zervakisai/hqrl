"""Post-hoc E1 argmax read-out: the 48 E1 final policies on the 80 e1_evaluation episodes (see PLAN.md).

Evaluation only: no training, no selection. The idea_e1 code tree (live hash 4a1f2e8e..., the tree E1 ran and its
approved freeze names) is imported unchanged, and every read-out goes through `p4_train.evaluate` itself, exactly as the
E1 trainer called it at the 30,000-decision point (point index 10, label e1-<arm>-b<block>-eval10), with
modes ['argmax'] (the main run) or ['sample'] (the precheck's reproduction of the stored E1 read-out). Writes only
inside this directory. Never writes into idea_e1 or idea_dq (sys.dont_write_bytecode; launch with python -B).

    <E1 python> -B runner.py test                               # unit tests + analysis.py --selftest
    <E1 python> -B runner.py precheck --workers 8 --tag _build  # before PLAN.md is hashed: cannot authorise a launch
    <E1 python> -B runner.py precheck --workers 8               # the precheck of record (needs PLAN.md hashed)
    nice -n 10 <E1 python> -B runner.py launch --workers 8      # the 48 argmax read-outs (resumable)

<E1 python> = /home/konstantinos/Documents/ChatGPT/idea_e1/.venv/bin/python, the interpreter E1 ran with
(output/study/gnss_qrl_v1/e1/run_logs/e1_chain.sh). CPU only: CUDA_VISIBLE_DEVICES='' is set before torch is imported
(the torch build is +cpu anyway), torch uses 1 thread per worker (the E1 config's torch_threads), and the process
lowers its own priority to nice >= 10, which every worker inherits.

Precheck (all must hold; precheck<tag>.json records everything):
  (a) both code trees at their expected hashes; the approved E1 freeze, campaign, baselines, selection and all 48
      evaluation jobs re-proven by the frozen E1 proof code (p4_train.require_approved_freeze, e1_study.load_campaign,
      e1_study.verify_selection_record, e1_study.proven_stage); baselines.json at its SHA-256 and the rule's weighted J
      of record re-derived by repr; the 48-policy table and its digest;
  (b) unit tests (test_runner.py) and analysis.py --selftest;
  (c) THE validity check: the stored SAMPLED read-out of every one of the 48 final policies re-run through this
      runner's own read-out path (`evaluate_policy`, mode 'sample'), every stored field of every one of the 3,840
      episodes equal by repr, and per-cell and weighted J equal by repr;
  (d) outcome-blind argmax probe: per family, block 10's final policy on the 8 realisation-0 episodes, argmax, run in
      two separate processes; only the wall time per episode and a SHA-256 of the 8 rows are kept (the rows are never
      written or printed); the two digests must agree. The same process also re-runs the same 8 episodes sampled
      (must equal the stored rows) to give the argmax / sampled time ratio (every cell is prepared first, so neither
      mode's timing holds the bank loading);
  (e) timing and the wall-time estimate of the main run; (f) memory (MemAvailable, max RSS per worker);
  (g) no file or directory under idea_e1 or idea_dq was created or modified during the precheck.

Launch refuses unless PLAN.md matches PLAN.sha256, every script has the SHA-256 PLAN.md lists, both code trees are at
their expected hashes, the 48-policy table re-proves with the digest PLAN.md lists, precheck.json (untagged) passed for
this plan and these scripts, and MemAvailable covers the workers. One launch at a time (flock on .launch.lock).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True                          # never write __pycache__ into idea_e1 / idea_dq
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''                 # before torch is imported, in this process and every child
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import resource
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
IDEA_E1 = Path('/home/konstantinos/Documents/ChatGPT/idea_e1')
IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq')
for _extra in (IDEA_E1 / 'src', IDEA_E1 / 'scripts', IDEA_E1 / 'scripts' / 'gnss_qrl_runtime'):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import torch                                                                                      # noqa: E402

import code_version                                                                               # noqa: E402
import e1_baselines                                                                               # noqa: E402
import e1_study                                                                                   # noqa: E402
import p4_train as p4                                                                             # noqa: E402
from aerial_qrl.training.smdp_ppo import file_sha256, load_checkpoint                             # noqa: E402

VERSION = 'e1-argmax-posthoc/1'
CAMPAIGN = IDEA_E1 / 'output/study/gnss_qrl_v1/e1/e1_campaign_v2'
BASELINES = CAMPAIGN / 'baselines' / 'baselines.json'
INTERPRETER_PREFIX = IDEA_E1 / '.venv'
# ------------------------------------------------------------------ DESIGN (stated in PLAN.md) -----------------------
EXPECTED_E1_TREE = '4a1f2e8e6a76cbc3ba286397139da9bb888c28f2e8a4bf84adb9f2494e46ddc5'
EXPECTED_DQ_TREE = 'ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952'
FREEZE_SHA256 = 'c80eb591ed5952de638c2bef4553608b8f25aefe1170f200936f882275467965'
CAMPAIGN_SHA256 = '76bc358d824567c0f413e946bccb5c6ab0fb1d84ef26d79f09459dd557f4dd56'
BASELINES_SHA256 = '00202039dda01d814146db77e85590996abda974b4e813bbb17c10d081572fd7'
RULE_J_OF_RECORD = 0.20564463190957744          # p3_rule/e1_evaluation weighted J in baselines.json
EXPECTED_CANDIDATES = {'quantum': 'lr3e-3_ent0', 'bottleneck': 'lr3e-3_ent0', 'mlp': 'lr3e-3_ent0',
                       'surrogate625': 'lr1e-3_ent0'}
FAMILIES = tuple(e1_study.FAMILIES)             # ('quantum', 'bottleneck', 'mlp', 'surrogate625')
BLOCKS = tuple(range(10, 22))
SUITE = 'e1_evaluation'
FINAL_POINT = 30_000                            # the last point of the E1 grid; Table 5's sampled read-out
MAIN_MODE = 'argmax'
PROBE_BLOCK, PROBE_REALIZATION = 10, 0
EPISODE_FIELDS = ('cell_id', 'seed', 'J', 'tp_timely', 'tp_spatial', 'fp_spatial', 'return_ok', 'fixes_attempted',
                  'decisions', 'termination_reason')
GIB_PER_WORKER, GIB_RESERVE = 0.8, 2.0          # launch refuses unless MemAvailable >= workers * 0.8 + 2 GiB
MAX_ATTEMPTS = 2                                # an infrastructure failure is retried once (read-outs are deterministic)
# ------------------------------------------------------------------ END DESIGN ---------------------------------------
if EPISODE_FIELDS != tuple(e1_baselines.EPISODE_FIELDS) or FAMILIES != ('quantum', 'bottleneck', 'mlp', 'surrogate625') \
        or tuple(e1_study.STUDY.blocks['e1_evaluation']) != BLOCKS or p4.CHECKPOINT_GRID_30K[-1] != FINAL_POINT:
    raise SystemExit('runner.py design constants differ from the E1 code they describe')
SCRIPTS = ('runner.py', 'analysis.py', 'test_runner.py')
PLAN = HERE / 'PLAN.md'
PLAN_HASH_FILE = HERE / 'PLAN.sha256'
PLAN_SHA256 = hashlib.sha256(PLAN.read_bytes()).hexdigest() if PLAN.exists() else None
LOCK = HERE / '.launch.lock'
RECORDS = HERE / 'records'
ROLE = p4.ARM_ROLES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes(here=HERE):
    return {f: sha(here / f) for f in SCRIPTS if (here / f).exists()}


def tree(root):
    """The code-tree hash of `root` by idea_e1's own rule (scripts/code_version.py; identical in idea_dq)."""
    return code_version.tree_hash(code_version.file_hashes(Path(root)))


def canonical(obj):
    return p4.canonical_sha256(json.loads(json.dumps(obj, default=float)))


def write_json_atomic(path, obj):
    """Temporary file in the same directory, fsync, os.replace: a crash leaves the old file or the new one."""
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with open(tmp, 'w') as handle:
        handle.write(json.dumps(obj, indent=1, default=float))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def mem_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2 ** 20
    return None


def free_m():
    try:
        return subprocess.run(['free', '-m'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def json_row(row):
    """An episode row as JSON stores it (the E1 metrics went through json.dumps(default=float))."""
    return json.loads(json.dumps({k: row[k] for k in EPISODE_FIELDS}, default=float))


def rows_digest(rows):
    return hashlib.sha256(json.dumps([json_row(r) for r in rows], sort_keys=True).encode()).hexdigest()


def compare_rows(new, stored):
    """Every stored field of every episode, by repr, in order. Returns a list of differences (empty = identical)."""
    diffs = []
    if len(new) != len(stored):
        diffs.append({'what': 'episode count', 'new': len(new), 'stored': len(stored)})
    for i, (a, b) in enumerate(zip(new, stored)):
        a = json.loads(json.dumps(a, default=float))
        if set(a) != set(b):
            diffs.append({'episode': i, 'what': 'fields', 'new': sorted(a), 'stored': sorted(b)})
        for k in sorted(set(a) | set(b)):
            if repr(a.get(k)) != repr(b.get(k)) or type(a.get(k)) is not type(b.get(k)):
                diffs.append({'episode': i, 'field': k, 'new': repr(a.get(k)), 'stored': repr(b.get(k))})
    return diffs


# ------------------------------------------------------------------ binding --------------------------------------------

def listed_scripts(plan_text):
    """{script: sha256} from the PLAN.md binding table rows  | `name` | `64-hex` |."""
    return {m.group(1): m.group(2) for m in re.finditer(r'^\|\s*`([\w.]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|',
                                                        plan_text, flags=re.M)}


def listed_table_digest(plan_text):
    m = re.search(r'policy table digest[^`]*`([0-9a-f]{64})`', plan_text)
    return m.group(1) if m else None


def bound_plan(here=HERE):
    """PLAN.md hashed (PLAN.sha256 names its SHA-256) and every script at the SHA-256 PLAN.md lists; else SystemExit."""
    plan, hash_file = here / 'PLAN.md', here / 'PLAN.sha256'
    if not plan.exists():
        raise SystemExit('PLAN.md does not exist: refused (PLAN_DRAFT.md is not a plan)')
    if not hash_file.exists():
        raise SystemExit('PLAN.sha256 does not exist: PLAN.md is not hashed: refused')
    digest = sha(plan)
    named = hash_file.read_text().split()
    if not named or named[0] != digest:
        raise SystemExit(f'PLAN.sha256 names {named[:1]}, PLAN.md is {digest}: refused')
    text = plan.read_text()
    listed = listed_scripts(text)
    if set(listed) != set(SCRIPTS):
        raise SystemExit(f'PLAN.md lists scripts {sorted(listed)}, expected {list(SCRIPTS)}: refused')
    current = code_hashes(here)
    bad = {k: (current.get(k), v) for k, v in listed.items() if current.get(k) != v}
    if bad:
        raise SystemExit(f'scripts differ from the SHA-256 PLAN.md lists: {bad}: refused')
    return {'sha256': digest, 'scripts': listed, 'policy_table_digest': listed_table_digest(text)}


# ------------------------------------------------------------------ environment ----------------------------------------

def module_origins():
    """Where every loaded module under ~/Documents/ChatGPT came from: only idea_e1, this folder or the venv packages."""
    base = Path('/home/konstantinos/Documents/ChatGPT')
    allowed = (IDEA_E1, HERE, base / 'idea' / '.venv')
    foreign = {}
    for name, module in list(sys.modules.items()):
        f = getattr(module, '__file__', None)
        if not f:
            continue
        p = Path(f).resolve()
        if p.is_relative_to(base) and not any(p.is_relative_to(a) for a in allowed):
            foreign[name] = str(p)
    import aerial_qrl
    import fast_mission
    import run_p3_cells
    key = {'p4_train': p4.__file__, 'e1_study': e1_study.__file__, 'fast_mission': fast_mission.__file__,
           'run_p3_cells': run_p3_cells.__file__, 'aerial_qrl': aerial_qrl.__file__,
           'code_version': code_version.__file__}
    return {'key_modules': key, 'key_modules_in_idea_e1': all(Path(v).resolve().is_relative_to(IDEA_E1)
                                                              for v in key.values()),
            'foreign_modules': foreign}


def environment(require_nice=True):
    import numpy
    nice = os.nice(0)
    env = {'executable': sys.executable, 'prefix': sys.prefix, 'python': sys.version.split()[0],
           'e1_interpreter': Path(sys.prefix).resolve() == INTERPRETER_PREFIX.resolve(),
           'dont_write_bytecode': bool(sys.dont_write_bytecode), 'flag_B': bool(sys.flags.dont_write_bytecode),
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cuda_available': torch.cuda.is_available(),
           'torch': torch.__version__, 'numpy': numpy.__version__, 'torch_threads': torch.get_num_threads(),
           'nice': nice, 'cpu_count': os.cpu_count()}
    problems = []
    if not env['e1_interpreter']:
        problems.append(f'not the E1 interpreter ({sys.prefix}; expected {INTERPRETER_PREFIX})')
    if not env['flag_B']:
        problems.append('python was not started with -B')
    if env['cuda_visible_devices'] != '' or env['cuda_available']:
        problems.append('CUDA is visible')
    if env['torch'] != '2.14.0+cpu':
        problems.append(f'torch {env["torch"]} is not the E1 run manifests\' 2.14.0+cpu')
    if require_nice and nice < 10:
        problems.append(f'nice {nice} < 10')
    env['problems'] = problems
    return env


def raise_priority_floor():
    """Lower this process to nice >= 10 (children inherit it)."""
    current = os.nice(0)
    if current < 10:
        os.nice(10 - current)
    return os.nice(0)


def provenance():
    return {'utc': p4.utc(), 'idea_e1_tree': tree(IDEA_E1), 'idea_dq_tree': tree(IDEA_DQ), 'scripts': code_hashes(),
            'plan_sha256_on_disk': sha(PLAN) if PLAN.exists() else None}


def trees_ok(prov):
    return prov['idea_e1_tree'] == EXPECTED_E1_TREE and prov['idea_dq_tree'] == EXPECTED_DQ_TREE


# ------------------------------------------------------------------ the 48 policies ------------------------------------

def e1_context():
    """The approved E1 freeze, its campaign, baselines, selection and evaluation stage, proven by E1's own code."""
    _, freeze_sha, record = p4.require_approved_freeze(study=e1_study.STUDY)
    if freeze_sha != FREEZE_SHA256 or record['copy_code_tree_sha256'] != EXPECTED_E1_TREE:
        raise SystemExit(f'the approved E1 freeze is {freeze_sha} / tree {record["copy_code_tree_sha256"]}: refused')
    campaign = e1_study.load_campaign(CAMPAIGN, record)
    if campaign['campaign_sha256'] != CAMPAIGN_SHA256:
        raise SystemExit(f'campaign {campaign["campaign_sha256"]} is not {CAMPAIGN_SHA256}: refused')
    rt = e1_study.runtime()
    baselines = e1_study.require_baselines(CAMPAIGN, record, rt)
    selection = json.loads((CAMPAIGN / 'tuning' / 'selection.json').read_text())
    e1_study.verify_selection_record(selection, freeze_sha, CAMPAIGN, None, rt)
    chosen = e1_study.selected_candidates(selection)
    if chosen != EXPECTED_CANDIDATES:
        raise SystemExit(f'the E1 selection is {chosen}, not {EXPECTED_CANDIDATES}: refused')
    proven = e1_study.proven_stage(CAMPAIGN / 'evaluation', 'evaluation', freeze_sha, record, rt, selection)
    return {'freeze_sha256': freeze_sha, 'record': record, 'campaign': campaign, 'runtime': rt,
            'baselines': baselines, 'selection': selection, 'chosen': chosen, 'proven': proven}


def policy_table(ctx):
    """The 48 E1 selected final policies, in family order then block order, each with its stored sampled read-out at
    the 30,000-decision point (the episodes of Table 5) and the checkpoint it was read from."""
    import p4_tuning
    grid = list(p4.CHECKPOINT_GRID_30K)
    table = []
    for family in FAMILIES:
        for block in BLOCKS:
            candidate = ctx['chosen'][family]
            summary, derived = ctx['proven'][(family, candidate, block)]
            job = e1_study.job_name(family, candidate, block)
            job_dir = CAMPAIGN / 'evaluation' / job
            attempt, resolved = p4_tuning.resolve(job_dir)
            if summary['status'] != 'completed' or resolved != summary:
                raise SystemExit(f'{job}: not a completed, resolved E1 run: refused')
            arm_dir = attempt / 'run' / family
            config = json.loads((job_dir / 'config.json').read_text())
            final = [c for c in summary['checkpoints'] if c['final']]
            at_point = [c for c in summary['checkpoints'] if FINAL_POINT in c['points']]
            with open(arm_dir / 'metrics.jsonl') as handle:
                events = [json.loads(l) for l in handle]
            evaluation = [e for e in events if e['kind'] == 'evaluation' and FINAL_POINT in e['points']]
            if len(final) != 1 or len(at_point) != 1 or len(evaluation) != 1:
                raise SystemExit(f'{job}: not exactly one final checkpoint, one {FINAL_POINT} checkpoint and one '
                                 f'{FINAL_POINT} evaluation: refused')
            curve = derived['curve'][FINAL_POINT]
            point_index = grid.index(evaluation[0]['points'][0])
            if not (final[0]['policy_decisions'] == at_point[0]['policy_decisions'] == curve['policy_decisions']
                    == evaluation[0]['policy_decisions'] == summary['policy_decisions']):
                raise SystemExit(f'{job}: the final checkpoint is not the state read at {FINAL_POINT}: refused')
            if evaluation[0]['results']['sample']['weighted_J'] != curve['J'] or list(evaluation[0]['results']) != \
                    ['sample'] or config['evaluation'] != {'suite': SUITE, 'modes': ['sample']}:
                raise SystemExit(f'{job}: the stored read-out is not the sampled {SUITE} read-out: refused')
            table.append({
                'job': job, 'family': family, 'role': ROLE[family], 'candidate': candidate, 'block': block,
                'job_dir': str(job_dir), 'arm_dir': str(arm_dir), 'config_sha256': p4.canonical_sha256(config),
                'final_checkpoint': {k: final[0][k] for k in ('path', 'sha256', 'policy_decisions')},
                'point_checkpoint': {k: at_point[0][k] for k in ('path', 'sha256', 'points')},
                'point_index': point_index, 'e1_eval_wall_s': evaluation[0]['wall_s'],
                'stored': {'weighted_J': curve['J'], 'episodes': curve['episodes'],
                           'per_cell_J': evaluation[0]['results']['sample']['per_cell_J']}})
    if len(table) != 48 or len({e['job'] for e in table}) != 48:
        raise SystemExit('the policy table is not 48 distinct policies: refused')
    return table


def table_digest(table):
    """The SHA-256 PLAN.md binds: job, both checkpoint hashes, point index and the stored sampled weighted J."""
    return canonical([{'job': e['job'], 'final_sha256': e['final_checkpoint']['sha256'],
                       'point_sha256': e['point_checkpoint']['sha256'], 'point_index': e['point_index'],
                       'stored_weighted_J': repr(e['stored']['weighted_J'])} for e in table])


def light(entry):
    """The table entry a worker needs (the stored episodes stay in the parent)."""
    return {k: v for k, v in entry.items() if k != 'stored'}


# ------------------------------------------------------------------ one read-out (a worker) ----------------------------

def load_policy(entry, cfg):
    """The E1 final checkpoint, loaded as e1_timing.py loads it, after proving it is the recorded file and the very
    state E1 read at 30,000 decisions (equal, tensor for tensor, to the checkpoint written at that point)."""
    arm_dir = Path(entry['arm_dir'])
    path = arm_dir / entry['final_checkpoint']['path']
    digest = file_sha256(path)
    if digest != entry['final_checkpoint']['sha256']:
        raise RuntimeError(f'{path}: SHA-256 {digest} is not the one its run recorded')
    model, _, payload = load_checkpoint(path)
    model = model.to(dtype=p4.DTYPES[cfg['dtype']])
    meta = payload['metadata']
    expected = {'final': True, 'arm': entry['family'], 'block': entry['block'], 'tool': e1_study.STUDY.tool,
                'policy_decisions': entry['final_checkpoint']['policy_decisions'],
                'protocol_freeze_sha256': FREEZE_SHA256}
    bad = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if bad or payload['parameter_dtype'] != cfg['dtype'] or model.config['actor_head'] != entry['family'] or \
            canonical(payload['train_config']) != canonical(cfg):
        raise RuntimeError(f'{path}: metadata or training configuration is not this policy\'s: {bad}')
    other_path = arm_dir / entry['point_checkpoint']['path']
    if file_sha256(other_path) != entry['point_checkpoint']['sha256']:
        raise RuntimeError(f'{other_path}: SHA-256 is not the one its run recorded')
    other = torch.load(other_path, map_location='cpu', weights_only=True)['model_state_dict']
    state = payload['model_state_dict']
    same = set(state) == set(other) and all(torch.equal(state[k], other[k]) for k in state)
    if not same:
        raise RuntimeError(f'{path}: not the state of the {FINAL_POINT}-decision checkpoint')
    return model, {'path': str(path), 'sha256': digest, 'policy_decisions': meta['policy_decisions'],
                   'equal_to_point_checkpoint': same, 'point_checkpoint': str(other_path),
                   'model_config': dict(model.config), 'parameters': sum(p.numel() for p in model.parameters())}


def setup(entry):
    cfg = p4.validate_config(json.loads((Path(entry['job_dir']) / 'config.json').read_text()))
    if p4.canonical_sha256(cfg) != entry['config_sha256']:
        raise RuntimeError(f'{entry["job"]}: config.json is not the assigned configuration')
    torch.set_num_threads(cfg['torch_threads'])
    # exactly p4_train.main's construction of the runtime from the job configuration
    rt = p4.CellRuntime(IDEA_E1 / cfg['freeze'], IDEA_E1 / cfg['execution_manifest'], IDEA_E1 / cfg['time_model'],
                        study=p4.study_of(cfg))
    model, info = load_policy(entry, cfg)
    label = f'e1-{entry["family"]}-b{entry["block"]}-eval{entry["point_index"]:02d}'
    return cfg, rt, model, info, label


def _start(task, kind):
    entry = task['entry']
    rec = {'version': VERSION, 'kind': kind, 'job': entry['job'], 'family': entry['family'], 'role': entry['role'],
           'candidate': entry['candidate'], 'block': entry['block'], 'mode': task.get('mode'),
           'purpose': task['purpose'], 'attempt': task.get('attempt', 1), 'plan_sha256': task['plan_sha256'],
           'plan_sha256_worker': PLAN_SHA256, 'pid': os.getpid(), 'utc_start': p4.utc(), 'status': 'started'}
    rec['start'] = provenance()
    rec['environment'] = environment(require_nice=task['purpose'] != 'test')
    rec['modules'] = module_origins()
    problems = list(rec['environment']['problems'])
    if not trees_ok(rec['start']):
        problems.append(f'code trees {rec["start"]["idea_e1_tree"]} / {rec["start"]["idea_dq_tree"]} differ')
    if rec['start']['scripts'] != task['scripts']:
        problems.append('scripts differ from the launching process\'s')
    if task['plan_sha256'] != PLAN_SHA256:
        problems.append(f'plan {PLAN_SHA256} in this worker is not the launching process\'s {task["plan_sha256"]}')
    if not rec['modules']['key_modules_in_idea_e1'] or rec['modules']['foreign_modules']:
        problems.append(f'modules outside idea_e1: {rec["modules"]["foreign_modules"]}')
    return rec, problems


def _finish(rec, t0, c0, task):
    rec['end'] = provenance()
    if rec['status'] == 'completed' and (not trees_ok(rec['end']) or rec['end']['scripts'] != task['scripts']):
        rec['status'] = 'refused'
        rec['error'] = 'a code tree or a script changed during the read-out'
    rec['utc_end'] = p4.utc()
    rec['wall_s'] = time.perf_counter() - t0
    rec['cpu_s'] = time.process_time() - c0
    rec['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec['load_average_end'] = list(os.getloadavg())
    return rec


def evaluate_policy(task):
    """One read-out of one policy: p4_train.evaluate on the e1_evaluation suite with modes [task['mode']]."""
    t0, c0 = time.perf_counter(), time.process_time()
    rec, problems = _start(task, 'read_out')
    rec['load_average_start'] = list(os.getloadavg())
    if problems:
        rec.update(status='refused', error='; '.join(problems))
        return _finish(rec, t0, c0, task)
    entry, mode = task['entry'], task['mode']
    try:
        if mode not in p4.EVALUATION_MODES:
            raise ValueError(f'mode {mode!r}')
        cfg, rt, model, info, label = setup(entry)
        rec['checkpoint'], rec['label'], rec['suite'], rec['point_index'] = info, label, SUITE, entry['point_index']
        rec['torch_threads_read_out'] = torch.get_num_threads()
        e0 = time.perf_counter()
        results = p4.evaluate(model, rt, {'suite': SUITE, 'modes': [mode]}, entry['point_index'], cfg['gamma_ref'],
                              cfg['reference_duration_s'], label)
        rec['evaluate_wall_s'] = time.perf_counter() - e0
        result = results[mode]
        per_cell, weighted = p4.aggregate_evaluation(rt, result['episodes'])
        if canonical([per_cell, weighted]) != canonical([result['per_cell_J'], result['weighted_J']]):
            raise RuntimeError('the read-out\'s weighted J is not its episodes\'')
        rec.update(episodes=[json_row(e) for e in result['episodes']], per_cell_J=result['per_cell_J'],
                   weighted_J=result['weighted_J'], status='completed')
    except Exception as exc:                                                    # infrastructure: retried once
        rec.update(status='infrastructure_failure', error=f'{type(exc).__name__}: {exc}',
                   traceback=traceback.format_exc()[-4000:])
    return _finish(rec, t0, c0, task)


def probe_policy(task):
    """Precheck (d), outcome-blind: argmax on the 8 realisation-0 episodes (wall time and a digest of the rows only;
    the rows are dropped here), then the same 8 episodes sampled (compared with the stored rows; wall time)."""
    t0, c0 = time.perf_counter(), time.process_time()
    rec, problems = _start(task, 'argmax_probe')
    if problems:
        rec.update(status='refused', error='; '.join(problems))
        return _finish(rec, t0, c0, task)
    entry = task['entry']
    try:
        cfg, rt, model, info, label = setup(entry)
        namespace, _ = rt.study.suites[SUITE]
        w = time.perf_counter()
        for cell in rt.cells:                           # prepare every cell first, so neither mode's timing holds it
            rt.static(cell)
        static_s = time.perf_counter() - w
        model.eval()                                    # as p4_train.evaluate does before its episodes
        arg_rows, arg_wall, samp_rows, samp_wall = [], [], [], []
        for cell in rt.cells:
            r = PROBE_REALIZATION
            seed = p4.evaluation_seed(rt, namespace, cell, r)
            w = time.perf_counter()
            run = p4.run_learned_episode(rt, cell, namespace, seed, model, None, True,
                                         f'{label}-argmax-c{cell["cell_index"]}-r{r}', -1, cfg['gamma_ref'],
                                         cfg['reference_duration_s'])
            arg_wall.append(time.perf_counter() - w)
            arg_rows.append(json_row(run['record']))
        for cell in rt.cells:
            r = PROBE_REALIZATION
            seed = p4.evaluation_seed(rt, namespace, cell, r)
            generator = torch.Generator().manual_seed(rt.study.stream(p4.EVAL_STREAM, entry['point_index'],
                                                                      cell['cell_index'], r))
            w = time.perf_counter()
            run = p4.run_learned_episode(rt, cell, namespace, seed, model, generator, False,
                                         f'{label}-sample-c{cell["cell_index"]}-r{r}', -1, cfg['gamma_ref'],
                                         cfg['reference_duration_s'])
            samp_wall.append(time.perf_counter() - w)
            samp_rows.append(json_row(run['record']))
        rec.update(checkpoint=info, label=label, cells=[c['cell_id'] for c in rt.cells], static_prepare_s=static_s,
                   torch_threads_read_out=torch.get_num_threads(), argmax_wall_s=arg_wall, argmax_rows_sha256=rows_digest(arg_rows),
                   argmax_keys=[[r['cell_id'], r['seed']] for r in arg_rows],
                   sample_wall_s=samp_wall, sample_rows=samp_rows, status='completed')
        del arg_rows                                                # never written, never printed
    except Exception as exc:
        rec.update(status='infrastructure_failure', error=f'{type(exc).__name__}: {exc}',
                   traceback=traceback.format_exc()[-4000:])
    return _finish(rec, t0, c0, task)


# ------------------------------------------------------------------ pool -----------------------------------------------

def drive(tasks, workers, fn, sink):
    """Run tasks on a spawn pool with one fresh interpreter per task (as each E1 job was a fresh process);
    sink(task, record) runs in the parent for every result. A worker that dies yields an infrastructure record."""
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(fn, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                rec = future.result()
            except Exception as exc:                                # e.g. a worker killed by the OOM killer
                rec = {'version': VERSION, 'job': task['entry']['job'], 'status': 'infrastructure_failure',
                       'mode': task.get('mode'), 'attempt': task.get('attempt', 1), 'purpose': task['purpose'],
                       'error': f'worker died: {type(exc).__name__}: {exc}'}
            sink(task, rec)


def check_record(rec, entry, plan_sha256, scripts, mode=MAIN_MODE):
    """Problems that keep a record out (empty list = the record enters): status, identity, plan, scripts and trees at
    start and end, the checkpoint, and the 80 episode keys in the stored suite order."""
    problems = []
    if not isinstance(rec, dict):
        return ['not a record']
    want = {'version': VERSION, 'kind': 'read_out', 'status': 'completed', 'mode': mode, 'job': entry['job'],
            'family': entry['family'], 'block': entry['block'], 'plan_sha256': plan_sha256}
    problems += [f'{k}: {rec.get(k)!r} != {v!r}' for k, v in want.items() if rec.get(k) != v]
    for when in ('start', 'end'):
        prov = rec.get(when) or {}
        if prov.get('idea_e1_tree') != EXPECTED_E1_TREE or prov.get('idea_dq_tree') != EXPECTED_DQ_TREE:
            problems.append(f'{when}: code trees differ')
        if prov.get('scripts') != scripts:
            problems.append(f'{when}: scripts differ from the plan\'s')
    if (rec.get('checkpoint') or {}).get('sha256') != entry['final_checkpoint']['sha256']:
        problems.append('checkpoint SHA-256 differs')
    stored = entry.get('stored', {}).get('episodes')
    episodes = rec.get('episodes') or []
    if stored is not None and [(e['cell_id'], e['seed']) for e in episodes] != \
            [(e['cell_id'], e['seed']) for e in stored]:
        problems.append('episode keys are not the stored suite, in order')
    if len(episodes) != 80 or any(set(e) != set(EPISODE_FIELDS) for e in episodes):
        problems.append('not 80 episodes with exactly the stored fields')
    return problems


# ------------------------------------------------------------------ test -----------------------------------------------

def run_tests(log_dir=None):
    """test_runner.py and analysis.py --selftest, each in its own -B interpreter. Returns {name: 'passed'|'failed'}."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    out = {}
    for name, args in (('test_runner.py', ['test_runner.py']), ('analysis.py --selftest', ['analysis.py', '--selftest'])):
        proc = subprocess.run([sys.executable, '-B', *args], cwd=HERE, env=env, capture_output=True, text=True)
        out[name] = 'passed' if proc.returncode == 0 else 'failed'
        if log_dir is not None:
            (Path(log_dir) / f'{args[0]}.log').write_text(proc.stdout + '\n--- stderr ---\n' + proc.stderr)
        else:
            print(proc.stdout[-3000:], proc.stderr[-3000:], sep='\n')
    return out


# ------------------------------------------------------------------ precheck -------------------------------------------

def files_touched_since(roots, since):
    """Files and directories under `roots` whose mtime or ctime is at or after `since` (epoch seconds)."""
    hits = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for name in [*dirnames, *filenames]:
                p = os.path.join(dirpath, name)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if st.st_mtime >= since or st.st_ctime >= since:
                    hits.append(p)
    return hits


def precheck(workers, tag=''):
    if not tag:
        plan = bound_plan()                                   # the untagged precheck is the precheck of record
        if plan['sha256'] != PLAN_SHA256:
            raise SystemExit('PLAN.md changed after this process started')
    out_json, out_dir = HERE / f'precheck{tag}.json', HERE / f'precheck{tag}'
    if out_json.exists() or out_dir.exists():
        raise SystemExit(f'{out_json.name} or {out_dir.name}/ exists: never overwritten')
    since = time.time() - 1.0
    t0 = time.perf_counter()
    raise_priority_floor()
    env = environment()
    if env['problems']:
        raise SystemExit(f'environment: {env["problems"]}')
    (out_dir / 'sample').mkdir(parents=True)
    (out_dir / 'probe').mkdir()
    scripts = code_hashes()
    result = {'version': VERSION, 'utc': p4.utc(), 'tag': tag, 'plan_sha256': PLAN_SHA256, 'code_sha256': scripts,
              'workers': workers, 'environment': env, 'load_average_start': list(os.getloadavg()),
              'mem_available_gib_start': mem_available_gib(), 'free_m_start': free_m()}
    # (a) static
    start = provenance()
    ctx = e1_context()
    table = policy_table(ctx)
    rule = ctx['baselines']['p3_rule/e1_evaluation']
    _, rule_j = p4.aggregate_evaluation(ctx['runtime'], rule['episodes'])
    a = {'start': start, 'trees_ok_start': trees_ok(start), 'freeze_sha256': ctx['freeze_sha256'],
         'campaign_sha256': ctx['campaign']['campaign_sha256'], 'selection': ctx['chosen'],
         'baselines_sha256': sha(BASELINES), 'baselines_sha256_ok': sha(BASELINES) == BASELINES_SHA256,
         'rule_weighted_J_repr': repr(rule_j), 'rule_J_of_record_ok': repr(rule_j) == repr(RULE_J_OF_RECORD),
         'rule_credited_timely_per_episode': sum(e1_report_credited(e) for e in rule['episodes']) / len(rule['episodes']),
         'policies': len(table), 'policy_table_digest': table_digest(table),
         'policy_table': [{k: e[k] for k in ('job', 'family', 'role', 'candidate', 'block', 'point_index',
                                             'final_checkpoint', 'point_checkpoint')}
                          | {'stored_weighted_J': e['stored']['weighted_J']} for e in table],
         'modules': module_origins()}
    if PLAN_SHA256 is not None:
        listed = listed_table_digest(PLAN.read_text())
        a['policy_table_digest_vs_plan'] = listed == a['policy_table_digest']
    result['a_static'] = a
    print(json.dumps({'a_static': {k: v for k, v in a.items() if k not in ('policy_table', 'start', 'modules')}},
                     indent=1, default=float), flush=True)
    # (b) tests
    tests = run_tests(out_dir)
    result['b_unit_tests'] = {'passed': all(v == 'passed' for v in tests.values()), 'results': tests}
    print(json.dumps(result['b_unit_tests']), flush=True)
    # (c) the sampled reproduction of all 48 policies
    by_job = {e['job']: e for e in table}
    rows = {}
    order = sorted(table, key=lambda e: -e['e1_eval_wall_s'])                 # longest first
    tasks = [{'entry': light(e), 'mode': 'sample', 'purpose': 'precheck', 'plan_sha256': PLAN_SHA256,
              'scripts': scripts} for e in order]
    c0 = time.perf_counter()

    def sink_sample(task, rec):
        entry = by_job[task['entry']['job']]
        write_json_atomic(out_dir / 'sample' / f'{entry["job"]}.json', rec)
        row = {'job': entry['job'], 'family': entry['family'], 'block': entry['block'], 'status': rec.get('status'),
               'error': rec.get('error'), 'wall_s': rec.get('wall_s'), 'evaluate_wall_s': rec.get('evaluate_wall_s'),
               'max_rss_kib': rec.get('max_rss_kib'), 'e1_eval_wall_s': entry['e1_eval_wall_s']}
        if rec.get('status') == 'completed':
            diffs = compare_rows(rec['episodes'], entry['stored']['episodes'])
            row.update(episodes=len(rec['episodes']), identical_episodes=sum(
                1 for i in range(len(rec['episodes'])) if not any(d.get('episode') == i for d in diffs)),
                differences=diffs[:20], n_differences=len(diffs),
                weighted_J_identical=repr(rec['weighted_J']) == repr(entry['stored']['weighted_J']),
                per_cell_J_identical=canonical(rec['per_cell_J']) == canonical(entry['stored']['per_cell_J']),
                record_problems=check_record(rec, entry, PLAN_SHA256, scripts, mode='sample'))
            row['identical'] = (not diffs and row['weighted_J_identical'] and row['per_cell_J_identical']
                                and not row['record_problems'])
        else:
            row['identical'] = False
        rows[entry['job']] = row
        print(f'[c] {len(rows):2d}/48 {entry["job"]:34s} {row["status"]:22s} identical={row["identical"]} '
              f'wall={row["wall_s"] or 0:.0f}s', flush=True)
    drive(tasks, workers, evaluate_policy, sink_sample)
    c_wall = time.perf_counter() - c0
    ordered = [rows[e['job']] for e in table]
    result['c_sample_reproduction'] = {
        'policies': len(ordered), 'policies_identical': sum(r['identical'] for r in ordered),
        'episodes': sum(r.get('episodes', 0) for r in ordered),
        'episodes_identical': sum(r.get('identical_episodes', 0) for r in ordered),
        'all_identical': len(ordered) == 48 and all(r['identical'] for r in ordered),
        'per_family': {f: {'policies_identical': sum(r['identical'] for r in ordered if r['family'] == f),
                           'mean_wall_s': _mean([r['wall_s'] for r in ordered if r['family'] == f])}
                       for f in FAMILIES},
        'makespan_s': c_wall, 'rows': ordered}
    print(json.dumps({k: v for k, v in result['c_sample_reproduction'].items() if k != 'rows'}, indent=1,
                     default=float), flush=True)
    # (d) outcome-blind argmax probe: one policy per family (block 10), two separate processes
    probes = {}
    ptasks = [{'entry': light(by_job[e1_study.job_name(f, ctx['chosen'][f], PROBE_BLOCK)]), 'purpose': 'precheck',
               'plan_sha256': PLAN_SHA256, 'scripts': scripts, 'copy': copy} for f in FAMILIES for copy in ('A', 'B')]
    d0 = time.perf_counter()

    def sink_probe(task, rec):
        write_json_atomic(out_dir / 'probe' / f'{task["entry"]["job"]}__{task["copy"]}.json', rec)
        probes[(task['entry']['family'], task['copy'])] = rec
        print(f'[d] {task["entry"]["job"]} {task["copy"]} {rec.get("status")}', flush=True)
    drive(ptasks, min(workers, len(ptasks)), probe_policy, sink_probe)
    d = {'block': PROBE_BLOCK, 'realization': PROBE_REALIZATION, 'makespan_s': time.perf_counter() - d0,
         'families': {}}
    for f in FAMILIES:
        entry = by_job[e1_study.job_name(f, ctx['chosen'][f], PROBE_BLOCK)]
        stored = {(e['cell_id'], e['seed']): e for e in entry['stored']['episodes']}
        pa, pb = probes.get((f, 'A'), {}), probes.get((f, 'B'), {})
        fam = {'job': entry['job'], 'status': [pa.get('status'), pb.get('status')]}
        if pa.get('status') == pb.get('status') == 'completed':
            subset_diffs = [compare_rows(p['sample_rows'], [stored[(r['cell_id'], r['seed'])] for r in p['sample_rows']])
                            for p in (pa, pb)]
            arg = [x for p in (pa, pb) for x in p['argmax_wall_s']]
            smp = [x for p in (pa, pb) for x in p['sample_wall_s']]
            fam.update(argmax_digest=pa['argmax_rows_sha256'],
                       argmax_digests_agree=pa['argmax_rows_sha256'] == pb['argmax_rows_sha256'],
                       argmax_keys=pa['argmax_keys'],
                       sample_subset_identical=not subset_diffs[0] and not subset_diffs[1],
                       sample_subset_differences=(subset_diffs[0] + subset_diffs[1])[:10],
                       argmax_wall_s_mean=_mean(arg), sample_wall_s_mean=_mean(smp),
                       argmax_over_sample_wall=_mean(arg) / _mean(smp) if _mean(smp) else None,
                       argmax_wall_s_A=pa['argmax_wall_s'], argmax_wall_s_B=pb['argmax_wall_s'],
                       max_rss_kib=max(pa['max_rss_kib'], pb['max_rss_kib']))
        d['families'][f] = fam
    d['passed'] = all(x.get('argmax_digests_agree') and x.get('sample_subset_identical') for x in d['families'].values())
    result['d_argmax_probe'] = d
    print(json.dumps(d, indent=1, default=float), flush=True)
    # (e) timing and the estimate of the main run
    est_tasks = []
    for e in table:
        r = rows[e['job']]
        ratio = d['families'][e['family']].get('argmax_over_sample_wall') or 1.0
        est_tasks.append((r['wall_s'] or 0.) * ratio)
    result['e_timing'] = {
        'note': ('the precheck (c) ran the 48 sampled read-outs at this worker count and load; the argmax estimate '
                 'scales each policy\'s measured sampled wall time by its family\'s argmax/sampled ratio from (d) '
                 '(8 episodes x 2 processes) and assumes the same load'),
        'sample_policy_wall_s': {f: result['c_sample_reproduction']['per_family'][f]['mean_wall_s'] for f in FAMILIES},
        'argmax_over_sample_wall': {f: d['families'][f].get('argmax_over_sample_wall') for f in FAMILIES},
        'estimated_argmax_task_s_sum': sum(est_tasks),
        'estimated_makespan_s': {str(w): lpt_makespan(est_tasks, w) for w in (6, 7, 8)},
        'measured_sample_makespan_s': c_wall, 'measured_at_workers': workers}
    # (f) memory
    rss = [r['max_rss_kib'] for r in ordered if r.get('max_rss_kib')] + \
          [x.get('max_rss_kib') for x in d['families'].values() if x.get('max_rss_kib')]
    result['f_memory'] = {'max_rss_gib_per_worker': max(rss) / 2 ** 20 if rss else None,
                          'mem_available_gib_end': mem_available_gib(), 'free_m_end': free_m(),
                          'launch_rule': f'MemAvailable >= workers x {GIB_PER_WORKER} GiB + {GIB_RESERVE} GiB',
                          'rule_covers_measured': bool(rss) and max(rss) / 2 ** 20 <= GIB_PER_WORKER}
    # (g) nothing written into the two code trees
    touched = files_touched_since([IDEA_E1, IDEA_DQ], since)
    result['g_trees_untouched'] = {'since_epoch': since, 'touched': touched[:50], 'n_touched': len(touched),
                                   'passed': not touched}
    end = provenance()
    a['end'], a['trees_ok_end'] = end, trees_ok(end)
    result['passed'] = bool(
        a['trees_ok_start'] and a['trees_ok_end'] and a['baselines_sha256_ok'] and a['rule_J_of_record_ok']
        and a['modules']['key_modules_in_idea_e1'] and not a['modules']['foreign_modules']
        and a.get('policy_table_digest_vs_plan', True)
        and end['scripts'] == scripts == start['scripts'] and end['plan_sha256_on_disk'] == start['plan_sha256_on_disk']
        and result['b_unit_tests']['passed'] and result['c_sample_reproduction']['all_identical']
        and d['passed'] and result['g_trees_untouched']['passed'] and result['f_memory']['rule_covers_measured'])
    result['wall_s'] = time.perf_counter() - t0
    result['load_average_end'] = list(os.getloadavg())
    write_json_atomic(out_json, result)
    print(json.dumps({'passed': result['passed'], 'tag': tag, 'plan_sha256': PLAN_SHA256,
                      'reproduced_policies': f'{result["c_sample_reproduction"]["policies_identical"]}/48',
                      'reproduced_episodes': f'{result["c_sample_reproduction"]["episodes_identical"]}/'
                                             f'{result["c_sample_reproduction"]["episodes"]}',
                      'argmax_probe': d['passed'], 'unit_tests': result['b_unit_tests']['passed'],
                      'trees_untouched': result['g_trees_untouched']['passed'],
                      'estimated_makespan_h': {k: v / 3600 for k, v in result['e_timing']['estimated_makespan_s'].items()},
                      'wall_s': result['wall_s']}, indent=1, default=float), flush=True)
    return result


def e1_report_credited(e):
    """Timely credited reports of one episode, exactly e1_report.credited: tp_timely if it returned, else 0."""
    return float(e['tp_timely']) if e['return_ok'] else 0.


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def lpt_makespan(durations, workers):
    """Makespan of longest-processing-time-first list scheduling of `durations` on `workers` machines."""
    loads = [0.] * workers
    for x in sorted(durations, reverse=True):
        i = loads.index(min(loads))
        loads[i] += x
    return max(loads)


# ------------------------------------------------------------------ launch ---------------------------------------------

def require_launchable(workers):
    plan = bound_plan()
    if plan['sha256'] != PLAN_SHA256:
        raise SystemExit('PLAN.md changed after this process started')
    for root, expected in ((IDEA_E1, EXPECTED_E1_TREE), (IDEA_DQ, EXPECTED_DQ_TREE)):
        if tree(root) != expected:
            raise SystemExit(f'the {root.name} code tree is {tree(root)}, not {expected}: refused')
    env = environment()
    if env['problems']:
        raise SystemExit(f'environment: {env["problems"]}: refused')
    pre = json.loads((HERE / 'precheck.json').read_text()) if (HERE / 'precheck.json').exists() else {}
    if not (pre.get('passed') and pre.get('plan_sha256') == PLAN_SHA256 and pre.get('tag') == ''):
        raise SystemExit('precheck.json (untagged) has not passed for this plan: refused')
    if pre.get('code_sha256') != code_hashes():
        raise SystemExit('a script changed since the precheck: refused')
    ctx = e1_context()
    table = policy_table(ctx)
    if table_digest(table) != plan['policy_table_digest'] or \
            table_digest(table) != pre['a_static']['policy_table_digest']:
        raise SystemExit(f'the policy table digest {table_digest(table)} is not the one PLAN.md and the precheck '
                         'name: refused')
    need = workers * GIB_PER_WORKER + GIB_RESERVE
    have = mem_available_gib()
    if have is None or have < need:
        raise SystemExit(f'MemAvailable {have} GiB < {need} GiB for {workers} workers: refused')
    return plan, table


def read_record(path, entry, plan_sha256, scripts):
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return rec if not check_record(rec, entry, plan_sha256, scripts) else None


def launch(workers):
    if not 1 <= workers <= 12:
        raise SystemExit('workers: 1-12')
    raise_priority_floor()
    plan, table = require_launchable(workers)
    lock = open(LOCK, 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('another launch holds .launch.lock: refused')
    scripts = code_hashes()
    attempts_dir, aside = RECORDS / 'attempts', RECORDS / 'aside'
    attempts_dir.mkdir(parents=True, exist_ok=True)
    pending, exhausted = [], []
    for e in table:
        path = RECORDS / f'{e["job"]}.json'
        if path.exists():
            if read_record(path, e, PLAN_SHA256, scripts) is not None:
                continue
            aside.mkdir(exist_ok=True)
            os.replace(path, aside / f'{e["job"]}.json.{p4.utc().replace(":", "")}')
        failed = len(list(attempts_dir.glob(f'{e["job"]}__attempt*.json')))
        if failed >= MAX_ATTEMPTS:
            exhausted.append(e['job'])
            continue
        pending.append((e, failed + 1))
    entry_log = {'utc': p4.utc(), 'plan_sha256': PLAN_SHA256, 'workers': workers, 'code_sha256': scripts,
                 'idea_e1_tree': EXPECTED_E1_TREE, 'idea_dq_tree': EXPECTED_DQ_TREE, 'pid': os.getpid(),
                 'nice': os.nice(0), 'pending': [e['job'] for e, _ in pending], 'exhausted': exhausted,
                 'mem_available_gib': mem_available_gib()}
    with open(HERE / 'launches.jsonl', 'a') as handle:
        handle.write(json.dumps(entry_log) + '\n')
    print(json.dumps({k: v for k, v in entry_log.items() if k != 'pending'} | {'pending': len(pending)}), flush=True)
    by_job = {e['job']: e for e in table}
    order = sorted(pending, key=lambda x: -x[0]['e1_eval_wall_s'])
    status = {}
    t0 = time.perf_counter()
    while order:
        tasks = [{'entry': light(e), 'mode': MAIN_MODE, 'purpose': 'main', 'attempt': k, 'plan_sha256': PLAN_SHA256,
                  'scripts': scripts} for e, k in order]
        retry = []

        def sink(task, rec):
            entry = by_job[task['entry']['job']]
            problems = check_record(rec, entry, PLAN_SHA256, scripts) if rec.get('status') == 'completed' else \
                [rec.get('status'), rec.get('error')]
            if not problems:
                write_json_atomic(RECORDS / f'{entry["job"]}.json', rec)
                status[entry['job']] = 'completed'
            else:
                rec['record_problems'] = problems
                write_json_atomic(attempts_dir / f'{entry["job"]}__attempt{task["attempt"]}.json', rec)
                status[entry['job']] = rec.get('status')
                if rec.get('status') == 'infrastructure_failure' and task['attempt'] < MAX_ATTEMPTS:
                    retry.append((entry, task['attempt'] + 1))
            done = sum(1 for v in status.values() if v == 'completed')
            print(f'{p4.utc()} {entry["job"]:34s} attempt {task["attempt"]} {status[entry["job"]]:22s} '
                  f'wall {rec.get("wall_s") or 0:.0f}s  ({done} completed this launch, '
                  f'{time.perf_counter() - t0:.0f}s)', flush=True)
        drive(tasks, workers, evaluate_policy, sink)
        order = retry
    final = {e['job']: ('completed' if read_record(RECORDS / f'{e["job"]}.json', e, PLAN_SHA256, scripts) else
                        status.get(e['job'], 'exhausted' if e['job'] in exhausted else 'missing')) for e in table}
    summary = {'utc': p4.utc(), 'final_status': {s: sum(1 for v in final.values() if v == s) for s in set(final.values())},
               'not_completed': sorted(j for j, v in final.items() if v != 'completed'),
               'wall_s': time.perf_counter() - t0, 'idea_e1_tree_end': tree(IDEA_E1), 'idea_dq_tree_end': tree(IDEA_DQ)}
    print(json.dumps(summary, indent=1), flush=True)
    with open(HERE / 'launches.jsonl', 'a') as handle:
        handle.write(json.dumps(dict(summary, event='launch/ended')) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('test')
    p = sub.add_parser('precheck')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tag', default='')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == 'test':
        out = run_tests()
        print(json.dumps(out, indent=1))
        return 0 if all(v == 'passed' for v in out.values()) else 1
    if args.command == 'precheck':
        if not 1 <= args.workers <= 12:
            raise SystemExit('workers: 1-12')
        if args.tag and not re.fullmatch(r'_[A-Za-z0-9_]+', args.tag):
            raise SystemExit('tag: an underscore and letters/digits, e.g. _build')
        return 0 if precheck(args.workers, args.tag)['passed'] else 1
    summary = launch(args.workers)
    return 0 if not summary['not_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
