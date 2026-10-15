"""P4: semi-Markov recurrent PPO of the learned arms on task flood-sar-village-gnss/1.

    .venv/bin/python scripts/gnss_qrl_runtime/p4_train.py --config <config.json> --out <new directory>

A development tool. A run of it is not a learning result and cannot enter P5 statistics. It never starts a
simulator: every episode is the fast cached-view tier on the r6 s0 banks, built by the SAME function the P3 rule
comparators ran through (`run_p3_cells.prepare_cell` / `cell_episode_parts`), so a learned arm meets exactly the
cells, drift, anchors, budgets and score the gate measured.

What is common to every arm (statistical protocol section 2)
    encoder obs-encoding/4-gnss and action-space/4-gnss with the frozen anchor order; RETURN rule gnss_exhausted/1;
    the GRU trunk and the critic head; the optimiser family; the reward `reward-gnss-timely/1`, placed on the
    decision records by `gnss_rewards.place_gnss_rewards`; gamma_ref = 1 with the time-aware trace
    lambda(dt) = lambda_ref ** (dt / reference_duration_s) (protocol section 4.2a); the training schedule of the
    block; the decision budget and the checkpoint grid. Only `actor_head` differs between arms.

The arms (statistical protocol section 2)
    HQ = 'quantum', C-matched = 'bottleneck' (415 vs 411 actor-head parameters at A = 27), C-strong = 'mlp',
    Surrogate625 = 'surrogate625'.

Pairing and seeds
    A block b fixes the training schedule every arm of that block consumes, in order: episode k of block b runs
    cell `schedule_cell(b, k)` (drawn with the frozen cell weights from SeedSequence([b, SCHEDULE_STREAM, k])) on
    the disturbance seed `main_training.start + b * BLOCK_STRIDE + k`. The CRN group is `crn_group(pair_id,
    'main_training', seed)`, exactly as in P3, and the re-observation seed is the same seed. An arm that needs more
    episodes to reach the decision budget simply consumes a longer prefix of the same schedule. Model
    initialisation, action sampling and minibatch order come from SeedSequence([b, stream]) per arm - independent
    named streams, as the protocol asks; identical initialisations across architectures are neither possible nor
    required.

Rewards and the identity they keep
    With gamma_ref = 1 the kept rollout rewards of an episode sum to J minus the reward mass no valid record could
    carry; `gnss_rewards.check_episode_reward_sum` asserts it on every episode without an invalid transition, and
    every episode's J, dropped mass and components are written to metrics.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import fast_mission                                                                              # noqa: E402
import run_p3_cells as runner                                                                    # noqa: E402
from aerial_qrl.mission.contracts import INFRASTRUCTURE                                          # noqa: E402
from aerial_qrl.mission.time_model import TimeModel                                              # noqa: E402
from aerial_qrl.training.actor_heads import QUANTUM_ACTOR_HEADS                                  # noqa: E402
from aerial_qrl.training.checkpoint_identity import (canonical_sha256, identity_for_gnss,        # noqa: E402
                                                     save_gnss_checkpoint)
from aerial_qrl.training.encoding import LAYOUT_V4_GNSS, ActionSpace                             # noqa: E402
from aerial_qrl.training.episodes import build_episode_transitions                              # noqa: E402
from aerial_qrl.training.gnss_rewards import (GNSS_REWARD_PLACEMENT_VERSION,                     # noqa: E402
                                              check_episode_reward_sum, place_gnss_rewards)
from aerial_qrl.training.reward_gnss import (REWARD_VERSION, SCORE_VERSION, check_reward_identity,  # noqa: E402
                                             episode_rewards)
from aerial_qrl.training.rl_policy import RLMissionPolicy                                        # noqa: E402
from aerial_qrl.training.smdp_ppo import (RecurrentActorCritic, RolloutBuffer, compute_advantages,  # noqa: E402
                                          file_sha256, ppo_update)

TOOL = 'gnss-p4-train/1'
CONFIG_VERSION = 'gnss-p4-train-config/1'
TRAINING_NAMESPACE = 'main_training'
#: Episode k of block b uses seed start + b * BLOCK_STRIDE + k; blocks and episodes can therefore never alias.
#: The frozen `main_training` namespace is 100,000 seeds wide, so 1,000 episodes per block for up to 100 blocks.
#: At the 30,000-decision budget that allows episodes as short as 30 decisions (an untrained policy averages
#: about 35); a block that needs more episodes is REFUSED, never wrapped into the next block's seeds.
BLOCK_STRIDE = 1_000
SCHEDULE_STREAM, INIT_STREAM, ACTION_STREAM, MINIBATCH_STREAM = 3, 0, 1, 2
RETURN_MASK_RULE = 'gnss_exhausted/1'
ARM_ROLES = {'quantum': 'HQ', 'bottleneck': 'C-matched', 'mlp': 'C-strong', 'surrogate625': 'Surrogate625'}
DTYPES = {'float32': torch.float32, 'float64': torch.float64}
PPO_KEYS = ('learning_rate', 'clip_range', 'value_coef', 'entropy_coef', 'max_grad_norm', 'epochs',
            'minibatch_transitions', 'advantage_std_floor', 'max_segment_length')
REQUIRED = ('config_version', 'freeze', 'execution_manifest', 'time_model', 'arms', 'critic', 'block',
            'total_policy_decisions', 'transitions_per_update', 'checkpoint_decisions', 'ppo', 'gamma_ref',
            'reference_duration_s', 'lambda_ref', 'lambda_mode')
OPTIONAL = ('hidden', 'memory', 'critic_gradient', 'dtype', 'torch_threads', 'note', 'max_updates', 'evaluation',
            'fixed_schedule', 'role', 'candidate', 'protocol_freeze_sha256', 'study')
#: review 23, P4 item 1: a tuning or pilot run exists only under the APPROVED protocol freeze. The approval is
#: reviewer-owned and lives outside the hashed code tree (like the P3 campaign approval), so writing it does not move
#: the tree it approves.
APPROVAL_PATH = ROOT / 'docs' / 'study' / 'gnss_qrl_v1' / 'approvals' / 'P4_PROTOCOL_APPROVAL.json'
APPROVAL_VERSION = 'gnss-p4-protocol-approval/1'
APPROVAL_FIELDS = ('approval_version', 'approved_freeze', 'approved_freeze_sha256', 'approved_by', 'review', 'utc')
EVAL_STREAM = 4
EVALUATION_MODES = ('sample', 'argmax')
#: THE block mapping and evaluation suites of the frozen P4 tuning-and-pilot protocol; `p4_seed_manifest.py`
#: emits exactly these. `pilot` realisation 0 belongs to the P3 smokes; `final_evaluation` is P5's alone.
BLOCKS = {'tuning': tuple(range(0, 2)), 'pilot': tuple(range(2, 8)), 'main': tuple(range(30, 80))}
SUITES = {'tuning_score': ('pilot', tuple(range(1, 6))), 'pilot_study': ('pilot', tuple(range(6, 16))),
          'main': ('final_evaluation', tuple(range(0, 10)))}
#: role -> (blocks, evaluation suite, budget, checkpoint grid). 'sanity' is the WP4 learning check only.
ROLES = {'tuning': ('tuning', 'tuning_score', 10000, [0, 10000]),
         'pilot': ('pilot', 'pilot_study', 30000, list(range(0, 30001, 3000)))}
PROTOCOL_CANDIDATES = {'c1': {'learning_rate': 3e-4, 'entropy_coef': 0.01},
                       'c2': {'learning_rate': 3e-4, 'entropy_coef': 0.0},
                       'c3': {'learning_rate': 1e-3, 'entropy_coef': 0.0}}
CHECKPOINT_GRID_30K = list(range(0, 30001, 3000))


@dataclass(frozen=True)
class Study:
    """Everything that distinguishes one learned-arm study from another; the trainer reads nothing else.

    `salt` is prefixed to EVERY SeedSequence the trainer draws (model initialisation, action sampling, minibatch
    order, training schedule, evaluation action streams). P4's salt is empty, so its streams are exactly the frozen
    P4 streams (a test regenerates the frozen P4 seed manifest byte for byte); E1's salt makes every E1 stream
    distinct from every P4 stream even where block, point, cell and realisation numbers coincide.
    """
    name: str
    tool: str
    salt: tuple
    training_namespace: str
    extra_namespaces: dict = field(default_factory=dict)
    blocks: dict = field(default_factory=dict)
    suites: dict = field(default_factory=dict)
    roles: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)
    evaluation_suites: tuple = ()
    approval_path: Path = None
    approval_version: str = ''
    approval_key: str = ''
    non_claim: str = ''
    allows_sanity: bool = False

    def stream(self, *parts):
        return stream(*self.salt, *parts)

    def seed_sequence(self, *parts):
        return np.random.SeedSequence([int(p) for p in (*self.salt, *parts)])


#: E1 (`E1_FAIR_COMPARISON_PROPOSAL_EN.md` revision 2): three new namespaces, each NAMESPACE_WIDTH wide, placed after
#: the last frozen P3 namespace (`demonstration`, index 6) by P3's own rule start = base * 1000 + index * width.
E1_SALT = 20260922
E1_NAMESPACES = {name: {'start': 20260921 * 1000 + index * 100000, 'end': 20260921 * 1000 + (index + 1) * 100000}
                 for index, name in ((7, 'e1_training'), (8, 'e1_development'), (9, 'e1_evaluation'))}
E1_CANDIDATES = {f'lr{lr}_ent{ent}': {'learning_rate': float(lr), 'entropy_coef': float(ent)}
                 for lr in ('3e-4', '1e-3', '3e-3') for ent in ('0', '0.01')}
STUDIES = {
    'p4': None,  # filled below, from the frozen P4 constants above
    'e1': Study(
        name='e1', tool='gnss-e1-train/1', salt=(E1_SALT,), training_namespace='e1_training',
        extra_namespaces=E1_NAMESPACES,
        blocks={'e1_tuning': (0, 1, 2), 'e1_evaluation': tuple(range(10, 22))},
        suites={'e1_development': ('e1_development', tuple(range(0, 5))),
                'e1_evaluation': ('e1_evaluation', tuple(range(0, 10)))},
        roles={'e1_tuning': ('e1_tuning', 'e1_development', 30000, CHECKPOINT_GRID_30K),
               'e1_evaluation': ('e1_evaluation', 'e1_evaluation', 30000, CHECKPOINT_GRID_30K)},
        candidates=E1_CANDIDATES, evaluation_suites=('e1_development', 'e1_evaluation'),
        approval_path=ROOT / 'docs' / 'study' / 'gnss_qrl_v1' / 'approvals' / 'E1_PROTOCOL_APPROVAL.json',
        approval_version='gnss-e1-protocol-approval/1', approval_key='e1_protocol_approval',
        non_claim=('An E1 run records interactions, losses and mission outcomes of one exploratory study '
                   '(E1_FAIR_COMPARISON_PROPOSAL_EN.md); it supports no confirmatory, superiority, equivalence or '
                   'quantum-advantage claim.')),
}


class FreezeBindingError(ValueError):
    """The run, the tool or its inputs are not the ones the approved protocol freeze names."""


def _sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_copy_tree():
    import code_version
    return code_version.tree_hash(code_version.file_hashes())


def approved_freeze(approval_path=None, study=None):
    """(freeze path, approved SHA-256) from the reviewer-owned approval; refuses a missing or malformed approval.

    Each study has its own approval file, key and version (P4: `P4_PROTOCOL_APPROVAL.json`; E1:
    `E1_PROTOCOL_APPROVAL.json`), so an approval of one study can never open another."""
    study = study or STUDIES['p4']
    path = Path(approval_path or study.approval_path)
    if not path.exists():
        raise FreezeBindingError(f'no {study.name.upper()} protocol is approved: {path} does not exist')
    payload = json.loads(path.read_text())
    approval = payload.get(study.approval_key) if isinstance(payload, dict) else None
    if not isinstance(approval, dict) or set(approval) != set(APPROVAL_FIELDS):
        raise FreezeBindingError(f'{path}: not exactly {list(APPROVAL_FIELDS)} under {study.approval_key!r}')
    if approval['approval_version'] != study.approval_version or approval['approved_by'] != 'reviewer':
        raise FreezeBindingError(f'{path}: version {approval["approval_version"]!r} by {approval["approved_by"]!r}')
    freeze = Path(approval['approved_freeze'])
    freeze = freeze if freeze.is_absolute() else ROOT / freeze
    return freeze, approval['approved_freeze_sha256']


def verify_freeze(freeze_path, approved_sha256):
    """Prove the approved freeze is what is on disk and what is running, before any work. Returns its record.

    The freeze file's own SHA-256, every member file it names, the seed manifest's internal hash and the LIVE code
    tree of this checkout must all equal what was approved; any difference is a refusal, never a warning.
    """
    freeze_path = Path(freeze_path)
    if _sha256_file(freeze_path) != approved_sha256:
        raise FreezeBindingError(f'{freeze_path} is not the approved freeze {approved_sha256[:12]}')
    record = json.loads(freeze_path.read_text())
    changed = [f for f, digest in record['files'].items() if _sha256_file(ROOT / f) != digest]
    if changed:
        raise FreezeBindingError(f'files differ from the approved freeze: {changed}')
    manifest = json.loads((ROOT / record['seed_manifest']).read_text())['seed_manifest']
    if canonical_sha256({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != \
            record['seed_manifest_sha256'] or manifest['manifest_sha256'] != record['seed_manifest_sha256']:
        raise FreezeBindingError('the seed manifest is not the frozen one')
    tree = live_copy_tree()
    if tree != record['copy_code_tree_sha256']:
        raise FreezeBindingError(f'the live code tree {tree[:12]} is not the frozen {record["copy_code_tree_sha256"][:12]}')
    return record


def require_approved_freeze(expected_sha256=None, approval_path=None, study=None):
    """The one gate every tuning/pilot (P4) or E1 entry point passes: approval, freeze, members, seeds, live tree."""
    freeze, approved = approved_freeze(approval_path, study)
    if expected_sha256 is not None and expected_sha256 != approved:
        raise FreezeBindingError(f'this run names freeze {str(expected_sha256)[:12]}, the approved one is '
                                 f'{approved[:12]}')
    return freeze, approved, verify_freeze(freeze, approved)


class ScheduleExhausted(ValueError):
    """A block needed more episodes than its seed range holds: an ALGORITHMIC non-completion (the policy's
    episodes were too short to reach the budget), recorded as such and never re-run on other seeds."""


#: THE failure rule of the P4 protocol (revision 3), stated once and applied by the trainer and the selector:
#:   terminal algorithmic failure - exactly these two: a non-finite value in training (FloatingPointError: a
#:       non-finite batch value caught by the update, or a non-finite parameter after an update) and
#:       ScheduleExhausted. The run stops, keeps its checkpoints, writes no final checkpoint and no AUC, is ineligible
#:       in tuning, contributes no d_i in the pilot, and is NEVER replaced by a run on other seeds;
#:   collapse, a falling or flat learning curve, a low score - NOT failures. Such a run is COMPLETED: final
#:       checkpoint, full curve, AUC; in tuning it is judged only by the declared score and guardrails, in the pilot
#:       and P5 it enters as it is. No collapse criterion exists, so nothing is ever excluded for "collapse";
#:   infrastructure failure - any other exception, which leaves NO summary.json: the same job is retried with the
#:       same configuration and seeds under the retry lineage of p4_tuning.py.
ALGORITHMIC_FAILURES = (FloatingPointError, ScheduleExhausted)
NON_CLAIM = ('A P4 run records interactions, losses and mission outcomes on development schedules; it is not a '
             'learning result, cannot select an arm and never enters P5 statistics.')
STUDIES['p4'] = Study(name='p4', tool=TOOL, salt=(), training_namespace=TRAINING_NAMESPACE, blocks=BLOCKS,
                      suites=SUITES, roles=ROLES, candidates=PROTOCOL_CANDIDATES,
                      evaluation_suites=('tuning_score', 'pilot_study'), approval_path=APPROVAL_PATH,
                      approval_version=APPROVAL_VERSION, approval_key='p4_protocol_approval', non_claim=NON_CLAIM,
                      allows_sanity=True)


def study_of(config):
    """The study a configuration belongs to. A configuration without `study` is P4's, byte for byte as frozen."""
    name = config.get('study', 'p4') if isinstance(config, dict) else 'p4'
    if name not in STUDIES:
        raise P4Error(f'study: one of {sorted(STUDIES)}, not {name!r}')
    return STUDIES[name]


class P4Error(ValueError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def stream(*parts):
    """An integer seed from SeedSequence(parts)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def validate_config(raw):
    if not isinstance(raw, dict):
        raise P4Error('config: expected an object')
    missing = [k for k in REQUIRED if k not in raw]
    unknown = sorted(set(raw) - set(REQUIRED) - set(OPTIONAL))
    if missing or unknown:
        raise P4Error(f'config: missing {missing}, unknown {unknown}')
    if raw['config_version'] != CONFIG_VERSION:
        raise P4Error(f'config_version: expected {CONFIG_VERSION}')
    config = dict({'hidden': 64, 'memory': 'gru', 'critic_gradient': 'stop', 'dtype': 'float64', 'torch_threads': 1,
                   'note': None, 'max_updates': None}, **raw)
    arms = config['arms']
    if not isinstance(arms, list) or not arms or len(set(arms)) != len(arms) or \
            any(a not in ARM_ROLES for a in arms):
        raise P4Error(f'arms: a non-empty list of distinct actor heads from {sorted(ARM_ROLES)}')
    # Protocol section 4.2a: gamma = 1 and the time-aware trace. These are not tuning knobs.
    if config['gamma_ref'] != 1.0 or config['lambda_mode'] != 'time':
        raise P4Error('gamma_ref must be 1.0 and lambda_mode "time" (statistical protocol section 4.2a)')
    for key in ('block', 'total_policy_decisions', 'transitions_per_update', 'hidden', 'torch_threads'):
        if type(config[key]) is not int or config[key] < 0 or (key != 'block' and config[key] == 0):
            raise P4Error(f'{key}: expected a {"non-negative" if key == "block" else "positive"} integer')
    grid = config['checkpoint_decisions']
    if not isinstance(grid, list) or any(type(p) is not int or p < 0 for p in grid) or grid != sorted(set(grid)):
        raise P4Error('checkpoint_decisions: strictly increasing non-negative integers')
    study = study_of(raw)
    if 'study' in raw and raw['study'] == 'p4':
        raise P4Error('a P4 configuration carries no study key (it is P4 byte for byte as frozen)')
    role = config.setdefault('role', 'sanity')
    fixed = config.get('fixed_schedule')
    if role == 'sanity' and not study.allows_sanity:
        raise P4Error(f'{study.name} has no sanity role; its roles are {sorted(study.roles)}')
    if role == 'sanity':
        if fixed is None or set(fixed) != {'cell_index', 'episode'} or \
                any(type(v) is not int or v < 0 for v in fixed.values()):
            raise P4Error('a sanity run repeats one cell and one disturbance: fixed_schedule {cell_index, episode}')
    elif role in study.roles:
        block_role, suite, budget, grid_expected = study.roles[role]
        if fixed is not None:
            raise P4Error(f'fixed_schedule is for the sanity run only, not a {role} run')
        if config['block'] not in study.blocks[block_role]:
            raise P4Error(f'a {role} run uses blocks {list(study.blocks[block_role])}, not {config["block"]}')
        if config['total_policy_decisions'] != budget or config['checkpoint_decisions'] != grid_expected:
            raise P4Error(f'a {role} run has budget {budget} and checkpoints {grid_expected} (frozen)')
        if (config.get('evaluation') or {}).get('suite') != suite:
            raise P4Error(f'a {role} run is evaluated on the {suite!r} suite (frozen)')
        if not isinstance(config.get('protocol_freeze_sha256'), str) or len(config['protocol_freeze_sha256']) != 64:
            raise P4Error(f'a {role} run names the approved protocol freeze it runs under (protocol_freeze_sha256)')
        candidate = config.get('candidate')
        if candidate not in study.candidates or any(config['ppo'][k] != v
                                                    for k, v in study.candidates[candidate].items()):
            raise P4Error(f'candidate: one of {sorted(study.candidates)}, with exactly its frozen values')
    elif study.name == 'p4':
        raise P4Error(f'role: one of sanity, {", ".join(ROLES)} (main runs belong to P5)')
    else:
        raise P4Error(f'role: one of {", ".join(study.roles)} for study {study.name}')
    evaluation = config.get('evaluation')
    if evaluation is not None:
        if set(evaluation) != {'suite', 'modes'}:
            raise P4Error('evaluation: exactly suite and modes')
        if evaluation['suite'] not in study.evaluation_suites:
            if study.name == 'p4':
                raise P4Error('evaluation.suite: tuning_score or pilot_study in P4 (the main suite is read in P5 only)')
            raise P4Error(f'evaluation.suite: one of {list(study.evaluation_suites)} in {study.name}')
        if not evaluation['modes'] or any(m not in EVALUATION_MODES for m in evaluation['modes']):
            raise P4Error(f'evaluation.modes: a non-empty subset of {EVALUATION_MODES}')
    ppo = config['ppo']
    if set(ppo) != set(PPO_KEYS):
        raise P4Error(f'ppo: exactly {sorted(PPO_KEYS)}')
    if config['dtype'] not in DTYPES:
        raise P4Error(f'dtype: one of {sorted(DTYPES)}')
    return config


class CellRuntime:
    """The frozen cells, their weights, and each cell's prepared static parts (loaded once, reused per episode)."""

    def __init__(self, freeze_path, manifest_path, time_model_path, study=None):
        self.study = study or STUDIES['p4']
        self.frozen, self.execution = runner.load_frozen(freeze_path, manifest_path)
        self.time_model = TimeModel.from_file(time_model_path)
        if runner.sha256_file(time_model_path) != self.execution['time_model_sha256']:
            raise P4Error('the time model differs from the execution manifest hash')
        self.cells = sorted(self.execution['cells'], key=lambda c: c['cell_index'])
        weights = np.asarray([c['weight'] for c in self.cells], float)
        self.probabilities = weights / weights.sum()
        self.namespaces = dict(self.execution['seed_namespaces'])
        for name, space in self.study.extra_namespaces.items():
            # A study's own namespaces must be new names and must not overlap ANY frozen range.
            if name in self.namespaces or any(space['start'] < other['end'] and other['start'] < space['end']
                                              for other in self.namespaces.values()):
                raise P4Error(f'{self.study.name} namespace {name} {space} collides with a frozen namespace')
            self.namespaces[name] = dict(space)
        self._static = {}

    def static(self, cell):
        if cell['cell_id'] not in self._static:
            self._static[cell['cell_id']] = runner.prepare_cell(self.frozen, cell)
        return self._static[cell['cell_id']]

    def anchor_ids(self):
        return tuple(a.anchor_id for a in runner.AnchorSet.from_dict(self.frozen['anchors']).anchors)

    def schedule(self, block, episode):
        """(cell, seed) of training episode `episode` of block `block`: identical for every arm of the block."""
        rng = np.random.default_rng(self.study.seed_sequence(block, SCHEDULE_STREAM, episode))
        cell = self.cells[int(rng.choice(len(self.cells), p=self.probabilities))]
        namespace = self.study.training_namespace
        space = self.namespaces[namespace]
        seed = space['start'] + int(block) * BLOCK_STRIDE + int(episode)
        if not seed < space['end']:
            raise P4Error(f'block {block} leaves the {namespace} namespace')
        if episode >= BLOCK_STRIDE:
            raise ScheduleExhausted(f'block {block} needed episode {episode}; its seed range holds {BLOCK_STRIDE}')
        return cell, seed

    def training_disturbance(self):
        """What the checkpoint identity binds: the set of drift configurations the schedule draws from."""
        configs = sorted({c['cell_id'].split('__')[0]: c['georeference'] for c in self.cells}.items())

        class TrainingDisturbance:
            process_model = self.cells[0]['georeference']['process_model']

            @staticmethod
            def to_dict():
                return {'training_cells': [c['cell_id'] for c in self.cells], 'drift_configs': configs}
        return TrainingDisturbance()


def build_model(arm, config, block):
    torch.manual_seed(study_of(config).stream(block, INIT_STREAM))
    model = RecurrentActorCritic(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, config['critic'], hidden=config['hidden'],
                                 critic_gradient=config['critic_gradient'], memory=config['memory'], actor_head=arm)
    return model.to(dtype=DTYPES[config['dtype']])


def run_learned_episode(runtime, cell, namespace, seed, model, generator, deterministic, episode_id, episode_number,
                        gamma_ref, tau):
    """One GNSS episode with a fresh RLMissionPolicy on the prepared cell; rewards placed and transitions built."""
    static = runtime.static(cell)
    backend_config, group, service = runner.cell_episode_parts(static, cell, namespace, seed)
    holder = {}

    def factory(briefing, obstacle_map, mission_config, reserve):
        holder['policy'] = RLMissionPolicy(briefing, obstacle_map, mission_config, reserve, model, generator=generator,
                                           deterministic=deterministic, compute_values=not deterministic,
                                           obs_encoding=LAYOUT_V4_GNSS.obs_version, return_mask_rule=RETURN_MASK_RULE,
                                           anchor_ids=runtime.anchor_ids())
        return holder['policy']
    started = time.perf_counter()
    result = fast_mission.run_fast_episode(static['context'], static['banks'], runtime.time_model, episode_id,
                                           backend_config=backend_config,
                                           executive_kwargs={'georeference': service, 'anchor_set': static['anchors']},
                                           policy_factory=factory, reports_for_matching=service.global_report_positions)
    wall = time.perf_counter() - started
    outcome, policy = result['outcome'], holder['policy']
    return_ok = outcome.termination_reason == 'mission_complete' and outcome.returned and outcome.landed
    times = [s.sim_time_s for s in service.submissions]
    breakdown = episode_rewards(service.global_report_positions(), times,
                                static['context']['evaluator'].targets, static['score'], return_ok)
    check_reward_identity(breakdown)
    placed = place_gnss_rewards(result['records'], breakdown, times)
    transitions = build_episode_transitions(policy.decisions, result['records'], placed.per_record,
                                            outcome.termination_reason, episode_number, gamma_ref, tau)
    invalid = any(not row['valid'] for row in transitions.rows)
    if not invalid:
        check_episode_reward_sum(transitions, breakdown, placed)
    kept = sum(row['reward'] for row in transitions.rows if row['valid'])
    final = breakdown.final
    return {'result': result, 'policy': policy, 'breakdown': breakdown, 'placed': placed, 'transitions': transitions,
            'wall_s': wall, 'record': {
                'cell_id': cell['cell_id'], 'seed': seed, 'crn_sha256': group['crn_sha256'],
                'termination_reason': outcome.termination_reason, 'return_ok': return_ok, 'J': breakdown.score,
                'tp_timely': final.tp_timely, 'tp_spatial': final.tp_spatial, 'fp_spatial': final.fp_spatial,
                'duplicates': final.duplicates, 'fixes_attempted': len(service.measurements),
                'decisions': outcome.decisions, 'rejections': outcome.rejections, 'elapsed_s': outcome.elapsed_s,
                'kept_reward': kept, 'dropped_reward': placed.dropped, 'invalid_rows': invalid,
                'valid_rows': transitions.valid_rows, 'wall_s': wall,
                'infrastructure': outcome.termination_reason == INFRASTRUCTURE,
                'policy_stats': {k: v for k, v in policy.stats.items() if not isinstance(v, dict)}}}


def evaluation_seed(runtime, namespace, cell, realization):
    """The disturbance seed of an evaluation episode: P3's own rule, so the two members of a pair share it."""
    import p3_design
    return p3_design.seed_for(runtime.namespaces, namespace, cell['seeds']['seed_cell_index'], realization)


def evaluate(model, runtime, evaluation, point_index, gamma_ref, tau, label):
    """Mean J of a frozen model over every core cell x realisation, per read-out mode, with the frozen cell weights.

    The sampled policy is the primary read-out (protocol section 4.4); its action stream for a given evaluation
    point, cell and realisation is SeedSequence([EVAL_STREAM, point, cell, realisation]) - the same for every arm.
    """
    namespace, realizations = runtime.study.suites[evaluation['suite']]
    was_training = model.training
    model.eval()
    results = {}
    for mode in evaluation['modes']:
        episodes = []
        for cell in runtime.cells:
            for r in realizations:
                seed = evaluation_seed(runtime, namespace, cell, r)
                generator = (torch.Generator().manual_seed(runtime.study.stream(EVAL_STREAM, point_index,
                                                                                 cell['cell_index'], r))
                             if mode == 'sample' else None)
                run = run_learned_episode(runtime, cell, namespace, seed, model, generator, mode == 'argmax',
                                          f'{label}-{mode}-c{cell["cell_index"]}-r{r}', -1, gamma_ref, tau)
                episodes.append({k: run['record'][k] for k in ('cell_id', 'seed', 'J', 'tp_timely', 'tp_spatial',
                                                               'fp_spatial', 'return_ok', 'fixes_attempted',
                                                               'decisions', 'termination_reason')})
        per_cell, weighted = aggregate_evaluation(runtime, episodes)
        results[mode] = {'weighted_J': weighted, 'per_cell_J': per_cell, 'episodes': episodes}
    model.train(was_training)
    return results


def aggregate_evaluation(runtime, episodes):
    """(per-cell mean J, frozen-weighted J) of one evaluation's episodes - THE aggregation.

    The trainer writes its evaluation with this function and the selector re-derives every score and every AUC
    from the episode rows with this same function (review 24), so a stored value that differs from its
    episodes by even one ULP is a refusal rather than a tolerance question. Episodes are grouped by cell in the
    frozen cell order and averaged in row order.
    """
    per_cell = {}
    for cell in runtime.cells:
        values = [e['J'] for e in episodes if e['cell_id'] == cell['cell_id']]
        per_cell[cell['cell_id']] = float(np.mean(values))
    weighted = float(sum(runtime.probabilities[i] * per_cell[c['cell_id']] for i, c in enumerate(runtime.cells)))
    return per_cell, weighted


def learning_curve_auc(points, values, budget):
    """AUC_ai of statistical protocol section 4.3: the trapezoid of mean J over the decision grid, divided by B."""
    pairs = sorted(zip(points, values))
    area = sum((b1 - b0) * (v1 + v0) / 2. for (b0, v0), (b1, v1) in zip(pairs, pairs[1:]))
    return area / budget


class JsonLines:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, record):
        with open(self.path, 'a') as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + '\n')


def train_arm(arm, config, runtime, arm_dir):
    """Train one arm of block `config['block']`; returns its summary. Checkpoints land on the declared grid."""
    block, ppo = config['block'], config['ppo']
    gamma, tau = config['gamma_ref'], config['reference_duration_s']
    study = study_of(config)
    if runtime.study is not study:
        raise P4Error(f'a {study.name} configuration needs a {study.name} runtime, not {runtime.study.name}')
    # P4's labels, tool and note are unchanged; another study writes its own everywhere they appear.
    label = 'p4' if study.name == 'p4' else study.name
    tool = study.tool
    note = f'P4 development block {block}' if study.name == 'p4' else f'{study.name.upper()} block {block}'
    torch.set_num_threads(config['torch_threads'])
    model = build_model(arm, config, block)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo['learning_rate'])
    action_generator = torch.Generator().manual_seed(study.stream(block, ACTION_STREAM))
    minibatch_generator = torch.Generator().manual_seed(study.stream(block, MINIBATCH_STREAM))
    # The identity reads the action space's version and action order from the SAME public stack an episode builds.
    context = runtime.static(runtime.cells[0])['context']
    stack = context['build_mission'](context['protocol'], context['briefing'], context['obstacle_data'],
                                     context['vehicle_manifest'])
    action_space = ActionSpace(context['briefing'], stack['config'], LAYOUT_V4_GNSS, RETURN_MASK_RULE,
                               anchor_ids=runtime.anchor_ids())
    identity = identity_for_gnss(model, action_space, runtime.training_disturbance(),
                                 runner.AnchorSet.from_dict(runtime.frozen['anchors']),
                                 runtime.execution['manifest_sha256'], REWARD_VERSION, SCORE_VERSION,
                                 LAYOUT_V4_GNSS.obs_version, seed=block, note=note)
    arm_dir.mkdir(parents=True)
    (arm_dir / 'checkpoints').mkdir()
    log = JsonLines(arm_dir / 'metrics.jsonl')
    totals = {'updates': 0, 'policy_decisions': 0, 'episodes': 0, 'empty_episodes': 0, 'infrastructure_episodes': 0,
              'dropped_reward': 0.}
    walls = {'rollout_s': 0., 'update_s': 0., 'checkpoint_s': 0.}
    # Process CPU time beside wall time (E1 section 6): CPU seconds are far less sensitive to the other jobs on the
    # machine than wall seconds, and the load average at start and end records how busy it was.
    cpu = {'rollout_s': 0., 'update_s': 0., 'checkpoint_s': 0., 'evaluation_s': 0.}
    load_average = {'start': list(os.getloadavg())}
    done_points, checkpoints = [], []
    start = time.perf_counter()

    curve = {mode: [] for mode in (config.get('evaluation') or {}).get('modes', [])}

    def evaluate_here(points):
        if not config.get('evaluation'):
            return
        t0, c0 = time.perf_counter(), time.process_time()
        point_index = config['checkpoint_decisions'].index(points[0])
        results = evaluate(model, runtime, config['evaluation'], point_index, gamma, tau,
                           f'{label}-{arm}-b{block}-eval{point_index:02d}')
        walls['evaluation_s'] = walls.get('evaluation_s', 0.) + time.perf_counter() - t0
        cpu['evaluation_s'] += time.process_time() - c0
        # Protocol 4.3's overshoot convention, frozen here: a grid point is evaluated at the FIRST update boundary
        # at or after it, the curve is indexed by the NOMINAL point, and the actual count is recorded beside it.
        # No interpolation.
        for mode, result in results.items():
            for point in points:                 # every nominal point this evaluation stands for, once each
                curve[mode].append((point, totals['policy_decisions'], result['weighted_J']))
        log.write({'kind': 'evaluation', 'arm': arm, 'points': points, 'policy_decisions': totals['policy_decisions'],
                   'results': results, 'wall_s': time.perf_counter() - t0})

    def checkpoint(points, final=False):
        t0, c0 = time.perf_counter(), time.process_time()
        name = f'{"final_" if final else ""}update{totals["updates"]:05d}_decisions{totals["policy_decisions"]:08d}.pt'
        path = arm_dir / 'checkpoints' / name
        save_gnss_checkpoint(path, model, optimizer, config, identity,
                             metadata={'tool': tool, 'arm': arm, 'role': ARM_ROLES[arm], 'block': block,
                                       'protocol_freeze_sha256': config.get('protocol_freeze_sha256'),
                                       'points': points, 'final': final, 'updates': totals['updates'],
                                       'policy_decisions': totals['policy_decisions'], 'episodes': totals['episodes'],
                                       'reward_placement': GNSS_REWARD_PLACEMENT_VERSION})
        walls['checkpoint_s'] += time.perf_counter() - t0
        cpu['checkpoint_s'] += time.process_time() - c0
        entry = {'kind': 'checkpoint', 'path': str(path.relative_to(arm_dir)), 'sha256': file_sha256(path),
                 'points': points, 'final': final, 'policy_decisions': totals['policy_decisions']}
        log.write(entry)
        checkpoints.append(entry)

    def due():
        hit = [p for p in config['checkpoint_decisions'] if p not in done_points and totals['policy_decisions'] >= p]
        done_points.extend(hit)
        return hit

    salt = ''.join(f'{s}, ' for s in study.salt)
    log.write({'kind': 'arm_start', 'utc': utc(), 'tool': tool, 'arm': arm, 'role': ARM_ROLES[arm], 'block': block,
               'study': study.name, 'model': model.describe(), 'identity': identity.to_dict(),
               'non_claim': study.non_claim,
               'seeds': {'model_init': f'SeedSequence([{salt}{block}, {INIT_STREAM}])',
                         'action_sampling': f'SeedSequence([{salt}{block}, {ACTION_STREAM}])',
                         'minibatch_order': f'SeedSequence([{salt}{block}, {MINIBATCH_STREAM}])',
                         'schedule': f'cell: SeedSequence([{salt}{block}, {SCHEDULE_STREAM}, episode]); seed: '
                                     f'{study.training_namespace}.start + {block} * {BLOCK_STRIDE} + episode'}})
    points = due()
    if points:
        checkpoint(points)
        evaluate_here(points)
    failure = None
    def one_update():
        buffer = RolloutBuffer(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, config['hidden'])
        rollout0, rollout_cpu0, kept, episodes_here, sim_s = time.perf_counter(), time.process_time(), 0, 0, 0.
        while kept < config['transitions_per_update']:
            number = totals['episodes']
            fixed = config.get('fixed_schedule')
            if fixed is None:
                cell, seed = runtime.schedule(block, number)
            else:
                # The learning-sanity run of WP4: one cell and one disturbance, repeated, so a working optimiser has
                # to show a rising J on an episode it sees again and again. Never a schedule of a real block.
                _, seed = runtime.schedule(block, fixed['episode'])
                cell = runtime.cells[fixed['cell_index']]
            run = run_learned_episode(runtime, cell, study.training_namespace, seed, model, action_generator, False,
                                      f'{label}-{arm}-b{block}-e{number:06d}', number, gamma, tau)
            totals['episodes'] += 1
            episodes_here += 1
            totals['dropped_reward'] += run['placed'].dropped
            totals['infrastructure_episodes'] += run['record']['infrastructure']
            sim_s += run['record']['elapsed_s'] or 0.
            if run['transitions'].valid_rows:
                run['transitions'].add_to(buffer)
                kept += run['transitions'].valid_rows
            else:
                totals['empty_episodes'] += 1
            log.write(dict(run['record'], kind='episode', arm=arm, episode=number, update_index=totals['updates'] + 1))
        rollout_wall, rollout_cpu = time.perf_counter() - rollout0, time.process_time() - rollout_cpu0
        batch = compute_advantages(buffer, gamma, tau, config['lambda_ref'], config['lambda_mode'],
                                   ppo['max_segment_length'])
        update0, update_cpu0 = time.perf_counter(), time.process_time()
        metrics = ppo_update(model, optimizer, batch, ppo['clip_range'], ppo['value_coef'], ppo['entropy_coef'],
                             ppo['max_grad_norm'], ppo['epochs'], generator=minibatch_generator,
                             minibatch_transitions=ppo['minibatch_transitions'],
                             advantage_std_floor=ppo['advantage_std_floor'])
        update_wall, update_cpu = time.perf_counter() - update0, time.process_time() - update_cpu0
        # review 23, P4 item 3: a non-finite PARAMETER after an update is a terminal algorithmic failure
        # here and now - otherwise the NaN would surface in the next rollout as some other exception and be
        # misread as an infrastructure crash with no summary.
        bad = [name for name, parameter in model.named_parameters() if not torch.isfinite(parameter).all()]
        if bad:
            raise FloatingPointError(f'non-finite parameters after update {totals["updates"] + 1}: {bad[:4]}')
        used = int(metrics['transitions'])
        if used != kept:
            raise P4Error(f'internal: {kept} kept transitions collected, {used} used by the update')
        totals['updates'] += 1
        totals['policy_decisions'] += used
        walls['rollout_s'] += rollout_wall
        walls['update_s'] += update_wall
        cpu['rollout_s'] += rollout_cpu
        cpu['update_s'] += update_cpu
        log.write({'kind': 'update', 'arm': arm, 'update_index': totals['updates'], 'metrics': metrics,
                   'transitions': used, 'episodes': episodes_here, 'sim_s': sim_s,
                   'wall': {'rollout_s': rollout_wall, 'update_s': update_wall,
                            'rollout_per_decision_s': rollout_wall / used, 'update_per_decision_s': update_wall / used},
                   'cumulative': dict(totals)})
        points = due()
        if points:
            checkpoint(points)
            evaluate_here(points)

    while failure is None and totals['policy_decisions'] < config['total_policy_decisions'] and \
            (config['max_updates'] is None or totals['updates'] < config['max_updates']):
        try:
            one_update()
        except ALGORITHMIC_FAILURES as error:
            # An outcome of the algorithm (protocol section 6): recorded and kept. The run stops here, keeps the
            # checkpoints it has, writes NO final checkpoint and is never replaced by a run on other seeds.
            failure = {'kind': type(error).__name__, 'message': str(error),
                       'policy_decisions': totals['policy_decisions'], 'updates': totals['updates']}
            log.write({'kind': 'algorithmic_failure', 'arm': arm, **failure})
    if failure is None:
        checkpoint([], final=True)
    wall = time.perf_counter() - start
    load_average['end'] = list(os.getloadavg())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {'arm': arm, 'role': ARM_ROLES[arm], 'block': block, 'run_role': config['role'],
               'candidate': config.get('candidate'), 'protocol_freeze_sha256': config.get('protocol_freeze_sha256'),
               'study': study.name,
               **totals, 'wall_s': wall, 'walls': walls, 'cpu': cpu, 'load_average': load_average,
               'process_rusage': {'user_s': usage.ru_utime, 'system_s': usage.ru_stime,
                                  'max_rss_kib': usage.ru_maxrss},
               'rollout_per_decision_s': walls['rollout_s'] / max(totals['policy_decisions'], 1),
               'update_per_decision_s': walls['update_s'] / max(totals['policy_decisions'], 1),
               'actor_head_parameters': sum(p.numel() for p in model.actor.parameters()),
               'model_parameters': sum(p.numel() for p in model.parameters()),
               'checkpoints': checkpoints,
               'status': 'completed' if failure is None else 'algorithmic_failure', 'failure': failure,
               'learning_curve': {mode: values for mode, values in curve.items()},
               # No AUC over a curve the algorithm did not finish: protocol section 6 forbids inventing the rest.
               'auc': None if failure is not None else {
                   mode: learning_curve_auc([p for p, _, _ in values], [v for _, _, v in values],
                                            config['total_policy_decisions'])
                   for mode, values in curve.items() if len(values) > 1}}
    log.write(dict(summary, kind='arm_end', utc=utc()))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; this tool never overwrites a run directory')
    raw = json.loads(args.config.read_text())
    config = validate_config(raw)
    study = study_of(config)
    freeze = None
    if config['role'] in study.roles:
        # Refused BEFORE the run directory exists: no approval, another freeze, a changed member file, another
        # seed manifest or another live tree (review 23, P4 item 1).
        freeze_path, approved, _ = require_approved_freeze(config['protocol_freeze_sha256'], study=study)
        freeze = {'path': str(freeze_path), 'sha256': approved}
    runtime = CellRuntime(ROOT / config['freeze'], ROOT / config['execution_manifest'], ROOT / config['time_model'],
                          study=study)
    args.out.mkdir(parents=True)
    (args.out / 'run_manifest.json').write_text(json.dumps(
        {'tool': study.tool, 'utc': utc(), 'study': study.name, 'config': config,
         'config_sha256': canonical_sha256(config), 'code_tree_sha256': live_copy_tree(), 'protocol_freeze': freeze,
         'non_claim': study.non_claim, 'host': {'cpu_count': os.cpu_count(), 'torch': torch.__version__}},
        indent=1) + '\n')
    summaries = [train_arm(arm, config, runtime, args.out / arm) for arm in config['arms']]
    (args.out / 'summary.json').write_text(json.dumps({'tool': study.tool, 'utc': utc(), 'arms': summaries}, indent=1,
                                                      default=float) + '\n')
    print(json.dumps([{k: s[k] for k in ('arm', 'role', 'policy_decisions', 'episodes', 'wall_s',
                                         'rollout_per_decision_s', 'update_per_decision_s', 'actor_head_parameters')}
                      for s in summaries], indent=1, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
