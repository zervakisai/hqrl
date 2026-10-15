"""Two-level identity for GNSS checkpoints (model contract invariant 11; reviews 1 R2 and 2).

The distinction this module exists to make
    Two questions look similar and are not the same, and review 2 was right that collapsing them is unsafe:

    1. **Are these runs part of the same paired comparison?** That asks whether they share the experimental
       condition: task, encoder, action order, reward, score, disturbance model and configuration, anchor set,
       data manifest and the common critic. It must IGNORE the actor head and the seed, because grouping
       `quantum`, `bottleneck`, `surrogate625` and `linear` into one paired block is the study's design.
       -> `ConditionIdentity`.

    2. **May these weights be loaded into this model?** That asks whether the parameter tensors mean the same
       thing. It must NOT ignore the actor head: a quantum head's weights are not a linear head's weights, and
       resuming or evaluating across them is a silent corruption, not a comparison.
       -> `CheckpointIdentity` = the condition, plus the live model architecture.

    Every GNSS checkpoint stores both, and both fingerprints, so a reader can answer either question without
    re-deriving anything.

Seeds
    A seed is provenance, never a compatibility field: a paired block is many seeds under one condition. Use
    `paired_block_key(condition, seed)` when a block needs a key; it is deliberately a separate function.

Fail closed
    `load_gnss_checkpoint` verifies the identity AFTER reading the payload and BEFORE the model is constructed
    or any weight is touched. A checkpoint with no identity is refused, not warned about.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

IDENTITY_VERSION = 'gnss-checkpoint-identity/2'
#: Experimental-condition fields. Two runs that agree on all of these belong in one paired comparison.
CONDITION_FIELDS = ('identity_version', 'task_id', 'obs_encoding_version', 'action_space_version', 'action_order',
                    'reward_version', 'score_version', 'disturbance_model', 'disturbance_config_sha256',
                    'anchor_set_sha256', 'manifest_sha256', 'critic')
#: Architecture fields. Two checkpoints that disagree on any of these must never load into one another.
MODEL_FIELDS = ('actor_head', 'model_config_sha256')
#: Recorded, never compared.
PROVENANCE_FIELDS = ('seed', 'note')


class CheckpointIdentityError(ValueError):
    """A checkpoint's promise does not match the run that is trying to use it."""


def canonical_sha256(payload):
    """Hash of a JSON-serialisable object, key order independent, NaN/inf refused."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def model_config_sha256(model_config):
    """Hash of the normalised model configuration that fixes parameter meaning and shape.

    `smdp_ppo.RecurrentActorCritic.config` holds exactly these: obs_dim, action_dim, critic, hidden,
    critic_gradient, memory, actor_head. Hashing the whole dict means a future architecture field is covered
    the day it is added, rather than the day someone remembers to list it here.
    """
    config = dict(model_config)
    missing = {'obs_dim', 'action_dim', 'critic', 'hidden', 'memory', 'actor_head'} - set(config)
    if missing:
        raise CheckpointIdentityError(f'model_config: missing architecture fields {sorted(missing)}')
    return canonical_sha256(config)


def _check_strings(owner, fields):
    for name in fields:
        value = getattr(owner, name)
        if name == 'action_order':
            if not isinstance(value, (tuple, list)) or not value or not all(isinstance(v, str) and v for v in value):
                raise CheckpointIdentityError('action_order: expected a non-empty tuple of names')
            object.__setattr__(owner, name, tuple(value))
            continue
        if not isinstance(value, str) or not value:
            raise CheckpointIdentityError(f'{name}: a non-empty string is required, got {value!r}')


@dataclass(frozen=True)
class ConditionIdentity:
    """The shared experimental condition. Actor head and seed are deliberately absent."""
    task_id: str
    obs_encoding_version: str
    action_space_version: str
    action_order: Tuple[str, ...]
    reward_version: str
    score_version: str
    disturbance_model: str
    disturbance_config_sha256: str
    anchor_set_sha256: str
    manifest_sha256: str
    critic: str
    identity_version: str = IDENTITY_VERSION

    def __post_init__(self):
        _check_strings(self, CONDITION_FIELDS)
        if self.identity_version != IDENTITY_VERSION:
            raise CheckpointIdentityError(f'identity_version: expected {IDENTITY_VERSION}')

    def to_dict(self):
        data = asdict(self)
        data['action_order'] = list(self.action_order)
        return data

    @classmethod
    def from_dict(cls, data):
        return _from_dict(cls, data, CONDITION_FIELDS)

    @property
    def fingerprint(self):
        """Groups the four arms of one paired block: identical for quantum, bottleneck, surrogate625, linear."""
        return canonical_sha256({f: (list(getattr(self, f)) if f == 'action_order' else getattr(self, f))
                                 for f in CONDITION_FIELDS})


@dataclass(frozen=True)
class CheckpointIdentity:
    """The condition plus the architecture the weights belong to."""
    condition: ConditionIdentity
    actor_head: str
    model_config_sha256: str
    seed: Optional[int] = None                   # provenance only
    note: Optional[str] = None                   # provenance only

    def __post_init__(self):
        condition = self.condition
        if not isinstance(condition, ConditionIdentity):
            object.__setattr__(self, 'condition', ConditionIdentity.from_dict(condition))
        _check_strings(self, MODEL_FIELDS)
        if self.seed is not None and not isinstance(self.seed, int):
            raise CheckpointIdentityError(f'seed: an integer or None is required, got {self.seed!r}')

    def to_dict(self):
        return {'identity_version': IDENTITY_VERSION, 'condition': self.condition.to_dict(),
                'actor_head': self.actor_head, 'model_config_sha256': self.model_config_sha256,
                'seed': self.seed, 'note': self.note,
                'condition_fingerprint': self.condition_fingerprint, 'checkpoint_fingerprint': self.checkpoint_fingerprint}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise CheckpointIdentityError('CheckpointIdentity: expected an object')
        payload = {k: v for k, v in data.items()
                   if k not in ('condition_fingerprint', 'checkpoint_fingerprint', 'identity_version')}
        missing = sorted({'condition', 'actor_head', 'model_config_sha256'} - set(payload))
        if missing:
            raise CheckpointIdentityError(f'CheckpointIdentity: missing fields {missing}')
        extra = sorted(set(payload) - {'condition', 'actor_head', 'model_config_sha256', 'seed', 'note'})
        if extra:
            raise CheckpointIdentityError(f'CheckpointIdentity: unknown fields {extra}')
        return cls(**payload)

    @property
    def condition_fingerprint(self):
        return self.condition.fingerprint

    @property
    def checkpoint_fingerprint(self):
        return canonical_sha256({'condition': self.condition.fingerprint,
                                 **{f: getattr(self, f) for f in MODEL_FIELDS}})


def _from_dict(cls, data, required):
    if not isinstance(data, dict):
        raise CheckpointIdentityError(f'{cls.__name__}: expected an object')
    names = set(cls.__dataclass_fields__)
    extra = sorted(set(data) - names)
    if extra:
        raise CheckpointIdentityError(f'{cls.__name__}: unknown fields {extra}')
    missing = sorted(f for f in required if f not in data and f != 'identity_version')
    if missing:
        raise CheckpointIdentityError(f'{cls.__name__}: missing fields {missing}')
    payload = dict(data)
    if 'action_order' in payload:
        payload['action_order'] = tuple(payload['action_order'])
    return cls(**payload)


def identity_for_gnss(model, action_space, georeference_config, anchor_set, manifest_sha256,
                      reward_version, score_version, obs_encoding_version, seed=None, note=None,
                      task_id='flood-sar-village-gnss/1'):
    """Build both levels from the LIVE objects, so neither can drift from what a run actually uses.

    `model` is a `smdp_ppo.RecurrentActorCritic` (anything exposing `.config`): the architecture is read off it
    rather than restated, so a changed head, hidden size or action dimension changes the checkpoint identity
    without anyone editing a constant.
    """
    config = dict(model.config)
    condition = ConditionIdentity(
        task_id=task_id,
        obs_encoding_version=obs_encoding_version,
        action_space_version=action_space.version,
        action_order=tuple(action_space.describe()['action_names']),
        reward_version=reward_version,
        score_version=score_version,
        disturbance_model=georeference_config.process_model,
        disturbance_config_sha256=canonical_sha256(georeference_config.to_dict()),
        anchor_set_sha256=canonical_sha256(anchor_set.to_dict()),
        manifest_sha256=manifest_sha256,
        critic=config['critic'])
    return CheckpointIdentity(condition=condition, actor_head=config['actor_head'],
                              model_config_sha256=model_config_sha256(config), seed=seed, note=note)


def condition_differences(saved, current):
    saved = saved if isinstance(saved, ConditionIdentity) else ConditionIdentity.from_dict(saved)
    current = current if isinstance(current, ConditionIdentity) else ConditionIdentity.from_dict(current)
    return [f for f in CONDITION_FIELDS if getattr(saved, f) != getattr(current, f)]


def checkpoint_differences(saved, current):
    saved = saved if isinstance(saved, CheckpointIdentity) else CheckpointIdentity.from_dict(saved)
    current = current if isinstance(current, CheckpointIdentity) else CheckpointIdentity.from_dict(current)
    return ([f'condition.{f}' for f in condition_differences(saved.condition, current.condition)]
            + [f for f in MODEL_FIELDS if getattr(saved, f) != getattr(current, f)])


def _raise(what, mismatched, saved, current, getter):
    detail = '; '.join(f'{f}: checkpoint {getter(saved, f)!r} vs run {getter(current, f)!r}' for f in mismatched)
    raise CheckpointIdentityError(f'{what}: incompatible identity on {mismatched} -> {detail}')


def check_condition_compatible(saved, current, what='paired block'):
    """Do these two runs belong to the same paired comparison? Ignores actor head and seed by design."""
    if saved is None:
        raise CheckpointIdentityError(f'{what}: no condition identity recorded')
    mismatched = condition_differences(saved, current)
    if mismatched:
        saved = saved if isinstance(saved, ConditionIdentity) else ConditionIdentity.from_dict(saved)
        current = current if isinstance(current, ConditionIdentity) else ConditionIdentity.from_dict(current)
        _raise(what, mismatched, saved, current, getattr)
    return True


def check_checkpoint_compatible(saved, current, what='checkpoint'):
    """May these weights be loaded here? Condition AND architecture must agree; the actor head is decisive."""
    if saved is None:
        raise CheckpointIdentityError(
            f'{what}: no task identity recorded. A checkpoint without an identity cannot be resumed or '
            'evaluated under flood-sar-village-gnss/1.')
    saved = saved if isinstance(saved, CheckpointIdentity) else CheckpointIdentity.from_dict(saved)
    current = current if isinstance(current, CheckpointIdentity) else CheckpointIdentity.from_dict(current)
    mismatched = checkpoint_differences(saved, current)
    if mismatched:
        def getter(identity, name):
            return getattr(identity.condition, name.split('.', 1)[1]) if name.startswith('condition.') \
                else getattr(identity, name)
        _raise(what, mismatched, saved, current, getter)
    return True


def paired_block_key(condition, seed):
    """The key of one paired block: the shared condition plus the seed, kept separate from compatibility."""
    fingerprint = condition.fingerprint if isinstance(condition, ConditionIdentity) else \
        ConditionIdentity.from_dict(condition).fingerprint
    if not isinstance(seed, int):
        raise CheckpointIdentityError(f'seed: an integer is required for a block key, got {seed!r}')
    return f'{fingerprint[:16]}:seed{seed}'


# --- checkpoint integration -------------------------------------------------------------------------------

IDENTITY_METADATA_KEY = 'gnss_task_identity'


def save_gnss_checkpoint(path, model, optimizer, train_config, identity, metadata=None, training_state=None):
    """`smdp_ppo.save_checkpoint` with the identity and both fingerprints written into the metadata.

    The identity's architecture is re-derived from the live model here, so a caller cannot save weights under
    an identity that describes a different network.
    """
    from .smdp_ppo import save_checkpoint
    if not isinstance(identity, CheckpointIdentity):
        identity = CheckpointIdentity.from_dict(identity)
    live = model_config_sha256(dict(model.config))
    if identity.model_config_sha256 != live or identity.actor_head != model.config['actor_head']:
        raise CheckpointIdentityError(
            f'save: the identity describes actor_head {identity.actor_head!r} / config {identity.model_config_sha256} '
            f'but the model is {model.config["actor_head"]!r} / {live}')
    payload = dict(metadata or {})
    if IDENTITY_METADATA_KEY in payload:
        raise CheckpointIdentityError(f'metadata already contains {IDENTITY_METADATA_KEY!r}')
    payload[IDENTITY_METADATA_KEY] = identity.to_dict()
    return save_checkpoint(path, model, optimizer, train_config, payload, training_state)


def identity_of_payload(payload):
    """The stored CheckpointIdentity, or None when the checkpoint carries none."""
    metadata = (payload or {}).get('metadata') or {}
    stored = metadata.get(IDENTITY_METADATA_KEY)
    return None if stored is None else CheckpointIdentity.from_dict(stored)


def load_gnss_checkpoint(path, expected, optimizer_factory=None, map_location='cpu', what=None):
    """Load a GNSS checkpoint only if its identity matches `expected`.

    The check runs on the raw payload BEFORE the model is constructed and before any weight is read, so an
    incompatible checkpoint never reaches `load_state_dict`. A checkpoint without an identity is refused.
    """
    from .smdp_ppo import load_checkpoint
    label = what or str(path)

    def payload_check(payload):
        check_checkpoint_compatible(identity_of_payload(payload), expected, label)

    model, optimizer, payload = load_checkpoint(path, optimizer_factory, map_location, payload_check=payload_check)
    return model, optimizer, payload, identity_of_payload(payload)


__all__ = ['CONDITION_FIELDS', 'IDENTITY_METADATA_KEY', 'IDENTITY_VERSION', 'MODEL_FIELDS', 'PROVENANCE_FIELDS',
           'CheckpointIdentity', 'CheckpointIdentityError', 'ConditionIdentity', 'canonical_sha256',
           'check_checkpoint_compatible', 'check_condition_compatible', 'checkpoint_differences',
           'condition_differences', 'identity_for_gnss', 'identity_of_payload', 'load_gnss_checkpoint',
           'model_config_sha256', 'paired_block_key', 'save_gnss_checkpoint']
