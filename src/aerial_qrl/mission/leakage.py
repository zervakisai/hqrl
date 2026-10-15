"""Name-based guard for public payloads: defence in depth, not a proof of no leakage.

Stdlib only and Python 3.9 compatible, because bridge servers run in other interpreters.
Only keys are inspected; values are the responsibility of the typed contracts.
"""
import re

FORBIDDEN_TOKENS = frozenset({
    'truth', 'gt', 'seed', 'target', 'targets', 'segmentation', 'semantic', 'visibility', 'visible',
    'correct', 'correctness', 'evaluator', 'oracle', 'scenario', 'privileged',
})


class PublicLeakError(ValueError):
    pass


def key_tokens(key):
    snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', key).lower()
    return {t for t in re.split(r'[^a-z0-9]+', snake) if t}


def find_forbidden_keys(payload, path='$'):
    hits = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            where = '%s.%s' % (path, key)
            if not isinstance(key, str) or key_tokens(key) & FORBIDDEN_TOKENS:
                hits.append(where)
            hits.extend(find_forbidden_keys(value, where))
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            hits.extend(find_forbidden_keys(value, '%s[%d]' % (path, i)))
    return hits


def assert_public_payload(payload):
    hits = find_forbidden_keys(payload)
    if hits:
        raise PublicLeakError('Truth-like keys in public payload: ' + ', '.join(hits[:10]))
    return payload
