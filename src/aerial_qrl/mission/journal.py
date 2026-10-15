"""Append-only on-disk lifecycle journal for crash and power-loss accounting.

Format      One JSON object per line: {seq, utc, monotonic_ns, kind, ...}. kind is 'intent'
            (written and fsync'ed before an operation runs), 'result' (after it returns or fails,
            referencing intent_seq), 'identity' (a process this run owns) or 'note'.
Clock       utc is wall time for humans; monotonic_ns orders records within one boot only.
Failures    A final line without newline or with invalid JSON is a torn tail (power loss) and is
            reported, not repaired. Invalid lines before the tail are corruption and raise.
Semantics   The journal never declares READY or completion. Recovery reads it to decide whether
            the last run ended with a successful stop; anything else is 'interrupted'.
Stdlib only, Python >= 3.9.
"""
import datetime
import json
import os
from pathlib import Path
import time

KINDS = ('intent', 'result', 'identity', 'note')


class JournalCorruption(ValueError):
    pass


def read_journal(path):
    """Return (records, torn_tail). Missing file -> ([], False)."""
    path = Path(path)
    if not path.exists():
        return [], False
    data = path.read_bytes()
    lines = data.split(b'\n')
    complete, tail = lines[:-1], lines[-1]
    records = []
    for index, line in enumerate(complete):
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or record.get('kind') not in KINDS:
                raise ValueError('bad record')
        except ValueError:
            if index == len(complete) - 1 and not tail:
                return records, True  # last complete-looking line is unreadable: torn write
            raise JournalCorruption(f'{path}: invalid record at line {index + 1}')
        records.append(record)
    return records, bool(tail)


class LifecycleJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records, torn = read_journal(self.path)
        if torn:
            raise JournalCorruption(f'{self.path}: torn tail; run recovery before appending')
        self.seq = records[-1]['seq'] if records else 0

    def append(self, kind, **fields):
        if kind not in KINDS:
            raise ValueError(f'unknown journal kind {kind}')
        record = {'seq': self.seq + 1, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'monotonic_ns': time.monotonic_ns(), 'kind': kind, **fields}
        line = json.dumps(record, sort_keys=True, allow_nan=False, default=str) + '\n'
        with open(self.path, 'a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.seq += 1
        return record


def summarize(records, torn_tail=False):
    """Classify a journal: 'empty', 'clean_stop' or 'interrupted', plus open intents and identities."""
    results = {r['intent_seq'] for r in records if r['kind'] == 'result'}
    open_intents = [r for r in records if r['kind'] == 'intent' and r['seq'] not in results]
    identities = [r for r in records if r['kind'] == 'identity']
    stops = [r for r in records if r['kind'] == 'result' and r.get('operation') == 'stop']
    if not records:
        status = 'empty'
    elif stops and stops[-1]['ok'] and not open_intents and not torn_tail and records[-1]['seq'] == stops[-1]['seq']:
        status = 'clean_stop'
    else:
        status = 'interrupted'
    return {'status': status, 'records': len(records), 'torn_tail': torn_tail, 'open_intents': open_intents,
            'identities': identities, 'last_record': records[-1] if records else None}
