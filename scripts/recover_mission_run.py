"""Recover an interrupted mission run from its on-disk lifecycle journal(s).

  .venv/bin/python scripts/recover_mission_run.py <run_dir> [--execute]

Reads every lifecycle_journal.jsonl below run_dir. A journal that did not end with a successful stop
is 'interrupted': its open intents and recorded process identities are reported, and each identity is
checked in /proc (pid, process group and start time must all match). Only with --execute is a verified
process group signalled (SIGINT, then SIGTERM, then SIGKILL), re-checking the identity before every
signal; a reused or foreign PID is never signalled. Recovery never marks anything READY or completed.
It writes RECOVERY_<utc>.json as a new file and leaves every run file unchanged.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aerial_qrl.mission.journal import read_journal, summarize

MAVLINK_UDP_PORTS = (14540, 14550, 14580, 18570)
ESCALATION = ((signal.SIGINT, 15.), (signal.SIGTERM, 10.), (signal.SIGKILL, 5.))


def proc_identity(pid):
    """Identity of a live process; zombies (state Z) count as not running."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'process_group': os.getpgid(pid), 'start_ticks': stat[19]}
    except (FileNotFoundError, ProcessLookupError, IndexError, PermissionError):
        return None


def identity_matches(recorded, current):
    return current is not None and all(str(recorded.get(k)) == str(current.get(k)) for k in ('pid', 'process_group', 'start_ticks'))


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_verified_group(recorded, execute, escalation=ESCALATION):
    current = proc_identity(recorded['pid'])
    action = {'role': recorded.get('role'), 'recorded': {k: recorded.get(k) for k in ('pid', 'process_group', 'start_ticks')}, 'current': current}
    if current is None:
        action['decision'] = 'not_running'
        return action
    if not identity_matches(recorded, current):
        action['decision'] = 'identity_mismatch_not_signalled'
        return action
    if not execute:
        action['decision'] = 'verified_alive_dry_run'
        return action
    signals = []
    for sig, wait_s in escalation:
        if not identity_matches(recorded, proc_identity(recorded['pid'])):
            break
        os.killpg(int(recorded['process_group']), sig)
        signals.append(sig.name)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and identity_matches(recorded, proc_identity(recorded['pid'])):
            time.sleep(.1)
    action.update(decision='signalled_verified_group', signals=signals,
                  leader_still_alive=identity_matches(recorded, proc_identity(recorded['pid'])),
                  group_still_alive=group_alive(int(recorded['process_group'])))
    return action


def busy_ports(ports=MAVLINK_UDP_PORTS):
    busy = []
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(('127.0.0.1', port))
        except OSError:
            busy.append(port)
        finally:
            s.close()
    return busy


def recover(run_dir, execute=False, escalation=ESCALATION):
    run_dir = Path(run_dir).resolve()
    journals = []
    for path in sorted(run_dir.rglob('lifecycle_journal.jsonl')):
        records, torn = read_journal(path)
        summary = summarize(records, torn)
        entry = {'journal': str(path), 'status': summary['status'], 'torn_tail': torn, 'records': summary['records'],
                 'open_intents': [{k: r.get(k) for k in ('seq', 'operation', 'from_state', 'generation', 'utc')} for r in summary['open_intents']],
                 'last_record': summary['last_record'], 'actions': []}
        # Identities are checked for every journal: even a clean stop must not have left owned processes alive.
        for identity in summary['identities']:
            entry['actions'].append(stop_verified_group(identity, execute and summary['status'] != 'clean_stop', escalation)
                                    if summary['status'] != 'clean_stop' else dict(stop_verified_group(identity, False), note='clean_stop_journal_check_only'))
        journals.append(entry)
    return {'recovery_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'run_dir': str(run_dir), 'execute': execute,
            'journals': journals, 'interrupted_journals': sum(j['status'] == 'interrupted' for j in journals),
            'mavlink_ports_busy_after': busy_ports(), 'claims': 'recovery never declares READY or completion'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    report = recover(args.run_dir, args.execute)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    with open(Path(args.run_dir) / f'RECOVERY_{stamp}.json', 'x') as f:
        f.write(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({'interrupted_journals': report['interrupted_journals'],
                      'decisions': [a['decision'] for j in report['journals'] for a in j['actions']],
                      'mavlink_ports_busy_after': report['mavlink_ports_busy_after']}, indent=2))


if __name__ == '__main__':
    main()
