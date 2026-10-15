"""Deterministic code version for an untracked repository (no git commits exist yet).

  .venv/bin/python scripts/code_version.py [--out <file.json>]

The version is the SHA256 of sorted "relative_path<TAB>file_sha256" lines over the code, tests and
configuration that determine behaviour. Outputs (output/, runs/, tmp/) and caches are excluded.
Reports cite the tree hash together with the JSON file listing every file hash.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [('src/aerial_qrl', '*.py'), ('scripts', '*.py'), ('scripts', '*.sh'), ('tests', '*.py'), ('configs', '*.json')]
SINGLE = ['pyproject.toml', 'requirements-lock.txt']


def file_hashes(root=ROOT):
    files = set()
    for folder, pattern in INCLUDE:
        files.update(p for p in (root / folder).rglob(pattern) if '__pycache__' not in p.parts)
    files.update(root / name for name in SINGLE if (root / name).exists())
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def tree_hash(hashes):
    lines = ''.join(f'{path}\t{digest}\n' for path, digest in sorted(hashes.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    args = parser.parse_args()
    hashes = file_hashes()
    record = {'code_tree_sha256': tree_hash(hashes), 'file_count': len(hashes), 'files': hashes}
    if args.out:
        with open(args.out, 'x') as f:
            f.write(json.dumps(record, indent=2) + '\n')
    print(record['code_tree_sha256'])


if __name__ == '__main__':
    main()
