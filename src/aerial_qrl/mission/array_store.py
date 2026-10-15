"""Lossless array storage for observation banks (fast-env audit B5; WebP form of the storage plan 2026-09-17).

A bank records every array by a logical `.npy` path and the SHA-256 of its canonical `.npy` bytes (numpy.save, no pickle).
The array itself may be stored as that `.npy` file, as a sibling `.npz` (numpy.savez_compressed, one member 'array') or, for
uint8 RGB images of shape (H, W, 3) with H, W <= 16383, as a sibling lossless `.webp` (Pillow, lossless=True, exact=True).
Readers resolve the stored file and never see a difference; identity checks hash the canonical `.npy` bytes of the loaded
array, which equal the original file bytes for arrays written by numpy.save with the same format version. Depth and label
arrays are never WebP (only uint8 HxWx3 is representable without loss).

Coexistence rule (two stored forms of one logical path, e.g. a crash between the rename and the removal of the source in a
transcode, or a transcode that was told to keep the source): a logical path is never served from an arbitrary one of them.
  * `stored_path` raises `StorageConflictError`: there is no single file behind the logical path any more.
  * `load_array` and `array_sha256` read EVERY form and serve the array only while all of them decode to the same canonical
    `.npy` bytes; otherwise they raise `StorageConflictError` naming the files. A wrong array is therefore never served
    silently, and a bank stays readable while a transcode has not yet removed the sources (at the cost of reading twice).
Repair of a conflict is a decision about which file is right, so no reader makes it: `scripts/transcode_bank_rgb.py` removes
a source only after the WebP copy has been verified against the recorded sha256.

Durability: a check that reads a file back right after writing it reads the page cache, not the disk, so a verified copy
whose source is then removed can still be lost to a power failure inside the writeback window. Every function here that
writes a stored file (`save_webp`) fsyncs it before closing, and `transcode_to_webp` fsyncs the directory after the rename
and again before it removes the source, so the WebP is on stable storage before the only other copy disappears.
`save_array` is the collector's write path, is never followed by a deletion, and is deliberately left without fsync.

Invariant behind `array_sha256`: a lone `.npy` is hashed as its file bytes, every other case as the re-serialised canonical
bytes of the loaded array. The two agree only because `save_array` writes exactly `canonical_npy_bytes(array)`; a `.npy`
written some other way (Fortran order, another format version) would change hash the moment a sibling form appears.
`transcode_to_webp` compares the canonical bytes before it renames anything, so it never creates such a pair.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np

COMPRESSED_SUFFIX = '.npz'
WEBP_SUFFIX = '.webp'
STORED_SUFFIXES = ('.npy', COMPRESSED_SUFFIX, WEBP_SUFFIX)  # read preference order
MEMBER = 'array'
WEBP_MAX_SIDE = 16383  # the WebP bitstream stores 14-bit image dimensions
WEBP_SAVE_KWARGS = {'lossless': True, 'exact': True}  # 'exact' keeps the RGB of fully transparent pixels; we store no alpha


class StorageConflictError(ValueError):
    """Several stored forms of one logical path, and they do not hold the same array (or one may not be read)."""


def canonical_npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def webp_storable(array):
    """True when an array is representable as lossless WebP: uint8, (H, W, 3), 1 <= H, W <= WEBP_MAX_SIDE."""
    array = np.asarray(array)
    return (array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 3
            and 1 <= array.shape[0] <= WEBP_MAX_SIDE and 1 <= array.shape[1] <= WEBP_MAX_SIDE)


def fsync_dir(path):
    """Force a directory entry to stable storage: a rename or an unlink is durable only after its directory is synced."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_webp(path, array):
    """Write a uint8 (H, W, 3) array as lossless WebP (exclusive create), fsynced. Returns (path, sha256 of canonical .npy bytes)."""
    from PIL import Image
    array = np.asarray(array)
    if not webp_storable(array):
        raise ValueError(f'{path}: WebP stores uint8 (H, W, 3) arrays with H, W <= {WEBP_MAX_SIDE}, got {array.dtype} {array.shape}')
    path = Path(path)
    with open(path, 'xb') as handle:  # never overwrite an existing stored form
        Image.fromarray(np.ascontiguousarray(array), 'RGB').save(handle, format='WEBP', **WEBP_SAVE_KWARGS)
        handle.flush()
        os.fsync(handle.fileno())     # the bytes, not only the page cache: a later check must read what survives a crash
    return path, hashlib.sha256(canonical_npy_bytes(array)).hexdigest()


def load_webp(path):
    """The array of a lossless WebP file: mode RGB, dtype uint8, C-contiguous, shape (H, W, 3)."""
    from PIL import Image
    with Image.open(path) as image:
        if image.format != 'WEBP':
            raise ValueError(f'{path}: not a WebP file (format {image.format})')
        if image.mode != 'RGB':
            raise ValueError(f'{path}: WebP mode {image.mode}, expected RGB (arrays are stored without alpha)')
        array = np.asarray(image, dtype=np.uint8)
    array = np.ascontiguousarray(array)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f'{path}: decoded {array.dtype} {array.shape}, expected uint8 (H, W, 3)')
    return array


def stored_paths(path):
    """Every existing stored form of a logical .npy path, in read preference order (.npy, .npz, .webp); [] when none exists."""
    path = Path(path)
    if path.suffix not in STORED_SUFFIXES:
        raise ValueError(f'{path}: a logical array path ends in one of {STORED_SUFFIXES}')
    return [p for p in (path.with_suffix(s) for s in STORED_SUFFIXES) if p.is_file()]


def _existing(path):
    found = stored_paths(path)
    if not found:
        raise FileNotFoundError(f'{path}: no stored form exists ({", ".join(STORED_SUFFIXES)})')
    return found


def stored_path(path):
    """The single file holding the array of a logical .npy path (the .npy itself, its .npz or its .webp sibling).

    Several forms at once are a conflict (see the module docstring): this raises StorageConflictError rather than choose one.
    """
    found = _existing(path)
    if len(found) > 1:
        raise StorageConflictError(f'{path}: {len(found)} stored forms exist ({[p.name for p in found]}); '
                                   'load_array serves them only while they hold the same array')
    return found[0]


def _load_stored(stored):
    if stored.suffix == COMPRESSED_SUFFIX:
        with np.load(stored, allow_pickle=False) as archive:
            if set(archive.files) != {MEMBER}:
                raise ValueError(f'{stored}: expected exactly one member {MEMBER!r}')
            return archive[MEMBER]
    if stored.suffix == WEBP_SUFFIX:
        return load_webp(stored)
    return np.load(stored, allow_pickle=False)


def load_array(path):
    """The array of a logical .npy path, whatever its stored form.

    While several forms exist, every one is decoded and they must hold the same array (StorageConflictError otherwise)."""
    found = _existing(path)
    array = _load_stored(found[0])
    if len(found) == 1:
        return array
    reference = canonical_npy_bytes(array)
    for other in found[1:]:
        if canonical_npy_bytes(_load_stored(other)) != reference:
            raise StorageConflictError(f'{path}: {found[0].name} and {other.name} hold different arrays; '
                                       'neither is served until the right one is decided')
    return array


def array_sha256(path):
    """SHA-256 of the canonical .npy bytes of a logical path (the file bytes for a single .npy, re-serialised otherwise)."""
    found = _existing(path)
    if len(found) == 1 and found[0].suffix == '.npy':
        digest = hashlib.sha256()
        with open(found[0], 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest()
    return hashlib.sha256(canonical_npy_bytes(load_array(path))).hexdigest()


def save_array(path, array, compress):
    """Write array under a logical .npy path (as .npz when compress); returns (stored path, sha256 of canonical .npy bytes)."""
    path = Path(path)
    if path.suffix != '.npy':
        raise ValueError(f'{path}: logical array paths end in .npy')
    data = canonical_npy_bytes(array)
    if not compress:
        with open(path, 'xb') as handle:
            handle.write(data)
        return path, hashlib.sha256(data).hexdigest()
    target = path.with_suffix(COMPRESSED_SUFFIX)
    with open(target, 'xb') as handle:
        np.savez_compressed(handle, **{MEMBER: np.asarray(array)})
    if hashlib.sha256(canonical_npy_bytes(load_array(target))).digest() != hashlib.sha256(data).digest():
        raise ValueError(f'{target}: compressed round trip differs from the array')
    return target, hashlib.sha256(data).hexdigest()


def compress_verified(job):
    """(source_root, target_root, relative logical path, recorded sha256) -> (source bytes, stored bytes). The source array must
    match the record and the written .npz must read back to the same canonical bytes (ValueError otherwise)."""
    source_root, target_root, relative, sha = job
    source = Path(source_root) / relative
    stored = stored_paths(source)
    if array_sha256(source) != sha:
        raise ValueError(f'{relative}: source sha256 differs from the record')
    target, digest = save_array(Path(target_root) / relative, load_array(source), compress=True)
    if digest != sha or array_sha256(Path(target_root) / relative) != sha:
        raise ValueError(f'{relative}: compressed copy fails verification')
    return sum(p.stat().st_size for p in stored), target.stat().st_size


def temporary_webp_path(logical, tmp_dir=None):
    """Where the temporary of a transcode is written: `<stem>.tmp.webp` beside the array, or a uniquely named file in
    `tmp_dir`. A bank part is a verified unit whose readers (scripts/run_bank_queue.py verify_bank, in every frozen code
    snapshot too) report ANY '.tmp.' file in public/ or evaluator/ as a leftover, so a transcode that may run beside such a
    reader puts its temporaries in a directory outside the part."""
    logical = Path(logical)
    if tmp_dir is None:
        return logical.with_name(logical.stem + '.tmp' + WEBP_SUFFIX)
    unique = hashlib.sha256(str(logical).encode()).hexdigest()[:16]
    return Path(tmp_dir) / f'{logical.stem}.{unique}.tmp{WEBP_SUFFIX}'


def transcode_to_webp(path, sha256=None, delete_source=False, tmp_dir=None):
    """Store the RGB array of a logical .npy path as lossless WebP, bit-exactly, and only then rename it into place.

    The source (.npy or .npz) is read, encoded to a temporary file (see temporary_webp_path), decoded again and compared
    byte for byte (shape, dtype and canonical .npy bytes) with the source array and, when `sha256` is given, with that
    record. Only then is the temporary file renamed to `<stem>.webp`, the stored file re-read and checked once more, and
    only then, and only with delete_source=True, is the source removed. Any failure leaves the source untouched and removes
    the temporary file. The WebP bytes are fsynced before the rename and the target directory after it, so the source is
    never removed while the only other copy of the array is still unwritten page cache.
    Returns a record dict (sizes, sha256, times, whether the source was removed).
    """
    import time
    path = Path(path)
    logical = path.with_suffix('.npy')
    found = stored_paths(logical)
    target = logical.with_suffix(WEBP_SUFFIX)
    if not found:
        raise FileNotFoundError(f'{logical}: no stored form to transcode')
    if target in found:
        raise FileExistsError(f'{target}: already stored as WebP')
    if len(found) > 1:
        raise StorageConflictError(f'{logical}: {[p.name for p in found]} exist; resolve the conflict before transcoding')
    source = found[0]
    array = _load_stored(source)
    if not webp_storable(array):
        raise ValueError(f'{logical}: not an RGB array ({np.asarray(array).dtype} {np.asarray(array).shape}); not transcoded')
    data = canonical_npy_bytes(array)
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f'{logical}: source sha256 {digest} differs from the record {sha256}; not transcoded')
    temporary = temporary_webp_path(logical, tmp_dir)
    t0 = time.monotonic()
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        save_webp(temporary, array)                 # fsynced inside save_webp
        encode_s = time.monotonic() - t0
        t1 = time.monotonic()
        back = load_webp(temporary)
        if back.shape != array.shape or back.dtype != array.dtype or not back.flags['C_CONTIGUOUS'] \
                or canonical_npy_bytes(back) != data:
            raise ValueError(f'{temporary}: decoded WebP differs from the source array; nothing renamed')
        stored_bytes = temporary.stat().st_size
        os.replace(temporary, target)
        fsync_dir(target.parent)                    # the rename is durable before anything else happens
        if temporary.parent != target.parent:
            fsync_dir(temporary.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if array_sha256(target) != digest:  # the renamed file alone, read again from disk
        raise ValueError(f'{target}: renamed WebP fails verification (source kept)')
    verify_s = time.monotonic() - t1
    source_bytes = source.stat().st_size
    removed = False
    if delete_source:
        fsync_dir(source.parent)         # the WebP is on stable storage before the only other copy is removed
        source.unlink()                  # the verified WebP holds the same canonical bytes
        fsync_dir(source.parent)
        removed = True
    return {'logical': str(logical), 'source': source.name, 'source_bytes': source_bytes, 'stored': target.name,
            'stored_bytes': stored_bytes, 'sha256': digest, 'shape': list(array.shape), 'encode_s': round(encode_s, 3),
            'verify_s': round(verify_s, 3), 'source_removed': removed}


__all__ = ['COMPRESSED_SUFFIX', 'STORED_SUFFIXES', 'StorageConflictError', 'WEBP_MAX_SIDE', 'WEBP_SAVE_KWARGS', 'WEBP_SUFFIX',
           'array_sha256', 'canonical_npy_bytes', 'compress_verified', 'fsync_dir', 'load_array', 'load_webp', 'save_array',
           'save_webp', 'stored_path', 'stored_paths', 'temporary_webp_path', 'transcode_to_webp', 'webp_storable']
