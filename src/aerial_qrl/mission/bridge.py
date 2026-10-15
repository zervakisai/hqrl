"""JSON-lines subprocess bridge, protocol aerial-qrl-bridge/1.

Transport between the Python 3.12 ML process and simulator-side processes that need other
interpreters (Gazebo bindings, AirSim client, pymavlink). Stdlib only, Python >= 3.9.

Request   {"protocol", "seq", "channel", "method", "params"}
Response  {"protocol", "seq", "channel", "ok", "result" | "error", "arrays", "stamp"}

Rules
- One JSON object per line; NaN/Infinity literals are rejected in both directions.
- seq starts at 1 and must be echoed exactly; the channel follows from the method name:
  methods under "evaluator." use the evaluator channel, every other method is public.
- Public results and array names are checked for truth-like keys on the server (fail closed
  before sending) and again on the client.
- Arrays never travel inline: {name, path, dtype, shape, sha256} references a raw file with
  C-order, little-endian bytes. read_array() verifies size and hash before use.
- After a timeout or protocol violation the client is unusable, because a late reply could
  otherwise be matched to the next request. Remote handler errors stay typed and usable.

Clocks
- stamp.server_monotonic_ns / stamp.server_unix_ns are host clocks taken when the response is
  encoded. CLOCK_MONOTONIC is shared by processes on one Linux host, so the client checks
  sent <= server <= received. They measure transport latency only; simulator time belongs
  inside results as sim_time_s.
"""
from collections import namedtuple
import hashlib
import json
import os
import re
import select
import subprocess
import time

from .leakage import FORBIDDEN_TOKENS, PublicLeakError, assert_public_payload, key_tokens

PROTOCOL = 'aerial-qrl-bridge/1'
CHANNELS = ('public', 'evaluator')
ARRAY_DTYPES = {'uint8': '<u1', 'uint16': '<u2', 'float32': '<f4', 'float64': '<f8'}
ITEM_BYTES = {'uint8': 1, 'uint16': 2, 'float32': 4, 'float64': 8}
MAX_LINE_BYTES = 16 * 1024 * 1024
_METHOD = re.compile(r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class BridgeError(RuntimeError):
    pass


class BridgeTimeout(BridgeError):
    pass


class BridgeProtocolError(BridgeError):
    pass


class BridgeLeakError(BridgeProtocolError):
    pass


class BridgeRemoteError(BridgeError):
    def __init__(self, kind, message):
        super().__init__('%s: %s' % (kind, message))
        self.kind = kind


class BridgeReply(namedtuple('BridgeReply', 'seq result arrays sent_monotonic_ns server_monotonic_ns received_monotonic_ns')):
    __slots__ = ()

    @property
    def round_trip_s(self):
        return (self.received_monotonic_ns - self.sent_monotonic_ns) / 1e9


def _reject_constant(name):
    raise BridgeProtocolError('non-finite JSON constant ' + name)


def dumps(obj):
    try:
        return json.dumps(obj, allow_nan=False, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise BridgeProtocolError('not JSON-serialisable without NaN/Infinity: %s' % exc)


def loads(text):
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise BridgeProtocolError('invalid JSON line: %s' % exc)


def channel_for(method):
    return 'evaluator' if method.startswith('evaluator.') else 'public'


def _is_int(value, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _check_seq(seq):
    if not _is_int(seq, 1):
        raise BridgeProtocolError('seq must be an integer >= 1')


def _check_method(method):
    if not isinstance(method, str) or not _METHOD.match(method):
        raise BridgeProtocolError('invalid method name')


def encode_request(seq, method, params):
    _check_seq(seq)
    _check_method(method)
    if not isinstance(params, dict):
        raise BridgeProtocolError('params must be an object')
    return dumps({'protocol': PROTOCOL, 'seq': seq, 'channel': channel_for(method), 'method': method, 'params': params}) + '\n'


def validate_array_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'name', 'path', 'dtype', 'shape', 'sha256'}:
        raise BridgeProtocolError('array reference needs exactly name, path, dtype, shape, sha256')
    if not isinstance(ref['name'], str) or not ref['name']:
        raise BridgeProtocolError('array name required')
    if not isinstance(ref['path'], str) or not os.path.isabs(ref['path']):
        raise BridgeProtocolError('array path must be absolute')
    if ref['dtype'] not in ARRAY_DTYPES:
        raise BridgeProtocolError('unsupported array dtype')
    shape = ref['shape']
    if not isinstance(shape, list) or not shape or not all(_is_int(n, 0) for n in shape):
        raise BridgeProtocolError('array shape must be a non-empty list of non-negative integers')
    if not isinstance(ref['sha256'], str) or not _SHA256.match(ref['sha256']):
        raise BridgeProtocolError('array sha256 must be lowercase hex')
    return ref


def write_array_file(path, name, dtype, shape, data):
    """Server helper: write raw little-endian C-order bytes and return the reference."""
    if dtype not in ARRAY_DTYPES:
        raise BridgeProtocolError('unsupported array dtype')
    count = 1
    for n in shape:
        count *= n
    data = bytes(data)
    if len(data) != count * ITEM_BYTES[dtype]:
        raise BridgeProtocolError('payload size does not match dtype and shape')
    with open(path, 'xb') as f:  # never overwrite an earlier sample
        f.write(data)
    return validate_array_ref({'name': name, 'path': os.path.abspath(path), 'dtype': dtype, 'shape': list(shape),
                               'sha256': hashlib.sha256(data).hexdigest()})


def read_array(ref):
    """Client helper: verify size and SHA256, then return a read-only numpy array."""
    import numpy as np
    validate_array_ref(ref)
    with open(ref['path'], 'rb') as f:
        data = f.read()
    expected = int(np.prod(ref['shape'], dtype=np.int64)) * ITEM_BYTES[ref['dtype']]
    if len(data) != expected:
        raise BridgeProtocolError('array file size mismatch for ' + ref['name'])
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise BridgeProtocolError('array file hash mismatch for ' + ref['name'])
    return np.frombuffer(data, dtype=ARRAY_DTYPES[ref['dtype']]).reshape(ref['shape'])


def _check_public(channel, result, arrays):
    if channel != 'public':
        return
    try:
        assert_public_payload(result)
    except PublicLeakError as exc:
        raise BridgeLeakError(str(exc))
    for ref in arrays:
        if key_tokens(ref['name']) & FORBIDDEN_TOKENS:
            raise BridgeLeakError('truth-like array name on public channel: ' + ref['name'])


def encode_response(seq, channel, result=None, error=None, arrays=()):
    """Server-side encoder; fails closed before truth-like data leaves the simulator side."""
    _check_seq(seq)
    if channel not in CHANNELS:
        raise BridgeProtocolError('unknown channel')
    arrays = [validate_array_ref(dict(a)) for a in arrays]
    message = {'protocol': PROTOCOL, 'seq': seq, 'channel': channel, 'arrays': arrays}
    if error is None:
        _check_public(channel, result, arrays)
        message.update(ok=True, result=result)
    else:
        if arrays:
            raise BridgeProtocolError('error responses carry no arrays')
        message.update(ok=False, error={'type': str(error[0]), 'message': str(error[1])})
    message['stamp'] = {'server_monotonic_ns': time.monotonic_ns(), 'server_unix_ns': time.time_ns()}
    return dumps(message) + '\n'


def decode_response(line, expected_seq, expected_channel, sent_monotonic_ns=None, received_monotonic_ns=None):
    """Validate a response line and return a BridgeReply, or raise."""
    message = loads(line)
    if not isinstance(message, dict):
        raise BridgeProtocolError('response must be an object')
    if message.get('protocol') != PROTOCOL:
        raise BridgeProtocolError('protocol version mismatch')
    if isinstance(message.get('seq'), bool) or message.get('seq') != expected_seq:
        raise BridgeProtocolError('sequence mismatch: expected %s' % expected_seq)
    if message.get('channel') != expected_channel:
        raise BridgeProtocolError('channel mismatch')
    ok = message.get('ok')
    if ok not in (True, False) or isinstance(ok, int) and not isinstance(ok, bool):
        raise BridgeProtocolError('ok must be boolean')
    expected_keys = {'protocol', 'seq', 'channel', 'ok', 'arrays', 'stamp', 'result' if ok else 'error'}
    if set(message) != expected_keys:
        raise BridgeProtocolError('unexpected response fields: %s' % sorted(set(message) ^ expected_keys))
    stamp = message['stamp']
    if not isinstance(stamp, dict) or set(stamp) != {'server_monotonic_ns', 'server_unix_ns'} or \
            not all(_is_int(v, 0) for v in stamp.values()):
        raise BridgeProtocolError('stamp needs non-negative integer server_monotonic_ns and server_unix_ns')
    server = stamp['server_monotonic_ns']
    if sent_monotonic_ns is not None and not sent_monotonic_ns <= server <= received_monotonic_ns:
        raise BridgeProtocolError('server stamp outside the request/response interval')
    if not isinstance(message['arrays'], list):
        raise BridgeProtocolError('arrays must be a list')
    arrays = [validate_array_ref(a) for a in message['arrays']]
    if not ok:
        error = message['error']
        if arrays or not isinstance(error, dict) or set(error) != {'type', 'message'} or not all(isinstance(v, str) for v in error.values()):
            raise BridgeProtocolError('malformed error response')
        raise BridgeRemoteError(error['type'], error['message'])
    _check_public(expected_channel, message['result'], arrays)
    return BridgeReply(expected_seq, message['result'], arrays, sent_monotonic_ns, server, received_monotonic_ns)


def handle_line(line, handlers):
    """Server step: returns (response_line, stop). handlers[method](params) -> (result, arrays)."""
    request = None
    try:
        request = loads(line)
        if not isinstance(request, dict) or set(request) != {'protocol', 'seq', 'channel', 'method', 'params'}:
            raise BridgeProtocolError('request needs exactly protocol, seq, channel, method, params')
        if request['protocol'] != PROTOCOL:
            raise BridgeProtocolError('protocol version mismatch')
        _check_seq(request['seq'])
        _check_method(request['method'])
        if request['channel'] != channel_for(request['method']) or not isinstance(request['params'], dict):
            raise BridgeProtocolError('channel/params mismatch')
    except BridgeProtocolError as exc:
        seq = request.get('seq') if isinstance(request, dict) else None
        return encode_response(seq if _is_int(seq, 1) else 1, 'public', error=('bad_request', exc)), False
    seq, method, channel = request['seq'], request['method'], request['channel']
    if method == 'shutdown':
        return encode_response(seq, channel, result={}), True
    handler = handlers.get(method)
    if handler is None:
        return encode_response(seq, channel, error=('unknown_method', method)), False
    try:
        result, arrays = handler(request['params'])
        return encode_response(seq, channel, result=result, arrays=arrays), False
    except Exception as exc:  # reported to the client as a typed remote error
        return encode_response(seq, channel, error=(exc.__class__.__name__, exc)), False


def serve(handlers, stdin, stdout):
    for line in stdin:
        response, stop = handle_line(line, handlers)
        stdout.write(response)
        stdout.flush()
        if stop:
            break


class BridgeClient:
    """Owns exactly one child process; close() never touches any other process."""

    def __init__(self, argv, timeout_s=10., cwd=None, env=None, stderr_path=None):
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s < float('inf'):
            raise ValueError('timeout_s must be positive and finite')
        self.timeout_s = float(timeout_s)
        self._stderr = open(stderr_path, 'ab') if stderr_path else subprocess.DEVNULL
        self._proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                      cwd=cwd, env=env, start_new_session=True)
        self._seq = 0
        self._buffer = b''
        self.broken = None

    @property
    def pid(self):
        return self._proc.pid

    def poll(self):
        return self._proc.poll()

    def call(self, method, params=None, timeout_s=None):
        if self.broken:
            raise BridgeError('bridge client unusable after ' + self.broken)
        self._seq += 1
        seq = self._seq
        line = encode_request(seq, method, {} if params is None else params)
        sent = time.monotonic_ns()
        try:
            self._proc.stdin.write(line.encode('utf-8'))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.broken = 'write failure'
            raise BridgeError('cannot write request: %s' % exc)
        raw = self._readline(self.timeout_s if timeout_s is None else timeout_s)
        try:
            return decode_response(raw, seq, channel_for(method), sent, time.monotonic_ns())
        except BridgeRemoteError:
            raise
        except BridgeError:
            self.broken = 'protocol violation'
            raise

    def _readline(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        fd = self._proc.stdout.fileno()
        while b'\n' not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.broken = 'timeout'
                raise BridgeTimeout('no response within %.3f s' % timeout_s)
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                self.broken = 'server exit'
                raise BridgeError('bridge server exited (code %s)' % self._proc.poll())
            self._buffer += chunk
            if len(self._buffer) > MAX_LINE_BYTES:
                self.broken = 'oversized response'
                raise BridgeProtocolError('response line exceeds limit')
        line, _, self._buffer = self._buffer.partition(b'\n')
        return line.decode('utf-8')

    def close(self, timeout_s=5.):
        """Shutdown request if usable, then EOF on stdin (servers clean up on EOF), then escalate."""
        if self._proc.poll() is None and not self.broken:
            try:
                self.call('shutdown', timeout_s=timeout_s)
            except BridgeError:
                pass
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        if self._proc.poll() is None:
            try:
                self._proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout_s)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
        for pipe in (self._proc.stdin, self._proc.stdout):
            if pipe:
                pipe.close()
        if self._stderr is not subprocess.DEVNULL:
            self._stderr.close()
        return self._proc.returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
