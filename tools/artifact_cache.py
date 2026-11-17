"""Content-addressed, locked tool artifacts. Never cache partial or unchecked output."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    return digest(path) if path.is_file() else None


def atomic_json(path, value, temporary_dir=None):
    """Commit a receipt only after every referenced artifact has been written."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=temporary_dir or path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def check_cancelled():
    if any(Path(p).exists() for p in json.loads(os.environ.get('VIBUILD_PARENT_STOPS', '[]'))):
        raise RuntimeError('Tool wait cancelled by project/session stop')
    if time.time() >= float(os.environ.get('VIBUILD_DEADLINE', 'inf')):
        raise TimeoutError('Tool wait exceeded session deadline')


class OutputProgress:
    """A locked tool-owned output, with immutable inputs and hash-bound commits.

    The caller holds an external output lock for the entire operation. Interrupted
    writes without a receipt can be replaced; committed artifacts cannot be changed.
    """
    marker = '.scene_resume.json'

    def __init__(self, output, request, allowed_names):
        self.output = Path(output)
        self.path = self.output / self.marker
        allowed = set(allowed_names) | {self.marker}
        if self.output.exists():
            existing = {p.name for p in self.output.iterdir()}
            if existing and not self.path.is_file():
                raise ValueError('Unknown nonempty output; preserve existing artifacts')
            if existing - allowed:
                raise ValueError('Unexpected files in resumable output; preserve existing artifacts')
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('version') != 1 or self.state.get('request') != request:
                raise ValueError('Output request or source changed; use a new output directory')
        else:
            self.output.mkdir(parents=True, exist_ok=True)
            self.state = dict(version=1, request=request, inputs=None, receipts={})
            atomic_json(self.path, self.state, self.output.parent)

    def bind_inputs(self, inputs):
        if self.state['inputs'] is None:
            self.state['inputs'] = inputs
            atomic_json(self.path, self.state, self.output.parent)
        elif self.state['inputs'] != inputs:
            raise ValueError('Output external inputs changed; use a new output directory')

    def get(self, key):
        receipt = self.state['receipts'].get(key)
        if receipt is None:
            return None
        if (receipt['inputs'] != fingerprint(self.state['inputs']) or
                any(Path(name).name != name or file_hash(self.output / name) != sha
                    for name, sha in receipt['files'].items())):
            raise ValueError('Completed output artifact changed; preserve it and use a new output directory')
        return receipt['value']

    def commit(self, key, names, value):
        if self.state['inputs'] is None:
            raise ValueError('Output inputs must be bound before committing artifacts')
        files = {}
        for name in names:
            if Path(name).name != name:
                raise ValueError('Output artifacts must be flat files')
            files[name] = digest(self.output / name)
        self.state['receipts'][key] = dict(files=files, value=value, inputs=fingerprint(self.state['inputs']))
        atomic_json(self.path, self.state, self.output.parent)


@contextmanager
def entry_lock(root, key, *, with_handle=False):
    root.mkdir(parents=True, exist_ok=True)
    with (root / (key + '.lock')).open('a') as handle:
        # Obey the enclosing session deadline even when another worker is computing.
        while True:
            check_cancelled()
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.1)
        yield (root / key, handle) if with_handle else root / key


def verified(entry):
    try:
        receipt = json.loads((entry / 'cache_receipt.json').read_text())
        return bool(receipt['files']) and all(
            Path(name).name == name and file_hash(entry / name) == sha
            for name, sha in receipt['files'].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def restore(entry, output):
    receipt = json.loads((entry / 'cache_receipt.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    for name in receipt['files']:
        # Copies isolate writable session artifacts from the shared cache.
        shutil.copy2(entry / name, output / name)


def publish(entry, output, names):
    tmp = Path(tempfile.mkdtemp(prefix='.pending-', dir=entry.parent))
    try:
        files = {}
        for name in names:
            if Path(name).name != name:
                raise ValueError('Cache artifacts must be flat files')
            shutil.copy2(output / name, tmp / name)
            files[name] = digest(tmp / name)
        (tmp / 'cache_receipt.json').write_text(json.dumps({'files': files}))
        if entry.exists():
            shutil.rmtree(entry)  # Only a tool-owned, locked content-addressed entry.
        tmp.rename(entry)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
