"""Host values the sandbox reads as `host.NAME`, from files in a store directory.

`VariableStore` pickles each value into its own file in a temporary directory. A
session given the store lets Deno read that directory (and only read it), and defines
`host` in the sandbox (`deno/host.py`), which loads a value when it is read:

    with VariableStore({"df": df}) as store:
        session = PyodideSession(store=store)
        session.start()
        session.run("len(host.df)")

Every read unpickles the file again, so it returns a fresh copy. The store can't be
changed after it is created. Unpickling in the sandbox is safe for the host. It needs
the values' modules in the sandbox (pandas and numpy, pinned on the host to the
sandbox's versions) and can't load classes defined only on the host.

The directory isn't mounted into Pyodide's filesystem. `host` reads it through Deno
instead (`Deno.readFileSync`), because Pyodide's NODEFS mount passes a cell's
`os.chmod` or `os.utime` straight to Deno, and Deno's refusal (it has no write
permission) breaks Pyodide for good. A refused read through Deno is an ordinary
exception in the cell.

Layout of the store's directory, ported from eryx-test's store (where it was a
read-only mount, with a copy of the accessor):

    view.json      {"format": 1, "epoch": 1, "entries": {name: entry}}
    blobs/<hash>   one value's pickle, named by its sha256; never changed
"""

import hashlib
import json
import keyword
import os
import pathlib
import pickle
import shutil
import tempfile
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any, Self

_FORMAT = 1


@dataclass(frozen=True)
class Entry:
    """One variable in the store, as the sandbox's `host.info(name)` also sees it."""

    name: str
    version: int
    codec: str
    """How `blob` is decoded in the sandbox: `"pickle"`."""
    blob: str
    """The sha256 of the stored bytes, which is also their file name."""
    type: str
    """The value's type name on the host, e.g. `DataFrame`."""
    size: int


class VariableStore:
    """Host values for the sandbox to read as `host.NAME` (see module docs).

    `variables` is copied and each value pickled when the store is created, so an
    unpicklable value fails here and later changes to the dict aren't seen. The
    directory is removed by `close()`, at the end of a `with` block, or when the
    store is garbage collected.
    """

    def __init__(self, variables: Mapping[str, Any] | None = None) -> None:
        values = dict(variables or {})
        bad = [n for n in values if not _is_name(n)]
        if bad:
            raise ValueError(f"variable names must be Python identifiers: {bad}")
        pickled = {name: pickle.dumps(value) for name, value in values.items()}
        # resolved, as Deno's --allow-read compares real paths
        self._dir = pathlib.Path(tempfile.mkdtemp(prefix="pyodide-store-")).resolve()
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, self._dir, ignore_errors=True
        )
        try:
            self._entries = self._write(
                {
                    name: (data, type(values[name]).__name__)
                    for name, data in pickled.items()
                }
            )
        except BaseException:
            self._finalizer()
            raise

    def _write(self, pickled: dict[str, tuple[bytes, str]]) -> dict[str, Entry]:
        blobs = self._dir / "blobs"
        blobs.mkdir()
        entries = {}
        for name, (data, type_name) in pickled.items():
            digest = hashlib.sha256(data).hexdigest()
            if not (blobs / digest).exists():
                _atomic_write(blobs / digest, data)
            entries[name] = Entry(name, 1, "pickle", digest, type_name, len(data))
        view = {
            "format": _FORMAT,
            "epoch": 1,
            "entries": {name: asdict(entry) for name, entry in entries.items()},
        }
        # Written last: the sandbox finds nothing until every blob is in place
        _atomic_write(self._dir / "view.json", json.dumps(view, indent=1).encode())
        return entries

    @property
    def directory(self) -> pathlib.Path:
        """Where the store is; a session gives Deno read access to it."""
        return self._dir

    def names(self) -> list[str]:
        """The variables' names, in the order they were given."""
        return list(self._entries)

    def info(self, name: str) -> Entry:
        return self._entries[name]

    def close(self) -> None:
        """Remove the store's directory. Sessions using it can no longer read it."""
        self._finalizer()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        listed = ", ".join(f"{e.name}: {e.type}" for e in self._entries.values())
        return f"VariableStore({listed})"


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write `data` to `path` so that a reader sees either the old file or the new."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _is_name(name: object) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)
