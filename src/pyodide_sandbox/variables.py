"""Host values the sandbox reads as `host.NAME`, from files in a store directory.

`VariableStore` pickles each value into its own file in a temporary directory. A
session given the store lets Deno read that directory (and only read it), and defines
`host` in the sandbox (`_deno/host.py`), which loads a value when it is read:

    with VariableStore({"df": df}) as store:
        session = PyodideSession(store=store)
        session.start()
        session.run("len(host.df)")

Every read decodes the file again, so it returns a fresh copy, and sees the store
as it is at that moment: the host can add, replace and remove values while sessions
run (`set`, `update`, `remove`), and the next read in a cell sees the change. A
copy a cell already made keeps the old value. Cells can't change the store.

A pandas DataFrame or Series goes as an Arrow IPC stream when pyarrow is installed
on the host (codec `arrow` or `arrow-series`). Arrow's format doesn't depend on the
pandas or pyarrow version, so the host can use any pandas; the sandbox reads it
with its own pyarrow, which the session then loads (`sandbox_packages`; about 5 s
more boot, as Pyodide compiles libarrow). Arrow keeps dtypes (tz-aware datetimes,
categoricals, nullable and string dtypes), the index and column names, and
`df.attrs`. Each frame is decoded back on the host as the sandbox will decode
it, and one that doesn't come back equal is pickled instead: one Arrow can't
convert (an object column of mixed types, duplicate column names, an int over 64
bits) or would change (a column of sets or dicts, mixed time zones, column names
of mixed types, attrs that aren't JSON). So is a subclass, or a Series whose name
isn't a plain scalar.

Everything else is pickled. Unpickling in the sandbox is safe for the host. It needs
the values' modules in the sandbox and can't load classes defined only on the host.
A numpy, pandas, scipy or statsmodels object pickled by another version than the
sandbox's (`distribution.SANDBOX_VERSIONS`) may not load there, so the store warns
(`VersionMismatchWarning`) when it pickles one; pin those versions on the host to
be sure.

The directory isn't mounted into Pyodide's filesystem. `host` reads it through Deno
instead (`Deno.readFileSync`), because Pyodide's NODEFS mount passes a cell's
`os.chmod` or `os.utime` straight to Deno, and Deno's refusal (it has no write
permission) breaks Pyodide for good. A refused read through Deno is an ordinary
exception in the cell.

Layout of the store's directory:

    view.json      {name: entry}, the fields of `Entry`
    blobs/<hash>   one value's bytes, named by their sha256

A change writes the new blobs first, then replaces `view.json` in one rename, so a
reader sees the whole old view or the whole new one, never a mix. Blobs no view
names any more are then deleted; a reader that read the old view just before then
finds its blob gone, reads the view again and gets the new value (`host.py`).
"""

import hashlib
import io
import json
import keyword
import os
import pathlib
import pickle
import shutil
import sys
import tempfile
import threading
import time
import types
import warnings
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any, Self

from pyodide_sandbox._deno.host import _from_arrow as from_arrow
from pyodide_sandbox.distribution import SANDBOX_VERSIONS


class VersionMismatchWarning(UserWarning):
    """A value was pickled by another numpy, pandas, scipy or statsmodels than the
    sandbox has."""


# A value's bytes, codec and type name on the host
_Encoded = tuple[bytes, str, str]


@dataclass(frozen=True)
class Entry:
    """One variable in the store, as the sandbox's `host.info(name)` also sees it."""

    name: str
    codec: str
    """How `blob` is decoded in the sandbox: `"pickle"`, `"arrow"` (a DataFrame) or
    `"arrow-series"`."""
    blob: str
    """The sha256 of the stored bytes, which is also their file name."""
    type: str
    """The value's type name on the host, e.g. `DataFrame`."""
    size: int
    generation: int
    """The store's `generation` when the value was set: 0 for the values it was
    created with."""


class VariableStore:
    """Host values for the sandbox to read as `host.NAME` (see module docs).

    `variables` is copied and each value pickled when the store is created, so an
    unpicklable value fails here and later changes to the dict aren't seen; the
    same goes for `set` and `update`. Changes are seen at once by every session
    reading the store. Thread-safe. The directory is removed by `close()`, at the
    end of a `with` block, or when the store is garbage collected.
    """

    def __init__(self, variables: Mapping[str, Any] | None = None) -> None:
        encoded = _encode_all(dict(variables or {}))
        self._lock = threading.Lock()
        self._generation = 0
        self._entries: dict[str, Entry] = {}
        # resolved, as Deno's --allow-read compares real paths
        self._dir = pathlib.Path(tempfile.mkdtemp(prefix="pyodide-store-")).resolve()
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, self._dir, ignore_errors=True
        )
        try:
            (self._dir / "blobs").mkdir()
            self._apply(encoded, ())
        except BaseException:
            self._finalizer()
            raise

    def set(self, name: str, value: Any) -> None:
        """Add `name`, or replace its value; cells see it on their next read."""
        self.update({name: value})

    def update(self, variables: Mapping[str, Any] | None = None, /, **kw: Any) -> None:
        """Add or replace several values at once: a cell sees all of them or none.

        New names go after the existing ones; a replaced value keeps its place. If
        any value can't be encoded, the store is left as it was.
        """
        values = {**(variables or {}), **kw}
        if values:
            self._change(_encode_all(values), ())

    def remove(self, *names: str) -> None:
        """Remove variables; a missing name raises `KeyError` and removes none."""
        if names:
            self._change({}, names)

    def _change(self, encoded: dict[str, _Encoded], removed: Sequence[str]) -> None:
        with self._lock:
            if not self._finalizer.alive:
                raise ValueError("the variable store is closed")
            if missing := [n for n in removed if n not in self._entries]:
                raise KeyError(f"no host variable {missing[0]!r}")
            self._generation += 1
            self._apply(encoded, removed)

    def _apply(self, encoded: dict[str, _Encoded], removed: Sequence[str]) -> None:
        """Write the new blobs, then the new view, then delete unnamed blobs."""
        blobs = self._dir / "blobs"
        entries = dict(self._entries)
        for name in removed:
            del entries[name]
        for name, (data, codec, type_name) in encoded.items():
            digest = hashlib.sha256(data).hexdigest()
            # Even if it exists: on Windows, a blob deleted while a session reads it
            # stays listed until the read ends, then goes. Readers of the one in
            # place see the old file or this one, with the same bytes.
            _replace_atomically(blobs / digest, data)
            entries[name] = Entry(
                name, codec, digest, type_name, len(data), self._generation
            )
        view = {name: asdict(entry) for name, entry in entries.items()}
        _replace_atomically(
            self._dir / "view.json", json.dumps(view, indent=1).encode()
        )
        self._entries = entries
        named = {e.blob for e in entries.values()}
        for blob in blobs.iterdir():
            if blob.name not in named and not blob.name.startswith("."):
                try:
                    blob.unlink()
                # Windows, while a session reads it: the next change deletes it
                except OSError:
                    pass

    @property
    def generation(self) -> int:
        """How many changes the store has had since it was created."""
        return self._generation

    @property
    def directory(self) -> pathlib.Path:
        """Where the store is; a session gives Deno read access to it."""
        return self._dir

    @property
    def sandbox_packages(self) -> tuple[str, ...]:
        """Pyodide packages the sandbox needs to decode the values: pyarrow for Arrow.

        It can grow when a value is added; a running session then loads the new
        package before its next cell (`PyodideSession.run`).
        """
        if any(e.codec.startswith("arrow") for e in self._entries.values()):
            return ("pyarrow",)
        return ()

    def names(self) -> list[str]:
        """The variables' names, in the order they were added."""
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


def _encode_all(values: dict[str, Any]) -> dict[str, _Encoded]:
    """Each value as Arrow if it can go that way, else pickled (see module docs)."""
    if bad := [n for n in values if not _is_name(n)]:
        raise ValueError(f"variable names must be Python identifiers: {bad}")
    encoded = {}
    for name, value in values.items():
        arrow = _to_arrow(value)
        if arrow is None:
            data, modules = _pickle(value)
            _check_versions(name, modules)
            arrow = (data, "pickle")
        encoded[name] = (*arrow, type(value).__name__)
    return encoded


def _replace_atomically(path: pathlib.Path, data: bytes) -> None:
    """Write `data` to `path` so that a reader sees the old file or the new one."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    temporary.write_bytes(data)
    # Windows refuses to replace a file a reader has open for a moment
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 49:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.01)


_ARROW_METADATA = b"pyodide_sandbox"
_SERIES_COLUMN = "values"


def _pyarrow() -> types.ModuleType | None:
    """pyarrow, if the host has it (the `arrow` extra)."""
    try:
        import pyarrow
    except ImportError:
        return None
    return pyarrow


def _to_arrow(value: Any) -> tuple[bytes, str] | None:
    """A DataFrame or Series as an Arrow IPC stream and its codec, else None.

    None too when the conversion fails or loses something (a column of sets or
    dicts, mixed time zones, column names of mixed types, attrs that aren't JSON):
    the stream is decoded back as the sandbox will, and must equal `value`.
    """
    pandas = sys.modules.get("pandas")
    # exact types: a subclass (a GeoDataFrame, say) would come back as its base
    if pandas is None or type(value) not in (pandas.DataFrame, pandas.Series):
        return None
    pa = _pyarrow()
    if pa is None:
        return None
    series = type(value) is pandas.Series
    if series and not isinstance(value.name, (str, int, float, bool, type(None))):
        return None
    try:
        # pyarrow warns of what it loses; the check below catches that anyway
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = _encode_arrow(pa, value, series)
            if not _same_frame(pandas, from_arrow(data, series), value):
                return None
    # anything: a value pyarrow can't take (an int over 64 bits raises
    # OverflowError) is pickled instead
    except Exception:  # noqa: BLE001
        return None
    return data, "arrow-series" if series else "arrow"


def _encode_arrow(pa: types.ModuleType, value: Any, series: bool) -> bytes:
    frame = value.to_frame(_SERIES_COLUMN) if series else value
    table = pa.Table.from_pandas(frame)
    if series:
        metadata = {**(table.schema.metadata or {})}
        metadata[_ARROW_METADATA] = json.dumps({"name": value.name}).encode()
        table = table.replace_schema_metadata(metadata)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _same_frame(pandas: types.ModuleType, back: Any, value: Any) -> bool:
    """Whether `back` equals `value` in values, dtypes, index, names and attrs."""
    if type(back) is not type(value) or back.attrs != value.attrs:
        return False
    check = (
        pandas.testing.assert_series_equal
        if type(value) is pandas.Series
        else pandas.testing.assert_frame_equal
    )
    try:
        check(back, value, check_exact=True)
    except AssertionError:
        return False
    return True


class _Pickler(pickle.Pickler):
    """Records the top-level modules of the classes and functions it pickles."""

    def __init__(self, file: io.BytesIO) -> None:
        super().__init__(file)
        self.modules: set[str] = set()

    def persistent_id(self, obj: Any) -> None:  # None: pickle `obj` as usual
        module = type(obj).__module__
        if isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType)):
            module = obj.__module__ or ""
        self.modules.add(module.partition(".")[0])


def _pickle(value: Any) -> tuple[bytes, set[str]]:
    buffer = io.BytesIO()
    pickler = _Pickler(buffer)
    pickler.dump(value)
    return buffer.getvalue(), pickler.modules


def _check_versions(name: str, modules: set[str]) -> None:
    for module in sorted(modules & SANDBOX_VERSIONS.keys()):
        host = getattr(sys.modules.get(module), "__version__", None)
        sandbox = SANDBOX_VERSIONS[module]
        if host != sandbox:
            hint = (
                " (or install pyarrow, so that DataFrames go as Arrow)"
                if module == "pandas" and _pyarrow() is None
                else ""
            )
            warnings.warn(
                f"{name!r} was pickled with {module} {host} but the sandbox has "
                f"{module} {sandbox}, so it may not load there; pin "
                f"{module}=={sandbox} on the host to be sure{hint}",
                VersionMismatchWarning,
                stacklevel=_stacklevel_outside_package(),
            )


# as code objects name their files
_PACKAGE_DIR = os.path.dirname(__file__) + os.sep


def _stacklevel_outside_package() -> int:
    """The `stacklevel` of the caller's first caller outside this package.

    So that a warning names the line that created the store, whether that was
    `VariableStore(...)` or `PyodideRepl(variables=...)`, whose dataclass-made
    `__init__` (file `<string>`) calls into the package too.
    """
    level, frame = 1, sys._getframe(1)
    while frame is not None and (
        frame.f_code.co_filename.startswith(_PACKAGE_DIR)
        or (frame.f_code.co_filename, frame.f_code.co_name) == ("<string>", "__init__")
    ):
        level, frame = level + 1, frame.f_back
    return level


def _is_name(name: object) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)
