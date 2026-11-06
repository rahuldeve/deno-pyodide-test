"""Host values the sandbox reads as `host.NAME`, from files in a store directory.

`VariableStore` pickles each value into its own file in a temporary directory. A
session given the store lets Deno read that directory (and only read it), and defines
`host` in the sandbox (`_deno/host.py`), which loads a value when it is read:

    with VariableStore({"df": df}) as store:
        session = PyodideSession(store=store)
        session.start()
        session.run("len(host.df)")

Every read decodes the file again, so it returns a fresh copy. The store can't be
changed after it is created.

A pandas DataFrame or Series goes as an Arrow IPC stream when pyarrow is installed
on the host (codec `arrow` or `arrow-series`). Arrow's format doesn't depend on the
pandas or pyarrow version, so the host can use any pandas; the sandbox reads it
with its own pyarrow, which the session then loads (`sandbox_packages`; about 5 s
more boot, as Pyodide compiles libarrow). Arrow keeps dtypes (tz-aware datetimes,
categoricals, nullable and string dtypes), the index and column names, and
`df.attrs`. A frame Arrow can't convert (an object column of mixed
types, duplicate column names), a subclass, or a Series whose name isn't a plain
scalar is pickled instead.

Everything else is pickled. Unpickling in the sandbox is safe for the host. It needs
the values' modules in the sandbox and can't load classes defined only on the host.
A numpy, pandas or scipy object pickled by another version than the sandbox's
(`distribution.SANDBOX_VERSIONS`) may not load there, so the store warns
(`VersionMismatchWarning`) when it pickles one; pin those versions on the host to
be sure.

The directory isn't mounted into Pyodide's filesystem. `host` reads it through Deno
instead (`Deno.readFileSync`), because Pyodide's NODEFS mount passes a cell's
`os.chmod` or `os.utime` straight to Deno, and Deno's refusal (it has no write
permission) breaks Pyodide for good. A refused read through Deno is an ordinary
exception in the cell.

Layout of the store's directory, written in full before any session can read it:

    view.json      {name: entry}, the fields of `Entry`
    blobs/<hash>   one value's bytes, named by their sha256
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
import types
import warnings
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any, Self

from pyodide_sandbox.distribution import SANDBOX_VERSIONS


class VersionMismatchWarning(UserWarning):
    """A value was pickled by another numpy, pandas or scipy than the sandbox has."""


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
        encoded = {}
        for name, value in values.items():
            arrow = _to_arrow(value)
            if arrow is None:
                data, modules = _pickle(value)
                _check_versions(name, modules)
                arrow = (data, "pickle")
            encoded[name] = arrow
        # resolved, as Deno's --allow-read compares real paths
        self._dir = pathlib.Path(tempfile.mkdtemp(prefix="pyodide-store-")).resolve()
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, self._dir, ignore_errors=True
        )
        try:
            self._entries = self._write(
                {
                    name: (data, codec, type(values[name]).__name__)
                    for name, (data, codec) in encoded.items()
                }
            )
        except BaseException:
            self._finalizer()
            raise

    def _write(self, encoded: dict[str, tuple[bytes, str, str]]) -> dict[str, Entry]:
        blobs = self._dir / "blobs"
        blobs.mkdir()
        entries = {}
        for name, (data, codec, type_name) in encoded.items():
            digest = hashlib.sha256(data).hexdigest()
            (blobs / digest).write_bytes(data)
            entries[name] = Entry(name, codec, digest, type_name, len(data))
        view = {name: asdict(entry) for name, entry in entries.items()}
        (self._dir / "view.json").write_text(json.dumps(view, indent=1))
        return entries

    @property
    def directory(self) -> pathlib.Path:
        """Where the store is; a session gives Deno read access to it."""
        return self._dir

    @property
    def sandbox_packages(self) -> tuple[str, ...]:
        """Pyodide packages the sandbox needs to decode the values: pyarrow for Arrow."""
        if any(e.codec.startswith("arrow") for e in self._entries.values()):
            return ("pyarrow",)
        return ()

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
    """A DataFrame or Series as an Arrow IPC stream and its codec, else None."""
    pandas = sys.modules.get("pandas")
    # exact types: a subclass (a GeoDataFrame, say) would come back as its base
    if pandas is None or type(value) not in (pandas.DataFrame, pandas.Series):
        return None
    pa = _pyarrow()
    if pa is None:
        return None
    if type(value) is pandas.Series:
        if not isinstance(value.name, (str, int, float, bool, type(None))):
            return None
        frame, codec = value.to_frame(_SERIES_COLUMN), "arrow-series"
        extra = {"name": value.name}
    else:
        frame, codec, extra = value, "arrow", {}
    try:
        table = pa.Table.from_pandas(frame)
    except (pa.ArrowException, TypeError, ValueError):
        return None
    if extra:
        metadata = {**(table.schema.metadata or {})}
        metadata[_ARROW_METADATA] = json.dumps(extra).encode()
        table = table.replace_schema_metadata(metadata)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes(), codec


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
