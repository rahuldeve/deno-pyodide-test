"""The sandbox side of a `VariableStore`: the `host` object that reads it.

main.ts loads this into a namespace of its own and gives `__main__` one `Host`, whose
`read` goes through Deno (see `runner.setup_host`). It uses only the standard
library, and pyarrow for Arrow values, which the session loads when the store has
any (imported on the first read, unless `import pandas` already did). Every access
reads `view.json` again, so it sees the host's latest changes, and a store closed on
the host fails the read. Ported from eryx-test's `host_accessor.py`, without
`host.tools` for now.
"""

import json
import pickle
import time
from collections.abc import Callable, Iterator
from typing import Any

# Reads of a value that the host keeps replacing between our two reads
_ATTEMPTS = 5
# How long to retry opening view.json while the host replaces it (Windows)
_VIEW_RETRY_S = 1.0


def _from_arrow(data: bytes, series: bool) -> Any:
    """A DataFrame (or Series) from the Arrow IPC stream `variables._to_arrow` wrote."""
    import pyarrow as pa

    table = pa.ipc.open_stream(data).read_all()
    frame = table.to_pandas()
    if not series:
        return frame
    extra = json.loads(table.schema.metadata[b"pyodide_sandbox"])
    return frame["values"].rename(extra["name"])


class Host:
    """Read-only access to the host's variables: `host.df` or `host["df"]`.

    Each read loads the value again, so it returns a fresh copy of the host's
    current value: changing it changes neither the host's value nor the next read.
    """

    __slots__ = ("_read", "_root")

    def __init__(self, root: str, read: Callable[[str], bytes]) -> None:
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_read", read)

    def _entries(self) -> dict[str, dict[str, Any]]:
        return json.loads(self._read_view())

    def _read_view(self) -> bytes:
        # Windows refuses to open a file for a moment while it is being replaced; a
        # closed store is FileNotFoundError, and fails at once
        deadline = time.monotonic() + _VIEW_RETRY_S
        while True:
            try:
                return self._read(f"{self._root}/view.json")
            except FileNotFoundError:
                raise
            except OSError:
                if time.monotonic() > deadline:
                    raise

    def _entry(self, name: str) -> dict[str, Any]:
        entries = self._entries()
        if name not in entries:
            available = ", ".join(entries) or "none"
            raise KeyError(f"no host variable {name!r}; available: {available}")
        return entries[name]

    def _fetch(self, name: str) -> tuple[dict[str, Any], bytes]:
        """The variable's entry and bytes; `KeyError` only if there is no such variable."""
        for _ in range(_ATTEMPTS):
            entry = self._entry(name)
            try:
                return entry, self._read(f"{self._root}/blobs/{entry['blob']}")
            except OSError:
                # The host replaced or removed the value between the two reads, and
                # deleted the old bytes: read the view again
                if self._entries().get(name, {}).get("blob") == entry["blob"]:
                    raise
        raise OSError(f"host variable {name!r} kept changing while it was read")

    def _decode(self, name: str, entry: dict[str, Any], data: bytes) -> Any:
        codec = entry["codec"]
        if codec == "pickle":
            return pickle.loads(data)
        if codec in ("arrow", "arrow-series"):
            return _from_arrow(data, series=codec == "arrow-series")
        raise ValueError(f"host variable {name!r} has unknown codec {codec!r}")

    def __getitem__(self, name: str) -> Any:
        return self._decode(name, *self._fetch(name))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        # Checked first, not by catching KeyError: unpickling can raise one too
        # (zoneinfo's ZoneInfoNotFoundError), which must reach the cell as it is
        try:
            entry, data = self._fetch(name)
        except KeyError as e:
            raise AttributeError(e.args[0]) from None
        return self._decode(name, entry, data)

    def info(self, name: str) -> dict[str, Any]:
        """The variable's entry: its type, size and how it is stored."""
        return dict(self._entry(name))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("host variables are read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("host variables are read-only")

    def __setitem__(self, name: str, value: Any) -> None:
        raise TypeError("host variables are read-only")

    def __delitem__(self, name: str) -> None:
        raise TypeError("host variables are read-only")

    def __contains__(self, name: object) -> bool:
        return name in self._entries()

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._entries()))

    def __len__(self) -> int:
        return len(self._entries())

    def __dir__(self) -> list[str]:
        return [*self._entries(), "info"]

    def __repr__(self) -> str:
        entries = self._entries()
        listed = ", ".join(f"{n} ({e['type']})" for n, e in entries.items())
        return f"host variables: {listed or 'none'}"
