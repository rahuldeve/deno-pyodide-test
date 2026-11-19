"""One Pyodide REPL session in a Deno process of its own, driven over JSON lines.

The process is the boundary, not Pyodide: code in a cell can reach JavaScript, so
everything worth protecting stays on this side. Deno runs with no network, write,
subprocess, FFI or environment access, can read only its own files, the Pyodide
files and the variable store, if any, and gets an environment without the host's
secrets. A cell over its time limit gets the process killed (SIGKILL, or
TerminateProcess on Windows), since Python inside Pyodide can catch an interrupt
and a long C call wouldn't see one. Every message from Deno is checked before use,
as a cell can write to the pipe itself.

Not yet done (later phases): a memory limit (Pyodide 314.0.7 has no
`maxMemoryBytes`; a cgroup is the plan), running as another user, seccomp.

The API is synchronous, like eryx's, so the process isn't tied to one event loop and
survives across `agent.run_sync()` calls.
"""

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import weakref
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from pyodide_sandbox.distribution import (
    default_dist_dir,
    ensure_distribution,
    extra_dependencies,
    import_names,
)
from pyodide_sandbox.variables import VariableStore

# main.ts and the Python it runs inside Pyodide, shipped in the package
DENO_DIR = Path(__file__).resolve().parent / "_deno"
# tzdata: pandas and zoneinfo find no time zone without it (no system zoneinfo in wasm)
DEFAULT_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "bokeh", "tzdata")

# Limits on what Deno sends back; the in-sandbox caps on each stream and on a cell's
# plots are the real ones. A reply may be this long plus `max_plot_bytes`.
_MAX_LINE_BYTES = 8 << 20
_STDERR_TAIL_LINES = 50
_MAX_FATAL_CHARS = 1000


def default_deno() -> str:
    """`$PYODIDE_SANDBOX_DENO`, else the pinned `deno` package's binary, else PATH's."""
    if configured := os.environ.get("PYODIDE_SANDBOX_DENO"):
        return configured
    try:
        import deno  # the `deno` dependency: Deno's binary in a platform wheel

        return deno.find_deno_bin()
    except (ImportError, FileNotFoundError):
        return shutil.which("deno") or "deno"


class SessionError(Exception):
    """The session can't run cells any more; start a new one."""


class SessionTimeout(SessionError):
    """A cell (or the boot) ran past its time limit, so the process was killed."""


class SessionCrashed(SessionError):
    """The Deno process exited or broke the protocol."""


@dataclass
class CellResult:
    error: str | None
    """The exception type's name if the cell raised, else None."""
    stdout: str
    stderr: str
    duration_ms: float
    plots: list[dict[str, Any]] = field(default_factory=list)
    """The bokeh objects the cell `show()`ed, in order, as `bokeh.embed.json_item`
    made them. Written in the sandbox, so untrusted (see `pyodide_sandbox.repl.Plot`)."""


class PyodideSession:
    """A Deno process running one Pyodide interpreter; variables persist between cells.

    Not thread-safe: run one cell at a time. After `SessionError` the session is
    dead and every later call raises it again. With `store`, cells read its values
    as `host.NAME`, and see the host's changes to it; the session keeps the store
    open while it holds it. With
    `warm_imports`, the boot also imports the packages' slow submodules
    (`_deno/warmup.py`), so the first cell to use them doesn't. With bokeh among the
    packages, a cell's `show(obj)` adds `obj` to its result's `plots`, up to
    `max_plot_bytes` of JSON per cell (more raises `ValueError` in the cell).
    """

    def __init__(
        self,
        packages: Sequence[str] = DEFAULT_PACKAGES,
        *,
        deno: str | None = None,
        dist_dir: str | Path | None = None,
        max_output_bytes: int = 1 << 20,
        max_plot_bytes: int = 10_000_000,
        store: VariableStore | None = None,
        download: bool = True,
        warm_imports: bool = True,
    ) -> None:
        self.packages = tuple(packages)
        self.warm_imports = warm_imports
        self.deno = deno or default_deno()
        self.dist_dir = Path(dist_dir) if dist_dir else default_dist_dir()
        self.max_output_bytes = max_output_bytes
        self.max_plot_bytes = max_plot_bytes
        self.store = store
        self.download = download
        self.boot_ms: float | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._next_id = 0
        self._dead: SessionError | None = None
        # loaded for the store beyond `packages` (pyarrow), at boot or since
        self._store_packages: set[str] = set()

    def _command(self, root: Path) -> list[str]:
        """Deno's command line; `root` holds the Pyodide files, with the lock file."""
        readable = [DENO_DIR, root]
        store = []
        if self.store is not None:
            readable.append(self.store.directory)
            store = ["--store", str(self.store.directory)]
        return [
            self.deno,
            "run",
            "--no-prompt",
            "--cached-only",
            "--no-remote",
            f"--config={DENO_DIR / 'deno.json'}",
            f"--allow-read={','.join(map(str, readable))}",
            str(DENO_DIR / "main.ts"),
            "--dist",
            str(root),
            # names checked against the lock file by ensure_distribution, which also
            # supplies the modules (dotted identifiers)
            "--packages",
            ",".join(
                [
                    *self.packages,
                    *extra_dependencies(self.packages, root),
                    *self._load_only(),
                ]
            ),
            "--imports",
            ",".join(import_names(self.packages, root)),
            "--max-output-bytes",
            str(self.max_output_bytes),
            "--max-plot-bytes",
            str(self.max_plot_bytes),
            *store,
            *([] if self.warm_imports else ["--no-warmup"]),
        ]

    def _load_only(self) -> list[str]:
        """Packages the store needs, loaded but not imported at boot (pyarrow)."""
        needed = self.store.sandbox_packages if self.store is not None else ()
        return [p for p in needed if p not in self.packages]

    @property
    def alive(self) -> bool:
        return (
            self._dead is None and self._proc is not None and self._proc.poll() is None
        )

    def start(self, timeout: float | None = 120.0) -> None:
        """Start Deno and wait until Pyodide and the packages are loaded.

        First downloads the Pyodide files the packages need into `dist_dir` if they
        aren't there yet, unless `download` is false (not counted in `timeout`; see
        `pyodide_sandbox.distribution`).
        """
        if self._proc is not None:
            raise RuntimeError("session already started")
        root = ensure_distribution(
            (*self.packages, *self._load_only()), self.dist_dir, download=self.download
        )
        command = self._command(root)
        self._store_packages = set(self._load_only())
        start = time.perf_counter()
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # nothing from the host's environment (API keys, say) reaches the process
            env=_deno_env(),
            # so a Ctrl-C in the host's terminal doesn't reach it: a new session on
            # POSIX, a new process group on Windows (where start_new_session is
            # ignored)
            start_new_session=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        weakref.finalize(self, _kill, self._proc)
        assert self._proc.stdout and self._proc.stderr
        threading.Thread(
            target=_read_messages,
            args=(
                self._proc.stdout,
                self._messages,
                _MAX_LINE_BYTES + self.max_plot_bytes,
            ),
            daemon=True,
        ).start()
        threading.Thread(
            target=_read_stderr,
            args=(self._proc.stderr, self._stderr_tail),
            daemon=True,
        ).start()
        self._wait(lambda m: m.get("op") == "ready", timeout, "boot")
        self.boot_ms = (time.perf_counter() - start) * 1000

    def run(
        self, code: str, timeout: float | None = None, cell_id: int | None = None
    ) -> CellResult:
        """Run a cell; `cell_id` names it in tracebacks (`<cell-N>`), and must increase.

        If the store gained a value that needs a package the session hasn't loaded
        (pyarrow, for a DataFrame added after boot), loads it first, downloading it
        if need be (`DistributionError`). The load has a `timeout` of its own.
        """
        if self._proc is None:
            raise RuntimeError("session not started")
        if self._dead is not None:
            raise self._dead
        if cell_id is None:
            cell_id = self._next_id + 1
        if cell_id <= self._next_id:
            raise ValueError(f"cell_id must be > {self._next_id}, got {cell_id}")
        self._load_store_packages(timeout)
        self._next_id = cell_id
        self._send({"id": cell_id, "op": "run", "code": code})
        reply = self._wait(lambda m: m.get("id") == cell_id, timeout, "cell")
        error, stdout, stderr = (
            reply.get("error"),
            reply.get("stdout"),
            reply.get("stderr"),
        )
        duration, plots = reply.get("duration_ms"), reply.get("plots")
        if not (
            (error is None or isinstance(error, str))
            and isinstance(stdout, str)
            and isinstance(stderr, str)
            and isinstance(duration, (int, float))
            and isinstance(plots, list)
            and all(isinstance(p, dict) for p in plots)
        ):
            raise self._die(SessionCrashed, "malformed reply from the sandbox")
        return CellResult(error, stdout, stderr, float(duration), plots)

    def _load_store_packages(self, timeout: float | None) -> None:
        missing = [p for p in self._load_only() if p not in self._store_packages]
        if not missing:
            return
        ensure_distribution(
            (*self.packages, *self._store_packages, *missing),
            self.dist_dir,
            download=self.download,
        )
        self._send({"op": "load", "packages": missing})
        self._wait(lambda m: m.get("op") == "loaded", timeout, "package load")
        self._store_packages.update(missing)

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        try:
            self._proc.stdin.write((json.dumps(message) + "\n").encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._die(SessionCrashed, "the sandbox process is gone") from None

    def close(self) -> None:
        if self._proc is not None:
            _kill(self._proc)
        if self._dead is None:
            self._dead = SessionCrashed("the session was closed")

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def _wait(self, wanted: Any, timeout: float | None, what: str) -> dict[str, Any]:
        """Return the first message `wanted` accepts, dropping others (stale replies)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                raise self._die(
                    SessionTimeout, f"{what} timed out after {timeout:g} s"
                ) from None
            if message is None:
                assert self._proc
                try:
                    code = self._proc.wait(timeout=1)
                    why = f"the sandbox process exited with code {code} during {what}"
                except subprocess.TimeoutExpired:
                    why = f"the sandbox broke the protocol during {what}"
                tail = self.stderr_tail()
                if tail and what == "boot":
                    why += f":\n{tail}"
                raise self._die(SessionCrashed, why)
            if message.get("op") == "fatal":
                # main.ts exits after this; a cell could also forge one, which only
                # ends its own session, as `Deno.exit` would
                reason = message.get("message")
                if not isinstance(reason, str):
                    reason = "no reason given"
                raise self._die(
                    SessionCrashed,
                    f"Pyodide failed during {what} and can't be used again: "
                    f"{reason[:_MAX_FATAL_CHARS]}",
                )
            if wanted(message):
                return message

    def _die(self, kind: type[SessionError], message: str) -> SessionError:
        assert self._proc
        _kill(self._proc)
        self._dead = kind(message)
        return self._dead


def _deno_env() -> dict[str, str]:
    env = {"NO_COLOR": "1", "PATH": os.environ.get("PATH", "")}
    # where Deno keeps its compile cache; it writes that itself, outside the permissions
    for name in ("HOME", "DENO_DIR", "XDG_CACHE_HOME"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _read_messages(
    stream: IO[bytes],
    messages: "queue.Queue[dict[str, Any] | None]",
    max_line_bytes: int,
) -> None:
    """Parse Deno's stdout into `messages`; None when it ends or misbehaves."""
    try:
        while line := stream.readline(max_line_bytes + 1):
            if len(line) > max_line_bytes or not line.endswith(b"\n"):
                break
            try:
                message = json.loads(line)
            except ValueError:
                break
            if not isinstance(message, dict):
                break
            messages.put(message)
    except (OSError, ValueError):
        pass
    messages.put(None)


def _read_stderr(stream: IO[bytes], tail: deque[str]) -> None:
    """Keep the last lines of Deno's stderr (boot errors), so the pipe never fills."""
    try:
        while line := stream.readline(64 << 10):
            tail.append(line.decode(errors="replace").rstrip("\n"))
    except (OSError, ValueError):
        pass
