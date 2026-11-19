"""A persistent Pyodide session as a pydantic-ai capability: a sandboxed Python REPL.

`PyodideRepl` gives the model two tools:

- `run_code` runs a cell in one Pyodide session, in a Deno process that lives as long
  as the capability, so variables carry over between calls and across `agent.run()`
  calls. It returns the cell's stdout and stderr (a traceback if the code raised),
  and echoes the value of a bare last expression, as a REPL would.
- `repl_history` returns the code of earlier cells, and their output on request.

With bokeh loaded (a default package), a cell's `show(p)` sends the plot to the host
as a `Plot` (bokeh's `json_item`), on `Cell.plots` and to `on_plot`, for the host
to render with BokehJS. The model is told its id (`[plot 3f9a2c1b]`), and `run_code`
returns the plots in its tool return's metadata, where they stay in the message
history; `pyodide_sandbox.plots` has the rest, for letting the model choose which
plots the user sees.

With `variables=` (or a shared `store=`), code in a cell reads host values as
`host.NAME` (see `pyodide_sandbox.variables`). The host can change them while the
REPL runs (`repl.store.set(...)`): the instructions list the current variables and
mark those changed after a cell the model ran, so it knows to read them again.

Every cell is kept on the host, and the instructions carry an index of them built
from that record rather than from the conversation. So whatever a compaction
strategy does to old tool calls and results (clears, summarises or drops them),
the model can still see what it ran and fetch it again with `repl_history`.
Compaction capabilities can be listed in any order relative to `PyodideRepl`.

    repl = PyodideRepl(variables={"df": df}, on_change="append")
    agent = Agent(
        "anthropic:claude-sonnet-5-5",
        capabilities=[ClearToolResults(max_tokens=50_000), repl],
    )

Ported from eryx-test's `EryxRepl`, without its `host.tools` and snapshots so far.
See `pyodide_sandbox.session` for the process boundary.
"""

import asyncio
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, InitVar, dataclass, field
from pathlib import Path
from typing import Any, Self

from pydantic_ai import FunctionToolset, RunContext, ToolReturn
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionChangePolicy, InstructionPart

from pyodide_sandbox.distribution import (
    PYTHON_VERSION,
    DistributionError,
    loaded_packages,
)
from pyodide_sandbox.plots import Plot, plots_metadata, report_plots
from pyodide_sandbox.session import (
    DEFAULT_PACKAGES,
    PyodideSession,
    SessionCrashed,
    SessionError,
    SessionTimeout,
)
from pyodide_sandbox.variables import VariableStore

INSTRUCTIONS = """\
You have a persistent, sandboxed Python REPL. Run code with `run_code`: \
variables, imports and functions carry over between calls. Each call returns the \
cell's stdout and stderr, and the value of a bare last expression is printed. \
Earlier calls and their results may be cleared, summarised or dropped from this \
conversation, but the list of cells below is complete: `repl_history(cell=N)` shows \
a cell's code, and `repl_history(cell=N, output=True)` also what it printed.

The sandbox is Python {python} in WebAssembly (Pyodide), with no network access and \
no way to install packages. Loaded: {packages}. Top-level `await` works.{numerics}"""

# What tests/test_session.py::test_numerics records for Pyodide 314.0.7's numpy
NUMERICS = """

numpy's floating-point errors are silent in this sandbox, because WebAssembly has no \
floating-point error flags. Array and float operations that would normally warn \
about division by zero, overflow or invalid values (such as `np.sqrt(-1)`) don't, \
and `np.errstate(all="raise")` doesn't raise for them. `np.linalg.inv`, `solve` and \
`cholesky` return NaNs instead of raising `LinAlgError`, so NaNs from them mean the \
call failed.{scipy_linalg} Integer array division by zero returns 0 without a warning. Plain \
Python arithmetic and `math` still raise as usual. Check computed values with \
`np.isfinite` before trusting them, and before NaN-skipping steps such as \
`np.nanmean` or `dropna` hide a failure. numpy's default integer is 32-bit here \
(`np.arange(3).dtype` is int32) and integer overflow wraps around silently, so pass \
`dtype=np.int64` for large integers."""

SCIPY_LINALG = """ `scipy.linalg`'s `inv`, `solve` and `cholesky` do raise \
`LinAlgError`, so use them when you need a matrix inverse or solve of your own."""

BOKEH = """

Plot with bokeh: `show(p)` (from `bokeh.plotting` or `bokeh.io`) saves a figure, \
widget or layout (`column(p1, p2)`, `gridplot`) as an interactive plot, and \
`run_code` reports its id, as in `[plot 3f9a2c1b]`. You can't see plots, so describe \
each one from the values you computed, and give its id when you report on it. A \
cell's plots can hold at most \
{max_plot_mb} MB of JSON: aggregate, bin or sample larger data first. \
`output_notebook` and `output_file` aren't needed; `bokeh.sampledata`, \
`export_png` and `export_svg` aren't available."""

# What tests/test_session.py::test_statsmodels_facts records
STATSMODELS = """

Fit statistical models with statsmodels rather than writing the linear algebra \
yourself: `statsmodels.formula.api` (`smf.ols("y ~ x + C(g)", df)`) or \
`statsmodels.api` for regression (including regularized fits), GLM, Logit, mixed \
models and time series (ARIMA, SARIMAX, Holt-Winters). Collinear predictors don't raise \
`LinAlgError` or warn in the maximum-likelihood models (`Logit`, `Probit`, \
`MNLogit`, `Poisson`, `NegativeBinomial` and the like): standard errors and \
p-values come back NaN, or finite but huge and meaningless. Before reading such a \
summary, check that `np.linalg.matrix_rank(result.model.exog)` equals \
`result.model.exog.shape[1]` and that `np.isfinite(np.asarray(result.bse)).all()` \
is true. The first `import statsmodels.api` takes several seconds."""


@dataclass
class Cell:
    """One `run_code` call: the code as the model wrote it and what it printed."""

    number: int
    code: str
    stdout: str
    stderr: str
    error: str | None
    """The exception type for code that raised, or why the sandbox stopped it."""
    duration_ms: float
    store_generation: int = 0
    """The store's `generation` when the cell started."""
    plots: list[Plot] = field(default_factory=list)
    """The plots the cell showed, in order."""

    def output(self) -> str:
        """What the cell printed, then the ids of the plots it showed."""
        return self.printed() + self.plot_lines()

    def printed(self) -> str:
        """stdout, stderr, and the error if stderr doesn't name it."""
        parts = [self.stdout, self.stderr]
        if self.error and self.error not in self.stderr:
            parts.append(self.error + "\n")
        return "".join(parts)

    def plot_lines(self) -> str:
        return "".join(f"[plot {p.id}]\n" for p in self.plots)

    def summary(self, width: int = 80) -> str:
        """The cell's index line: number, ok/error and its first line of code."""
        first = next((line for line in self.code.splitlines() if line.strip()), "")
        if len(first) > width:
            first = first[: width - 3] + "..."
        status = f"error ({self.error})" if self.error else "ok"
        return f"{self.number} {status}: {first}"


@dataclass
class PyodideRepl(AbstractCapability[Any]):
    """Capability giving the model a persistent, sandboxed Python REPL (see module docs).

    The session (one Deno process) starts on the first `run_code` call, or ahead of
    it with `warm_up()`, and is kept, with the cell history, until `reset()`. A cell that runs past
    `execution_timeout_ms`, or that crashes the process, gets the process killed; the
    next cell starts a new one, without the earlier variables. If a session can't
    start, `run_code` raises `RuntimeError`, which ends the run. Every new session
    gives the cells `host`, reading `store`; host values live on the host, so they
    survive restarts, and changes to `store` reach the running session.

    An instance belongs to the first agent it is given to; another agent gets a
    `ValueError`. It should also serve one conversation at a time, which isn't
    checked: concurrent runs of its agent share variables and history. Tool calls
    in one model response run one at a time, in order. Cancelling `run_code` doesn't
    stop its cell (it runs in a thread): the next cell, and `reset()`, wait for it to
    finish or time out, and it is still recorded.
    """

    packages: Sequence[str] = DEFAULT_PACKAGES
    """Pyodide packages loaded and imported when a session starts, by their names in
    Pyodide's package list (e.g. `scikit-learn`); each adds startup time."""

    _: KW_ONLY

    store: VariableStore | None = None
    """Host values the model reads as `host.NAME` (see `pyodide_sandbox.variables`).

    `None` creates one from `variables`. A store passed in belongs to the caller,
    and can be shared by several REPLs. It can't be replaced afterwards, but its
    values can be changed (`store.set`, `update`, `remove`) at any time."""

    variables: InitVar[Mapping[str, Any] | None] = None
    """Shorthand for `store=VariableStore(variables)`. The values are copied and
    pickled when the capability is created, so later changes to the objects on the
    host aren't seen; change them through `store` instead. Unpickling happens in the sandbox, when a cell reads one, so it needs the
    values' modules (e.g. pandas, among `packages`) and can't load classes defined
    only on the host. Only an init argument: read `store` afterwards, as this
    attribute stays `None`."""

    execution_timeout_ms: int | None = 60_000
    """Time limit for one cell; at the limit the process is killed. `None` removes it."""

    startup_timeout_ms: int = 120_000
    """Time limit for starting a session (Pyodide plus `packages`)."""

    max_output_chars: int = 20_000
    """Longer cell output is clipped in the `run_code` return; history keeps it all."""

    max_output_bytes: int = 1 << 20
    """Each of a cell's stdout and stderr is cut to this inside the sandbox, so a
    cell can't flood the host; history keeps what is left."""

    max_plot_bytes: int = 10_000_000
    """The most JSON a cell's bokeh plots can add up to; a `show()` past it raises
    `ValueError` in the cell, and the instructions tell the model the limit."""

    on_plot: Callable[[Plot], None] | None = None
    """Called with each plot a cell shows, once the cell has finished and is in
    `cells`, in order. It runs in the worker thread that ran the cell, and an
    exception from it ends the agent run. The plots are also kept on `Cell.plots`."""

    index_cells: int = 50
    """Most recent cells listed in the instructions; `0` lists none, only how many
    there are."""

    on_change: InstructionChangePolicy = "rewrite"
    """How changes to the instructions (the cell index) reach the model.

    `'rewrite'` resends them in the prompt prefix on each request. `'append'` keeps
    the prefix and adds each change to history (as a diff where shorter), so the
    prompt cache survives new cells; the full text is resent after compaction. See
    pydantic-ai's `InstructionPart.on_change`."""

    deno: str | None = None
    """The Deno executable; `None` uses `pyodide_sandbox.session.default_deno()`."""

    dist_dir: str | Path | None = None
    """Where the Pyodide files are kept; `None` uses `$PYODIDE_DIST_DIR`, else the
    user's cache (`pyodide_sandbox.distribution.default_dist_dir()`)."""

    download: bool = True
    """Download the Pyodide files `packages` need when the first session starts, if
    they aren't in `dist_dir` yet (about 53 MB for the default packages). With
    `False`, a missing file fails the start; fetch them ahead of time with
    `python -m pyodide_sandbox fetch`."""

    warm_imports: bool = True
    """Import the packages' slow submodules (`scipy.stats`, `statsmodels.api`; see
    `_deno/warmup.py`) when a session starts. That takes about 5 s for the default
    packages, which the first cell to use them would otherwise pay. Call `warm_up()`
    to have it done before the first cell."""

    _session: PyodideSession | None = field(default=None, init=False, repr=False)
    _cells: list[Cell] = field(default_factory=list, init=False, repr=False)
    # held while a cell runs, so a cancelled cell can't overlap the next one
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _agent: weakref.ref[AbstractAgent[Any, Any]] | None = field(
        default=None, init=False, repr=False
    )

    def __setattr__(self, name: str, value: Any) -> None:
        # A running session keeps the store it was started with, so a new one would
        # put the instructions and `host` out of step
        if name == "store" and self.__dict__.get("store") is not None:
            raise AttributeError("PyodideRepl.store can't be changed after creation")
        if name == "variables":
            raise AttributeError("PyodideRepl.variables is only an init argument")
        super().__setattr__(name, value)

    def __post_init__(self, variables: Mapping[str, Any] | None) -> None:
        # Appended instruction changes are addressed by the capability's id
        if self.id is None:
            self.id = "pyodide_repl"
        if self.index_cells < 0:
            raise ValueError(f"index_cells must be >= 0, got {self.index_cells}")
        if self.store is not None and variables is not None:
            raise ValueError("pass variables or store, not both")
        if self.store is None:
            # Pickled now so an unpicklable value fails here, not on the first call
            self.store = VariableStore(variables)
        self.packages = tuple(self.packages)
        # Each call runs alone, in the order the model made them: the session can
        # only run one cell at a time, and later cells may use earlier ones' variables
        self._toolset = FunctionToolset[Any](
            [self.run_code, self.repl_history], sequential=True
        )

    def for_agent(self, agent: AbstractAgent[Any, Any]) -> Self:
        """Bind to `agent`, refusing any other while it is alive."""
        bound = self._agent() if self._agent else None
        if bound is not None and bound is not agent:
            raise ValueError(
                f"this PyodideRepl is already used by agent {bound.name or bound!r}; "
                "create one PyodideRepl per agent"
            )
        self._agent = weakref.ref(agent)
        return self

    @property
    def cells(self) -> list[Cell]:
        return list(self._cells)

    @property
    def plots(self) -> list[Plot]:
        """Every cell's plots, in the order they were shown."""
        return [plot for cell in self._cells for plot in cell.plots]

    @property
    def session(self) -> PyodideSession | None:
        """The running session, if any (for its `boot_ms`, say)."""
        return self._session

    def warm_up(self) -> None:
        """Start the session in the background, so the first cell doesn't wait for it.

        Returns at once. Call it when the capability is created, or when a
        conversation opens: the boot (10 to 15 s for the default packages) then runs
        while the model writes its first cell, and a cell that arrives sooner waits
        for the rest of it. A start that fails is tried again by the first cell,
        which raises the error. Does nothing while a session is running.
        """

        def start() -> None:
            with self._lock:
                try:
                    self._get_session()
                except RuntimeError:
                    pass

        threading.Thread(target=start, name="pyodide-repl-warm-up", daemon=True).start()

    def reset(self) -> None:
        """Drop the session and the cell history; the next cell starts fresh."""
        with self._lock:
            self._close_session()
            self._cells.clear()

    def close(self) -> None:
        """Stop the session's process; the next cell starts a new one."""
        with self._lock:
            self._close_session()

    @property
    def _host(self) -> VariableStore:
        assert self.store is not None, "set in __post_init__"
        return self.store

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def get_instructions(self) -> Any:
        return self._instructions

    async def run_code(self, ctx: RunContext[Any], code: str) -> str | ToolReturn:
        """Run Python code in the persistent REPL and return what it printed.

        Args:
            code: Python source. Variables, imports and functions persist between
                calls; a bare last expression is printed, as in a REPL.
        """
        cell = await asyncio.to_thread(self._execute, code)
        printed = cell.printed()
        if len(printed) > self.max_output_chars:
            printed = (
                printed[: self.max_output_chars]
                + f"\n... {len(printed) - self.max_output_chars} more characters, "
                f"see repl_history(cell={cell.number}, output=True)\n"
            )
        # after the clipping, so the model always learns the plots' ids
        output = printed + cell.plot_lines() or "(no output)\n"
        text = f"[cell {cell.number}]\n{output}"
        if not cell.plots:
            return text
        report_plots(cell.plots)  # to the delegations running this cell, if any
        # kept in the history with the result, but not sent to the model
        return ToolReturn(return_value=text, metadata=plots_metadata(cell.plots))

    async def repl_history(
        self,
        ctx: RunContext[Any],
        cell: int | None = None,
        last: int = 5,
        output: bool = False,
    ) -> str:
        """Show the code of earlier REPL cells, and optionally their full output.

        Args:
            cell: The cell number to show. Omit it to show the most recent cells.
            last: How many of the most recent cells to show when `cell` is omitted.
            output: Also show each cell's full output (stdout, stderr and error).
        """
        if cell is not None:
            if not 1 <= cell <= len(self._cells):
                return f"No cell {cell}: cells run so far are 1-{len(self._cells)}."
            shown = [self._cells[cell - 1]]
        else:
            shown = self._cells[-max(last, 1) :]
        if not shown:
            return "No cells have been run yet."
        return "\n\n".join(_render_cell(c, output) for c in shown)

    def _instructions(self, ctx: RunContext[Any]) -> str | InstructionPart:
        text = self._instructions_text()
        if self.on_change == "append":
            return InstructionPart(text, name="repl", on_change="append")
        return text

    def _instructions_text(self) -> str:
        # What the session loads, not what `packages` spells: statsmodels brings
        # numpy and scipy with it, and `Statsmodels` loads too
        loaded = loaded_packages(self.packages, self.dist_dir)
        text = INSTRUCTIONS.format(
            python=PYTHON_VERSION,
            packages=", ".join(self.packages) or "the standard library only",
            numerics=NUMERICS.format(
                scipy_linalg=SCIPY_LINALG if "scipy" in loaded else ""
            )
            if "numpy" in loaded
            else "",
        )
        if "statsmodels" in loaded:
            text += STATSMODELS
        if "bokeh" in loaded:
            text += BOKEH.format(max_plot_mb=f"{self.max_plot_bytes / 1e6:.3g}")
        if names := self._host.names():
            entries = [self._host.info(n) for n in names]
            changed = {e.name: self._changed_since(e.generation) for e in entries}
            listed = ", ".join(
                f"{e.name} ({e.type}"
                + (f", changed since cell {n})" if (n := changed[e.name]) else ")")
                for e in entries
            )
            text += (
                f"\n\nHost variables (read-only): {listed}. Read one with "
                f"`host.NAME`, e.g. `{names[0]} = host.{names[0]}`; each read "
                "returns a fresh copy."
            )
            if any(changed.values()):
                text += (
                    " The host may add, replace or remove them between your calls. "
                    "One changed since cell N was set after that cell started: read "
                    "it again rather than using a copy made before."
                )
        if not self._cells:
            return text
        # -0 would slice from the start and list every cell
        shown = self._cells[-self.index_cells :] if self.index_cells else []
        lines = [c.summary() for c in shown]
        if earlier := len(self._cells) - len(shown):
            lines.insert(0, f"({earlier} earlier cells, see repl_history)")
        return text + "\n\nCells run so far:\n" + "\n".join(lines)

    def _changed_since(self, generation: int) -> int | None:
        """The last cell that started before the store's `generation`, if any."""
        for cell in reversed(self._cells):
            if cell.store_generation < generation:
                return cell.number
        return None

    def _get_session(self) -> PyodideSession:
        if self._session is None or not self._session.alive:
            session = PyodideSession(
                self.packages,
                deno=self.deno,
                dist_dir=self.dist_dir,
                download=self.download,
                max_output_bytes=self.max_output_bytes,
                max_plot_bytes=self.max_plot_bytes,
                store=self._host,
                warm_imports=self.warm_imports,
            )
            # Failures here are host setup problems (no Deno, a failed download, a
            # package that won't load, a closed store), not something for the model
            # to see as a cell error
            try:
                session.start(timeout=self.startup_timeout_ms / 1000)
            except (SessionError, OSError, RuntimeError, ValueError) as e:
                session.close()
                raise RuntimeError(
                    f"starting the PyodideRepl session failed: {e}"
                ) from e
            self._session = session
        return self._session

    def _close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _restart(self) -> str:
        """Drop a session that was killed or crashed."""
        self._close_session()
        note = "; the sandbox was restarted and variables from earlier cells are gone"
        if self._host.names():
            note += "; host variables are still there"
        return note

    def _execute(self, code: str) -> Cell:
        with self._lock:
            session = self._get_session()
            number = len(self._cells) + 1
            timeout = (
                None
                if self.execution_timeout_ms is None
                else self.execution_timeout_ms / 1000
            )
            stdout = stderr = ""
            items: list[dict[str, Any]] = []
            # before the cell, so a change during it counts as after it started
            generation = self._host.generation
            start = time.perf_counter()
            try:
                result = session.run(code, timeout=timeout, cell_id=number)
                stdout, stderr, error = result.stdout, result.stderr, result.error
                items = result.plots
            except DistributionError as e:
                # pyarrow for a DataFrame added since the start; as for the start,
                # a host problem, not the model's
                raise RuntimeError(
                    f"loading the packages the host variables need failed: {e}"
                ) from e
            except SessionTimeout:
                error = (
                    f"timed out after {self.execution_timeout_ms} ms" + self._restart()
                )
            except SessionCrashed as e:
                error = f"sandbox crashed: {e}" + self._restart()
            cell = Cell(
                number=number,
                code=code,
                stdout=stdout,
                stderr=stderr,
                error=error,
                duration_ms=(time.perf_counter() - start) * 1000,
                store_generation=generation,
            )
            shown = sum(len(c.plots) for c in self._cells)
            cell.plots = [
                Plot(number=shown + i, cell=number, json=item)
                for i, item in enumerate(items, 1)
            ]
            self._cells.append(cell)
            if self.on_plot is not None:
                for plot in cell.plots:
                    self.on_plot(plot)
            return cell


def _render_cell(cell: Cell, output: bool) -> str:
    text = (
        f"[cell {cell.number}{f', {cell.error}' if cell.error else ''}]\n"
        f">>> code\n{cell.code.rstrip()}\n"
    )
    if output:
        text += f">>> output\n{cell.output() or '(no output)'}"
    return text
