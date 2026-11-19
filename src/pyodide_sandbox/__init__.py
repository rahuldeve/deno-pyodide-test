"""A sandboxed Python REPL for pydantic-ai agents: Pyodide in a Deno process per session."""

from pyodide_sandbox.distribution import (
    PYODIDE_VERSION,
    SANDBOX_VERSIONS,
    DistributionError,
    ensure_distribution,
)
from pyodide_sandbox.plots import (
    PLOTS_KEY,
    Plot,
    PlotAnswer,
    plots_in,
    split_answer,
    subagent_return,
)
from pyodide_sandbox.repl import Cell, PyodideRepl
from pyodide_sandbox.session import (
    CellResult,
    PyodideSession,
    SessionCrashed,
    SessionError,
    SessionTimeout,
)
from pyodide_sandbox.variables import VariableStore, VersionMismatchWarning

__all__ = [
    "PLOTS_KEY",
    "PYODIDE_VERSION",
    "SANDBOX_VERSIONS",
    "Cell",
    "CellResult",
    "DistributionError",
    "Plot",
    "PlotAnswer",
    "PyodideRepl",
    "PyodideSession",
    "SessionCrashed",
    "SessionError",
    "SessionTimeout",
    "VariableStore",
    "VersionMismatchWarning",
    "ensure_distribution",
    "plots_in",
    "split_answer",
    "subagent_return",
]
