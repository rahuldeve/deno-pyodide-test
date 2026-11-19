"""Runs one REPL cell inside Pyodide; loaded by main.ts into a namespace of its own.

Pyodide already keeps state between calls and can return a cell's last expression
(`CodeRunner`). What it lacks for an agent REPL is added here:

- the value of a bare last expression is printed as its `repr`, as Python's REPL does
- each cell is compiled as `<cell-N>` and its source registered with `linecache`, so
  tracebacks show the model's own line numbers and code
- tracebacks start at the cell's first frame, without this runner's or Pyodide's
- the exception type is returned on its own, for the cell index

Output isn't captured here: main.ts captures the process's stdout and stderr per
cell, which also catches what C extensions write. The cell's variables live in
`__main__`, so classes and functions the model defines can be pickled.

With a variable store, `setup_host` also defines `host` there (see `host.py`). With
bokeh loaded, `setup_plots` makes its `show()` collect plots for the host instead of
opening a browser (see `plots.py`); main.ts sends them with the cell's result.
"""

import ast
import importlib.util
import linecache
import sys
import traceback

import js
from pyodide.code import CodeRunner
from pyodide.ffi import JsException

import __main__

_CELL_PREFIX = "<cell-"

# The running cell's bokeh plots (plots.Plots), with bokeh loaded
_plots = None


def _load_module(source: str, filename: str) -> dict:
    """Run `source` in a namespace of its own, with its lines in tracebacks."""
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": f"pyodide_{filename.strip('<>')}"}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace


def setup_host(source: str, root: str) -> None:
    """Define `host` in `__main__`, reading the store at `root` through Deno.

    `source` is host.py, run in a namespace of its own so that only `host` reaches
    the cells. The store is read once here, so a missing one fails the boot.
    """
    host = _load_module(source, "<host>")["Host"](root, _read_through_deno)
    len(host)
    __main__.host = host


def setup_plots(source: str, max_bytes: int) -> None:
    """Make bokeh's `show()` collect plots, up to `max_bytes` a cell; `source` is
    plots.py. Does nothing without bokeh."""
    global _plots
    if importlib.util.find_spec("bokeh") is None:
        return
    _plots = _load_module(source, "<plots>")["Plots"](max_bytes)
    _plots.install()


def take_plots() -> str:
    """The plots shown since the last call, as a JSON array of bokeh json_items."""
    return "[]" if _plots is None else _plots.take()


def _read_through_deno(path: str) -> bytes:
    # Not a mount: see pyodide_sandbox.variables for why
    try:
        return js.Deno.readFileSync(path).to_bytes()
    except JsException as e:
        # NotFound: the store was closed, or a value's old bytes deleted (host.py)
        kind = FileNotFoundError if e.name == "NotFound" else OSError
        raise kind(f"can't read the host's variable store: {e}") from None


async def run_cell(code: str, number: int) -> str | None:
    """Run `code` as cell `number`; return the exception type's name, or None."""
    filename = f"{_CELL_PREFIX}{number}>"
    # mtime None keeps linecache.checkcache() from dropping the entry
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    try:
        runner = CodeRunner(
            code,
            filename=filename,
            return_mode="last_expr",
            # Python's REPL echoes `x;` too
            quiet_trailing_semicolon=False,
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        ).compile()
        value = await runner.run_async(__main__.__dict__)
        if value is not None:
            print(repr(value))
        return None
    # SystemExit and KeyboardInterrupt end the cell, not the process
    except BaseException as e:  # noqa: BLE001
        sys.stderr.write(_format(e))
        return type(e).__qualname__
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def _format(e: BaseException) -> str:
    """Format `e` from the cell's first frame; a SyntaxError has none, so no frames."""
    tb = e.__traceback__
    while tb is not None and not tb.tb_frame.f_code.co_filename.startswith(
        _CELL_PREFIX
    ):
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(e), e, tb))
