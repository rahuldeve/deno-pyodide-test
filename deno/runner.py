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

With a variable store, `setup_host` also defines `host` there (see `host.py`).
"""

import ast
import linecache
import sys
import traceback

import js
from pyodide.code import CodeRunner
from pyodide.ffi import JsException

import __main__

_CELL_PREFIX = "<cell-"
_HOST_FILENAME = "<host>"


def setup_host(source: str, root: str) -> None:
    """Define `host` in `__main__`, reading the store at `root` through Deno.

    `source` is host.py, run in a namespace of its own so that only `host` reaches
    the cells. The store is read once here, so a missing one fails the boot.
    """
    linecache.cache[_HOST_FILENAME] = (
        len(source),
        None,
        source.splitlines(True),
        _HOST_FILENAME,
    )
    namespace = {"__name__": "pyodide_host"}
    exec(compile(source, _HOST_FILENAME, "exec"), namespace)  # noqa: S102
    host = namespace["Host"](root, _read_through_deno)
    len(host)
    __main__.host = host


def _read_through_deno(path: str) -> bytes:
    # Not a mount: see pyodide_sandbox.variables for why
    try:
        return js.Deno.readFileSync(path).to_bytes()
    except JsException as e:
        raise OSError(f"can't read the host's variable store: {e}") from None


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
