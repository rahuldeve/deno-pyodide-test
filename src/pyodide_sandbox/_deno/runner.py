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
opening a browser (see `plots.py`); main.ts sends them with the cell's result. With
host modules, `install_modules` writes them to Pyodide's in-memory file system,
imports them and defines their public names in `__main__` (see
`pyodide_sandbox.modules`).
"""

import ast
import importlib
import importlib.util
import json
import linecache
import os
import sys
import traceback

import js
from pyodide.code import CodeRunner
from pyodide.ffi import JsException

import __main__

_CELL_PREFIX = "<cell-"
# Host modules live here, in Pyodide's in-memory file system: writes there never
# reach Deno (see pyodide_sandbox.variables for what a write that did would do)
MODULES_DIR = "/host_modules"

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


def install_modules(files_json: str, exports_json: str) -> str | None:
    """Install the host's modules and define their public names in `__main__`.

    `files_json` maps paths under `MODULES_DIR` to source, `exports_json` module
    names to their public names. Returns None, or why it failed (a traceback for an
    import that raised). Their files are real files, so tracebacks show their lines.
    """
    files = json.loads(files_json)
    exports = json.loads(exports_json)
    # appended to sys.path, so a host module can't replace a package or the
    # standard library; one with the same name would silently never load, so
    # refuse it instead
    for top in sorted({path.split("/")[0].removesuffix(".py") for path in files}):
        spec = importlib.util.find_spec(top)
        if spec is not None:
            where = spec.origin or "the sandbox"
            return (
                f"host module {top!r} has the name of a module the sandbox "
                f"already has ({where}); rename it"
            )
    for path, source in files.items():
        full = os.path.join(MODULES_DIR, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(source)
    sys.path.append(MODULES_DIR)
    importlib.invalidate_caches()
    for name, names in exports.items():
        try:
            module = importlib.import_module(name)
            for public in names:
                setattr(__main__, public, getattr(module, public))
        except BaseException:  # noqa: BLE001
            return f"importing host module {name} failed:\n{traceback.format_exc()}"
    return None


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
