# pyodide-sandbox

A pydantic-ai capability, `PyodideRepl`, that gives an agent a persistent, sandboxed
Python REPL: Pyodide 314.0.7 (Python 3.14) running in its own Deno process, with
numpy 2.4.6 and pandas 3.0.2 loaded from local files. Code in a cell reads host
values (a DataFrame, say) as `host.NAME`. Phase 0 of the plan in
[Pyodide on Deno as an eryx backend](<Pyodide on Deno as an eryx backend.md>); the
capability is a port of eryx-test's `EryxRepl`.

```python
from pydantic_ai import Agent
from pyodide_sandbox import PyodideRepl

# packages=("numpy", "pandas", "tzdata") by default
repl = PyodideRepl(variables={"df": df}, on_change="append")
agent = Agent("anthropic:claude-sonnet-5-5", capabilities=[repl])
```

## Setup

```sh
# Deno 2.9.8 in ~/.deno/bin (found on PATH, or set PYODIDE_SANDBOX_DENO)
uv sync
uv run poe setup     # downloads the 337 MB Pyodide distribution (sha256-checked) and the npm package
uv run poe test      # 109 tests, ~3 min; the sandbox ones skip without Deno or the distribution
uv run poe agent     # live demo, needs ANTHROPIC_API_KEY
```

## How it fits together

| Piece | Role |
|---|---|
| `src/pyodide_sandbox/repl.py` | `PyodideRepl`: `run_code`, `repl_history`, the cell index in the instructions, restart on timeout or crash |
| `src/pyodide_sandbox/session.py` | `PyodideSession`: starts and kills the Deno process, speaks JSON lines, checks every reply |
| `src/pyodide_sandbox/variables.py` | `VariableStore`: host values pickled into a temporary directory that only the session's Deno can read, and only read |
| `deno/main.ts` | Loads Pyodide and the packages, captures each cell's stdout and stderr (including C-level writes), caps them at 1 MB |
| `deno/runner.py` | Runs a cell in Pyodide: `CodeRunner` as `<cell-N>`, echoes the last expression's `repr`, trims tracebacks to the cell's frames |
| `deno/host.py` | The `host` object in the sandbox: reads the store's `view.json` and unpickles a value on every read, so each read is a fresh copy |

Deno runs with `--no-prompt --cached-only --no-remote --allow-read=<deno dir>,<pyodide-dist>,<store>`,
so it has no network, write, subprocess, FFI or env access. It gets none of the host's
environment variables. The time limit is a SIGKILL of the process, and the next cell
starts a new process. If Pyodide itself fails for good (`os._exit`, or a JavaScript
error inside one of its syscalls), `main.ts` reports the cause and exits, and the next
cell starts a new process too. A JavaScript error that nothing catches but that
leaves Pyodide usable goes to the running cell's stderr, and the session goes on.

Host variables are pickled on the host and unpickled in the sandbox, so the host pins
numpy and pandas to the sandbox's versions. A class defined only on the host fails
when a cell reads it. The store isn't mounted into Pyodide's filesystem: `host` reads
it with `Deno.readFileSync`. Through a NODEFS mount, a cell's `os.chmod` or `os.utime`
reaches Deno, and Deno's refusal breaks Pyodide for good, even with the files
read-only on the host. Through `Deno.readFileSync`, every refused call is an ordinary
exception in the cell.

## Measured on this machine (WSL2)

| What | Result |
|---|---|
| Boot, core only / + numpy / + numpy and pandas | 2.2 s / 3.2 s / 6.3 s (eryx's pandas factory: 2.12 s) |
| SIGKILL during `while True` with a 1 s limit | stops at 1.06 s, and `except BaseException` can't evade it |
| SIGKILL during one `np.linalg.svd` call with a 0.5 s limit | process gone within 1 s |
| `js.fetch`, reading `/etc`, writing, `Deno.env`, `Deno.Command`, `Deno.dlopen` from a cell | all refused by Deno (`NotCapable`) |
| Python `open()` | Emscripten's in-memory FS; host files, even those Deno can read, aren't visible |
| numpy errors | Float and array errors are silent: no warnings, `errstate(all="raise")` doesn't raise, `inv`/`solve`/`cholesky` give NaN, integer array `// 0` gives 0. Integer scalars and plain Python still warn or raise. Default int is **int32**, and overflow wraps |
| `maxMemoryBytes` | Not in the 314.0.7 release |
| Boot with `tzdata` as well | About +0.2 s. Without it, pandas and `zoneinfo` know no time zones |
| `pyodide.mountNodeFS` under Deno | Reads work. A write, `chmod` or `utime` that reaches Deno is fatal to Pyodide (not used) |

## Not in phase 0

- `host.tools`
- Snapshots and a warm pool, for startup time
- A memory limit (needs a cgroup)
- Running as another user, or seccomp
- Escape tests beyond Deno's permission flags, such as `ctypes.CDLL(None)`

A cell can write to the protocol pipe itself, so it can forge its own reply. That
only misleads the model about its own cell, but the host still treats every message
as untrusted.
