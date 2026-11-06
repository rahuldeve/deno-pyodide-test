# pyodide-sandbox

A pydantic-ai capability, `PyodideRepl`, that gives an agent a persistent, sandboxed
Python REPL: Pyodide 314.0.7 (Python 3.14) running in its own Deno process, with
numpy 2.4.6 and pandas 3.0.2 loaded from local files. Phase 0 of the plan in
[Pyodide on Deno as an eryx backend](<Pyodide on Deno as an eryx backend.md>); the
capability is a port of eryx-test's `EryxRepl`.

```python
from pydantic_ai import Agent
from pyodide_sandbox import PyodideRepl

repl = PyodideRepl(on_change="append")  # packages=("numpy", "pandas")
agent = Agent("anthropic:claude-sonnet-5-5", capabilities=[repl])
```

## Setup

```sh
# Deno 2.9.8 in ~/.deno/bin (found on PATH, or set PYODIDE_SANDBOX_DENO)
uv sync
uv run poe setup     # downloads the 337 MB Pyodide distribution (sha256-checked) and the npm package
uv run poe test      # 62 tests, ~2.5 min; skipped without Deno or the distribution
uv run poe agent     # live demo, needs ANTHROPIC_API_KEY
```

## How it fits together

| Piece | Role |
|---|---|
| `src/pyodide_sandbox/repl.py` | `PyodideRepl`: `run_code`, `repl_history`, the cell index in the instructions, restart on timeout or crash |
| `src/pyodide_sandbox/session.py` | `PyodideSession`: starts and kills the Deno process, speaks JSON lines, checks every reply |
| `deno/main.ts` | Loads Pyodide and the packages, captures each cell's stdout and stderr (including C-level writes), caps them at 1 MB |
| `deno/runner.py` | Runs a cell in Pyodide: `CodeRunner` as `<cell-N>`, echoes the last expression's `repr`, trims tracebacks to the cell's frames |

Deno runs with `--no-prompt --cached-only --no-remote --allow-read=<deno dir>,<pyodide-dist>`,
so it has no network, write, subprocess, FFI or env access. It gets none of the host's
environment variables. The time limit is a SIGKILL of the process, and the next cell
starts a new process.

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

## Not in phase 0

- Host variables (`store` / `host.NAME`)
- `host.tools`
- Snapshots and a warm pool, for startup time
- A memory limit (needs a cgroup)
- Running as another user, or seccomp
- Escape tests beyond Deno's permission flags, such as `ctypes.CDLL(None)`

A cell can write to the protocol pipe itself, so it can forge its own reply. That
only misleads the model about its own cell, but the host still treats every message
as untrusted.
