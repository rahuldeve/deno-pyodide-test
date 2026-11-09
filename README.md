# pyodide-sandbox

A pydantic-ai capability, `PyodideRepl`, that gives an agent a persistent, sandboxed
Python REPL: Pyodide 314.0.7 (Python 3.14) running in its own Deno process, with
numpy 2.4.6, pandas 3.0.2, scipy 1.18.0 and statsmodels 0.14.6 loaded from local
files. Code in a cell
reads host values (a DataFrame, say) as `host.NAME`. Phase 0 of the plan in
[Pyodide on Deno as an eryx backend](<Pyodide on Deno as an eryx backend.md>); the
capability is a port of eryx-test's `EryxRepl`.

```python
from pydantic_ai import Agent
from pyodide_sandbox import PyodideRepl

# packages=("numpy", "pandas", "scipy", "statsmodels", "tzdata") by default
repl = PyodideRepl(variables={"df": df}, on_change="append")
agent = Agent("anthropic:claude-sonnet-5-5", capabilities=[repl])
```

## Using it as a library

```sh
uv add "pyodide-sandbox @ git+ssh://…/pyodide-sandbox"   # or a path, or an index
```

That is all the install there is, on Linux, macOS or Windows:

- **Deno** comes from the `deno` package on PyPI (pinned to 2.9.8), which ships the
  binary as a platform wheel. `$PYODIDE_SANDBOX_DENO` overrides it.
- **`packages`** are names from Pyodide's package list (`scikit-learn`, `pyyaml`);
  their modules (`sklearn`, `yaml`) are imported at boot, as the lock file names
  them.
- **Pyodide** is downloaded when the first session starts: the runtime (14 MB) and
  the wheels of `packages` with their dependencies (30 MB for the defaults), not the
  337 MB release. They go to `$PYODIDE_DIST_DIR`, else
  `~/.cache/pyodide-sandbox/314.0.7` (`%LOCALAPPDATA%\pyodide-sandbox\314.0.7` on
  Windows; `$XDG_CACHE_HOME` is respected), and later
  sessions use them offline. A package added later is downloaded then. The
  runtime's sha256s are pinned in `distribution.py`, and the pinned
  `pyodide-lock.json` holds every wheel's, so only the release's own files are kept.
  Files are renamed into place when complete, under a lock file, so sessions can
  share a cache. The download is not counted in `startup_timeout_ms`.
- **The download source** is jsDelivr's Pyodide CDN. `$PYODIDE_INDEX_URL` (or
  `index_url=` on `ensure_distribution`) points it at a mirror of the release
  instead, e.g. an internal artifact store; the hashes still apply.
- **To fetch ahead of time**, e.g. in a Docker build step, so that sessions never
  need the network, run `python -m pyodide_sandbox fetch [PACKAGE ...] [--dir DIR]`
  and set `PyodideRepl(download=False)`. Then a missing file fails the start
  instead.
- **Host DataFrames and Series** go to the sandbox as Arrow IPC when the host has
  pyarrow (`pyodide-sandbox[arrow]`), so any host pandas works. They keep their
  dtypes, index, column names and `attrs`. The sandbox reads them with Pyodide's
  pyarrow 22, which is downloaded (10 MB) and loaded only for a store that holds
  one.
- **Other host values** are pickled. numpy, pandas, scipy or statsmodels objects
  (a fitted model, say) load reliably
  only with the sandbox's versions (`SANDBOX_VERSIONS`), so a
  `VersionMismatchWarning` names the value and the version to pin. This also covers
  a frame that Arrow can't convert (mixed-type object columns, duplicate column
  names), a DataFrame subclass, and any frame when the host has no pyarrow.

## Developing

```sh
uv sync
uv run poe fetch     # optional; the first session downloads the same files
uv run poe test      # 126 tests, ~4.5 min; the sandbox ones skip without Deno
uv run poe agent     # live demo, needs ANTHROPIC_API_KEY
```

## How it fits together

| Piece | Role |
|---|---|
| `src/pyodide_sandbox/repl.py` | `PyodideRepl`: `run_code`, `repl_history`, the cell index in the instructions, restart on timeout or crash |
| `src/pyodide_sandbox/session.py` | `PyodideSession`: starts and kills the Deno process, speaks JSON lines, checks every reply |
| `src/pyodide_sandbox/variables.py` | `VariableStore`: host values (DataFrames as Arrow, the rest pickled) in a temporary directory that only the session's Deno can read, and only read |
| `src/pyodide_sandbox/distribution.py` | Downloads and checks the Pyodide files the packages need; `python -m pyodide_sandbox fetch` |
| `src/pyodide_sandbox/_deno/main.ts` | Imports Pyodide from those files (no npm), loads it and the packages, captures each cell's stdout and stderr (including C-level writes), caps them at 1 MB |
| `src/pyodide_sandbox/_deno/runner.py` | Runs a cell in Pyodide: `CodeRunner` as `<cell-N>`, echoes the last expression's `repr`, trims tracebacks to the cell's frames |
| `src/pyodide_sandbox/_deno/host.py` | The `host` object in the sandbox: reads the store's `view.json` and unpickles a value on every read, so each read is a fresh copy |

Deno runs with `--no-prompt --cached-only --no-remote --allow-read=<_deno>,<Pyodide files>,<store>`,
so it has no network, write, subprocess, FFI or env access. It gets none of the host's
environment variables. The time limit is a SIGKILL of the process (TerminateProcess
on Windows), and the next cell
starts a new process. If Pyodide itself fails for good (`os._exit`, or a JavaScript
error inside one of its syscalls), `main.ts` reports the cause and exits, and the next
cell starts a new process too. A JavaScript error that nothing catches but that
leaves Pyodide usable goes to the running cell's stderr, and the session goes on.

Host values other than DataFrames and Series are pickled on the host and unpickled
in the sandbox (see "Using it as a library"). A class defined only on the host fails
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
| numpy errors | Float and array errors are silent: no warnings, `errstate(all="raise")` doesn't raise, `inv`/`solve`/`cholesky` give NaN, integer array `// 0` gives 0. `scipy.linalg`'s `inv`/`solve`/`cholesky` raise `LinAlgError` as usual. Integer scalars and plain Python still warn or raise. Default int is **int32**, and overflow wraps |
| `maxMemoryBytes` | Not in the 314.0.7 release |
| Boot with `tzdata` as well | About +0.2 s. Without it, pandas and `zoneinfo` know no time zones |
| Boot with `scipy` as well | About +1.7 s (8.2 s for all four). Only `import scipy` runs at boot; the first cell to import `scipy.linalg` and `optimize` pays about 1.9 s more, `stats` about 1.7 s more |
| Boot with a DataFrame in the store | About +5 s (13.5 s with the default packages), for loading pyarrow: Pyodide compiles libarrow's WebAssembly, and `import pandas` then imports pyarrow. A read then takes 0.1 s. Arrow IPC from pyarrow 26 on the host arrives intact in pyarrow 22 |
| Boot with `statsmodels` as well | About +1.5 s (11–12 s for all five). Only `import statsmodels` runs at boot; the first cell to import `statsmodels.api` or `statsmodels.formula.api` pays about 5.5 s more |
| statsmodels | OLS and formulas (patsy), Logit, GLM, MixedLM, ARIMA and SARIMAX (Cython state space), Holt-Winters, lowess and `anova_lm` work. A fitted model pickled on the host loads and predicts in the sandbox |
| scipy's C and C++ code | qhull, the k-d tree, HiGHS (`linprog`, `milp`), sparse solvers and signal filters work, and their errors are ordinary exceptions |
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
