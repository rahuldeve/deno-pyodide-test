"""PyodideSession: one Deno process running Pyodide, driven over JSON lines."""

import os
import signal
import sys
import tempfile
import time
from collections.abc import Iterator

import pytest

from pyodide_sandbox.distribution import DistributionError
from pyodide_sandbox.session import (
    DENO_DIR,
    PyodideSession,
    SessionCrashed,
    SessionTimeout,
)


@pytest.fixture(scope="module")
def numpy_session() -> Iterator[PyodideSession]:
    session = PyodideSession()
    session.start()
    yield session
    session.close()


@pytest.fixture
def bare() -> Iterator[PyodideSession]:
    session = PyodideSession(())
    session.start()
    yield session
    session.close()


def test_packages_load_offline(numpy_session: PyodideSession) -> None:
    result = numpy_session.run(
        "import numpy, pandas\nnumpy.__version__, pandas.__version__"
    )
    assert result.error is None
    assert result.stdout == "('2.4.6', '3.0.2')\n"
    assert numpy_session.boot_ms is not None


SCIPY_CELL = """\
import numpy as np
from scipy import integrate, optimize, signal, sparse, stats
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull, cKDTree

points = np.random.default_rng(0).random((50, 2))
b, a = signal.butter(3, 0.2)
print(scipy.__version__)
print(round(stats.norm.ppf(0.975), 4), round(stats.ttest_ind([1, 2, 3, 4], [2, 3, 4, 9]).pvalue, 4))
print(optimize.minimize(lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2, [0, 0]).x.round(4).tolist())
print(round(integrate.quad(np.sin, 0, np.pi)[0], 6))
print(spsolve(sparse.csr_matrix([[4.0, 1], [1, 3]]), np.array([1.0, 2])).round(4).tolist())
print(signal.filtfilt(b, a, np.arange(20.0))[:2].round(3).tolist())
# C++: qhull, the k-d tree, and HiGHS behind linprog
print(len(ConvexHull(points).vertices), int(cKDTree(points).query([0.5, 0.5])[1]))
print(optimize.linprog([-1, -2], A_ub=[[1, 1]], b_ub=[4], bounds=[(0, 3), (0, 3)]).x.tolist())
"""


def test_scipy(numpy_session: PyodideSession) -> None:
    result = numpy_session.run("import scipy\n" + SCIPY_CELL, timeout=60)
    assert result.error is None, result.stderr
    assert result.stdout.splitlines() == [
        "1.18.0",
        "1.96 0.2797",
        "[3.0, -1.0]",
        "2.0",
        "[0.0909, 0.6364]",
        "[0.001, 0.995]",
        "11 37",
        "[1.0, 3.0]",
    ]


def test_scipy_errors_from_native_code(numpy_session: PyodideSession) -> None:
    # qhull reports its error from C; the session carries on
    result = numpy_session.run(
        "from scipy.spatial import ConvexHull\nConvexHull([[0, 0], [1, 1], [2, 2]])"
    )
    assert result.error == "QhullError"
    assert numpy_session.run("1 + 1").stdout == "2\n"


STATSMODELS_CELL = """\
import numpy as np, pandas as pd
import statsmodels, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=200), "g": rng.choice(list("abc"), 200)})
df["y"] = 2 * df.x + 1 + rng.normal(size=200)
df["hi"] = (df.y > 1).astype(int)
print(statsmodels.__version__)
# formulas go through patsy; ARIMA through the Cython state-space code
print(smf.ols("y ~ x + C(g)", df).fit().params.round(3).tolist())
print(smf.logit("hi ~ x", df).fit(disp=0).params.round(3).tolist())
print(sm.GLM(df.hi, sm.add_constant(df.x), family=sm.families.Binomial()).fit().params.round(3).tolist())
print(ARIMA(np.cumsum(rng.normal(size=100)), order=(1, 1, 0)).fit().params.round(3).tolist())
"""


def test_statsmodels(numpy_session: PyodideSession) -> None:
    result = numpy_session.run(STATSMODELS_CELL, timeout=60)
    assert result.error is None, result.stderr
    assert result.stdout.splitlines() == [
        "0.14.6",
        "[0.975, 0.037, 0.013, 1.877]",
        "[0.033, 4.587]",
        "[0.033, 4.587]",
        "[-0.054, 0.807]",
    ]


def test_state_persists(bare: PyodideSession) -> None:
    bare.run("x = 40")
    bare.run("x += 2")
    assert bare.run("x").stdout == "42\n"


@pytest.mark.parametrize(
    ("code", "printed"),
    [
        ("x = 2\nx * 21", "42\n"),
        ("{\n    'a': 1,\n}", "{'a': 1}\n"),
        ("'text'", "'text'\n"),
        ("None", ""),
        ("x = 1", ""),
        ("print('p')", "p\n"),
        ("x = 1; x", "1\n"),
        ("x = 3\nx;", "3\n"),  # unlike IPython, a trailing semicolon still echoes
        ("s = 'é'; s  # comment", "'é'\n"),
        ("for i in range(2):\n    i", ""),  # only a top-level last expression
    ],
)
def test_last_expression_is_echoed(
    bare: PyodideSession, code: str, printed: str
) -> None:
    result = bare.run(code)
    assert (result.error, result.stdout) == (None, printed)


def test_exceptions_keep_output_and_name_the_cell(bare: PyodideSession) -> None:
    result = bare.run('print("before")\ndef f():\n    return 1/0\n\nf()', cell_id=7)
    assert result.error == "ZeroDivisionError"
    assert result.stdout == "before\n"
    assert result.stderr.startswith("Traceback (most recent call last):\n")
    assert '  File "<cell-7>", line 5, in <module>\n    f()\n' in result.stderr
    assert '  File "<cell-7>", line 3, in f\n    return 1/0\n' in result.stderr
    assert result.stderr.endswith("ZeroDivisionError: division by zero\n")
    # nothing of the runner's or Pyodide's
    assert "runner" not in result.stderr and "_pyodide" not in result.stderr


def test_syntax_error(bare: PyodideSession) -> None:
    result = bare.run("def f(:\n    pass", cell_id=3)
    assert result.error == "SyntaxError"
    assert result.stderr.startswith('  File "<cell-3>", line 1\n')
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("raise SystemExit(1)", "SystemExit"),
        ("import sys; sys.exit()", "SystemExit"),
        ("raise KeyboardInterrupt", "KeyboardInterrupt"),
        ("class E(Exception): pass\nraise E", "E"),
    ],
)
def test_base_exceptions_end_the_cell_not_the_session(
    bare: PyodideSession, code: str, error: str
) -> None:
    assert bare.run(code).error == error
    assert bare.run("1 + 1").stdout == "2\n"


def test_top_level_await(bare: PyodideSession) -> None:
    result = bare.run("import asyncio\nawait asyncio.sleep(0.01)\n'slept'")
    assert (result.error, result.stdout) == (None, "'slept'\n")


def test_stderr_and_c_level_output(numpy_session: PyodideSession) -> None:
    result = numpy_session.run(
        "import sys, os\nprint('err', file=sys.stderr)\nos.write(1, b'raw\\n')\nNone"
    )
    assert result.stderr == "err\n"
    assert result.stdout == "raw\n"


def test_output_is_capped_in_the_sandbox() -> None:
    session = PyodideSession((), max_output_bytes=1000)
    session.start()
    try:
        result = session.run("print('x' * 5000)")
        assert (
            result.stdout == "x" * 1000 + "\n[... 4001 more bytes of output dropped]\n"
        )
        # the cap is per cell
        assert session.run("print('y')").stdout == "y\n"
    finally:
        session.close()


def test_background_task_output_is_not_the_next_cells(bare: PyodideSession) -> None:
    bare.run(
        "import asyncio\n"
        "async def later():\n    await asyncio.sleep(0.05)\n    print('late')\n"
        "task = asyncio.create_task(later())"
    )
    time.sleep(0.3)
    assert bare.run("print('now')").stdout == "now\n"


def test_console_output_stays_off_the_protocol(bare: PyodideSession) -> None:
    # Deno prints these to stdout, which carries only the protocol's messages
    calls = ["log", "info", "debug", "dir", "dirxml", "table", "group", "trace"]
    code = "import js\n" + "\n".join(f"js.console.{c}('x')" for c in calls)
    result = bare.run(code + "\nNone")
    assert (result.error, result.stdout) == (None, ""), result.stderr
    assert bare.run("1 + 1").stdout == "2\n"


def test_packages_are_imported_by_their_modules_names() -> None:
    # python-dateutil's module is `dateutil`; names are matched as pip does
    session = PyodideSession(("Python_DateUtil",))
    session.start()
    try:
        result = session.run("import sys\n'dateutil' in sys.modules")
        assert (result.error, result.stdout) == (None, "True\n")
    finally:
        session.close()


def test_cell_ids_must_increase(bare: PyodideSession) -> None:
    bare.run("1", cell_id=5)
    with pytest.raises(ValueError, match="cell_id"):
        bare.run("1", cell_id=5)


# The time limit is a SIGKILL of the whole process


def test_timeout_kills_a_python_loop(bare: PyodideSession) -> None:
    start = time.perf_counter()
    with pytest.raises(SessionTimeout):
        bare.run("while True: pass", timeout=1)
    assert time.perf_counter() - start < 1.5
    assert not bare.alive
    # SIGKILL; on Windows TerminateProcess, whose exit code is 1
    killed = 1 if sys.platform == "win32" else -signal.SIGKILL
    assert bare._proc is not None and bare._proc.returncode == killed
    with pytest.raises(SessionTimeout):
        bare.run("1")


def test_timeout_kills_a_long_numpy_call() -> None:
    # One C call that a KeyboardInterrupt couldn't stop until it returned
    session = PyodideSession(("numpy",))
    session.start()
    try:
        session.run("import numpy as np\na = np.random.rand(2000, 2000)")
        start = time.perf_counter()
        with pytest.raises(SessionTimeout):
            session.run("np.linalg.svd(a)", timeout=0.5)
        assert time.perf_counter() - start < 1.0
        assert not session.alive
    finally:
        session.close()


def test_timeout_cannot_be_caught(bare: PyodideSession) -> None:
    code = "while True:\n    try:\n        while True: pass\n    except BaseException:\n        pass"
    with pytest.raises(SessionTimeout):
        bare.run(code, timeout=0.5)
    assert not bare.alive


# The process is the boundary: what Deno's flags allow from inside a cell

# Targets that exist on the platform, so that Deno refuses rather than not finding
# them
if sys.platform == "win32":
    _SYSTEM = os.environ.get("SYSTEMROOT", r"C:\Windows")
    _READABLE = rf"{_SYSTEM}\win.ini"
    _WRITABLE = os.path.join(tempfile.gettempdir(), "pyodide-sandbox-test")
    _PROGRAM, _LIBRARY = rf"{_SYSTEM}\System32\whoami.exe", "kernel32.dll"
else:
    _READABLE, _WRITABLE = "/etc/hostname", "/tmp/pyodide-sandbox-test"
    _PROGRAM, _LIBRARY = "true", "libc.so.6"

ESCAPES = {
    "network": "await js.fetch('https://example.com')",
    "read outside": f"js.Deno.readTextFileSync({_READABLE!r})",
    "write": f"js.Deno.writeTextFileSync({_WRITABLE!r}, 'x')",
    "env": "js.Deno.env.get('PATH')",
    "subprocess": f"js.Deno.Command.new({_PROGRAM!r}).outputSync()",
    "ffi": f"js.Deno.dlopen({_LIBRARY!r}, js.Object.new())",
}


@pytest.mark.parametrize("what", ESCAPES)
def test_deno_permissions_refuse(bare: PyodideSession, what: str) -> None:
    code = f"import js\n{ESCAPES[what]}"
    result = bare.run(code)
    assert result.error is not None, result.stdout
    assert "NotCapable" in result.stderr or "PermissionDenied" in result.stderr, (
        result.stderr
    )


def test_python_file_access_stays_in_pyodides_memory_fs(bare: PyodideSession) -> None:
    # Emscripten's in-memory filesystem, not the host's: even files Deno may read
    # aren't mounted
    result = bare.run(
        "import os\nopen('/tmp/x', 'w').write('hi')\n"
        f"os.path.exists('/etc/hostname'), os.path.exists({str(DENO_DIR / 'main.ts')!r})"
    )
    assert result.error is None
    assert result.stdout == "(False, False)\n"


def test_host_environment_is_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyodide_sandbox.session import _deno_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    assert "ANTHROPIC_API_KEY" not in _deno_env()


def test_process_exit_is_a_crash(bare: PyodideSession) -> None:
    with pytest.raises(SessionCrashed, match="exited with code 3"):
        bare.run("import js\njs.Deno.exit(3)")
    assert not bare.alive


@pytest.mark.parametrize(
    ("code", "cause"),
    [
        (
            "import pyodide_js, js\npyodide_js._api.fatal_error(js.Error.new('boom'))",
            "Error: boom",
        ),
        # A JS error inside a syscall: Deno refuses the chmod on a NODEFS mount
        (
            (
                "import pyodide_js, os\n"
                f"pyodide_js.mountNodeFS('/mnt', {str(DENO_DIR)!r})\n"
                "os.chmod('/mnt/main.ts', 0o777)"
            ),
            "NotCapable: Requires write access",
        ),
        ("import os\nos._exit(3)", "Exit: Program terminated with exit(3)"),
    ],
)
def test_pyodide_failing_for_good_is_a_crash_with_its_cause(
    bare: PyodideSession, code: str, cause: str
) -> None:
    # main.ts reports the cause and exits, rather than answer every later cell with
    # an error from a Pyodide that can't run anything
    with pytest.raises(SessionCrashed) as raised:
        bare.run(code, timeout=10)
    assert str(raised.value).startswith(
        f"Pyodide failed during cell and can't be used again: {cause}"
    )
    assert not bare.alive


def test_uncaught_js_errors_go_to_the_cell(bare: PyodideSession) -> None:
    # Deno would exit on them, but Pyodide is fine, so the session goes on
    throw = "js.Function.new('throw new Error(\"thrown\")')"
    reject = "js.Function.new('return Promise.reject(new Error(\"rejected\"))')"
    result = bare.run(
        "import asyncio, js\nx = 1\n"
        f"js.setTimeout({throw}, 0)\njs.setTimeout({reject}, 0)\n"
        "await asyncio.sleep(0.2)"
    )
    assert result.error is None
    assert result.stderr == (
        "Uncaught JavaScript error: Error: thrown\n"
        "Uncaught JavaScript error: Error: rejected\n"
    )
    assert bare.run("x").stdout == "1\n"


def test_garbage_on_the_pipe_is_a_crash(bare: PyodideSession) -> None:
    code = (
        "import js\n"
        "js.Deno.stdout.writeSync(js.TextEncoder.new().encode('not json\\n'))"
    )
    with pytest.raises(SessionCrashed, match="protocol"):
        bare.run(code, timeout=5)
    assert not bare.alive


def test_unknown_package() -> None:
    session = PyodideSession(("no_such_package",))
    with pytest.raises(ValueError, match="not in Pyodide 314.0.7: no_such_package"):
        session.start()
    assert session._proc is None


def test_missing_files_without_download(tmp_path) -> None:
    session = PyodideSession(dist_dir=tmp_path, download=False)
    with pytest.raises(DistributionError, match="python -m pyodide_sandbox fetch"):
        session.start()
    assert list(tmp_path.iterdir()) == []


# Recorded facts about Pyodide 314.0.7's numpy, which the REPL's instructions repeat
# (repl.NUMERICS). Native numpy warns or raises in every case marked "silent" here,
# except integer overflow, which wraps silently there too.

NUMERICS_PROBE = """\
import math, warnings, numpy as np, scipy.linalg

def outcome(f, raise_=False):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            if raise_:
                with np.errstate(all="raise"):
                    r = f()
            else:
                r = f()
        except Exception as e:
            return f"raises {type(e).__name__}"
    if w:
        return "warns"
    if isinstance(r, np.ndarray) and r.dtype.kind == "f" and np.isnan(r).all():
        return "nan"
    return f"silent {r.tolist() if hasattr(r, 'tolist') else r!r}"

f64 = np.float64
cases = {
    # float errors: numpy reads them from the FPU's flags, which wasm doesn't have
    "float div by zero": lambda: f64(1) / f64(0),
    "float overflow": lambda: f64(1e308) * 10,
    "invalid sqrt": lambda: np.sqrt(f64(-1)),
    "array div by zero": lambda: np.array([1.0]) / np.array([0.0]),
    "int array // 0": lambda: np.array([1]) // np.array([0]),
    "inv singular": lambda: np.linalg.inv(np.zeros((2, 2))),
    "solve singular": lambda: np.linalg.solve(np.zeros((2, 2)), np.ones(2)),
    "cholesky not PD": lambda: np.linalg.cholesky(-np.eye(2)),
    # scipy checks LAPACK's info code, so these still raise
    "scipy inv singular": lambda: scipy.linalg.inv(np.zeros((2, 2))),
    "scipy solve singular": lambda: scipy.linalg.solve(np.zeros((2, 2)), np.ones(2)),
    "scipy cholesky not PD": lambda: scipy.linalg.cholesky(-np.eye(2)),
    # checked in software, so unaffected
    "int scalar // 0": lambda: np.int64(1) // np.int64(0),
    "python 1.0 / 0": lambda: 1.0 / 0,
    "math.sqrt(-1)": lambda: math.sqrt(-1),
    # wraps natively too, but the default int is narrower
    "default int overflow": lambda: np.array([2**31 - 1]) + 1,
}
out = {name: outcome(f) for name, f in cases.items()}
for name in ["float div by zero", "float overflow", "invalid sqrt", "int scalar // 0"]:
    out[name + " errstate"] = outcome(cases[name], raise_=True)
out["default int"] = str(np.arange(3).dtype)
for name, value in out.items():
    print(f"{name}: {value}")
"""


def test_numerics(numpy_session: PyodideSession) -> None:
    result = numpy_session.run(NUMERICS_PROBE)
    assert result.error is None, result.stderr
    seen = dict(line.split(": ", 1) for line in result.stdout.splitlines())
    assert seen == {
        "float div by zero": "silent inf",
        "float overflow": "silent inf",
        "invalid sqrt": "silent nan",
        "array div by zero": "silent [inf]",
        "int array // 0": "silent [0]",
        "inv singular": "nan",
        "solve singular": "nan",
        "cholesky not PD": "nan",
        "scipy inv singular": "raises LinAlgError",
        "scipy solve singular": "raises LinAlgError",
        "scipy cholesky not PD": "raises LinAlgError",
        "int scalar // 0": "warns",
        "python 1.0 / 0": "raises ZeroDivisionError",
        "math.sqrt(-1)": "raises ValueError",
        "default int overflow": "silent [-2147483648]",
        "float div by zero errstate": "silent inf",
        "float overflow errstate": "silent inf",
        "invalid sqrt errstate": "silent nan",
        "int scalar // 0 errstate": "raises FloatingPointError",
        "default int": "int32",
    }
