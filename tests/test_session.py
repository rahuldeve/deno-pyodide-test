"""PyodideSession: one Deno process running Pyodide, driven over JSON lines."""

import time
from collections.abc import Iterator

import pytest

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
    assert bare._proc is not None and bare._proc.returncode == -9
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

ESCAPES = {
    "network": "await js.fetch('https://example.com')",
    "read outside": "js.Deno.readTextFileSync('/etc/hostname')",
    "write": "js.Deno.writeTextFileSync('/tmp/pyodide-sandbox-test', 'x')",
    "env": "js.Deno.env.get('HOME')",
    "subprocess": "js.Deno.Command.new('true').outputSync()",
    "ffi": "js.Deno.dlopen('libc.so.6', js.Object.new())",
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
    from pyodide_sandbox.session import DENO_DIR

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


def test_boot_failure_reports_stderr() -> None:
    session = PyodideSession(("no_such_package",))
    with pytest.raises(SessionCrashed, match="exited with code 1 during boot:\n"):
        session.start()


def test_missing_distribution(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="poe setup"):
        PyodideSession(dist_dir=tmp_path).start()


# Recorded facts about Pyodide 314.0.7's numpy, which the REPL's instructions repeat
# (repl.NUMERICS). Native numpy warns or raises in every case marked "silent" here,
# except integer overflow, which wraps silently there too.

NUMERICS_PROBE = """\
import math, warnings, numpy as np

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
