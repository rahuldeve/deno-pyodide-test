"""Host modules preloaded into the sandbox (`sandbox_modules`)."""

import importlib
import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pyodide_sandbox import modules as modules_module
from pyodide_sandbox.modules import SandboxModules, instructions
from pyodide_sandbox.repl import PyodideRepl
from pyodide_sandbox.session import PyodideSession, SessionError
from pyodide_sandbox.variables import VariableStore

Tree = dict[str, str]


def write(root: Path, files: Tree) -> None:
    for path, source in files.items():
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(textwrap.dedent(source))


class Importer:
    """Imports modules from a directory of their own, and forgets them afterwards."""

    def __init__(self, root: Path) -> None:
        self.root = root
        sys.path.insert(0, str(root))

    def __call__(self, files: Tree, name: str) -> types.ModuleType:
        write(self.root, files)
        importlib.invalidate_caches()
        return importlib.import_module(name)

    def close(self) -> None:
        sys.path.remove(str(self.root))
        for name in list(sys.modules):
            file = getattr(sys.modules[name], "__file__", None) or ""
            if file.startswith(str(self.root)):
                del sys.modules[name]


@pytest.fixture
def load(tmp_path: Path) -> Iterator[Importer]:
    importer = Importer(tmp_path)
    yield importer
    importer.close()


# -- on the host ---------------------------------------------------------------

PLAIN = """\
import json
from math import sqrt

LIMIT = 3


def scale(xs, factor=2.0):
    \"\"\"Multiply each value by factor.

    Second paragraph, left out of the summary.\"\"\"
    return [x * factor for x in xs]


def _private():
    pass


class Box:
    def __init__(self, size: int) -> None:
        self.size = size
"""


def test_plain_module(load: Importer) -> None:
    module = load({"sbx_plain.py": PLAIN}, "sbx_plain")
    collected = SandboxModules.collect([module])

    assert collected.files == {"sbx_plain.py": PLAIN}
    # imported names, constants and private names aren't public
    assert collected.exports == {"sbx_plain": ("scale", "Box")}
    assert collected.descriptions == {
        "sbx_plain": (
            "scale(xs, factor=2.0): Multiply each value by factor.",
            # no `-> None` from `__init__`, and not object's docstring
            "class Box(size: int)",
        )
    }


def test_dunder_all(load: Importer) -> None:
    module = load(
        {
            "sbx_all.py": "__all__ = ['LIMIT', 'f']\nLIMIT = 3\ndef f(): pass\ndef g(): pass"
        },
        "sbx_all",
    )
    collected = SandboxModules.collect([module])
    assert collected.exports == {"sbx_all": ("LIMIT", "f")}
    assert collected.descriptions["sbx_all"] == ("LIMIT: int", "f()")

    missing = load({"sbx_all_bad.py": "__all__ = ['nope']"}, "sbx_all_bad")
    with pytest.raises(ValueError, match="names 'nope', which it doesn't define"):
        SandboxModules.collect([missing])


def test_package(load: Importer) -> None:
    files = {
        "sbx_pkg/__init__.py": "from .core import run\nfrom ._util import helper\nimport os\n",
        "sbx_pkg/core.py": "def run(): pass\n",
        "sbx_pkg/_util.py": "def helper(): pass\n",
        "sbx_pkg/sub/__init__.py": "",
        "sbx_pkg/sub/deep.py": "X = 1\n",
        "sbx_pkg/data.csv": "not python\n",
    }
    module = load(files, "sbx_pkg")
    (Path(module.__file__).parent / "__pycache__" / "junk.py").parent.mkdir(
        exist_ok=True
    )
    (Path(module.__file__).parent / "__pycache__" / "junk.py").write_text("")

    collected = SandboxModules.collect([module])
    assert sorted(collected.files) == [
        "sbx_pkg/__init__.py",
        "sbx_pkg/_util.py",
        "sbx_pkg/core.py",
        "sbx_pkg/sub/__init__.py",
        "sbx_pkg/sub/deep.py",
    ]
    # functions the package's submodules define count as its own
    assert collected.exports == {"sbx_pkg": ("run", "helper")}


def test_submodule_ships_empty_parents(load: Importer) -> None:
    files = {
        "sbx_app/__init__.py": "API_KEY = 'secret'\n",
        "sbx_app/web.py": "import secrets\n",
        "sbx_app/lib/__init__.py": "",
        "sbx_app/lib/tools.py": "def f(): pass\n",
    }
    tools = load(files, "sbx_app.lib.tools")
    collected = SandboxModules.collect([tools])
    assert collected.files == {
        "sbx_app/lib/tools.py": "def f(): pass\n",
        # not the real ones, which the sandbox doesn't need
        "sbx_app/__init__.py": "",
        "sbx_app/lib/__init__.py": "",
    }

    # a parent shipped too is the real one, whatever the order
    lib = importlib.import_module("sbx_app.lib")
    for order in ([tools, lib], [lib, tools]):
        collected = SandboxModules.collect(order)
        assert sorted(collected.files) == [
            "sbx_app/__init__.py",
            "sbx_app/lib/__init__.py",
            "sbx_app/lib/tools.py",
        ]
        assert collected.files["sbx_app/__init__.py"] == ""


def test_long_summary_is_clipped(load: Importer) -> None:
    module = load({"sbx_long.py": f"def f():\n    '''{'word ' * 100}'''\n"}, "sbx_long")
    (line,) = SandboxModules.collect([module]).descriptions["sbx_long"]
    assert len(line) == len("f(): ") + 300 and line.endswith("...")


def test_refused(load: Importer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sys has no source file"):
        SandboxModules.collect([sys])
    with pytest.raises(TypeError, match="takes modules, got function"):
        SandboxModules.collect([test_refused])  # type: ignore[list-item]

    one = load({"sbx_one.py": "def f(): pass"}, "sbx_one")
    two = load({"sbx_two.py": "def f(): pass"}, "sbx_two")
    with pytest.raises(ValueError, match="sbx_one and sbx_two both export 'f'"):
        SandboxModules.collect([one, two])

    host = load({"sbx_host.py": "def host(): pass"}, "sbx_host")
    with pytest.raises(ValueError, match="exports 'host'"):
        SandboxModules.collect([host])

    (tmp_path / "sbx_latin.py").write_bytes(b"# coding: latin-1\n# \xe9\n")
    latin = load({}, "sbx_latin")
    with pytest.raises(ValueError, match="isn't UTF-8"):
        SandboxModules.collect([latin])


def test_size_limit(load: Importer, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load({"sbx_big.py": "X = '" + "x" * 100 + "'"}, "sbx_big")
    monkeypatch.setattr(modules_module, "MAX_SOURCE_BYTES", 50)
    with pytest.raises(ValueError, match="not the whole app"):
        SandboxModules.collect([module])


def test_a_module_passed_twice_counts_once(load: Importer) -> None:
    module = load({"sbx_twice.py": "def f(): pass"}, "sbx_twice")
    assert SandboxModules.collect([module, module]).exports == {"sbx_twice": ("f",)}


def test_repl_instructions(load: Importer) -> None:
    module = load({"sbx_plain2.py": PLAIN}, "sbx_plain2")
    repl = PyodideRepl((), sandbox_modules=[module])
    text = repl._instructions_text()
    assert text.endswith(
        "\n\nsbx_plain2:\n"
        "- scale(xs, factor=2.0): Multiply each value by factor.\n"
        "- class Box(size: int)"
    )
    assert "Host modules: the host preloaded `sbx_plain2`." in text
    assert "Host modules" not in PyodideRepl(())._instructions_text()
    assert instructions(SandboxModules()) == ""

    with pytest.raises(AttributeError, match="can't be changed"):
        repl.sandbox_modules = []


def test_repl_refuses_a_module_it_cant_ship() -> None:
    with pytest.raises(ValueError, match="no source file"):
        PyodideRepl((), sandbox_modules=[sys])


# -- in a session --------------------------------------------------------------

LIB = {
    "sbx_lib/__init__.py": """\
        \"\"\"Helpers.\"\"\"
        from .stats import zscore, boom
        print("imported sbx_lib")


        class Point:
            def __init__(self, x, y):
                self.x, self.y = x, y

            def norm(self):
                return (self.x**2 + self.y**2) ** 0.5
        """,
    "sbx_lib/stats.py": """\
        import statistics


        def zscore(xs):
            m, s = statistics.fmean(xs), statistics.stdev(xs)
            return [round((x - m) / s, 3) for x in xs]


        def boom():
            raise ValueError("from the module")
        """,
    "sbx_outer/__init__.py": "SECRET = 'not for the sandbox'\n",
    "sbx_outer/inner.py": "def twice(x):\n    return 2 * x\n",
}


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PyodideSession]:
    importer = Importer(tmp_path_factory.mktemp("modules"))
    lib = importer(LIB, "sbx_lib")
    inner = importlib.import_module("sbx_outer.inner")
    store = VariableStore({"p": lib.Point(3, 4)})
    session = PyodideSession(
        (), modules=SandboxModules.collect([lib, inner]), store=store
    )
    session.start()
    yield session
    session.close()
    store.close()
    importer.close()


def test_names_are_predefined(session: PyodideSession) -> None:
    result = session.run("zscore([1, 2, 3]), twice(4), Point(6, 8).norm()")
    assert (result.error, result.stdout) == (None, "([-1.0, 0.0, 1.0], 8, 10.0)\n")


def test_importing_them_prints_nothing_into_a_cell(session: PyodideSession) -> None:
    result = session.run("pass")
    assert (result.stdout, result.stderr) == ("", "")


def test_import_brings_back_an_overwritten_name(session: PyodideSession) -> None:
    result = session.run(
        "zscore = None\nfrom sbx_lib import zscore\nimport sbx_outer.inner\n"
        "zscore([0, 2]), sbx_outer.inner.twice(1)"
    )
    assert (result.error, result.stdout) == (None, "([-0.707, 0.707], 2)\n")


def test_parents_are_empty(session: PyodideSession) -> None:
    result = session.run("import sbx_outer\nhasattr(sbx_outer, 'SECRET')")
    assert result.stdout == "False\n"


def test_traceback_shows_the_modules_lines(session: PyodideSession) -> None:
    result = session.run("boom()")
    assert result.error == "ValueError"
    assert '  File "/host_modules/sbx_lib/stats.py", line 10, in boom\n' in (
        result.stderr
    )
    assert '    raise ValueError("from the module")\n' in result.stderr


def test_host_variable_of_a_shipped_class(session: PyodideSession) -> None:
    result = session.run("p = host.p\ntype(p).__module__, p.norm()")
    assert (result.error, result.stdout) == (None, "('sbx_lib', 5.0)\n")


def test_a_module_that_shadows_the_sandboxs_fails_the_start() -> None:
    session = PyodideSession(
        (),
        modules=SandboxModules(
            files={"json.py": "def f(): pass\n"},
            exports={"json": ("f",)},
            descriptions={"json": ("f()",)},
        ),
    )
    with pytest.raises(SessionError, match="'json' has the name of a module"):
        session.start()
    assert not session.alive


def test_a_module_that_fails_to_import_fails_the_start() -> None:
    session = PyodideSession(
        (),
        modules=SandboxModules(
            files={"sbx_needs.py": "import sbx_not_a_package\n"},
            exports={"sbx_needs": ()},
            descriptions={"sbx_needs": ()},
        ),
    )
    with pytest.raises(SessionError) as raised:
        session.start()
    message = str(raised.value)
    assert message.startswith("importing host module sbx_needs failed:\n")
    assert "ModuleNotFoundError: No module named 'sbx_not_a_package'" in message


# -- through the capability ----------------------------------------------------


def scripted(*codes: str) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    calls = [ToolCallPart("run_code", {"code": code}) for code in codes]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if calls:
            return ModelResponse(parts=[calls.pop(0)])
        return ModelResponse(parts=[TextPart("done")])

    return model


def test_modules_survive_a_restart(load: Importer) -> None:
    module = load({"sbx_restart.py": "def f():\n    return 'ok'\n"}, "sbx_restart")
    repl = PyodideRepl((), sandbox_modules=[module], execution_timeout_ms=1000)
    try:
        agent = Agent(FunctionModel(scripted("while True: pass", "f()")))
        agent.run_sync("go", capabilities=[repl])
    finally:
        repl.close()

    timed_out, after = repl.cells
    assert timed_out.error is not None
    assert timed_out.error.endswith(
        "variables from earlier cells are gone; host modules are still there"
    )
    assert after.stdout == "'ok'\n"


def test_a_failed_import_ends_the_run(load: Importer) -> None:
    # importing it on the host would raise too, so the module is made by hand
    write(load.root, {"sbx_broken.py": "raise RuntimeError('nope')\n"})
    module = types.ModuleType("sbx_broken")
    module.__file__ = str(load.root / "sbx_broken.py")
    repl = PyodideRepl((), sandbox_modules=[module])
    agent = Agent(FunctionModel(scripted("1")))
    with pytest.raises(RuntimeError, match="starting the PyodideRepl session failed"):
        agent.run_sync("go", capabilities=[repl])
    repl.close()
