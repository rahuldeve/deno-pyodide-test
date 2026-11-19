"""Host modules preloaded into the sandbox (`sandbox_modules`)."""

import importlib
import sys
import textwrap
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyodide_sandbox import modules as modules_module
from pyodide_sandbox.modules import SandboxModules

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
