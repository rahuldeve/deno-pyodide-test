"""Downloading only the Pyodide files a session needs.

The mirror here is a `file://` URL to the cache that the sessions use, so only the
`cache` fixture touches the network, and only once per cache.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pyodide_sandbox.distribution import (
    CORE_FILES,
    LOCK_FILE,
    PYODIDE_VERSION,
    SANDBOX_VERSIONS,
    DistributionError,
    _locked,
    default_dist_dir,
    ensure_distribution,
    import_names,
)
from pyodide_sandbox.session import DEFAULT_PACKAGES

NUMPY = "numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl"
PANDAS_AND_DEPENDENCIES = {
    "pandas-3.0.2-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
    "python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
    "six-1.17.0-py2.py3-none-any.whl",
    "pytz-2026.1.post1-py2.py3-none-any.whl",
}


@pytest.fixture(scope="module")
def cache() -> Path:
    return ensure_distribution(DEFAULT_PACKAGES)


@pytest.fixture(scope="module")
def mirror(cache: Path) -> str:
    return cache.as_uri() + "/"


def files(root: Path) -> set[str]:
    return {p.name for p in root.iterdir()} - {".lock"}


def test_only_what_the_packages_need(tmp_path: Path, mirror: str) -> None:
    root = ensure_distribution(["numpy"], tmp_path, index_url=mirror)
    assert root == tmp_path.resolve()
    assert files(tmp_path) == set(CORE_FILES) | {NUMPY}
    # Nothing is fetched again: an unreachable mirror is never asked
    ensure_distribution(["numpy"], tmp_path, index_url="file:///nonexistent/")
    ensure_distribution(["numpy"], tmp_path, download=False)
    # A package added later brings its dependencies, and only those
    ensure_distribution(["numpy", "pandas"], tmp_path, index_url=mirror)
    assert files(tmp_path) == set(CORE_FILES) | {NUMPY} | PANDAS_AND_DEPENDENCIES


def test_a_file_that_doesnt_match_its_hash_is_not_kept(
    tmp_path: Path, cache: Path
) -> None:
    bad = tmp_path / "mirror"
    bad.mkdir()
    shutil.copy(cache / LOCK_FILE, bad)
    (bad / "pyodide.mjs").write_text("export const loadPyodide = 1;\n")
    for name in set(CORE_FILES) - {LOCK_FILE, "pyodide.mjs"}:
        shutil.copy(cache / name, bad)
    shutil.copy(cache / NUMPY, bad)
    dest = tmp_path / "dest"
    with pytest.raises(DistributionError, match="sha256 mismatch for .*pyodide.mjs"):
        ensure_distribution(["numpy"], dest, index_url=bad.as_uri())
    assert "pyodide.mjs" not in files(dest)
    assert not [n for n in files(dest) if n.endswith(".part")]


def test_a_tampered_lock_file_is_not_kept(tmp_path: Path, cache: Path) -> None:
    bad = tmp_path / "mirror"
    bad.mkdir()
    lock = json.loads((cache / LOCK_FILE).read_text())
    lock["packages"]["numpy"]["sha256"] = "0" * 64
    (bad / LOCK_FILE).write_text(json.dumps(lock))
    with pytest.raises(DistributionError, match="sha256 mismatch"):
        ensure_distribution(["numpy"], tmp_path / "dest", index_url=bad.as_uri())
    assert files(tmp_path / "dest") == set()


def test_unreachable_mirror(tmp_path: Path) -> None:
    with pytest.raises(DistributionError, match="downloading Pyodide into"):
        ensure_distribution(["numpy"], tmp_path, index_url="file:///nonexistent/")


def test_unknown_package(tmp_path: Path, mirror: str) -> None:
    with pytest.raises(ValueError, match="not in Pyodide 314.0.7: nope, nor_this"):
        ensure_distribution(["numpy", "nope", "nor_this"], tmp_path, index_url=mirror)


def test_missing_files_without_download(tmp_path: Path, mirror: str) -> None:
    with pytest.raises(DistributionError, match=f"missing from .*: {LOCK_FILE};"):
        ensure_distribution(["numpy"], tmp_path, download=False)
    ensure_distribution([], tmp_path, index_url=mirror)
    with pytest.raises(DistributionError, match=f"missing from .*: {NUMPY};"):
        ensure_distribution(["numpy"], tmp_path, download=False)


def test_import_names(cache: Path) -> None:
    assert import_names(
        ["scikit-learn", "PyYAML", "python_dateutil", "numpy"], cache
    ) == [
        "sklearn",
        "_yaml",
        "yaml",
        "dateutil",
        "numpy",
    ]
    assert import_names(["libopenblas"], cache) == []  # a shared library: no modules
    with pytest.raises(ValueError, match="not in Pyodide 314.0.7: nope"):
        import_names(["nope"], cache)


def test_sandbox_versions_match_the_lock_file(cache: Path) -> None:
    packages = json.loads((cache / LOCK_FILE).read_text())["packages"]
    assert {n: packages[n]["version"] for n in SANDBOX_VERSIONS} == SANDBOX_VERSIONS


# Another process holding the lock for 1.5 s
HOLD_THE_LOCK = """
import sys, time
from pathlib import Path
from pyodide_sandbox.distribution import _locked
with _locked(Path(sys.argv[1])):
    print("locked", flush=True)
    time.sleep(1.5)
"""


def test_the_download_lock_waits_for_another_process(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_THE_LOCK, str(tmp_path / ".lock")],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "locked\n"
        start = time.perf_counter()
        with _locked(tmp_path / ".lock"):
            waited = time.perf_counter() - start
    finally:
        holder.wait(timeout=10)
    assert waited > 0.5
    with _locked(tmp_path / ".lock"):  # released, so taken again at once
        pass


def test_default_dist_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("PYODIDE_DIST_DIR", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
        expected = tmp_path / "local"
    else:
        expected = Path.home() / ".cache"
    assert default_dist_dir() == expected / "pyodide-sandbox" / PYODIDE_VERSION
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_dist_dir() == tmp_path / "xdg" / "pyodide-sandbox" / PYODIDE_VERSION
    monkeypatch.setenv("PYODIDE_DIST_DIR", str(tmp_path / "dist"))
    assert default_dist_dir() == tmp_path / "dist"
