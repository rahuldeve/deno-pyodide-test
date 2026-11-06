import pytest

from pyodide_sandbox.session import default_deno, default_dist_dir


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    import shutil

    if (
        shutil.which(default_deno())
        and (default_dist_dir() / "pyodide-lock.json").exists()
    ):
        return
    skip = pytest.mark.skip(reason="needs Deno and the Pyodide distribution: poe setup")
    for item in items:
        item.add_marker(skip)
