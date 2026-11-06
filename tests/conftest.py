import shutil

import pytest

from pyodide_sandbox.session import default_deno


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Pyodide's files are downloaded by the first session (or test_distribution)
    if shutil.which(default_deno()):
        return
    skip = pytest.mark.skip(
        reason="needs Deno: the `deno` package, or $PYODIDE_SANDBOX_DENO"
    )
    for item in items:
        item.add_marker(skip)
