"""The read-only variable store: on the host, and through a PyodideSession.

Ported from eryx-test's tests/test_variables.py.
"""

import datetime
import gc
import hashlib
import importlib.util
import json
import pickle
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from pyodide_sandbox.distribution import SANDBOX_VERSIONS
from pyodide_sandbox.session import DENO_DIR, PyodideSession, SessionCrashed
from pyodide_sandbox.variables import VariableStore, VersionMismatchWarning

# deno/host.py runs in the sandbox, but only needs the standard library
_spec = importlib.util.spec_from_file_location("pyodide_host", DENO_DIR / "host.py")
assert _spec and _spec.loader
host_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(host_module)
Host = host_module.Host


def host_for(store: VariableStore) -> Any:
    return Host(str(store.directory))


def test_layout() -> None:
    with VariableStore({"x": 1, "data": {"a": [1, 2]}}) as store:
        root = store.directory
        view = json.loads((root / "view.json").read_text())
        assert view["format"] == 1
        assert list(view["entries"]) == ["x", "data"]
        x = view["entries"]["x"]
        data = pickle.dumps(1)
        assert x == {
            "name": "x",
            "version": 1,
            "codec": "pickle",
            "blob": hashlib.sha256(data).hexdigest(),
            "type": "int",
            "size": len(data),
        }
        assert (root / "blobs" / x["blob"]).read_bytes() == data
        assert sorted(p.name for p in root.iterdir()) == ["blobs", "view.json"]
        assert not list(root.rglob(".*.tmp"))


def test_passed_dict_is_copied() -> None:
    variables: dict[str, object] = {"x": 1}
    store = VariableStore(variables)
    variables["x"] = [1, 2, 3]
    variables["y"] = "added later"
    assert store.names() == ["x"]
    assert store.info("x").type == "int"
    assert host_for(store).x == 1


@pytest.mark.parametrize("name", ["not-a-name", "class", 1])
def test_bad_names(name: object) -> None:
    with pytest.raises(ValueError, match="Python identifiers"):
        VariableStore({name: 1})  # type: ignore[dict-item]


def test_unpicklable_value() -> None:
    with pytest.raises((pickle.PicklingError, AttributeError)):
        VariableStore({"f": lambda: 0})


def test_close_and_context_manager_remove_the_directory() -> None:
    store = VariableStore({"x": 1})
    root = store.directory
    store.close()
    store.close()
    assert not root.exists()
    with VariableStore({"x": 1}) as store:
        root = store.directory
    assert not root.exists()


def test_garbage_collection_removes_the_directory() -> None:
    store = VariableStore({"x": 1})
    root = store.directory
    del store
    gc.collect()
    assert not root.exists()


def test_identical_values_share_a_blob() -> None:
    with VariableStore({"a": [1, 2], "b": [1, 2]}) as store:
        assert store.info("a").blob == store.info("b").blob
        assert len(list((store.directory / "blobs").iterdir())) == 1


def test_empty_store() -> None:
    with VariableStore() as store:
        assert store.names() == []
        host = host_for(store)
        assert list(host) == []
        assert repr(host) == "host variables: none"


def test_read_is_a_fresh_copy() -> None:
    with VariableStore({"data": [1, 2]}) as store:
        host = host_for(store)
        host.data.append(3)
        host["data"].append(4)
        assert host.data == [1, 2]


def test_host_access() -> None:
    with VariableStore({"x": 1, "data": {"a": 1}}) as store:
        host = host_for(store)
        assert "x" in host
        assert "y" not in host
        assert list(host) == ["x", "data"]
        assert len(host) == 2
        assert {"x", "data", "info"} <= set(dir(host))
        assert host.info("x")["type"] == "int"
        assert repr(host) == "host variables: x (int), data (dict)"


def test_missing_name_lists_available() -> None:
    with VariableStore({"x": 1}) as store:
        host = host_for(store)
        with pytest.raises(AttributeError, match="no host variable 'y'; available: x"):
            _ = host.y
        with pytest.raises(KeyError, match="available: x"):
            _ = host["y"]
        with pytest.raises(AttributeError):
            _ = host._private


class RaisesKeyErrorOnLoad:
    def __reduce__(self) -> tuple[Any, ...]:
        return (dict.__getitem__, ({}, "inner"))


def test_a_key_error_while_loading_is_not_a_missing_variable() -> None:
    # zoneinfo's ZoneInfoNotFoundError is a KeyError; eryx's accessor turned it into
    # "no such variable" for `host.NAME`
    with (
        VariableStore({"x": RaisesKeyErrorOnLoad()}) as store,
        pytest.raises(KeyError, match="inner"),
    ):
        _ = host_for(store).x


def test_warns_when_pickled_by_another_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SANDBOX_VERSIONS, "numpy", "0.0")
    with pytest.warns(
        VersionMismatchWarning,
        match="'arr' was pickled with numpy 2.4.6 but the sandbox has numpy 0.0",
    ) as record:
        VariableStore({"x": [1], "arr": np.arange(3)}).close()
    assert [w.filename for w in record] == [__file__]
    # also inside a container, and through pandas
    with pytest.warns(VersionMismatchWarning, match="'data' was pickled with numpy"):
        VariableStore({"data": {"df": pd.DataFrame({"a": [1]})}}).close()


def test_no_warning_for_the_sandboxs_versions() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        VariableStore(
            {"arr": np.arange(3), "df": weather(), "s": scipy.sparse.eye_array(2)}
        ).close()


def test_host_is_read_only() -> None:
    with VariableStore({"x": 1}) as store:
        host = host_for(store)
        with pytest.raises(AttributeError, match="read-only"):
            host.x = 2
        with pytest.raises(AttributeError, match="read-only"):
            del host.x
        with pytest.raises(TypeError, match="read-only"):
            host["x"] = 2


# In the sandbox


def weather() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "day": pd.date_range("2026-10-01", periods=n, freq="D", tz="Asia/Kolkata"),
            "city": pd.Categorical(rng.choice(["Delhi", "Mumbai", "Pune"], n)),
            "station": [f"s{i}" for i in range(n)],
            "temp": (30 + 5 * np.sin(np.linspace(0, 3 * np.pi, n))).round(2),
            "rain_mm": rng.integers(0, 40, n),
        }
    )


@pytest.fixture(scope="module")
def store() -> Iterator[VariableStore]:
    with VariableStore(
        {
            "x": 1,
            "data": {"a": [1, 2]},
            "df": weather(),
            "arr": np.arange(6, dtype=np.int64).reshape(2, 3),
            "when": datetime.datetime(2026, 11, 6, 12, 30),
            "sparse": scipy.sparse.csr_array([[0, 2.5], [1, 0]]),
        }
    ) as store:
        yield store


@pytest.fixture(scope="module")
def session(store: VariableStore) -> Iterator[PyodideSession]:
    session = PyodideSession(store=store)
    session.start()
    yield session
    session.close()


def test_values_arrive_intact(session: PyodideSession) -> None:
    # The same answers computed on the host
    df = weather()
    expected = (
        df.rain_mm.sum(),
        df.groupby("city", observed=True).temp.mean().round(4).to_dict(),
        str(df.day.iloc[0]),
        dict(df.dtypes.astype(str)),
    )
    result = session.run(
        "df = host.df\n"
        "(df.rain_mm.sum(), "
        "df.groupby('city', observed=True).temp.mean().round(4).to_dict(), "
        "str(df.day.iloc[0]), dict(df.dtypes.astype(str)))"
    )
    assert result.error is None, result.stderr
    assert result.stdout == repr(expected) + "\n"


def test_plain_values(session: PyodideSession) -> None:
    result = session.run(
        "host.x, host['data'], host.arr.tolist(), host.when, "
        "type(host.sparse).__name__, host.sparse.toarray().tolist()"
    )
    assert result.error is None, result.stderr
    assert result.stdout == (
        "(1, {'a': [1, 2]}, [[0, 1, 2], [3, 4, 5]], "
        "datetime.datetime(2026, 11, 6, 12, 30), 'csr_array', [[0.0, 2.5], [1.0, 0.0]])\n"
    )


def test_setup_leaves_only_host(session: PyodideSession) -> None:
    result = session.run(
        "[n for n in ('Host', 'pickle', 'json', 'setup_host', 'js', 'read_file')"
        " if n in globals()], repr(host)"
    )
    assert result.stdout == (
        "([], 'host variables: x (int), data (dict), df (DataFrame), "
        "arr (ndarray), when (datetime), sparse (csr_array)')\n"
    )


def test_traceback_shows_the_accessor(session: PyodideSession) -> None:
    result = session.run("host.missing")
    assert result.error == "AttributeError"
    assert 'File "<host>", line' in result.stderr
    assert "raise AttributeError(e.args[0]) from None" in result.stderr
    assert result.stderr.endswith(
        "AttributeError: no host variable 'missing'; available: "
        "x, data, df, arr, when, sparse\n"
    )


@pytest.mark.parametrize(
    "code",
    [
        # through Deno, which has no write permission
        "js.Deno.writeFileSync(f'{root}/view.json', js.Uint8Array.new(1))",
        "js.Deno.removeSync(f'{root}/view.json')",
        "js.Deno.chmodSync(f'{root}/view.json', 0o777)",
        # Python's own file functions see Pyodide's memory FS, where it isn't
        "open(f'{root}/view.json')",
        "os.chmod(f'{root}/view.json', 0o777)",
        "os.utime(f'{root}/view.json')",
    ],
)
def test_store_cannot_be_changed(
    session: PyodideSession, store: VariableStore, code: str
) -> None:
    before = (store.directory / "view.json").read_bytes()
    result = session.run(f"import js, os\nroot = host._root\n{code}")
    assert result.error is not None, result.stdout
    assert (store.directory / "view.json").read_bytes() == before
    assert session.alive
    assert session.run("host.x").stdout == "1\n"


def test_store_closed_before_the_session_starts() -> None:
    store = VariableStore({"x": 1})
    store.close()
    session = PyodideSession((), store=store)
    with pytest.raises(SessionCrashed, match="during boot:\n"):
        session.start()


def test_store_closed_while_the_session_runs() -> None:
    store = VariableStore({"x": 1})
    session = PyodideSession((), store=store)
    session.start()
    try:
        store.close()
        result = session.run("host.x")
        assert result.error == "OSError"
        assert "can't read the host's variable store" in result.stderr
    finally:
        session.close()


def test_no_store_means_no_host() -> None:
    session = PyodideSession(())
    session.start()
    try:
        assert session.run("host").error == "NameError"
    finally:
        session.close()
