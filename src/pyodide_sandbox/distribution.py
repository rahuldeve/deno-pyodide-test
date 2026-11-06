"""The parts of the Pyodide distribution a session needs, downloaded on first use.

The full Pyodide release is 337 MB. A session needs only the runtime (about 14 MB)
and the wheels of its packages and their dependencies (about 22 MB for numpy,
pandas, scipy and tzdata), so `ensure_distribution` downloads just those into a
cache directory, from Pyodide's CDN by default:

    $PYODIDE_DIST_DIR, else $XDG_CACHE_HOME/pyodide-sandbox/<version>, else
    ~/.cache/pyodide-sandbox/<version>

Every file is checked before it is kept. The runtime's hashes are pinned here, and
`pyodide-lock.json` (pinned too) holds the hash of every package's wheel, so a
mirror given as `index_url` must serve the release's exact files. Files are
renamed into place only when complete, and a lock file keeps two processes from
downloading the same files, so a cache can be shared by concurrent sessions. A
package added later is downloaded then; nothing already there is fetched again.

`python -m pyodide_sandbox fetch` does the same ahead of time, e.g. while building
an image, so that sessions start without network access.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PYODIDE_VERSION = "314.0.7"
DEFAULT_INDEX_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
LOCK_FILE = "pyodide-lock.json"
# From the release's sha256-checked full archive (pyodide-314.0.7.tar.bz2)
CORE_FILES = {
    LOCK_FILE: "5dc2fc119108bc148c7457dc86e7675b5c87e1cafd420b9c34c1eaef7b36c010",
    "pyodide.mjs": "6f1d60f7bf529beb300f0f47983c921d3982363640ba20af0e38efdddbc66109",
    "pyodide.asm.mjs": "f7cdc8ece80678ceb712f8e65ebe6d3a83203a180c399865f49612a051693635",
    "pyodide.asm.wasm": "cc36e3cab04fdfc9a63ff13eb52eae2b911bf46c025cc7b281f394bd3de1d5e6",
    "python_stdlib.zip": "fa1957e5777068fc4f7437f96d860ae2fbe9c19732ba06c84e004ec16dd7dd7a",
}
# The versions of the default packages in the sandbox; values pickled on the host
# load reliably only from the same versions (see `pyodide_sandbox.variables`)
SANDBOX_VERSIONS = {"numpy": "2.4.6", "pandas": "3.0.2", "scipy": "1.18.0"}

_TIMEOUT = 60.0
_WORKERS = 4
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")

log = logging.getLogger(__name__)


class DistributionError(RuntimeError):
    """The Pyodide files a session needs aren't there and couldn't be downloaded."""


def default_dist_dir() -> Path:
    """`$PYODIDE_DIST_DIR`, else a per-version directory in the user's cache."""
    if configured := os.environ.get("PYODIDE_DIST_DIR"):
        return Path(configured)
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "pyodide-sandbox" / PYODIDE_VERSION


def default_index_url() -> str:
    """`$PYODIDE_INDEX_URL` (a mirror of the release), else Pyodide's CDN."""
    return os.environ.get("PYODIDE_INDEX_URL") or DEFAULT_INDEX_URL


def ensure_distribution(
    packages: Iterable[str],
    dist_dir: str | Path | None = None,
    *,
    index_url: str | None = None,
    download: bool = True,
) -> Path:
    """Make sure `dist_dir` holds the runtime and `packages` (with their dependencies).

    Returns the directory, resolved. Downloads whatever is missing unless `download`
    is false, when a missing file raises `DistributionError` instead. A package not
    in this Pyodide release raises `ValueError`.
    """
    root = Path(dist_dir) if dist_dir is not None else default_dist_dir()
    packages = list(packages)
    if not download:
        missing = _missing(root, packages)
        if missing:
            names = sorted(missing)
            listed = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
            raise DistributionError(
                f"Pyodide files missing from {root}: {listed}; "
                "run `python -m pyodide_sandbox fetch`"
            )
        return root.resolve()
    if not _missing(root, packages):
        return root.resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # another process may be downloading
            base = index_url or default_index_url()
            _download_missing(root, base, {LOCK_FILE: CORE_FILES[LOCK_FILE]})
            _download_missing(root, base, _required(root, packages))
    except OSError as e:
        raise DistributionError(f"downloading Pyodide into {root} failed: {e}") from e
    return root.resolve()


def _missing(root: Path, packages: list[str]) -> set[str]:
    if not (root / LOCK_FILE).is_file():
        return {LOCK_FILE}
    return {name for name in _required(root, packages) if not (root / name).is_file()}


def _required(root: Path, packages: list[str]) -> dict[str, str]:
    lock = _read_lock(root)
    files = dict(CORE_FILES)
    for name in _with_dependencies(lock, packages):
        entry = lock[name]
        file_name, sha256 = entry.get("file_name"), entry.get("sha256")
        if not (isinstance(file_name, str) and _FILE_NAME.fullmatch(file_name)):
            raise DistributionError(f"bad file name for {name} in {LOCK_FILE}")
        files[file_name] = str(sha256)
    return files


def _read_lock(root: Path) -> dict[str, Any]:
    try:
        packages = json.loads((root / LOCK_FILE).read_text())["packages"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise DistributionError(f"can't read {root / LOCK_FILE}: {e}") from e
    return packages


def _with_dependencies(lock: dict[str, Any], packages: list[str]) -> list[str]:
    """`packages` and everything they depend on, as the lock file names them."""
    found: list[str] = []
    unknown = []

    def visit(name: str) -> None:
        key = name.lower()
        if key in found:
            return
        if key not in lock:
            unknown.append(name)
            return
        found.append(key)
        for dependency in lock[key].get("depends", []):
            visit(dependency)

    for name in packages:
        visit(name)
    if unknown:
        raise ValueError(f"not in Pyodide {PYODIDE_VERSION}: {', '.join(unknown)}")
    return found


def _download_missing(root: Path, base: str, files: dict[str, str]) -> None:
    missing = {n: h for n, h in files.items() if not (root / n).is_file()}
    if not missing:
        return
    log.info("downloading %d Pyodide files into %s", len(missing), root)
    with ThreadPoolExecutor(_WORKERS) as pool:
        # list() so the first failure raises here
        list(
            pool.map(
                lambda item: _download(base, root, *item),
                sorted(missing.items()),
            )
        )


def _download(base: str, root: Path, name: str, sha256: str) -> None:
    url = base.rstrip("/") + "/" + name
    part = root / f".{name}.{os.getpid()}.part"
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(url, timeout=_TIMEOUT) as response,
            part.open("wb") as out,
        ):
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise DistributionError(
                f"sha256 mismatch for {url}: got {digest.hexdigest()}, "
                f"expected {sha256}"
            )
        os.replace(part, root / name)
    finally:
        part.unlink(missing_ok=True)
    log.info("downloaded %s", name)
