"""Download the full Pyodide distribution once, so the sandbox loads packages offline.

The npm `pyodide` package in `deno/deno.json` holds only the core; numpy, pandas and
the other packages come from this distribution, served from local files. The two
must be the same release, so bump `VERSION`, `SHA256` and `deno/deno.json` together.
"""

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "314.0.7"
SHA256 = "192b5864e6e6d30ab074861af800cb8b4acb0998ef0f9342c3367448aeb86645"
URL = (
    f"https://github.com/pyodide/pyodide/releases/download/{VERSION}/"
    f"pyodide-{VERSION}.tar.bz2"
)
ROOT = Path(__file__).resolve().parent.parent
DEST = Path(os.environ.get("PYODIDE_DIST_DIR", ROOT / "pyodide-dist"))


def main() -> None:
    marker = DEST / ".version"
    if marker.exists() and marker.read_text().strip() == VERSION:
        print(f"Pyodide {VERSION} already in {DEST}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "pyodide.tar.bz2"
        print(f"Downloading {URL}")
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL) as response, archive.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != SHA256:
            sys.exit(f"sha256 mismatch: got {digest.hexdigest()}, expected {SHA256}")
        extract = Path(tmp) / "extract"
        with tarfile.open(archive) as tar:
            tar.extractall(extract, filter="data")
        # the archive holds a single top-level `pyodide/` directory
        (top,) = extract.iterdir()
        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.move(top, DEST)
    marker.write_text(VERSION + "\n")
    print(f"Pyodide {VERSION} extracted to {DEST}")


if __name__ == "__main__":
    main()
