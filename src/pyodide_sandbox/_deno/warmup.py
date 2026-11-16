# Run once at boot, after the packages' own modules are imported (main.ts --imports).
#
# `import scipy` and `import statsmodels` load little: the first cell to use
# `scipy.stats` or `statsmodels.api` paid for the rest, about 5 s for a cell using
# both. Importing them here moves that into the boot, which the host can start
# ahead of the first cell (PyodideRepl.warm_up).
#
# Only for packages the session has imported: a module is skipped unless its
# top-level package is in sys.modules. Nothing reaches __main__, so a cell still has
# to import what it uses.

import importlib
import sys

# Seconds are what each added to the boot here (WSL2), in this order; the modules
# without one were already imported by the others, and are listed so that they stay
# imported if that changes.
SUBMODULES = (
    "scipy.linalg",
    "scipy.special",
    "scipy.sparse",
    "scipy.optimize",  # 1.0
    "scipy.stats",  # 1.6
    "scipy.signal",  # 0.3
    "scipy.interpolate",
    "scipy.integrate",
    "scipy.spatial",
    "scipy.cluster",
    "scipy.fft",
    "scipy.ndimage",
    "statsmodels.api",  # 1.8
    "statsmodels.formula.api",
    "statsmodels.tsa.api",
    "statsmodels.stats.api",
)

for name in SUBMODULES:
    if name.partition(".")[0] not in sys.modules:
        continue
    try:
        importlib.import_module(name)
    # not fatal to the boot: the cell that imports it gets the error itself
    except Exception as e:  # noqa: BLE001
        print(f"warmup: import {name} failed: {e!r}", file=sys.stderr)
