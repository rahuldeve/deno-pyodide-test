# Run once at boot, after the packages' own modules are imported (main.ts --imports).
#
# `import scipy` and `import statsmodels` load little: the first cell to use
# `scipy.stats` or `statsmodels.api` paid for the rest, about 5 s for a cell using
# both. Importing them here moves that into the boot, which the host can start
# ahead of the first cell (PyodideRepl.warm_up).
#
# Only for packages the session has imported: a module is skipped unless its
# top-level package is in sys.modules, or the package in NEEDED_BY is. Nothing
# reaches __main__, so a cell still has to import what it uses.

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
    "bokeh.plotting",  # plots.py has already imported most of bokeh
    "bokeh.models",
    "bokeh.palettes",
    "bokeh.transform",
    "narwhals.stable.v1",  # 0.5; bokeh imports it for ColumnDataSource(df)
)
# Imported for another package, which imports it only when first needed
NEEDED_BY = {"narwhals.stable.v1": "bokeh"}

for name in SUBMODULES:
    if NEEDED_BY.get(name, name).partition(".")[0] not in sys.modules:
        continue
    try:
        importlib.import_module(name)
    # not fatal to the boot: the cell that imports it gets the error itself
    except Exception as e:  # noqa: BLE001
        print(f"warmup: import {name} failed: {e!r}", file=sys.stderr)
