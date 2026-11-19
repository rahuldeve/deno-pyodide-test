"""Sends the bokeh plots a cell shows to the host; loaded by runner.setup_plots.

bokeh's `show()` saves an HTML file and opens a browser, and this sandbox has
neither (Pyodide's `webbrowser` reaches for `js.window`, which Deno lacks). Here
`show(obj)` serializes `obj` with `bokeh.embed.json_item` instead, and main.ts sends
a cell's items to the host with its result, where BokehJS renders them
(`Bokeh.embed.embed_item`).

`show()` calls `bokeh.io.showing._show_with_state` by its module-level name, so
replacing that catches every way of importing `show` and every output mode a cell
picks (`output_file`, `output_notebook`, `reset_output`). `output_notebook()` would
also load Jupyter's display machinery, so the "jupyter" hook's `load` does nothing
here, and its `app` hook refuses a Bokeh server app.
"""

import json
from collections.abc import Sequence
from typing import Any

from bokeh.embed import json_item
from bokeh.io import notebook, showing
from bokeh.io.state import State
from bokeh.layouts import column


class Plots:
    """The plots shown in the running cell, as JSON, up to `max_bytes` in all."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._items: list[str] = []
        self._size = 0

    def install(self) -> None:
        showing._show_with_state = self._show
        notebook.install_notebook_hook(
            "jupyter", _load, self._show_doc, _show_app, overwrite=True
        )
        # Showing an app or a function asserts a notebook type, which only
        # output_notebook() sets and reset_output() clears: "jupyter" is always set,
        # so that it reaches _show_app instead
        notebook_type = State.notebook_type
        State.notebook_type = property(
            lambda state: notebook_type.fget(state) or "jupyter", notebook_type.fset
        )

    def take(self) -> str:
        """The plots shown since the last call, as a JSON array."""
        text = "[" + ",".join(self._items) + "]"
        self._items = []
        self._size = 0
        return text

    def _show(
        self,
        obj: Any,
        state: Any,
        browser: str | None = None,
        new: str = "tab",
        notebook_handle: bool = False,
    ) -> None:
        if isinstance(obj, Sequence):
            # bokeh would save the list as one page, one object under another
            obj = column(*obj)
        document = obj.document
        item = json_item(obj)
        if document is None and obj.document is not None:
            # json_item leaves `obj` in a document of its own, so a later layout
            # holding it would fail ("Models must be owned by only a single
            # document"), as it does with bokeh's own show(); `show(p)` and then
            # `show(column(p, q))` is common enough in a REPL to allow
            obj.document.remove_root(obj)
        # ASCII, so its length is its size in bytes; main.ts re-encodes it no larger
        text = json.dumps(item, allow_nan=False, separators=(",", ":"))
        if self._size + len(text) > self.max_bytes:
            raise ValueError(
                f"this plot is {_megabytes(len(text))} of JSON, and a cell's plots "
                f"can be at most {_megabytes(self.max_bytes)} in all: plot less data "
                "(aggregate, bin or sample it first)"
            )
        self._items.append(text)
        self._size += len(text)

    def _show_doc(self, obj: Any, state: Any, notebook_handle: bool = False) -> None:
        self._show(obj, state)


def _load(*args: Any, **kwargs: Any) -> None:
    pass


def _show_app(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "Bokeh server apps can't run in this sandbox: show a figure, widget or "
        "layout instead"
    )


def _megabytes(n: int) -> str:
    return f"{n / 1e6:.3g} MB"
