"""Plots made in the sandbox, carried through pydantic-ai runs to the app that shows them.

A cell's bokeh `show(p)` becomes a `Plot` with an id the model sees (`[plot 3f9a2c1b]`).
`PyodideRepl.run_code` puts the cell's plots in its tool return's metadata, under
`PLOTS_KEY`. That metadata is kept in message history but never sent to the model,
so the plots live where the conversation does, and survive compaction and a JSON
round trip of the history.

The model picks which plots the user sees:

- A subagent with a REPL reports the ids it made; the tool that delegated to it
  returns `subagent_return(result)`, which lists them for the calling agent and moves
  the plots into its history.
- The agent that answers the user gets `PlotAnswer()`. It places a plot by writing
  `{{plot:ID}}` on a line of its own in its final answer; an id it never saw is sent
  back for a retry.
- The app cuts the final answer into text and plots with `split_answer`, and renders
  each plot itself, e.g. in a sandboxed iframe from `Plot.html()`.

        @main.tool
        async def ask_analyst(ctx: RunContext[Deps], question: str) -> ToolReturn:
            result = await analyst.run(
                question, usage=ctx.usage, capabilities=[ctx.deps.analyst_repl]
            )
            return subagent_return(result)

        result = await main.run(prompt, message_history=history, deps=deps)
        for piece in split_answer(result.output, result.all_messages()):
            ...  # str: markdown; Plot: render plot.json
"""

import html
import json
import re
import secrets
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import ModelRetry, RunContext, ToolReturn
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.output import OutputContext

PLOTS_KEY = "pyodide_sandbox.plots"
"""The key in a tool return's `metadata` that holds its plots, as `Plot.to_dict()`s."""

PLOT_ANSWER = """\
The user sees only your final answer, not tool results. Plots made in a Python REPL, \
yours or a subagent's, have ids such as `3f9a2c1b`, reported by the tool that made \
them. To show one to the user, put `{{plot:3f9a2c1b}}` on a line of its own in your \
final answer, where it belongs; a plot you don't place this way isn't shown. Use only \
ids a tool reported, and say in words what each plot shows."""

# An id the model wrote, however malformed, so that it can be checked
_MARKER = re.compile(r"\{\{\s*plot\s*:\s*([^{}\s]*)\s*\}\}")
_BOKEH_VERSION = re.compile(r"\d+\.\d+\.\d+")
# The BokehJS bundles a json_item may need: widgets and tables for their models,
# gl for `output_backend="webgl"`, mathjax for LaTeX labels
_BOKEH_BUNDLES = ("", "-gl", "-widgets", "-tables", "-mathjax")
_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
{scripts}
</head>
<body>
<div id="plot"></div>
<script type="application/json" id="plot-json">{item}</script>
<script>
Bokeh.embed.embed_item(
  JSON.parse(document.getElementById("plot-json").textContent), "plot");
</script>
</body>
</html>
"""


def _new_id() -> str:
    return secrets.token_hex(4)


@dataclass
class Plot:
    """A bokeh object that a cell `show()`ed, for the host to render with BokehJS.

    `json` is what `bokeh.embed.json_item` made in the sandbox: pass it to
    `Bokeh.embed.embed_item(item, element_id)` in a page that loads BokehJS of the
    same version (`json["version"]`), or use `html()`. It was written by code in
    the sandbox, so treat it as untrusted: a plot can carry JavaScript
    (`CustomJS`) and HTML (`Div`), which run when it is rendered. Render it where
    that can't reach the app or its users' sessions, e.g. a sandboxed iframe on
    another origin.
    """

    number: int
    """Counts its REPL's plots from 1, in the order they were shown."""
    cell: int
    """The cell that showed it."""
    json: dict[str, Any] = field(repr=False)
    id: str = field(default_factory=_new_id)
    """Random, so unique across REPLs and agents; the model names the plot by it."""

    @property
    def title(self) -> str | None:
        """The text of the first `Title` in the plot, if any."""
        return next(
            (
                text
                for model in _models(self.json.get("doc"))
                if model.get("name") == "Title"
                and isinstance(text := model.get("attributes", {}).get("text"), str)
                and text
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON types, as kept in a tool return's metadata."""
        return {
            "id": self.id,
            "number": self.number,
            "cell": self.cell,
            "json": self.json,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Plot | None":
        """The plot `to_dict()` made, or None for anything else."""
        if not isinstance(data, dict):
            return None
        id, number, cell, item = (data.get(k) for k in ("id", "number", "cell", "json"))
        if not (
            isinstance(id, str)
            and isinstance(number, int)
            and isinstance(cell, int)
            and isinstance(item, dict)
        ):
            return None
        return cls(number=number, cell=cell, json=item, id=id)

    def html(self, title: str = "Bokeh plot") -> str:
        """A standalone page showing the plot, loading BokehJS from Bokeh's CDN.

        The page runs whatever the plot carries (see the class docs), so open it
        accordingly. Raises `ValueError` if `json` names no plain bokeh version.
        """
        version = self.json.get("version")
        if not (isinstance(version, str) and _BOKEH_VERSION.fullmatch(version)):
            raise ValueError(f"not a bokeh version: {version!r}")
        scripts = "\n".join(
            '<script src="https://cdn.bokeh.org/bokeh/release/'
            f'bokeh{bundle}-{version}.min.js" crossorigin="anonymous"></script>'
            for bundle in _BOKEH_BUNDLES
        )
        # `<` only occurs inside JSON strings, where < means the same, so no
        # `</script>` or `<!--` in the plot can end the script element early
        item = json.dumps(self.json).replace("<", "\\u003c")
        return _PAGE.format(title=html.escape(title), scripts=scripts, item=item)


def _models(value: Any) -> Iterator[dict[str, Any]]:
    """Every serialized bokeh model in a json_item's `doc`, depth first."""
    if isinstance(value, dict):
        if value.get("type") == "object":
            yield value
        for item in value.values():
            yield from _models(item)
    elif isinstance(value, list):
        for item in value:
            yield from _models(item)


def plots_metadata(plots: Sequence[Plot]) -> dict[str, Any]:
    """A tool return's `metadata` carrying `plots`."""
    return {PLOTS_KEY: [p.to_dict() for p in plots]}


def plots_in(messages: Sequence[ModelMessage]) -> dict[str, Plot]:
    """Every plot the tool returns in `messages` carry, by id, in the order made."""
    found: dict[str, Plot] = {}
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and isinstance(part.metadata, dict):
                items = part.metadata.get(PLOTS_KEY)
                for item in items if isinstance(items, list) else ():
                    if (plot := Plot.from_dict(item)) is not None:
                        found[plot.id] = plot
    return found


def subagent_return(result: AgentRunResult[Any]) -> ToolReturn:
    """What a tool that ran a subagent returns, so the plots it made aren't lost.

    The return value is the subagent's output, followed by the ids (and titles) of
    the plots made during that run, for the calling agent to choose from. The plots
    themselves go in the metadata, so they are in the caller's history too.
    """
    plots = list(plots_in(result.new_messages()).values())
    text = result.output if isinstance(result.output, str) else str(result.output)
    if plots:
        listed = ", ".join(
            p.id + (f" ({json.dumps(p.title)})" if p.title else "") for p in plots
        )
        text += f"\n\nPlots made: {listed}"
    return ToolReturn(return_value=text, metadata=plots_metadata(plots))


@dataclass
class PlotAnswer(AbstractCapability[Any]):
    """For the agent that answers the user: it places plots in its final answer.

    It is told to write `{{plot:ID}}` where a plot belongs, and an id that no tool
    return in the conversation carries is sent back to it for a retry. Only a `str`
    output is checked; the app then cuts it up with `split_answer`.
    """

    def get_instructions(self) -> str:
        return PLOT_ANSWER

    async def after_output_process(
        self,
        ctx: RunContext[Any],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        if not isinstance(output, str):
            return output
        known = plots_in(ctx.messages)
        unknown = [i for i in _MARKER.findall(output) if i not in known]
        if unknown:
            available = ", ".join(known) or "none so far"
            raise ModelRetry(
                f"No plot has the id {', '.join(map(repr, unknown))}. Plots available: "
                f"{available}. Use only those, as {{{{plot:ID}}}}, or leave the plot out."
            )
        return output


def split_answer(text: str, messages: Sequence[ModelMessage]) -> list[str | Plot]:
    """`text` cut at its `{{plot:ID}}` markers: text pieces, and the plots they name.

    Ids are looked up in `messages` (`plots_in`); a marker naming none stays in the
    text. Blank lines around a plot are dropped, and so are empty pieces.
    """
    plots = plots_in(messages)
    pieces: list[str | Plot] = []
    pending = ""
    position = 0
    for match in _MARKER.finditer(text):
        plot = plots.get(match.group(1))
        pending += text[position : match.start()]
        position = match.end()
        if plot is None:
            pending += match.group(0)
            continue
        if pending.strip():
            pieces.append(pending.strip("\n"))
        pieces.append(plot)
        pending = ""
    pending += text[position:]
    if pending.strip():
        pieces.append(pending.strip("\n"))
    return pieces
