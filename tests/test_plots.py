"""Plots through pydantic-ai runs: history metadata, subagents, and the final answer.

The models are scripted (`FunctionModel`), so no API key is needed; the REPLs load
only bokeh, for a quicker boot.
"""

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import SubAgent, SubAgents
from pydantic_ai_harness.compaction import ClearToolResults

from pyodide_sandbox import (
    PLOTS_KEY,
    DelegatedPlots,
    Plot,
    PlotAnswer,
    PyodideRepl,
    plots_in,
    split_answer,
    subagent_return,
)

SHOW = """\
from bokeh.plotting import figure, show
p = figure(title={title!r})
p.line([1, 2, 3], [3, 1, 2])
show(p)
"""

Reply = Callable[[list[ModelMessage]], ModelResponse]


def scripted(*replies: ModelResponse | Reply) -> FunctionModel:
    """A model giving `replies` in turn (a callable sees the messages), then "done"."""
    pending = list(replies)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not pending:
            return ModelResponse(parts=[TextPart("done")])
        reply = pending.pop(0)
        return reply(messages) if callable(reply) else reply

    return FunctionModel(model)


def call(tool: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args)])


def returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, ToolReturnPart)
    ]


@pytest.fixture
def make_repl() -> Iterator[Callable[..., PyodideRepl]]:
    made: list[PyodideRepl] = []

    def make(**kwargs: Any) -> PyodideRepl:
        made.append(PyodideRepl(("bokeh",), **kwargs))
        return made[-1]

    yield make
    for repl in made:
        repl.close()


def test_plot_dict_round_trip() -> None:
    plot = Plot(2, 5, {"version": "3.9.0", "doc": {}})
    assert re.fullmatch("[0-9a-f]{8}", plot.id)
    assert plot.id != Plot(2, 5, {}).id
    assert Plot.from_dict(json.loads(json.dumps(plot.to_dict()))) == plot
    assert plot.title is None
    for junk in [None, [], {"id": 1, "number": 1, "cell": 1, "json": {}}, {"id": "x"}]:
        assert Plot.from_dict(junk) is None


def test_plot_html() -> None:
    plot = Plot(1, 1, {"version": "3.9.0", "doc": {"t": "</script><!-- & é"}})
    page = plot.html(title="<b>&")
    assert "<title>&lt;b&gt;&amp;</title>" in page
    assert page.count("</script>") == 7  # 5 bundles, the item and the embed
    assert "<!--" not in page
    start = page.index('id="plot-json">') + len('id="plot-json">')
    item = page[start : page.index("</script>", start)]
    assert json.loads(item) == plot.json
    assert "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.9.0.min.js" in page
    for version in [None, '3.9.0.dev1"><script>', 3]:
        with pytest.raises(ValueError, match="not a bokeh version"):
            Plot(1, 1, {"version": version}).html()


def test_plots_are_kept_in_history_not_shown_to_the_model(
    make_repl: Callable[..., PyodideRepl],
) -> None:
    repl = make_repl()
    agent = Agent(
        scripted(
            call("run_code", code=SHOW.format(title="a")),
            call("run_code", code="1 + 1"),
            call("run_code", code=SHOW.format(title="b")),
            call("run_code", code="2 + 2"),
        ),
        # clears all but the last result before each request
        capabilities=[ClearToolResults(max_messages=2, keep_pairs=0), repl],
    )
    result = agent.run_sync("go")

    parts = returns(result.all_messages())
    # the model gets the text; the plot JSON only rides along in metadata
    assert all(isinstance(p.content, str) for p in parts)
    assert parts[0].content == "[tool result cleared]"
    assert parts[0].metadata[PLOTS_KEY] == [repl.plots[0].to_dict()]
    assert parts[1].metadata is None or PLOTS_KEY not in parts[1].metadata
    assert plots_in(result.all_messages()) == {p.id: p for p in repl.plots}
    # and a stored conversation still has them
    stored = ModelMessagesTypeAdapter.validate_json(
        ModelMessagesTypeAdapter.dump_json(result.all_messages())
    )
    assert plots_in(stored) == {p.id: p for p in repl.plots}


def test_main_agent_places_a_subagents_plot(
    make_repl: Callable[..., PyodideRepl],
) -> None:
    repl = make_repl()
    analyst = Agent(
        scripted(
            call("run_code", code=SHOW.format(title="Rain by city")),
            ModelResponse(parts=[TextPart("Oslo is wettest.")]),
        )
    )

    def answer(messages: list[ModelMessage]) -> ModelResponse:
        (reported,) = [p for p in returns(messages) if p.tool_name == "ask_analyst"]
        plot_id = re.search(r"Plots made: ([0-9a-f]{8})", str(reported.content))
        assert plot_id is not None
        return ModelResponse(
            parts=[TextPart(f"Oslo.\n\n{{{{plot:{plot_id[1]}}}}}\n\nAs shown.")]
        )

    main = Agent(
        scripted(
            call("ask_analyst", question="rain?"),
            # an id it made up is sent back, with the ones it can use
            ModelResponse(parts=[TextPart("{{plot:deadbeef}}")]),
            answer,
        ),
        capabilities=[PlotAnswer()],
    )

    @main.tool
    async def ask_analyst(ctx: RunContext[None], question: str) -> ToolReturn:
        result = await analyst.run(question, usage=ctx.usage, capabilities=[repl])
        return subagent_return(result)

    result = main.run_sync("Which city is wettest?")

    (plot,) = repl.plots
    (reported,) = [p for p in returns(result.all_messages())]
    assert (
        reported.content
        == f'Oslo is wettest.\n\nPlots made: {plot.id} ("Rain by city")'
    )
    (retry,) = [
        part
        for m in result.all_messages()
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, RetryFeedbackPart)
    ]
    assert f"No plot has the id 'deadbeef'. Plots available: {plot.id}." in str(
        retry.content
    )
    assert split_answer(result.output, result.all_messages()) == [
        "Oslo.",
        plot,
        "As shown.",
    ]
    # the instructions tell the main agent how to place plots
    assert "{{plot:3f9a2c1b}}" in str(result.all_messages()[0])


def test_plots_come_back_through_harness_subagents(
    make_repl: Callable[..., PyodideRepl],
) -> None:
    repl = make_repl()
    analyst = Agent(
        scripted(
            call("run_code", code=SHOW.format(title="Rain by city")),
            ModelResponse(parts=[TextPart("Oslo is wettest.")]),
        ),
        name="analyst",
        deps_type=PyodideRepl,
        # the REPL comes with the deps SubAgents passes on
        capabilities=[lambda ctx: ctx.deps],
    )

    def answer(messages: list[ModelMessage]) -> ModelResponse:
        (reported,) = [p for p in returns(messages) if p.tool_name == "delegate_task"]
        plot_id = re.search(r"Plots made: ([0-9a-f]{8})", str(reported.content))
        assert plot_id is not None
        return ModelResponse(parts=[TextPart(f"Oslo.\n{{{{plot:{plot_id[1]}}}}}")])

    main = Agent(
        scripted(
            call("delegate_task", agent_name="analyst", task="rain?"),
            answer,
        ),
        deps_type=PyodideRepl,
        capabilities=[
            SubAgents(agents=[SubAgent(analyst)]),
            DelegatedPlots(),
            PlotAnswer(),
        ],
    )
    result = main.run_sync("Which city is wettest?", deps=repl)

    (plot,) = repl.plots
    (reported,) = returns(result.all_messages())
    assert (
        reported.content
        == f'Oslo is wettest.\n\nPlots made: {plot.id} ("Rain by city")'
    )
    assert reported.metadata == {PLOTS_KEY: [plot.to_dict()]}
    assert split_answer(result.output, result.all_messages()) == ["Oslo.", plot]


def test_subagent_without_plots() -> None:
    sub = Agent(scripted(ModelResponse(parts=[TextPart("nothing to plot")])))
    returned = subagent_return(sub.run_sync("hi"))
    assert returned.return_value == "nothing to plot"
    assert returned.metadata == {PLOTS_KEY: []}


def test_split_answer() -> None:
    a, b = Plot(1, 1, {}), Plot(2, 1, {})
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "run_code",
                    "",
                    tool_call_id="1",
                    metadata={PLOTS_KEY: [a.to_dict(), b.to_dict(), "junk"]},
                )
            ]
        )
    ]
    text = (
        f"Intro:\n\n{{{{plot:{a.id}}}}}\n{{{{ plot: {b.id} }}}}\n\n"
        "    indented code\n{{plot:nope}} stays\n\n"
        f"{{{{plot:{a.id}}}}}"
    )
    assert split_answer(text, history) == [
        "Intro:",
        a,
        b,
        "    indented code\n{{plot:nope}} stays",
        a,
    ]
    assert split_answer("no plots", history) == ["no plots"]
    assert split_answer("", history) == []
