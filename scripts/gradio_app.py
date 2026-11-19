"""A Gradio chat over pydantic-ai agents, with the plots the main agent picks inline.

The main agent answers the user, but can't run code: it asks an analyst subagent,
whose Pyodide REPL reads the weather frame from `run_agent_repl.py` as `host.df`.
The analyst's bokeh plots reach the main agent through `subagent_return`, and the
main agent places the ones worth showing in its answer as `{{plot:ID}}`
(`PlotAnswer`). The app cuts the answer up with `split_answer` and renders each plot
in a sandboxed iframe, since a plot is code written in the sandbox.

Both agents are global; each browser session has its own `Conversation`: the main
agent's message history and the analyst's REPL, started when the page loads and
closed when the session ends.

Runs on OpenRouter like `run_agent_repl.py` (OPENROUTER_API_KEY, AGENT_MODEL), and
serves on http://localhost:7860 (GRADIO_SERVER_PORT to change it).
"""

import html
from dataclasses import dataclass, field

import gradio as gr
from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.messages import ModelMessage
from run_agent_repl import MODEL, make_weather  # also loads .env.secret

from pyodide_sandbox import Plot, PlotAnswer, PyodideRepl, split_answer, subagent_return

# bokeh's default figure height, for a plot that sets none; plus room for margins
DEFAULT_HEIGHT = 600
FRAME_MARGIN = 30


@dataclass
class Conversation:
    repl: PyodideRepl
    history: list[ModelMessage] = field(default_factory=list)
    chat: list[gr.ChatMessage] = field(default_factory=list)

    @classmethod
    def start(cls) -> "Conversation":
        repl = PyodideRepl(variables={"df": make_weather()}, on_change="append")
        repl.warm_up()  # boots while the user types
        return cls(repl)


analyst = Agent(
    MODEL,
    instructions=(
        "You analyse data by running code in the Python REPL; `host.df` holds daily "
        "weather (date, city, temp, rain) for three cities. Make a bokeh plot when it "
        "helps. Report the numbers you found and, for each plot, its id and what it "
        "shows."
    ),
    model_settings={"cache": True},
)

main = Agent(
    MODEL,
    deps_type=Conversation,
    instructions=(
        "You answer questions about daily weather data for Oslo, Lisbon and Cairo. "
        "You can't see the data: ask the analyst, who computes with it and can make "
        "plots. Keep answers short, and show a plot when it helps."
    ),
    capabilities=[PlotAnswer()],
    model_settings={"cache": True},
)


@main.tool
async def ask_analyst(ctx: RunContext[Conversation], question: str) -> ToolReturn:
    """Ask the analyst to compute or plot something from the weather data.

    Args:
        question: What to find out or plot, with any detail the analyst needs.
    """
    # the session's REPL, attached to the global analyst for this run only
    result = await analyst.run(question, usage=ctx.usage, capabilities=[ctx.deps.repl])
    return subagent_return(result)


def frame_height(plot: Plot) -> int:
    """The height the plot's root (a figure, or a layout of them) asks for, if any."""
    roots = plot.json.get("doc", {}).get("roots", [])
    height = roots[0].get("attributes", {}).get("height") if roots else None
    return (height if isinstance(height, int) else DEFAULT_HEIGHT) + FRAME_MARGIN


def plot_frame(plot: Plot) -> gr.HTML:
    """The plot's page in an iframe whose scripts can't reach this app's page."""
    page = html.escape(plot.html(title=plot.title or "Plot"), quote=True)
    label = html.escape(plot.title or "plot", quote=True)
    return gr.HTML(
        f'<iframe sandbox="allow-scripts" title="{label}" srcdoc="{page}" '
        f'style="width:100%;height:{frame_height(plot)}px;border:0"></iframe>'
    )


def answer_message(text: str, history: list[ModelMessage]) -> gr.ChatMessage:
    content = [
        piece if isinstance(piece, str) else plot_frame(piece)
        for piece in split_answer(text, history)
    ]
    return gr.ChatMessage(role="assistant", content=content or "(no answer)")


async def respond(message: str, conv: Conversation | None):
    conv = conv or Conversation.start()
    conv.chat += [
        gr.ChatMessage(role="user", content=message),
        gr.ChatMessage(role="assistant", content="Working…"),
    ]
    yield conv.chat, conv, ""
    try:
        result = await main.run(message, message_history=conv.history, deps=conv)
    except Exception as e:  # noqa: BLE001 - shown to the user, the app goes on
        conv.chat[-1] = gr.ChatMessage(role="assistant", content=f"Error: {e}")
    else:
        conv.history = result.all_messages()
        conv.chat[-1] = answer_message(result.output, conv.history)
    yield conv.chat, conv, ""


def close(conv: Conversation | None) -> None:
    if conv is not None:
        conv.repl.close()


with gr.Blocks(title="Weather analyst") as demo:
    state = gr.State(None, delete_callback=close)
    chatbot = gr.Chatbot(label="Weather analyst", height=720)
    box = gr.Textbox(
        placeholder="e.g. Plot the 7-day rolling mean temperature of each city",
        show_label=False,
        submit_btn=True,
    )
    demo.load(Conversation.start, outputs=state)
    box.submit(respond, inputs=[box, state], outputs=[chatbot, state, box])


if __name__ == "__main__":
    demo.launch()
