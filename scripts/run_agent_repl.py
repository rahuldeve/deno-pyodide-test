"""Let a pydantic-ai agent analyse a host DataFrame through the Pyodide REPL capability.

Ported from eryx-test's script of the same name. The weather frame is built on the
host and passed in as a host variable, so the agent's cells read it as `host.df`.
Three questions are asked as three `agent.run()` calls sharing one conversation and one
Pyodide session; the last asks for a bokeh plot, which `on_plot` saves as an HTML page
(in the system's temp directory) to open in a browser. `ClearToolResults` is set to clear aggressively so that, by the second
question, early cell outputs are gone from the conversation; the model can still
find them through the cell index in the instructions and `repl_history`.
Instruction changes are appended (`on_change="append"`), so the prompt cache
survives each new cell.

Runs on OpenRouter and needs OPENROUTER_API_KEY, read from the environment or from
`.env.secret` at the repo root (gitignored; the environment wins). Set AGENT_MODEL to
pick another OpenRouter model (`openrouter:<provider>/<model>`) or any other
pydantic-ai model string. OpenRouter has no automatic prompt caching, so the
agent turns it on with the `cache` model setting; routes that can't cache ignore it.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai_harness.compaction import ClearToolResults

from pyodide_sandbox import Plot, PyodideRepl

load_dotenv(Path(__file__).resolve().parent.parent / ".env.secret")

MODEL = os.environ.get("AGENT_MODEL", "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free")
CLEARED = "[tool result cleared]"  # ClearToolResults' default placeholder
PLOTS_DIR = Path(tempfile.gettempdir()) / "pyodide-sandbox-plots"


def make_weather() -> pd.DataFrame:
    """Daily weather for three cities over 120 days."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-01-01", periods=120, freq="D", tz="Europe/Lisbon")
    return pd.concat(
        pd.DataFrame(
            {
                "date": days,
                "city": city,
                "temp": base + 8 * np.sin(np.arange(120) / 19) + rng.normal(0, 2, 120),
                "rain": np.where(rng.random(120) < p, rng.gamma(2, 3, 120), 0.0),
            }
        )
        for city, base, p in [
            ("Oslo", 2, 0.45),
            ("Lisbon", 15, 0.25),
            ("Cairo", 22, 0.03),
        ]
    )


QUESTIONS = [
    (
        "`host.df` holds daily weather for three cities. Which city had the most rain "
        "in total, and on how many days did it rain there?"
    ),
    "For that city, what was the warmest week (7-day rolling mean of temp)?",
    "Plot the 7-day rolling mean temperature of all three cities on one chart.",
]


def save_plot(plot: Plot) -> None:
    """Write the plot as a standalone page; it loads BokehJS from Bokeh's CDN."""
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / f"plot-{plot.number}.html"
    path.write_text(plot.html(title=f"Plot {plot.number}"), encoding="utf-8")
    print(f"[plot {plot.number} from cell {plot.cell} saved to {path}]")


def main() -> None:
    repl = PyodideRepl(
        variables={"df": make_weather()}, on_change="append", on_plot=save_plot
    )
    repl.warm_up()  # boot while the model writes its first cell
    agent = Agent(
        MODEL,
        instructions="Answer questions about the data by running code in the REPL.",
        capabilities=[ClearToolResults(max_messages=6, keep_pairs=2), repl],
        model_settings={"cache": True},
    )

    history = None
    try:
        for question in QUESTIONS:
            print(f"\n### {question}")
            with capture_run_messages() as messages:
                result = agent.run_sync(question, message_history=history)
            history = result.all_messages()
            print(result.output)
            cleared = [
                part.tool_call_id
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.content == CLEARED
            ]
            print(f"[{len(cleared)} REPL results cleared so far]")
            usage = result.usage
            print(
                f"[tokens: {usage.input_tokens} input, "
                f"{usage.cache_read_tokens} read from cache]"
            )

        if repl.session is not None:
            print(f"\n[session boot: {repl.session.boot_ms:.0f} ms]")
        print("\n### Cells run")
        for cell in repl.cells:
            print(f"--- cell {cell.number} ({cell.duration_ms:.0f}ms)")
            print(cell.code.rstrip())
            print(">>>", cell.output().rstrip() or "(no output)")
    finally:
        repl.close()


if __name__ == "__main__":
    main()
