"""Let a pydantic-ai agent analyse data through the Pyodide REPL capability.

Ported from eryx-test's script of the same name. Phase 0 has no host variables, so
the agent's first cell builds the weather data itself, from a seed. Two questions
are asked as two `agent.run()` calls sharing one conversation and one Pyodide
session. `ClearToolResults` is set to clear aggressively so that, by the second
question, early cell outputs are gone from the conversation; the model can still
find them through the cell index in the instructions and `repl_history`.
Instruction changes are appended (`on_change="append"`), so the prompt cache
survives each new cell.

Needs ANTHROPIC_API_KEY (or set AGENT_MODEL to another pydantic-ai model).
"""

import os

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai_harness.compaction import ClearToolResults

from pyodide_sandbox import PyodideRepl

MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5")
CLEARED = "[tool result cleared]"  # ClearToolResults' default placeholder

SETUP = """\
Start by building the data in the REPL, exactly as follows:

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
days = pd.date_range("2026-01-01", periods=120, freq="D")
df = pd.concat(
    pd.DataFrame({
        "date": days,
        "city": city,
        "temp": base + 8 * np.sin(np.arange(120) / 19) + rng.normal(0, 2, 120),
        "rain": np.where(rng.random(120) < p, rng.gamma(2, 3, 120), 0.0),
    })
    for city, base, p in [("Oslo", 2, 0.45), ("Lisbon", 15, 0.25), ("Cairo", 22, 0.03)]
)
```
"""

QUESTIONS = [
    "Which city had the most rain in total, and on how many days did it rain there?",
    "For that city, what was the warmest week (7-day rolling mean of temp)?",
]


def main() -> None:
    repl = PyodideRepl(on_change="append")
    agent = Agent(
        MODEL,
        instructions="Answer questions about the data by running code in the REPL.",
        capabilities=[ClearToolResults(max_messages=6, keep_pairs=2), repl],
    )

    history = None
    try:
        for i, question in enumerate(QUESTIONS):
            prompt = SETUP + "\n" + question if i == 0 else question
            print(f"\n### {question}")
            with capture_run_messages() as messages:
                result = agent.run_sync(prompt, message_history=history)
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
