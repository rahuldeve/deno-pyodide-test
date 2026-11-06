"""The Pyodide REPL capability, driven by a scripted model (no API key needed).

Ported from eryx-test's tests/test_pydantic_capability.py. Most tests load no
packages, so each session boots in about 2 s instead of 6.
"""

import asyncio
import gc
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.compaction import ClearToolResults

from pyodide_sandbox.repl import NUMERICS, PyodideRepl

Call = tuple[str, dict[str, object]]


class Script:
    """A model that makes `calls` one per request, then answers, recording what it saw."""

    def __init__(self, calls: list[Call]) -> None:
        self.calls = list(calls)
        self.requests: list[tuple[list[ModelMessage], AgentInfo]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append((messages, info))
        if self.calls:
            name, args = self.calls.pop(0)
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart("done")])

    @property
    def returns(self) -> dict[str, str]:
        """The last request's tool results, by tool call id."""
        messages, _ = self.requests[-1]
        return {
            part.tool_call_id: str(part.content)
            for m in messages
            if isinstance(m, ModelRequest)
            for part in m.parts
            if isinstance(part, ToolReturnPart)
        }


def run_cells(*codes: str) -> list[Call]:
    return [("run_code", {"code": code}) for code in codes]


@pytest.fixture
def repl() -> Iterator[PyodideRepl]:
    repl = PyodideRepl((), execution_timeout_ms=2000)
    yield repl
    repl.close()


def make_agent(
    repl: PyodideRepl, script: Script, *capabilities: AbstractCapability[Any]
) -> Agent:
    return Agent(FunctionModel(script), capabilities=[*capabilities, repl])


def tool_returns(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, ToolReturnPart)
    ]


def test_output_and_errors(repl: PyodideRepl) -> None:
    script = Script(
        run_cells(
            'import sys\nprint("out")\nprint("err", file=sys.stderr)',
            'print("before")\n1/0',
            "pass",
        )
    )
    result = make_agent(repl, script).run_sync("go")

    ok, raised, empty = tool_returns(result.all_messages())
    assert ok == "[cell 1]\nout\nerr\n"
    assert raised.startswith("[cell 2]\nbefore\nTraceback (most recent call last):\n")
    assert '  File "<cell-2>", line 2, in <module>\n' in raised
    assert raised.endswith("ZeroDivisionError: division by zero\n")
    assert empty == "[cell 3]\n(no output)\n"
    assert [c.error for c in repl.cells] == [None, "ZeroDivisionError", None]


def test_state_persists_across_calls_and_runs(repl: PyodideRepl) -> None:
    script = Script(run_cells("x = 40", "x += 1"))
    agent = make_agent(repl, script)
    first = agent.run_sync("go")

    script.calls = run_cells("x + 1")
    agent.run_sync("again", message_history=first.all_messages())
    assert repl.cells[-1].number == 3
    assert repl.cells[-1].stdout == "42\n"


def test_session_survives_separate_event_loops(repl: PyodideRepl) -> None:
    # run_sync and asyncio.run each use a loop of their own
    make_agent(repl, Script(run_cells("x = 1"))).run_sync("go")
    asyncio.run(repl.run_code(None, "x += 1"))  # type: ignore[arg-type]
    make_agent(repl, Script(run_cells("x"))).run_sync("go")
    assert repl.cells[-1].stdout == "2\n"


def test_instructions_index_cells(repl: PyodideRepl) -> None:
    script = Script(run_cells("x = 1\ny = 2", "1/0"))
    make_agent(repl, script).run_sync("go")

    _, info = script.requests[-1]
    assert info.instructions is not None
    assert info.instructions.endswith(
        "Cells run so far:\n1 ok: x = 1\n2 error (ZeroDivisionError): 1/0"
    )


def test_instructions_describe_the_sandbox() -> None:
    bare = PyodideRepl(())._instructions_text()
    assert "Loaded: the standard library only." in bare
    assert "Numerical errors" not in bare
    full = PyodideRepl()._instructions_text()
    assert "Loaded: numpy, pandas." in full
    assert full.endswith(NUMERICS)


def test_instructions_index_no_cells() -> None:
    repl = PyodideRepl((), index_cells=0)
    script = Script(run_cells("x = 1", "x"))
    try:
        make_agent(repl, script).run_sync("go")
    finally:
        repl.close()

    _, info = script.requests[-1]
    assert info.instructions is not None
    assert info.instructions.endswith(
        "Cells run so far:\n(2 earlier cells, see repl_history)"
    )


def test_negative_index_cells() -> None:
    with pytest.raises(ValueError, match="index_cells"):
        PyodideRepl(index_cells=-1)


def test_long_output_is_clipped_but_kept(repl: PyodideRepl) -> None:
    repl.max_output_chars = 100
    script = Script(
        [
            *run_cells("print('x' * 500)"),
            ("repl_history", {"cell": 1, "output": True}),
        ]
    )
    result = make_agent(repl, script).run_sync("go")

    clipped, history = tool_returns(result.all_messages())
    assert clipped.endswith(
        "... 401 more characters, see repl_history(cell=1, output=True)\n"
    )
    assert "x" * 500 + "\n" in history


def test_history(repl: PyodideRepl) -> None:
    script = Script(
        [
            *run_cells("print(1)", "1/0", "print(3)"),
            ("repl_history", {"cell": 2}),
            ("repl_history", {"cell": 2, "output": True}),
            ("repl_history", {"last": 2}),
            ("repl_history", {"last": 2, "output": True}),
            ("repl_history", {"cell": 9}),
        ]
    )
    result = make_agent(repl, script).run_sync("go")

    code, with_output, last_two, last_two_output, missing = tool_returns(
        result.all_messages()
    )[3:]
    assert code == "[cell 2, ZeroDivisionError]\n>>> code\n1/0\n"
    assert with_output.startswith(
        "[cell 2, ZeroDivisionError]\n>>> code\n1/0\n>>> output\n"
    )
    assert "ZeroDivisionError: division by zero" in with_output
    assert last_two == (
        "[cell 2, ZeroDivisionError]\n>>> code\n1/0\n\n\n[cell 3]\n>>> code\nprint(3)\n"
    )
    assert last_two_output.endswith("\n\n[cell 3]\n>>> code\nprint(3)\n>>> output\n3\n")
    assert missing == "No cell 9: cells run so far are 1-3."


@pytest.mark.parametrize("clear_first", [True, False])
def test_clearing_in_any_order(repl: PyodideRepl, clear_first: bool) -> None:
    script = Script(
        [
            *run_cells("print('one')", "print('two')", "print('three')"),
            ("repl_history", {"cell": 1, "output": True}),
        ]
    )
    clear = ClearToolResults(max_messages=2, keep_pairs=1)
    capabilities = [clear, repl] if clear_first else [repl, clear]
    Agent(FunctionModel(script), capabilities=capabilities).run_sync("go")

    seen = list(script.returns.values())
    assert seen[:3] == ["[tool result cleared]"] * 3
    assert seen[-1] == "[cell 1]\n>>> code\nprint('one')\n>>> output\none\n"
    _, info = script.requests[-1]
    assert info.instructions is not None
    assert "Cells run so far:\n1 ok: print('one')" in info.instructions


def test_append_instructions() -> None:
    repl = PyodideRepl((), on_change="append")
    script = Script(run_cells("x = 1", "y = 2"))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Instruction block")
            make_agent(repl, script).run_sync("go")
    finally:
        repl.close()

    messages, info = script.requests[-1]
    # the prompt prefix keeps the index as it was when the run started
    assert info.instructions is not None
    assert "Cells run so far" not in info.instructions
    updates = [
        str(part.content)
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, UserPromptPart) and "pyodide_repl" in str(part.content)
    ]
    assert "1 ok: x = 1" in updates[0]
    assert "<context_diff" in updates[-1] and "+2 ok: y = 2" in updates[-1]


def test_default_id() -> None:
    assert PyodideRepl().id == "pyodide_repl"
    assert PyodideRepl(id="analysis").id == "analysis"


def test_timeout_restarts_session(repl: PyodideRepl) -> None:
    script = Script(run_cells("x = 1", "while True: pass", "x", "y = 2\ny"))
    make_agent(repl, script).run_sync("go")

    _, timed_out, lost, fresh = repl.cells
    assert timed_out.error == (
        "timed out after 2000 ms; the sandbox was restarted "
        "and variables from earlier cells are gone"
    )
    assert lost.error == "NameError"
    # the new process still names cells by their number in the history
    assert '"<cell-3>"' in lost.stderr
    assert fresh.stdout == "2\n"


def test_crash_restarts_session(repl: PyodideRepl) -> None:
    make_agent(
        repl, Script(run_cells("x = 1", "import js\njs.Deno.exit(1)", "x"))
    ).run_sync("go")
    _, crashed, lost = repl.cells
    assert crashed.error is not None
    assert crashed.error.startswith(
        "sandbox crashed: the sandbox process exited with code 1"
    )
    assert crashed.error.endswith("variables from earlier cells are gone")
    assert lost.error == "NameError"


def test_cancelled_cell_finishes_before_the_next(repl: PyodideRepl) -> None:
    # the cancelled cell keeps running in its thread; the next one must wait for it
    slow = (
        "import time\nt = time.time()\nwhile time.time() - t < 1: pass\nprint('slow')"
    )

    async def main() -> str:
        task = asyncio.create_task(repl.run_code(None, slow))  # type: ignore[arg-type]
        await asyncio.sleep(0.2)
        task.cancel()
        return await repl.run_code(None, "print(1)")  # type: ignore[arg-type]

    # the first call also waits for the session to boot
    asyncio.run(repl.run_code(None, "pass"))  # type: ignore[arg-type]
    assert asyncio.run(main()) == "[cell 3]\n1\n"
    _, slow_cell, _ = repl.cells
    assert slow_cell.stdout == "slow\n"


def test_session_that_cannot_start() -> None:
    repl = PyodideRepl(("no_such_package",))
    with pytest.raises(RuntimeError, match="starting the PyodideRepl session failed"):
        make_agent(repl, Script(run_cells("1"))).run_sync("go")
    assert repl.cells == []


def test_reset(repl: PyodideRepl) -> None:
    make_agent(repl, Script(run_cells("x = 1"))).run_sync("go")
    first = repl.session
    repl.reset()
    assert first is not None and not first.alive
    make_agent(repl, Script(run_cells("x"))).run_sync("go")
    assert [c.number for c in repl.cells] == [1]
    assert repl.cells[0].error == "NameError"


def test_dropping_the_repl_stops_its_process() -> None:
    repl = PyodideRepl(())
    make_agent(repl, Script(run_cells("1"))).run_sync("go")
    assert repl.session is not None
    proc = repl.session._proc
    assert proc is not None and proc.poll() is None
    del repl
    gc.collect()
    assert proc.poll() is not None


def test_parallel_calls_run_in_order(repl: PyodideRepl) -> None:
    calls = [
        *run_cells("x = 1", "x += 1", "x"),
        ("repl_history", {"last": 3, "output": True}),
    ]
    responses = [
        ModelResponse(parts=[ToolCallPart(name, args) for name, args in calls]),
        ModelResponse(parts=[TextPart("done")]),
    ]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return responses.pop(0)

    result = Agent(FunctionModel(model), capabilities=[repl]).run_sync("go")

    assert [(c.number, c.code) for c in repl.cells] == [
        (1, "x = 1"),
        (2, "x += 1"),
        (3, "x"),
    ]
    assert repl.cells[-1].stdout == "2\n"
    assert (
        "[cell 3]\n>>> code\nx\n>>> output\n2\n"
        in tool_returns(result.all_messages())[-1]
    )


def test_bound_to_one_agent(repl: PyodideRepl) -> None:
    script = Script([])
    agent = make_agent(repl, script)
    with pytest.raises(ValueError, match="already used by agent"):
        make_agent(repl, script)

    # binding again to the same agent is fine
    agent.run_sync("go")
    with agent.override(model=FunctionModel(script)):
        agent.run_sync("go")
    # a run capability is bound to the run's agent
    other = Agent(FunctionModel(script))
    with pytest.raises(ValueError, match="already used by agent"):
        other.run_sync("go", capabilities=[repl])


def test_rebinds_after_agent_is_gone(repl: PyodideRepl) -> None:
    make_agent(repl, Script([]))
    gc.collect()
    make_agent(repl, Script(run_cells("1 + 1"))).run_sync("go")
    assert repl.cells[0].stdout == "2\n"


def test_numpy_and_pandas() -> None:
    repl = PyodideRepl()
    try:
        script = Script(
            run_cells(
                "import pandas as pd\n"
                "df = pd.DataFrame({'city': ['a', 'b', 'a'], 'rain': [1.0, 2.5, 3.0]})",
                "df.groupby('city').rain.sum().to_dict()",
            )
        )
        make_agent(repl, script).run_sync("go")
        assert repl.cells[-1].stdout == "{'a': 4.0, 'b': 2.5}\n"
    finally:
        repl.close()
