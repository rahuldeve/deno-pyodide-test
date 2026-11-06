# Pyodide on Deno can match EryxRepl's features, but not its boundary

Every feature `EryxRepl` has today has a design on Pyodide under Deno, and most of the capability carries over unchanged, because the REPL, cell index, history, compaction handling and tool selection are host code. What doesn't carry over is the security boundary. Python in Pyodide can run arbitrary JavaScript, and a 2026 study found a route through `ctypes` to Emscripten's own symbols that also got past Deno's subprocess check. Projects that relied on Pyodide plus Deno permissions as the boundary were archived, removed or moved to a process boundary. So Pyodide is only usable here inside a disposable per-session process with OS-level limits. That is also what [Safe sandbox host data transfer](</home/rahul/eryx-test/reports/Safe sandbox host data transfer.md>) recommends for eryx, after guest code aborted the host process once. The process wrapper is therefore shared work, and the choice between backends comes down to package breadth against startup time and memory enforcement. Three things decide whether Pyodide is viable at all, and none is proven: a fast start with numpy and pandas loaded, a memory cap that holds, and a reliable kill. The recommendation is to build the backend interface and the process wrapper first, which eryx needs anyway, and to run the Pyodide spike only if the code is trusted or package breadth is worth the weaker boundary.

Nothing here was run. Deno and Pyodide aren't installed on this machine. Everything about them comes from documentation, READMEs and vendor posts read on 2026-11-04 through a summarizing fetch tool, so check quotes against the source before repeating them. Evidence for each claim is in [research notes](</home/rahul/eryx-test/research_notes/Pyodide on Deno as an eryx backend/sources.md>). Claims about this repo come from reading `src/eryx_sandbox/` at `39af867`.

| Question | Answer | Confidence |
|---|---|---|
| Can it match the feature set? | Yes for the REPL and tools, with host code mostly unchanged; startup, memory limits and fuel are at risk | Design only, untested |
| Is Pyodide plus Deno a sandbox by itself? | No: Python reaches JS, and the `ctypes` route got past Deno's `--allow-run` check | Documented, several independent sources |
| What has to surround it? | One process per session, run as non-root, killed on a wall clock, with a cgroup memory limit and Deno's permission flags at the minimum | Recommendation; eryx needs the same |
| Is there prior art to adopt? | Nothing combines a persistent REPL, host tools and numpy. LocalSandbox is closest and is beta and unaudited | READMEs read; source not read |
| What do we gain? | scipy, scikit-learn, matplotlib and pyarrow prebuilt; no cross-compiling; probably C++ exceptions | Package list read; exceptions not tested |

## The boundary moves from Wasm to the process

In eryx, guest code reaches the host only through mounts and callbacks. In Pyodide, the `js` module gives Python the JavaScript runtime it runs in. Pydantic archived `mcp-run-python` on 2026-01-30 because "there's just no safe way to run Python within pyodide safely with reasonable latency", and its notice says Deno "has no good way" to limit memory use. CVE-2026-25905 is that bridge used to take over its MCP server, with no fix coming.

Hiding `js` doesn't close the gap. The Cyera study presented at DEF CON 34 traced seven products to one flaw: Python-level restrictions "did not account for ctypes and Emscripten's exported symbols", so `ctypes.CDLL(None)` reaches `emscripten_run_script_string`, `system` and `popen`. In langchain-sandbox that path reached `system()` through Emscripten's syscall emulation, bypassing Deno's `allow_run`. Grist restricted `jsglobals` and was still escaped through class-hierarchy traversal and `ctypes`. These findings come from Cyera's article and weren't cross-checked against the CVE records (see [comparable systems](</home/rahul/eryx-test/research_notes/Safe sandbox host data transfer/comparable_systems.md>)).

Deno's documentation agrees. Code on one thread shares one privilege level, and for untrusted code it recommends combining all of: limited permissions with `--frozen` and `--cached-only`, a Web Worker with reduced permissions, OS sandboxing such as cgroups and seccomp, and a VM or microVM. Per-worker permissions are marked unstable. The documentation sets no memory, CPU or time limits.

The design that follows:
- **One Deno process per session.** A compromised session can only damage its own state, which already belongs to the model.
- **Nothing worth stealing in that process.** Tools, `deps`, secrets and the cell record stay in the Python host, which treats every message from Deno as hostile, as `CellTools` already treats `invoke`.
- **Minimal Deno permissions.** `--allow-read` limited to the Pyodide files and the store, and no `--allow-net`, `--allow-write`, `--allow-run`, `--allow-ffi` or `--allow-env`. Use `--no-prompt`, `--cached-only` and `--no-remote`. Deno treats `--allow-run` and `--allow-ffi` as equivalent to `--allow-all`.
- **An OS layer around the process, run as non-root.** Deno's permissions limit the APIs, not what the JS engine itself can be attacked through. Terrarium's CVE-2026-5752 reports root-level code execution on a host process.
- **Optionally, strip dangerous symbols from the Wasm binary,** as Cyera recommends. That means a custom Pyodide build, which gives back part of the "no cross-compiling" gain.

## Most of EryxRepl ports as host code

| Feature today | On Pyodide | Risk |
|---|---|---|
| Cell index, `repl_history`, `on_change`, compaction in any order, agent binding, the `store`/`tools` setters | Unchanged host code | None |
| Persistent session and the echoed last expression | One Pyodide instance per session; `prepare_cell` unchanged; top-level `await` through `runPythonAsync` | Low |
| stdout, stderr and the error type | `setStdout`/`setStderr`. A small runner reports the exception type, which replaces `_exception_name` and the `_WRAPPER_FRAME` regex | Low |
| `_RESULT_VAR` workaround | Not needed; eryx-specific | None |
| `snapshot_state` and `restore_state` | pyeryx documents it as a pickle of user variables, so a runner function can do the same | Low |
| `host.tools` | `host_accessor.Tools` takes an `invoke` and a callback name, so the runner supplies an `invoke` that sends a message. A reader thread calls `CellTools.dispatch`, which already runs off the agent's loop | Low |
| `max_tool_result_bytes`, `max_callback_invocations` | Counted on the host | Low |
| `VariableStore` read-only mount | `mountNodeFS` at `/eryx-store`, with writes refused because Deno has no `--allow-write`; otherwise blobs fetched over messages | Medium: `mountNodeFS` under Deno undocumented |
| Timeout, then restart with "variables are gone" | SIGKILL the process at the deadline and start a new one; same note to the model | Medium: see the kill gate |
| `max_memory_bytes` | `maxMemoryBytes` for the Wasm heap plus a cgroup for the process | **High** |
| `load_factory` memory snapshot | Pyodide's private `_makeSnapshot`/`_loadSnapshot`, tied to the exact build | **High** |
| `max_fuel` | Nothing equivalent; wall clock plus cgroup `cpu.max` at best | Accepted loss |
| No network, no package installs | No `--allow-net`; packages served from local files | Low |

`host_tools.py`'s module docstring explains eryx-specific choices: one dispatcher, turning off `callback_timeout_ms`, and the deadline margin. Most of those reasons go away, but the dispatcher design still fits, so the code can stay.

Pyodide also adds problems eryx doesn't have:
- **Background tasks outlive the cell.** Pyodide's event loop keeps running, so a task the model starts can print into the next cell or call a tool after its cell ends. Tag every message with its cell number and drop stale ones. `_cell_tools` already refuses calls outside a cell.
- **Output can flood the pipe.** Limit per-cell output and message size on the host before anything is buffered.
- **The worker can probably write to the pipe directly,** so `main.ts` isn't a boundary. Only the Python host is.
- **A shared startup snapshot shares its entropy.** Cloudflare reseeds the random number generator after each restore and accepts a fixed hash seed for each Worker. Sessions restored from one snapshot would share both unless this is handled.

## Three gates decide whether it is viable

**Startup.** eryx loads the pandas factory in 2.12 s ([q1d.log](/home/rahul/eryx-test/wasi-build/work/nanoarrow-eval/transfer-probes/logs/q1d.log)). Pydantic's own benchmark puts Pyodide in Deno at 2,700 ms to run `1 + 1` through `mcp-run-python` (Pyodide 0.28.2), plus about 200 ms to install numpy. That is a vendor benchmark with an older Pyodide and without pandas. Pyodide's snapshot options are private, marked `@ignore` in `pyodide.ts`, and a snapshot from another build "will fail badly". An older snapshot PR crashed with binary extensions loaded. Cloudflare shows it can work with dynamic libraries, cutting one Worker's start from about 10 s to 1 s, but it needed a patched loader and allocator, a check of the function table, and JS references replayed from `globalThis`. The pages read don't say whether that code is open source. The fallback is a warm process pool. Deno fixes permissions at process start, though, so a warm process couldn't mount a store created later and would have to fetch values over messages.

**Memory.** `pyodide.ts` on the main branch has `maxMemoryBytes`, which makes Wasm heap growth beyond it fail with `MemoryError`, but it may not be in the 314.0.7 release. That covers only the Wasm heap, and Python can allocate JavaScript objects through the bridge outside it. The process needs a cgroup. The eryx report found that a 4 GiB `RLIMIT_AS` broke eryx's address reservations, and V8 reserves address space for Wasm too, so a cgroup is the likelier bound for both backends. It is untested for either, and cgroup support varies on WSL2.

**Kill.** Pyodide's interrupt buffer raises `KeyboardInterrupt` and needs a `SharedArrayBuffer` shared with a worker. Model code can catch `KeyboardInterrupt`. It probably can't stop a long numpy call either, because Python checks for signals between bytecodes; that is an inference, not tested. Deno's docs say `Worker.terminate()` stops a worker "at once" but don't cover a tight loop. A host-side SIGKILL is the bound that should hold, and the eryx report found a wall-clock kill was its only consistent limit as well.

## What Pyodide gains

- **Packages without a build.** The Pyodide 314.0.7 list (Python 3.14.2 since 314.0.0) has numpy 2.4.6, pandas 3.0.2, scipy 1.18.0, bokeh 3.9.0, scikit-learn 1.8.0, matplotlib 3.10.8, pyarrow 22.0.0 and pydantic 2.12.5. `wasi-build/` takes 4.1 GB on disk, including build output.
- **Version gaps to resolve.** The host pins pandas 2.3.3, and pickles across pandas major versions aren't reliable, so the host would move to pandas 3.0.2 or values would need another codec. bokeh is one minor version behind, and it is pure Python, so 3.10.1 can be installed from its wheel.
- **More of the standard library.** Since 314.0.0, `sqlite3` and `lzma` are bundled. `ssl` is a stub.
- **Probably C++ exceptions,** which would make [the tabled exceptions work](/home/rahul/.claude/projects/-home-rahul-eryx-test/memory/wasi-cpp-exceptions-tabled.md) unnecessary. Not tested.
- **Probably not numerical errors.** Wasm has no floating-point flags whatever the toolchain, so numpy's silent NaN is likely to remain, and the warning in `INSTRUCTIONS` should only change after testing.
- **Later, sync tool calls.** Stack switching might let plain functions and `df.apply` call tools. Pyodide's changelog covers it, but nothing read says whether it works under Deno.

## Nothing combines a REPL, host tools and numpy

| Project | Variables kept | Code calls host tools | numpy and pandas | Status |
|---|---|---|---|---|
| LocalSandbox (Deno, Pyodide) | Not guaranteed: "shouldn't be depended on"; files persist | Yes, a synchronous `call()` relayed over stdio JSON | Pyodide packages preloaded; neither named | Beta, unaudited, one maintainer; 0.3.2, April 2026 |
| langchain-sandbox (Deno, Pyodide) | Pickled `session_bytes` | Not documented | numpy in an example | Archived January 2026; `ctypes` escape |
| mcp-run-python (Deno, Pyodide) | Not checked | Not documented | Installed in a separate write-enabled step | Archived January 2026; CVE-2026-25905 |
| Cohere Terrarium (Node, Pyodide) | None; recycled per call | Not documented | Yes, plus scipy and scikit-learn | Archived April 2026; CVE-2026-5752, then CVE-2026-61522 |
| smolagents `WasmExecutor` (Deno, Pyodide) | Not checked | Not checked | Not checked | Removed after CVE-2026-10613, a default `--allow-run`, per Cyera |
| Grist formulas (Pyodide) | n/a | n/a | n/a | Moved under Deno with restricted permissions in 1.7.9 after CVE-2026-24002 |
| n8n Code node (Pyodide) | n/a | n/a | n/a | Moved Python to an external runner after CVE-2025-68668 |
| Cloudflare Python Workers | Not an agent REPL | n/a | numpy named for dynamic libraries | Active, hosted; the startup snapshot reference |
| Monty and `CodeMode` | Not checked | Yes, the purpose of `CodeMode` | No third-party libraries | Active |

None documents an equivalent of `host.NAME` or of a cell index that survives compaction. LocalSandbox's design notes are worth reading before building the message protocol: `docs/design/pyodide-tool-bridge.md` and `docs/design/python-execution.md` in its repo. How its `call()` is synchronous could also answer the `df.apply` limit. `host_tools.py` already imports its helpers from `pydantic_ai_harness._code_tools`, the module behind `CodeMode`, so the tool-calling half shares its lineage with Monty. Monty supplies that half without numpy, Pyodide has the packages behind a weaker boundary, and eryx is what currently gives both in one sandbox.

## Recommended order of work

1. **Backend interface.** Pull a `SandboxRepl` out of `EryxRepl`, over a small session protocol (run a cell with a deadline, snapshot, restore, close), with no behaviour change. Rework the test fixtures, which import `eryx` directly, so the capability tests can run against any backend. This is useful even if Pyodide is dropped.
2. **Process wrapper for eryx.** Run each session in a child process with a wall-clock kill and a cgroup memory limit. This answers the eryx abort finding, and a Pyodide backend would reuse it.
3. **Pyodide spike, only if the decision below favours it.** A throwaway script, outside `src/`, with go or no-go gates:

| Gate | Passes when |
|---|---|
| Boot | numpy, pandas and bokeh import with no network, reads limited, local files only |
| Startup | A snapshot restores with pandas loaded, or a warm pool is fast enough, against eryx's 2.12 s |
| Kill | SIGKILL meets the deadline during a Python loop and during a long numpy call |
| Memory | The cap holds for Wasm and for JS objects made from Python, on WSL2 |
| Mount | The store reads from Python, and writes fail |
| Escape | `import js`, `ctypes.CDLL(None)` with `emscripten_run_script_string` and `system`, class-hierarchy traversal, file reads, `fetch`, `Deno.Command` and forged messages all fail or stay inside the process |

4. **Pyodide backend,** if the spike passes: a Deno main module, a worker, a Python runner, then the store and tools, then both backends through the shared tests.
5. **Packages and accuracy.** Pin the Pyodide distribution, run the numpy and pandas suites to produce new known-failure lists, and rewrite `sandbox-limitations.md` and `INSTRUCTIONS` from the results.

The decision this report can't make: **is the code untrusted?** If it is, Pyodide's history argues for staying on eryx and spending the effort on its package gaps. If the code is trusted, or the model only works on trusted data, Pyodide's packages may be worth a spike.

## What was not answered or tested

| Open question | Status | What would settle it |
|---|---|---|
| Does `_makeSnapshot` work with numpy and pandas loaded, and how fast is a restore? | Undocumented; private API | Spike: Startup gate |
| Is `maxMemoryBytes` in 314.0.7, and what bounds JS allocations? | Seen only on the main branch | Spike: Memory gate |
| Does `Worker.terminate()` or the interrupt buffer stop a long C call? | Not on the pages | Spike: Kill gate |
| Does `mountNodeFS` work under Deno? | Not on the pages | Spike: Mount gate |
| Does stack switching work under Deno? | The stack-switching docs page returned 404 | Read the docs or spike |
| Does Pyodide's numpy still return NaN from a singular `inv` without raising? | Not tested | Run the checks in [numpy-silent-nan-open](/home/rahul/.claude/projects/-home-rahul-eryx-test/memory/numpy-silent-nan-open.md) |
| Do C++ exceptions work in Pyodide extensions? | Not tested | A scipy or pandas call that throws in C++ |
| How does LocalSandbox make tool calls synchronous? | Design notes not read | Read its two design notes |
| Is Cloudflare's snapshot code public? | Not stated on the pages read | Search workerd and Pyodide sources |
| Do the Cyera CVEs match their records? | Seen only through Cyera | Read the CVE records |
| Does a cgroup memory limit work on this WSL2 machine, for either backend? | Not tested | Shared with the eryx report's open questions |

## Conclusion

The question started as "can Pyodide do what eryx does" and turned into "where does the boundary sit". The features mostly port, because the capability's work happens on the host. The boundary doesn't port: with Pyodide it is a process and the OS, not Wasm. Eryx is heading the same way after its host abort, so the process wrapper and the backend interface are the first steps either way. After that, whether the code is trusted decides whether Pyodide is worth a spike.
