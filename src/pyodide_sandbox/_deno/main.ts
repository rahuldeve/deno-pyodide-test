// One Pyodide REPL session, driven over JSON lines by the Python host (PyodideSession).
//
//   host -> deno (stdin):   {"id": 1, "op": "run", "code": "..."}
//   deno -> host (stdout):  {"op": "ready", "boot_ms": ...} once, then per cell
//                           {"id": 1, "error": null | "ZeroDivisionError",
//                            "stdout": "...", "stderr": "...", "duration_ms": ...}
//                           or, when Pyodide can't go on, {"op": "fatal", "message": "..."}
//                           just before the process exits (code 70)
//
// Stdout carries nothing but these messages: console.log is sent to stderr, and so
// is Pyodide's own output outside a cell. The host enforces the time limit by
// killing this process, so nothing here tries to interrupt a running cell. This file
// isn't a security boundary: code in a cell can reach JavaScript (`import js`) and
// write to the pipe itself, so the host treats every message as untrusted.
//
// Usage: deno run <flags> main.ts --dist DIR [--packages numpy,pandas] [--max-output-bytes N]
//                                [--store DIR]
//
// With --store, cells read the host's variables in DIR as `host.NAME` (runner.setup_host).

const t0 = performance.now();
const args = parseArgs(Deno.args);
// From the distribution itself, not npm, so the two are always the same release
const { loadPyodide } = await import(
  "file://" + args.dist.split("/").map(encodeURIComponent).join("/") +
    "pyodide.mjs"
);
const encoder = new TextEncoder();
const stdoutWriter = Deno.stdout.writable.getWriter();

// Anything printed with console.log (Pyodide's own messages, say) would corrupt the protocol
console.log = console.error;
console.info = console.error;

async function send(message: unknown): Promise<void> {
  await stdoutWriter.write(encoder.encode(JSON.stringify(message) + "\n"));
}

/**
 * Report why Pyodide can't go on, then exit, so the host starts a new process.
 *
 * Pyodide can fail for good, sometimes from inside a cell: a JS error thrown in a
 * syscall does it, e.g. `os.chmod` on a NODEFS directory the cell mounted itself.
 * Every call into it fails after that. Synchronous, so it can run inside Pyodide's
 * fatal-error hook, before the follow-on errors that escape it.
 */
function die(cause: unknown): never {
  const message = describe(cause);
  console.error(`pyodide-sandbox: exiting: ${message}`);
  try {
    // a message's own writes are awaited before the next, so none is half done
    const line = encoder.encode(
      JSON.stringify({ op: "fatal", message }) + "\n",
    );
    for (let done = 0; done < line.length;) {
      done += Deno.stdout.writeSync(line.subarray(done));
    }
  } catch {
    // the host sees the exit anyway
  }
  Deno.exit(70);
}

function describe(cause: unknown): string {
  const text = cause instanceof Error
    ? `${cause.name}: ${cause.message}`
    : String(cause);
  const firstLine = text.split("\n", 1)[0];
  return firstLine.length > 500 ? firstLine.slice(0, 500) + "..." : firstLine;
}

/** A cell's output stream, kept to `limit` bytes; what is dropped is counted. */
class Capture {
  private chunks: Uint8Array[] = [];
  private kept = 0;
  private dropped = 0;
  constructor(private limit: number) {}

  write(buffer: Uint8Array): number {
    const room = this.limit - this.kept;
    if (room > 0) {
      const chunk = buffer.slice(0, room); // copied: Pyodide reuses its buffer
      this.chunks.push(chunk);
      this.kept += chunk.length;
    }
    this.dropped += Math.max(0, buffer.length - Math.max(room, 0));
    return buffer.length;
  }

  take(): string {
    const all = new Uint8Array(this.kept);
    let offset = 0;
    for (const chunk of this.chunks) {
      all.set(chunk, offset);
      offset += chunk.length;
    }
    let text = new TextDecoder().decode(all); // invalid UTF-8 becomes U+FFFD
    if (this.dropped) {
      text += `\n[... ${this.dropped} more bytes of output dropped]\n`;
    }
    this.chunks = [];
    this.kept = 0;
    this.dropped = 0;
    return text;
  }
}

const pyodide = await loadPyodide({
  indexURL: args.dist,
  stdout: (msg: string) => console.error(msg),
  stderr: (msg: string) => console.error(msg),
});
if (args.packages.length) {
  await pyodide.loadPackage(args.packages, {
    messageCallback: () => {},
    errorCallback: (msg: string) => console.error(msg),
  });
}

const runnerSource = await Deno.readTextFile(
  new URL("./runner.py", import.meta.url),
);
const runnerNamespace = pyodide.globals.get("dict")();
pyodide.runPython(runnerSource, {
  globals: runnerNamespace,
  filename: "<runner>",
});
const runCell = runnerNamespace.get("run_cell");
if (args.store) {
  const hostSource = await Deno.readTextFile(
    new URL("./host.py", import.meta.url),
  );
  runnerNamespace.get("setup_host")(hostSource, args.store);
}
for (const pkg of args.packages) {
  pyodide.runPython(`import ${pkg}`); // so the first cell doesn't pay for it
}

// Private API: Pyodide calls it with the cause of a fatal error, then throws
// (pyodide.asm.mjs, API.fatal_error). Without it, the listeners below still exit.
// deno-lint-ignore no-explicit-any
(pyodide as any)._api.on_fatal = die;
const stdout = new Capture(args.maxOutputBytes);
const stderr = new Capture(args.maxOutputBytes);

// An error nothing caught, e.g. thrown by a JS callback a cell scheduled. Deno would
// exit; if Pyodide is still usable, it goes to the running cell's stderr instead
// (between cells, it is dropped with the rest of that output). After boot only:
// until then, Deno's own report goes to stderr, which the host shows.
function uncaught(event: Event, cause: unknown): void {
  if (!pyodideUsable()) die(cause);
  event.preventDefault();
  stderr.write(
    encoder.encode(`Uncaught JavaScript error: ${describe(cause)}\n`),
  );
}
globalThis.addEventListener("error", (e) => uncaught(e, e.error ?? e.message));
globalThis.addEventListener("unhandledrejection", (e) => uncaught(e, e.reason));

function pyodideUsable(): boolean {
  try {
    // after a fatal error, Pyodide's public API throws on every access
    pyodide.globals;
    return true;
  } catch {
    return false;
  }
}
pyodide.setStdout({ write: (b: Uint8Array) => stdout.write(b) });
pyodide.setStderr({ write: (b: Uint8Array) => stderr.write(b) });

await send({ op: "ready", boot_ms: performance.now() - t0 });

for await (const line of readLines(Deno.stdin.readable)) {
  if (!line.trim()) continue;
  const request = JSON.parse(line);
  if (request.op !== "run") {
    await send({
      id: request.id,
      op: "error",
      message: `unknown op ${request.op}`,
    });
    continue;
  }
  // Output written between cells (by a task a cell left running) is not this cell's
  stdout.take();
  stderr.take();
  const start = performance.now();
  let error: string | null = null;
  try {
    error = (await runCell(request.code, request.id)) ?? null;
  } catch (e) {
    // run_cell catches all the cell's own exceptions, so this is Pyodide failing,
    // and its state can't be trusted
    die(e);
  }
  await send({
    id: request.id,
    error,
    stdout: stdout.take(),
    stderr: stderr.take(),
    duration_ms: performance.now() - start,
  });
}

async function* readLines(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<string> {
  let pending = "";
  for await (const text of stream.pipeThrough(new TextDecoderStream())) {
    pending += text;
    let newline;
    while ((newline = pending.indexOf("\n")) >= 0) {
      yield pending.slice(0, newline);
      pending = pending.slice(newline + 1);
    }
  }
  if (pending) yield pending;
}

function parseArgs(argv: string[]) {
  const options = {
    dist: "",
    packages: [] as string[],
    maxOutputBytes: 1 << 20,
    store: "",
  };
  for (let i = 0; i < argv.length; i++) {
    const value = argv[i + 1];
    switch (argv[i]) {
      case "--dist":
        options.dist = value.endsWith("/") ? value : value + "/";
        i++;
        break;
      case "--packages":
        options.packages = value.split(",").filter(Boolean);
        i++;
        break;
      case "--store":
        options.store = value;
        i++;
        break;
      case "--max-output-bytes":
        options.maxOutputBytes = Number(value);
        i++;
        break;
      default:
        throw new Error(`unknown argument ${argv[i]}`);
    }
  }
  if (!options.dist) throw new Error("--dist is required");
  return options;
}
