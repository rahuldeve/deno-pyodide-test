"""Host modules preloaded into the sandbox: code from the host that cells call directly.

`PyodideRepl(sandbox_modules=[my_lib])` ships `my_lib`'s source to every session it
starts. There it can be imported under its own name, and its public names are already
defined in the cell namespace. The instructions list them, each with its signature and
the first paragraph of its docstring.

They run inside the sandbox like the model's own code, so they get its limits too: no
network, no host files, no secrets and no `deps`, and only the standard library and
the sandbox's packages to import, at the sandbox's versions. They suit pure
computation: domain formulas, cleaning steps, plotting presets. Arguments never cross
the process boundary, so a DataFrame goes to one as it is, and plain code (`df.apply`)
can call them. Code that needs the host is a job for host tools, which aren't done yet.

The whole module is shipped, and the model can read it (`inspect.getsource`) or
replace its names, so keep these modules free of anything secret, and treat what
they return like any other output from the sandbox. A dedicated module, such as
`myapp/sandbox_lib.py`, keeps the rest of the app out.

What is shipped:

- a plain module: its `.py` file
- a package: every `.py` file under its directory, so relative imports work
- a submodule (`myapp.sandbox_lib`): its file, plus an empty `__init__.py` for each
  parent package, so it keeps its dotted name. Pickles name classes by module, so a
  host variable holding an instance of a class from a shipped module loads in the
  sandbox, which a class defined only on the host can't.

The public names are `__all__` if the module defines it, else its top-level
functions and classes without a leading underscore that it defines itself (not ones
it imports). The source is read when `SandboxModules` is created; later edits to the
files reach only REPLs created after them.
"""

import inspect
import types
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A guard against shipping a whole app by passing its top package
MAX_SOURCE_BYTES = 10_000_000
# Names a module can't put in the cell namespace
RESERVED_NAMES = frozenset({"host"})
_MAX_SUMMARY_CHARS = 300


@dataclass(frozen=True)
class SandboxModules:
    """The source and public names of the host modules a session preloads.

    Create it with `SandboxModules.collect(modules)`; `PyodideRepl` does this for its
    `sandbox_modules`.
    """

    files: dict[str, str] = field(default_factory=dict)
    """Source by path relative to the sandbox's module directory (`pkg/mod.py`)."""
    exports: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Each module's public names, by module name, in the order they're defined in
    the cell namespace."""
    descriptions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """A line per public name for the instructions, by module name."""

    def __bool__(self) -> bool:
        return bool(self.exports)

    @classmethod
    def collect(cls, modules: Iterable[types.ModuleType]) -> "SandboxModules":
        """Read the modules' source and public names; `ValueError` for one that can't
        be shipped, or for two that define the same public name."""
        files: dict[str, str] = {}
        stubs: set[str] = set()
        exports: dict[str, tuple[str, ...]] = {}
        descriptions: dict[str, tuple[str, ...]] = {}
        owner: dict[str, str] = {}
        for module in modules:
            if not isinstance(module, types.ModuleType):
                raise TypeError(
                    f"sandbox_modules takes modules, got {type(module).__name__}"
                )
            name = module.__name__
            if name in exports:
                continue
            for path, source in _module_files(module).items():
                if path in files and path not in stubs and files[path] != source:
                    raise ValueError(f"two sandbox modules ship different {path}")
                files[path] = source
                stubs.discard(path)
            # empty parent packages, unless a module shipped the real ones
            parts = name.split(".")
            for depth in range(1, len(parts)):
                init = "/".join(parts[:depth]) + "/__init__.py"
                if init not in files:
                    files[init] = ""
                    stubs.add(init)
            names = public_names(module)
            for public in names:
                if public in RESERVED_NAMES:
                    raise ValueError(
                        f"sandbox module {name} exports {public!r}, which the "
                        "sandbox already defines"
                    )
                if public in owner:
                    raise ValueError(
                        f"sandbox modules {owner[public]} and {name} both export "
                        f"{public!r}"
                    )
                owner[public] = name
            exports[name] = names
            descriptions[name] = tuple(describe(module, n) for n in names)
        total = sum(len(s.encode()) for s in files.values())
        if total > MAX_SOURCE_BYTES:
            raise ValueError(
                f"sandbox modules add up to {total} bytes of source, over "
                f"{MAX_SOURCE_BYTES}; ship a dedicated module, not the whole app"
            )
        return cls(files, exports, descriptions)


def _module_files(module: types.ModuleType) -> dict[str, str]:
    name = module.__name__
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise ValueError(
            f"sandbox module {name} has no source file (a built-in module or a "
            "namespace package)"
        )
    path = Path(origin)
    prefix = name.replace(".", "/")
    if hasattr(module, "__path__"):  # a package
        if path.name != "__init__.py":
            raise ValueError(f"sandbox package {name} isn't plain Python ({path})")
        sources = {
            f"{prefix}/{file.relative_to(path.parent).as_posix()}": file
            for file in sorted(path.parent.rglob("*.py"))
            if not any(
                part == "__pycache__" or part.startswith(".")
                for part in file.relative_to(path.parent).parts
            )
        }
    else:
        if path.suffix != ".py":
            raise ValueError(f"sandbox module {name} isn't plain Python ({path})")
        sources = {f"{prefix}.py": path}
    files = {}
    for relative, file in sources.items():
        try:
            files[relative] = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"sandbox module file {file} isn't UTF-8") from None
    return files


def public_names(module: types.ModuleType) -> tuple[str, ...]:
    """`__all__`, else the public functions and classes `module` defines itself."""
    declared = getattr(module, "__all__", None)
    if declared is not None:
        names = tuple(declared)
        for name in names:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"{module.__name__}.__all__ holds {name!r}")
            if not hasattr(module, name):
                raise ValueError(
                    f"{module.__name__}.__all__ names {name!r}, which it doesn't define"
                )
        return names
    own = module.__name__
    return tuple(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and (
            value.__module__ == own or value.__module__.startswith(own + ".")
            if hasattr(module, "__path__")
            else value.__module__ == own
        )
    )


def describe(module: types.ModuleType, name: str) -> str:
    """`name`'s line in the instructions: signature, then its docstring's summary."""
    value: Any = getattr(module, name)
    if inspect.isclass(value) or callable(value):
        try:
            parsed = inspect.signature(value)
            if inspect.isclass(value):  # `__init__`'s `-> None` says nothing
                parsed = parsed.replace(return_annotation=inspect.Signature.empty)
            signature = str(parsed)
        except (TypeError, ValueError):
            signature = "(...)"
        head = ("class " if inspect.isclass(value) else "") + name + signature
        # the object's own docstring: inspect.getdoc would give a class without one
        # its base's (object's, say)
        doc = value.__doc__ if isinstance(value.__doc__, str) else ""
    else:
        head, doc = f"{name}: {type(value).__name__}", ""
    summary = " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[: _MAX_SUMMARY_CHARS - 3] + "..."
    return f"{head}: {summary}" if summary else head


def _module_list(names: Iterable[str]) -> str:
    return ", ".join(f"`{n}`" for n in names)


INSTRUCTIONS = """

Host modules: the host preloaded {modules}. Their public names, below, are already \
defined in the REPL, so call them directly; `import` works too, and \
`from MODULE import NAME` brings back a name you overwrote. They run inside the \
sandbox like your own code. `help(NAME)` shows the full documentation."""


def instructions(modules: SandboxModules) -> str:
    """The instructions' section on `modules`, or "" without any."""
    if not modules:
        return ""
    text = INSTRUCTIONS.format(modules=_module_list(modules.exports))
    for name, lines in modules.descriptions.items():
        text += f"\n\n{name}:" + "".join(f"\n- {line}" for line in lines)
        if not lines:
            text += " (no public names)"
    return text
