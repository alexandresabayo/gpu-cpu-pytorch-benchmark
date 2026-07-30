"""
richlog.core — a collapsible, multi-step Docker/BuildKit-style logger.

Design in one sentence: there is a *stack* of currently-open Steps; only the
deepest one ever renders live detail (rolling log window + progress bar).
Nothing is committed to permanent scrollback until a step (and everything
nested under it) is actually finished — at which point the whole thing
prints as one block, the step's own summary line first, followed by
whatever survived underneath it, in the order it happened.

What survives into that permanent block:
  - the step's own one-line summary (marker, title, dots, elapsed time)
  - any warn()/error()/prompt() lines it logged directly
  - any block() calls it made (e.g. a metrics table)
  - its children's own finalized blocks, recursively, in the same shape

What does NOT survive (plain info() lines): those only ever exist in the
live rolling window while their step is the active leaf. They're real-time
"here's what's happening" narration, not part of the permanent record —
they're still written to the file log in full, just not the terminal.

A plain timestamped file sink runs alongside the terminal renderer and
receives *everything*, in real time, regardless of what the terminal ends
up committing or when — a crash mid-run still leaves a complete file log.

Usage:

    with Logger(log_dir="logs") as log:
        with log.step("Training on CPU") as step:
            prog = step.progress(total=200, label="epoch")
            for epoch in range(200):
                ...
                step.info(f"epoch {epoch}: loss={loss:.4f}")   # transient
                prog.advance()
            step.warn("early stopping triggered")               # survives
"""

from __future__ import annotations

import contextvars
import io
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, List, Optional

from rich.console import Console, Group
from rich.padding import Padding
from rich.text import Text

WINDOW_SIZE = 8          # how many sub-lines stay visible under a live step
INDENT = "  "            # per-depth-level indent (auto-applied, never manual)
BAR_WIDTH = 24

# (marker, style) for messages logged *inside* an open step
_MSG_STYLE = {
    "info":    ("", "dim"),
    "warn":    ("[!]", "yellow"),
    "error":   ("[x]", "bold red"),
    "prompt":  ("[?]", "bold cyan"),
}

# (marker, style) for the permanent line printed when a step *closes*
_CLOSE_STYLE = {
    "success": ("[+]", "bold white"),
    "failure": ("[-]", "bold red"),
}


@dataclass
class ProgressState:
    total: int
    label: str = ""
    current: int = 0


@dataclass
class StepState:
    title: str
    depth: int
    start: float
    # Transient, live-view-only: info() lines, plus warn/error/prompt while
    # this step is still the rendered leaf. Bounded, never written anywhere
    # permanent from here (the file already got everything eagerly).
    lines: Deque[Text] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    progress: Optional[ProgressState] = None
    # What actually survives to the final printed block for this step:
    # warn/error/prompt Text lines, block() renderables, and closed
    # children's own finalized Groups — all in the order they happened.
    permanent: List[Any] = field(default_factory=list)


class ProgressHandle:
    """Returned by StepHandle.progress(); update it as work proceeds."""

    def __init__(self, logger: "Logger", state: StepState):
        self._logger = logger
        self._state = state

    def advance(self, n: int = 1) -> None:
        p = self._state.progress
        p.current = min(p.total, p.current + n)
        self._logger._refresh()

    def update(self, current: int) -> None:
        p = self._state.progress
        p.current = max(0, min(p.total, current))
        self._logger._refresh()


class StepHandle:
    """What you get inside `with logger.step(...) as step:`."""

    def __init__(self, logger: "Logger", state: StepState):
        self._logger = logger
        self._state = state

    def _log(self, kind: str, msg: str) -> None:
        marker, style = _MSG_STYLE[kind]
        prefix = f"{marker} " if marker else "    "
        text = Text(prefix, style=style)
        text.append(msg, style=style if kind != "info" else "dim")

        # File gets everything, immediately, regardless of kind.
        self._logger._log_to_file(f"{marker or '   '} {INDENT * (self._state.depth + 1)}{msg}")

        # Live view shows everything while this step is the active leaf.
        self._state.lines.append(text)
        self._logger._refresh()

        # Only warn/error/prompt earn a permanent place once this step
        # (or an ancestor) finally flushes — plain info() is ephemeral.
        if kind != "info":
            padded = Text(INDENT * (self._state.depth + 1))
            padded.append_text(text)
            self._state.permanent.append(padded)

    def info(self, msg: str) -> None:
        self._log("info", msg)

    def warn(self, msg: str) -> None:
        self._log("warn", msg)

    def error(self, msg: str) -> None:
        self._log("error", msg)

    def prompt(self, msg: str) -> None:
        self._log("prompt", msg)

    def progress(self, total: int, label: str = "") -> ProgressHandle:
        self._state.progress = ProgressState(total=total, label=label)
        self._logger._refresh()
        return ProgressHandle(self._logger, self._state)

    def child(self, title: str):
        """Open a nested step under this one (same mechanism, deeper stack)."""
        return self._logger.step(title)

    def block(self, renderable, indent: Optional[int] = None, style: Optional[str] = None) -> None:
        """Queue a permanent block (e.g. a metrics table) to appear in this
        step's own finalized output, in the position it occurred — not
        printed immediately, since this step (or an ancestor) might still
        be open and we want everything to commit together, header first.

        Still written to the file log immediately, same as everything else.
        
        Args:
            renderable: The Rich renderable to display.
            indent: Optional number of indent levels. Each level is `len(INDENT)`
                    spaces, applied as left padding via Rich's Padding.
            style: Optional style string to apply to the renderable.
        """
        if style is not None:
            if isinstance(renderable, str):
                renderable = Text(renderable, style=style)
        if indent is not None:
            renderable = Padding(renderable, (0, 0, 0, len(INDENT) * indent))
        self._logger._write_block_to_file(renderable)
        self._state.permanent.append(renderable)


class NullProgressHandle:
    """No-op stand-in for `ProgressHandle`, returned by
    `NullStepHandle.progress()`. `.advance()`/`.update()` do nothing."""

    def advance(self, n: int = 1) -> None:
        pass

    def update(self, current: int) -> None:
        pass


class _NullStepContext:
    """No-op context manager returned by `NullStepHandle.child()`. Yields
    the same shared `NullStepHandle` instance, so nested
    `with step.child(...) as sub:` code works unchanged and `sub` is still
    a no-op handle."""

    def __init__(self, handle: "NullStepHandle"):
        self._handle = handle

    def __enter__(self) -> "NullStepHandle":
        return self._handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False  # never swallow exceptions


class NullStepHandle:
    """No-op stand-in for `StepHandle`, used as the default `step`/`log`
    argument throughout the training/data/visualization pipeline so every
    function works the same way with logging fully disabled.

    This is a full no-op: it does not touch the console *or* the file log
    (unlike a real `StepHandle`, whose `info()` lines still land in the
    file log even though they don't survive to the terminal). A
    quiet-but-still-file-logged mode would be a separate, smaller change to
    `Logger.__init__`, not this class.

    Stateless and safe to share; see `NULL_STEP` below.
    """

    def info(self, msg: str) -> None:
        pass

    def warn(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass

    def prompt(self, msg: str) -> None:
        pass

    def progress(self, total: int, label: str = "") -> NullProgressHandle:
        return NullProgressHandle()

    def child(self, title: str) -> _NullStepContext:
        """Open a no-op nested step under this one (mirrors
        `StepHandle.child`); returns a context manager yielding `self`."""
        return _NullStepContext(self)

    def block(self, renderable, indent: Optional[int] = None, style: Optional[str] = None) -> None:
        pass


# Shared instance: NullStepHandle is stateless, so one instance can be
# reused everywhere as the default value of the context stack below.
NULL_STEP = NullStepHandle()

# Holds whichever StepHandle (or root Logger) is innermost right now.
# Logger.__enter__ sets itself as the root value; each nested
# `with logger.step(...)`/`with step.child(...)` pushes its own StepHandle
# on top and restores the previous value on exit. Falls back to NULL_STEP
# when nothing is open at all.
_current: contextvars.ContextVar = contextvars.ContextVar("richlog_current", default=NULL_STEP)


def current():
    """Return the innermost open StepHandle, the root Logger if no step is
    open, or NULL_STEP if no Logger is open either."""
    return _current.get()


class _StepContext:
    def __init__(self, logger: "Logger", title: str):
        self._logger = logger
        self._title = title
        self._token = None

    def __enter__(self) -> StepHandle:
        state = self._logger._push(self._title)
        handle = StepHandle(self._logger, state)
        self._token = _current.set(handle)
        return handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "failure" if exc_type else "success"
        self._logger._pop(status)
        _current.reset(self._token)
        return False  # never swallow exceptions


class Logger:
    def __init__(self, log_dir: str = "logs", run_timestamp: Optional[str] = None,
                 console: Optional[Console] = None):
        self.console = console or Console(highlight=False)
        self._stack: List[StepState] = []
        self._live = None
        self._token = None

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stamp = run_timestamp or datetime.now().strftime("%Y-%m-%d_%Hh%M%Ss")
        self._file = open(log_path / f"run_{stamp}.log", "w")

    # -- public -----------------------------------------------------------
    def step(self, title: str) -> _StepContext:
        return _StepContext(self, title)

    def block(self, renderable, indent: Optional[int] = None, style: Optional[str] = None) -> None:
        """Print a permanent block straight to scrollback, right now.

        Only call this with no step open (e.g. a final report after every
        step has already closed) — it prints immediately rather than
        queuing, since there's no enclosing step to commit alongside.
        Called from inside a step, use step.block() instead.
        
        Args:
            renderable: The Rich renderable to display.
            indent: Optional number of indent levels. Each level is `len(INDENT)`
                    spaces, applied as left padding via Rich's Padding.
            style: Optional style string to apply to the renderable.
        """
        if style is not None:
            if isinstance(renderable, str):
                renderable = Text(renderable, style=style)
        if indent is not None:
            renderable = Padding(renderable, (0, 0, 0, len(INDENT) * indent))
        self.console.print(renderable)
        self._write_block_to_file(renderable)

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._file.close()

    def __enter__(self) -> "Logger":
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _current.reset(self._token)
        self.close()

    # -- internals ----------------------------------------------------------
    def _log_to_file(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._file.write(f"{ts} {line}\n")
        self._file.flush()

    def _write_block_to_file(self, renderable) -> None:
        # Render through a plain, colorless, fixed-width Console so the file
        # gets clean text regardless of what the real console's settings
        # are — works for a bare string, a Text, a Table, a Group, anything.
        plain_console = Console(file=io.StringIO(), width=100, no_color=True,
                                 force_terminal=False, highlight=False)
        with plain_console.capture() as capture:
            plain_console.print(renderable)
        text = capture.get()
        for line in text.splitlines() or [""]:
            self._log_to_file(f"    {line}")

    def _push(self, title: str) -> StepState:
        from rich.live import Live  # local import keeps module import light

        depth = len(self._stack)
        state = StepState(title=title, depth=depth, start=time.perf_counter())
        self._stack.append(state)
        self._log_to_file(f"[~] {INDENT * depth}{title}")

        if self._live is None:
            self._live = Live(
                self._build_renderable(),
                console=self.console,
                refresh_per_second=12,
                transient=True,
            )
            self._live.start()
        else:
            self._refresh()
        return state

    def _pop(self, status: str) -> None:
        state = self._stack.pop()
        elapsed = time.perf_counter() - state.start
        marker, style = _CLOSE_STYLE[status]
        summary_line = self._format_close_line(state, marker, style, elapsed)
        self._log_to_file(f"{marker} {INDENT * state.depth}{state.title} ({elapsed:.2f}s)")

        # This step's fully finalized output: its own summary line first,
        # then whatever survived inside it (warns/errors/blocks/children's
        # own finalized blocks), in the order it actually happened.
        finalized = Group(summary_line, *state.permanent)

        if self._stack:
            # An ancestor is still open — hand our finalized block up to
            # its buffer instead of printing. It isn't done yet, and we
            # want its own summary to appear before ours once everything
            # commits together as one block.
            self._stack[-1].permanent.append(finalized)
            self._refresh()
        else:
            # Nothing above us: this is the moment to actually commit,
            # for real, to the terminal.
            self._live.stop()
            self._live = None
            self.console.print(finalized)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_renderable())

    # -- rendering ----------------------------------------------------------
    def _format_close_line(self, state: StepState, marker: str, style: str, elapsed: float) -> Text:
        indent = INDENT * state.depth
        time_str = f"{elapsed:.2f}s"
        width = self.console.size.width
        used = len(indent) + len(marker) + 1 + len(state.title) + 1 + len(time_str) + 1
        dots = "." * max(2, width - used)

        line = Text(f"{indent}{marker} ", style=style)
        line.append(state.title, style="bold")
        line.append(f" {dots} ", style="dim")
        line.append(time_str, style="dim")
        return line

    def _render_progress(self, state: StepState) -> Text:
        p = state.progress
        filled = int(BAR_WIDTH * p.current / p.total) if p.total else 0
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        indent = INDENT * (state.depth + 1)
        label = f" {p.label}" if p.label else ""
        return Text(f"{indent}[{bar}] {p.current}/{p.total}{label}", style="dim")

    def _build_renderable(self) -> Group:
        rendered = []
        last = len(self._stack) - 1
        for i, state in enumerate(self._stack):
            is_leaf = i == last
            indent = INDENT * i
            marker_style = "bold white"
            marker = "[~] "
            title_line = Text(f"{indent}{marker}", style=marker_style)
            title_line.append(state.title, style="bold")
            rendered.append(title_line)

            if is_leaf:
                for sub in state.lines:
                    padded = Text(INDENT * (i + 1))
                    padded.append_text(sub)
                    rendered.append(padded)
                if state.progress is not None:
                    rendered.append(self._render_progress(state))

        return Group(*rendered)
