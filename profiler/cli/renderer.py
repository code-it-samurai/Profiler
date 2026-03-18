"""Rich terminal renderer for the Profiler CLI.

Renders structured progress events as a tree-style display with
timing, checkmarks, and spinners.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Phase name mapping from internal keys to display names
PHASE_NAMES = {
    "discovery": ("1/5", "Discovery"),
    "broad_search": ("1/5", "Discovery"),
    "extract": ("2/5", "Extract & Enrich"),
    "scraping": ("2/5", "Extract & Enrich"),
    "extracting": ("2/5", "Extract & Enrich"),
    "narrowing": ("3/5", "Narrowing"),
    "analyze": ("3/5", "Narrowing"),
    "filter": ("3/5", "Narrowing"),
    "deep_enrich": ("4/5", "Deep Enrichment"),
    "deep_scrape": ("4/5", "Deep Enrichment"),
    "compile": ("5/5", "Compiling Dossier"),
}

PHASE_COLORS = {
    "1/5": "cyan",
    "2/5": "blue",
    "3/5": "yellow",
    "4/5": "magenta",
    "5/5": "green",
}


def _fmt_time(seconds: float) -> str:
    """Format elapsed time compactly."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.0f}s"


@dataclass
class TaskState:
    label: str
    status: str = "running"  # running, done, failed
    detail: str = ""
    start_time: float = 0.0
    elapsed: Optional[float] = None

    def finish(self, status: str = "done", detail: str = ""):
        self.status = status
        if detail:
            self.detail = detail
        self.elapsed = time.monotonic() - self.start_time


@dataclass
class PhaseState:
    name: str
    number: str  # "1/5", "2/5", etc.
    status: str = "running"
    start_time: float = 0.0
    tasks: list[TaskState] = field(default_factory=list)
    elapsed: Optional[float] = None

    def finish(self):
        self.status = "done"
        self.elapsed = time.monotonic() - self.start_time


class ProfilerRenderer:
    """Renders structured progress events to the terminal."""

    def __init__(self, console: Console, verbosity: int = 1):
        self.console = console
        self.verbosity = verbosity
        self.phases: list[PhaseState] = []
        self.current_phase: Optional[PhaseState] = None
        self._task_starts: dict[str, float] = {}
        self._total_start: float = time.monotonic()
        self._last_phase_key: Optional[str] = None

    def _get_or_create_phase(self, phase_key: str) -> PhaseState:
        """Get existing phase or create a new one."""
        info = PHASE_NAMES.get(phase_key, ("?", phase_key))
        phase_num, phase_name = info

        # Check if we're still in the same phase
        if self.current_phase and self.current_phase.number == phase_num:
            return self.current_phase

        # Check if this phase already exists
        for p in self.phases:
            if p.number == phase_num:
                self.current_phase = p
                return p

        # Finish previous phase
        if self.current_phase and self.current_phase.status == "running":
            self.current_phase.finish()
            self._print_phase_done(self.current_phase)

        # Create new phase
        phase = PhaseState(
            name=phase_name,
            number=phase_num,
            start_time=time.monotonic(),
        )
        self.phases.append(phase)
        self.current_phase = phase
        self._print_phase_start(phase)
        return phase

    def _print_phase_start(self, phase: PhaseState):
        """Print the phase header line."""
        color = PHASE_COLORS.get(phase.number, "white")
        self.console.print(
            f"\n[{color} bold]Phase {phase.number} \u00b7 {phase.name}[/{color} bold]"
        )

    def _print_phase_done(self, phase: PhaseState):
        """Print phase completion with timing."""
        if self.verbosity < 1:
            return
        color = PHASE_COLORS.get(phase.number, "white")
        elapsed = _fmt_time(phase.elapsed) if phase.elapsed else ""
        self.console.print(
            f"[green]\u2713[/green] [{color}]Phase {phase.number} \u00b7 {phase.name} "
            f"complete[/{color}] [dim]{elapsed}[/dim]"
        )

    def _print_task(self, task: TaskState, is_last: bool = False):
        """Print a sub-task line with tree connector."""
        if self.verbosity < 1:
            return
        prefix = "  \u2514\u2500\u2500 " if is_last else "  \u251c\u2500\u2500 "
        if task.status == "done":
            icon = "[green]\u2713[/green]"
        elif task.status == "failed":
            icon = "[red]\u2717[/red]"
        else:
            icon = "[cyan]\u2022[/cyan]"

        elapsed = ""
        if task.elapsed is not None:
            elapsed = f" [dim]{_fmt_time(task.elapsed)}[/dim]"

        detail = f" {task.detail}" if task.detail else ""
        self.console.print(f"{prefix}{task.label} {icon}{detail}{elapsed}")

    def _print_info(self, detail: str, is_last: bool = False):
        """Print an info line with tree connector."""
        if self.verbosity < 1:
            return
        prefix = "  \u2514\u2500\u2500 " if is_last else "  \u251c\u2500\u2500 "
        self.console.print(f"{prefix}[dim]{detail}[/dim]")

    def on_event(
        self,
        phase: str,
        event: str,
        detail: str,
        pct: Optional[int] = None,
        meta: Optional[dict] = None,
    ):
        """Process a progress event from the agent."""
        current = self._get_or_create_phase(phase)

        if event == "start":
            # Phase start — already handled by _get_or_create_phase
            pass

        elif event == "task_start":
            task = TaskState(
                label=detail,
                status="running",
                start_time=time.monotonic(),
            )
            current.tasks.append(task)
            self._task_starts[detail] = time.monotonic()
            if self.verbosity >= 2:
                self._print_info(f"{detail}...")

        elif event == "task_done":
            task = self._find_or_create_task(current, detail)
            summary = ""
            if meta:
                parts = []
                if "count" in meta and "total" in meta:
                    parts.append(f"{meta['count']}/{meta['total']}")
                elif "count" in meta:
                    parts.append(str(meta["count"]))
                if "queries" in meta:
                    parts.append(f"from {meta['queries']} queries")
                if "success" in meta and "failed" in meta:
                    parts.append(f"{meta['success']} ok, {meta['failed']} failed")
                if "tool" in meta:
                    parts.append(f"via {meta['tool']}")
                if "emails" in meta and isinstance(meta["emails"], int):
                    parts.append(f"{meta['emails']} emails")
                if "urls" in meta and isinstance(meta["urls"], int):
                    parts.append(f"{meta['urls']} URLs")
                if "allowed" in meta and "blocked" in meta:
                    parts.append(
                        f"{meta['allowed']} allowed, {meta['blocked']} blocked"
                    )
                summary = ", ".join(parts)
            task.finish("done", summary)
            self._print_task(task)

        elif event == "task_fail":
            task = self._find_or_create_task(current, detail)
            error_msg = meta.get("error", "") if meta else ""
            task.finish("failed", error_msg)
            self._print_task(task)

        elif event == "phase_done":
            summary = ""
            if meta:
                parts = []
                for k, v in meta.items():
                    parts.append(f"{k}: {v}")
                summary = ", ".join(parts)
            if summary:
                self._print_info(summary, is_last=True)
            current.finish()
            self._print_phase_done(current)

        elif event == "info":
            # Info event — old-style or supplementary detail
            if self.verbosity >= 1:
                # For old-style calls, just print as a sub-item
                if meta:
                    # Structured info (like field coverage)
                    parts = []
                    for k, v in meta.items():
                        if k != "total":
                            total = meta.get("total", "")
                            parts.append(f"{k}: {v}/{total}" if total else f"{k}: {v}")
                    if parts:
                        self._print_info(" \u00b7 ".join(parts))
                elif detail:
                    self._print_info(detail)

    def _find_or_create_task(self, phase: PhaseState, label: str) -> TaskState:
        """Find an existing running task by label, or create one."""
        for task in phase.tasks:
            if task.label == label and task.status == "running":
                return task
        # Create on-the-fly (task_done without prior task_start)
        start = self._task_starts.get(label, time.monotonic())
        task = TaskState(label=label, status="running", start_time=start)
        phase.tasks.append(task)
        return task

    def print_header(
        self,
        target_name: str,
        target_type: str,
        known_facts: dict,
        direct_urls: list[str],
        context: str = "",
    ):
        """Print the initial search header panel."""
        lines = [f"[bold]Target:[/bold] {target_name} ({target_type})"]
        if known_facts:
            facts_str = ", ".join(f"{k}={v}" for k, v in known_facts.items())
            lines.append(f"[bold]Known:[/bold] {facts_str}")
        if context:
            lines.append(f"[bold]Context:[/bold] {context}")
        if direct_urls:
            lines.append(f"[bold]URLs:[/bold] {', '.join(direct_urls[:3])}")

        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold cyan]Profiler[/bold cyan]",
                border_style="cyan",
            )
        )

    def print_narrowing_question(
        self,
        question: str,
        options: Optional[list[str]],
        n_candidates: int,
        round_num: int,
        history: Optional[list[dict]] = None,
    ):
        """Print a narrowing question as an interactive panel."""
        header = f"{n_candidates} candidates \u00b7 Round {round_num}"
        if history:
            last = history[-1]
            header += f" [dim](last: {last['before']} \u2192 {last['after']} by {last['field']})[/dim]"

        self.console.print(
            Panel(
                f"[bold]{question}[/bold]",
                title=f"[yellow]{header}[/yellow]",
                border_style="yellow",
            )
        )

        if options:
            for i, opt in enumerate(options, 1):
                self.console.print(f"  [cyan]{i}.[/cyan] {opt}")
            self.console.print(
                "  [dim]Type a number to select, or type your own answer[/dim]"
            )

    def print_narrowing_result(self, before: int, after: int, field: str, answer: str):
        """Print the result of a narrowing round."""
        self._print_info(
            f"{before} \u2192 {after} candidates remaining "
            f'(filtered by {field}="{answer}")',
            is_last=True,
        )

    def print_final_summary(
        self,
        profile_data: dict,
        data_sources: list[str],
        narrowing_history: list[dict],
    ):
        """Print the final results summary panel."""
        # Finish any running phase
        if self.current_phase and self.current_phase.status == "running":
            self.current_phase.finish()
            self._print_phase_done(self.current_phase)

        total_elapsed = time.monotonic() - self._total_start

        lines = []
        if profile_data.get("narrowing_summary"):
            lines.append(
                f"[bold]Search Journey:[/bold] {profile_data['narrowing_summary']}"
            )
        found = profile_data.get("candidates_found", 0)
        remaining = profile_data.get("candidates_remaining", 0)
        if found or remaining:
            lines.append(
                f"[bold]Candidates:[/bold] {found} found \u2192 {remaining} remaining"
            )
        if data_sources:
            lines.append(f"[bold]Tools used:[/bold] {', '.join(data_sources)}")
        lines.append(f"[bold]Total time:[/bold] {_fmt_time(total_elapsed)}")

        confidence = profile_data.get("confidence_score", 0)
        color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
        lines.append(f"[bold]Confidence:[/bold] [{color}]{confidence:.0%}[/{color}]")

        self.console.print()
        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold green]Results[/bold green]",
                border_style="green",
            )
        )
