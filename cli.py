#!/usr/bin/env python3
"""
Profiler CLI -- Interactive OSINT search from the terminal.

Usage:
    python cli.py "John Smith"
    python cli.py "Acme Corp" --type company
    python cli.py "Jane Doe" --context "lives in Austin, went to UT"
    python cli.py "John Smith" --location "Portland" --employer "Nike"
    python cli.py "John Smith" --twitter "@jsmith" --linkedin "https://linkedin.com/in/jsmith"
    python cli.py "John Smith" --output ./results/
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from profiler.config import settings
from profiler.models.enums import TargetType, SessionStatus
from profiler.agent.graph import build_graph, get_checkpointer
from profiler.agent.state import AgentState

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]PROFILER[/bold cyan]\n"
            "[dim]Agentic OSINT Person & Company Search[/dim]",
            border_style="cyan",
        )
    )


def print_profile(profile_data: dict, target_name: str):
    """Pretty-print the final profile to the terminal."""
    console.print()
    console.print(
        Panel(
            f"[bold green]Profile Complete: {target_name}[/bold green]",
            border_style="green",
        )
    )

    # Summary
    if profile_data.get("summary"):
        console.print(f"\n[bold]Summary[/bold]")
        console.print(profile_data["summary"])

    # Social profiles table
    socials = profile_data.get("social_profiles", [])
    if socials:
        table = Table(title="Social Profiles", show_lines=True)
        table.add_column("Platform", style="cyan")
        table.add_column("URL", style="blue")
        table.add_column("Bio", max_width=40)
        for sp in socials:
            table.add_row(
                sp.get("platform", ""),
                sp.get("url", ""),
                (sp.get("bio") or "")[:40],
            )
        console.print(table)

    # Key facts
    for label, key in [
        ("Locations", "locations"),
        ("Education", "education"),
        ("Employment", "employment"),
        ("Associated Entities", "associated_entities"),
    ]:
        items = profile_data.get(key, [])
        if items:
            console.print(f"\n[bold]{label}:[/bold] {', '.join(items)}")

    # Sources
    sources = profile_data.get("sources", [])
    if sources:
        console.print(f"\n[bold]Sources ({len(sources)}):[/bold]")
        for s in sources[:10]:
            console.print(f"  [dim]- {s.get('url', '')}[/dim]")
        if len(sources) > 10:
            console.print(f"  [dim]... and {len(sources) - 10} more[/dim]")

    # Confidence
    confidence = profile_data.get("confidence_score", 0)
    color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
    console.print(f"\n[bold]Confidence:[/bold] [{color}]{confidence:.0%}[/{color}]")


def save_profile_json(profile_data: dict, target_name: str, output_dir: str) -> Path:
    """Save profile as a JSON file. Returns the file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in target_name)
    safe_name = safe_name.strip().replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"profile_{safe_name}_{timestamp}.json"

    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(profile_data, f, indent=2, default=str)

    return filepath


async def check_ollama():
    """Verify Ollama is running, model is available, and pre-warm it."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if not any(settings.ollama_model in name for name in model_names):
                console.print(
                    f"[red]Model '{settings.ollama_model}' not found in Ollama.[/red]"
                )
                console.print(
                    f"[yellow]Run: ollama pull {settings.ollama_model}[/yellow]"
                )
                return False

        # Pre-warm: load model into memory with a trivial request.
        # Cold-loading a 6.6GB model from disk can take 30-60s and causes
        # connection drops if the first real LLM call hits before it's ready.
        console.print("[dim]Loading model into memory...[/dim]")
        async with httpx.AsyncClient(
            timeout=float(settings.ollama_timeout_seconds),
        ) as client:
            await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_ctx": 512, "num_predict": 1},
                },
            )
        console.print("[dim]Model ready.[/dim]")
        return True
    except Exception as e:
        console.print(f"[red]Cannot connect to Ollama: {e}[/red]")
        console.print("[yellow]Run: ollama serve[/yellow]")
        return False


async def run_search(
    args: argparse.Namespace, target_type: TargetType, output_dir: str
):
    """Run the full agent pipeline interactively."""

    # Pre-flight check
    if not await check_ollama():
        sys.exit(1)

    name = args.name

    # Build known_facts from structured input
    known_facts = {}
    if args.location:
        known_facts["location"] = args.location
    if args.school:
        known_facts["school"] = args.school
    if args.employer:
        known_facts["employer"] = args.employer

    # Build direct URLs list
    direct_urls = []
    if args.facebook:
        direct_urls.append(args.facebook)
    if args.linkedin:
        direct_urls.append(args.linkedin)
    if args.website:
        direct_urls.append(args.website)
    if args.twitter:
        handle = args.twitter.lstrip("@")
        direct_urls.append(f"https://x.com/{handle}")
    if args.instagram:
        handle = args.instagram.lstrip("@")
        direct_urls.append(f"https://instagram.com/{handle}")

    # Build context string from all inputs for the LLM
    context_parts = []
    if args.context:
        context_parts.append(args.context)
    if args.email:
        context_parts.append(f"email: {args.email}")
    if args.twitter:
        context_parts.append(f"twitter: {args.twitter}")
    context = "; ".join(context_parts) if context_parts else ""

    session_id = str(uuid4())
    checkpointer = await get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    thread_config = {"configurable": {"thread_id": session_id}}

    initial_state: AgentState = {
        "target_name": name,
        "target_type": target_type,
        "initial_context": context,
        "session_id": session_id,
        "known_facts": known_facts,
        "candidates": [],
        "eliminated": [],
        "search_history": [],
        "narrowing_round": 0,
        "current_question": None,
        "user_answer": None,
        "_raw_search_results": [],
        "direct_urls": direct_urls,
        "final_profile": None,
        "status": SessionStatus.SEARCHING,
        "error": None,
    }

    console.print(f"\n[bold]Searching for:[/bold] {name}")
    if context:
        console.print(f"[bold]Context:[/bold] {context}")
    if known_facts:
        console.print(f"[bold]Known facts:[/bold] {known_facts}")
    if direct_urls:
        console.print(f"[bold]Direct URLs:[/bold] {', '.join(direct_urls)}")
    console.print()

    # --- Run the graph until first interrupt ---
    current_state = dict(initial_state)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Broad search -- querying Google and social media...", total=None
        )

        async for event in graph.astream(initial_state, config=thread_config):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    current_state = {**current_state, **node_output}

                if node_name == "broad_search":
                    progress.update(
                        task,
                        description="Scraping and extracting profiles...",
                    )
                elif node_name == "extract_and_normalize":
                    n = len(current_state.get("candidates", []))
                    progress.update(
                        task,
                        description=f"Found {n} candidates. Analyzing...",
                    )
                elif node_name == "analyze_candidates":
                    progress.update(task, description="Preparing narrowing question...")

    # Check for early failure
    if current_state.get("status") == SessionStatus.FAILED:
        console.print(
            f"\n[red]Search failed:[/red] {current_state.get('error', 'Unknown error')}"
        )
        sys.exit(1)

    # Check if it finished without needing narrowing
    if current_state.get("status") == SessionStatus.DONE:
        profile = current_state["final_profile"]
        profile_data = profile.model_dump()
        print_profile(profile_data, name)
        filepath = save_profile_json(profile_data, name, output_dir)
        console.print(f"\n[green]Saved to:[/green] {filepath}")
        return

    # --- Narrowing loop ---
    while True:
        snapshot = await graph.aget_state(thread_config)
        state = snapshot.values

        if state.get("status") == SessionStatus.DONE:
            break
        if state.get("status") == SessionStatus.FAILED:
            console.print(f"\n[red]Error:[/red] {state.get('error')}")
            sys.exit(1)

        question = state.get("current_question")
        if not question:
            break

        # Display the narrowing question
        n_candidates = len(state.get("candidates", []))
        round_num = state.get("narrowing_round", 0) + 1

        console.print(
            Panel(
                f"[bold yellow]Narrowing Round {round_num}[/bold yellow] -- "
                f"{n_candidates} candidates remaining",
                border_style="yellow",
            )
        )

        console.print(f"\n[bold]{question['question']}[/bold]")

        options = question.get("options")
        if options:
            console.print("[dim]Suggestions:[/dim]")
            for i, opt in enumerate(options, 1):
                console.print(f"  [cyan]{i}.[/cyan] {opt}")
            console.print(
                f"  [dim]Type a number to select, or type your own answer[/dim]"
            )

        # Get user input
        answer = Prompt.ask("\n[bold green]Your answer[/bold green]")

        # Handle numbered selection
        if options and answer.strip().isdigit():
            idx = int(answer.strip()) - 1
            if 0 <= idx < len(options):
                answer = options[idx]
                console.print(f"[dim]Selected: {answer}[/dim]")

        # Resume the graph
        await graph.aupdate_state(
            thread_config,
            {"user_answer": answer, "status": SessionStatus.NARROWING},
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Filtering candidates...", total=None)

            async for event in graph.astream(None, config=thread_config):
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        state = {**state, **node_output}

                    if node_name == "filter_candidates":
                        n = len(state.get("candidates", []))
                        progress.update(
                            task,
                            description=f"{n} candidates remaining...",
                        )
                    elif node_name == "analyze_candidates":
                        progress.update(
                            task,
                            description="Analyzing for next question...",
                        )
                    elif node_name == "deep_scrape":
                        progress.update(
                            task,
                            description="Deep scraping final candidates...",
                        )
                    elif node_name == "compile_profile":
                        progress.update(
                            task,
                            description="Compiling final profile...",
                        )

    # --- Output ---
    final_snapshot = await graph.aget_state(thread_config)
    final_state = final_snapshot.values
    profile = final_state.get("final_profile")

    if profile:
        profile_data = profile.model_dump()
        print_profile(profile_data, name)
        filepath = save_profile_json(profile_data, name, output_dir)
        console.print(f"\n[green]Saved to:[/green] {filepath}")
    else:
        console.print("\n[red]Failed to compile a profile.[/red]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Profiler -- Agentic OSINT Person & Company Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "Elon Musk"
  python cli.py "Jane Doe" --context "lives in Austin, works at Dell"
  python cli.py "Acme Corp" --type company
  python cli.py "John Smith" --location "Portland" --employer "Nike"
  python cli.py "John Smith" --twitter "@jsmith" --facebook "https://facebook.com/jsmith"
  python cli.py "John Smith" --email "jsmith@example.com" --output ./results/
        """,
    )
    parser.add_argument("name", help="Person or company name to search")
    parser.add_argument(
        "--type",
        "-t",
        choices=["person", "company"],
        default="person",
        help="Target type (default: person)",
    )
    parser.add_argument(
        "--context",
        "-c",
        default="",
        help="Additional context to narrow initial search (e.g., 'lives in Portland')",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Directory to save the JSON profile (default: ./output)",
    )
    # Structured optional fields
    parser.add_argument("--email", "-e", default=None, help="Email address")
    parser.add_argument(
        "--location", "-l", default=None, help="City, state, or country"
    )
    parser.add_argument("--school", default=None, help="School or university name")
    parser.add_argument("--employer", default=None, help="Current or past employer")
    parser.add_argument(
        "--twitter", default=None, help="Twitter/X handle (with or without @)"
    )
    parser.add_argument("--facebook", default=None, help="Facebook profile URL")
    parser.add_argument("--linkedin", default=None, help="LinkedIn profile URL")
    parser.add_argument("--instagram", default=None, help="Instagram handle")
    parser.add_argument("--website", "-w", default=None, help="Personal website URL")

    args = parser.parse_args()

    print_banner()

    target_type = TargetType.PERSON if args.type == "person" else TargetType.COMPANY

    try:
        asyncio.run(run_search(args, target_type, args.output))
    except KeyboardInterrupt:
        console.print("\n[yellow]Search cancelled.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
