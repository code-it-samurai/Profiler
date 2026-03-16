from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from profiler.agent.state import AgentState
from profiler.agent.nodes import (
    broad_search,
    extract_and_normalize,
    analyze_candidates,
    filter_candidates,
    deep_scrape,
    compile_profile,
)
from profiler.config import settings
from profiler.models.enums import SessionStatus


def should_continue_narrowing(state: AgentState) -> str:
    """Router: decide whether to keep narrowing or move to deep scrape."""
    candidates = state.get("candidates", [])
    narrowing_round = state.get("narrowing_round", 0)
    status = state.get("status")

    # If analyze_candidates decided to compile directly (no more fields)
    if status == SessionStatus.COMPILING:
        return "deep_scrape"

    # Threshold reached
    if len(candidates) <= settings.candidate_threshold:
        return "deep_scrape"

    # Max rounds reached
    if narrowing_round >= settings.max_narrowing_rounds:
        return "deep_scrape"

    return "ask_user"


def after_filter(state: AgentState) -> str:
    """Router: after filtering, decide next step."""
    candidates = state.get("candidates", [])
    narrowing_round = state.get("narrowing_round", 0)

    if len(candidates) <= settings.candidate_threshold:
        return "deep_scrape"
    if narrowing_round >= settings.max_narrowing_rounds:
        return "deep_scrape"

    return "analyze_candidates"


def after_broad_search(state: AgentState) -> str:
    """Router: check if broad search failed."""
    if state.get("status") == SessionStatus.FAILED:
        return END
    return "extract_and_normalize"


def build_graph(checkpointer=None):
    """Build and compile the LangGraph agent graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Required for the ASK_USER interrupt to work.

    Returns:
        Compiled graph.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("broad_search", broad_search)
    graph.add_node("extract_and_normalize", extract_and_normalize)
    graph.add_node("analyze_candidates", analyze_candidates)
    graph.add_node("ask_user", lambda state: state)  # no-op; interrupt happens here
    graph.add_node("filter_candidates", filter_candidates)
    graph.add_node("deep_scrape", deep_scrape)
    graph.add_node("compile_profile", compile_profile)

    # Set entry point
    graph.set_entry_point("broad_search")

    # Edges
    graph.add_conditional_edges("broad_search", after_broad_search)
    graph.add_edge("extract_and_normalize", "analyze_candidates")
    graph.add_conditional_edges("analyze_candidates", should_continue_narrowing)
    graph.add_edge("ask_user", "filter_candidates")
    graph.add_conditional_edges("filter_candidates", after_filter)
    graph.add_edge("deep_scrape", "compile_profile")
    graph.add_edge("compile_profile", END)

    # Compile with interrupt_before on ask_user
    # This pauses the graph before executing ask_user,
    # allowing us to send the question to the user via SSE
    # and resume when they respond.
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["ask_user"],
    )

    return compiled


async def get_checkpointer():
    """Create an async SQLite checkpointer for LangGraph state persistence.

    Returns an AsyncSqliteSaver backed by a persistent SQLite file.
    The caller is responsible for closing the connection when done.
    """
    import aiosqlite

    conn = await aiosqlite.connect("profiler_checkpoints.db")
    checkpointer = AsyncSqliteSaver(conn=conn)
    await checkpointer.setup()
    return checkpointer
