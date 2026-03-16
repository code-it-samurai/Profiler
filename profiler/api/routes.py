import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from profiler.config import settings
from profiler.models.session import SearchRequest, AnswerRequest, SearchSession
from profiler.models.enums import SessionStatus
from profiler.agent.graph import build_graph, get_checkpointer
from profiler.api.sse import (
    status_update,
    question_event,
    profile_ready_event,
    error_event,
)
from profiler.api.dependencies import check_rate_limit, track_session, release_session
from profiler.db.repository import (
    create_session,
    update_session,
    get_session,
    save_profile,
    get_profile,
    delete_session,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# In-memory store for pending user answers (session_id -> asyncio.Event + answer)
_pending_answers: dict[str, dict] = {}


@router.post("/search")
async def start_search(req: SearchRequest, request: Request):
    """Start a new profiling session."""
    await check_rate_limit(request)

    # Build known_facts from structured fields
    known_facts = {}
    if req.location:
        known_facts["location"] = req.location
    if req.school:
        known_facts["school"] = req.school
    if req.employer:
        known_facts["employer"] = req.employer

    # Build context with extra structured info
    context_parts = []
    if req.context:
        context_parts.append(req.context)
    if req.email:
        context_parts.append(f"email: {req.email}")
    if req.twitter_handle:
        context_parts.append(f"twitter: {req.twitter_handle}")
    context = "; ".join(context_parts) if context_parts else req.context

    session = SearchSession(
        target_name=req.name,
        target_type=req.target_type,
        context=context,
        known_facts=known_facts,
    )
    await create_session(session)

    client_ip = request.client.host if request.client else "unknown"
    track_session(client_ip)

    return {"session_id": str(session.id), "status": "created"}


@router.get("/search/{session_id}/stream")
async def stream_search(session_id: str, request: Request):
    """SSE stream that runs the agent and yields events."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    async def event_generator():
        try:
            checkpointer = await get_checkpointer()
            graph = build_graph(checkpointer=checkpointer)

            thread_config = {"configurable": {"thread_id": session_id}}

            # Build known_facts and direct_urls from structured request fields
            import json as _json

            known_facts = {}
            direct_urls = []
            # Decode stored known_facts from session if present
            stored_facts = session.get("known_facts", "{}")
            if isinstance(stored_facts, str):
                try:
                    known_facts = _json.loads(stored_facts)
                except _json.JSONDecodeError:
                    pass
            elif isinstance(stored_facts, dict):
                known_facts = stored_facts

            # Initial state
            initial_state = {
                "target_name": session["target_name"],
                "target_type": session["target_type"],
                "initial_context": session["context"] or "",
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

            yield status_update("searching", "Starting broad search...", 0)

            # Run graph -- it will pause at interrupt_before=["ask_user"]
            state = None
            async for event in graph.astream(initial_state, config=thread_config):
                # event is a dict of {node_name: state_update}
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        state = {**initial_state, **(state or {}), **node_output}

                current_status = state.get("status") if state else None

                if current_status == SessionStatus.FAILED:
                    yield error_event(state.get("error", "Unknown error"))
                    return

                if current_status == SessionStatus.SEARCHING:
                    yield status_update(
                        "searching",
                        "Scraping and extracting profiles...",
                        len(state.get("candidates", [])),
                    )

                if current_status == SessionStatus.NARROWING:
                    yield status_update(
                        "narrowing",
                        "Analyzing candidates...",
                        len(state.get("candidates", [])),
                    )

            # Graph paused at ask_user interrupt -- enter narrowing loop
            while True:
                # Get current state from checkpointer
                snapshot = await graph.aget_state(thread_config)
                current_state = snapshot.values

                if current_state.get("status") == SessionStatus.DONE:
                    break

                if current_state.get("status") == SessionStatus.FAILED:
                    yield error_event(current_state.get("error", "Unknown error"))
                    return

                question = current_state.get("current_question")
                if not question:
                    break

                # Send question to user
                yield question_event(
                    question=question["question"],
                    field=question["field"],
                    options=question.get("options"),
                    round_num=current_state.get("narrowing_round", 0),
                )

                await update_session(
                    session_id,
                    status="asking_user",
                    narrowing_round=current_state.get("narrowing_round", 0),
                    candidates_count=len(current_state.get("candidates", [])),
                )

                # Wait for user answer
                answer_event_obj = asyncio.Event()
                _pending_answers[session_id] = {
                    "event": answer_event_obj,
                    "answer": None,
                }

                try:
                    await asyncio.wait_for(
                        answer_event_obj.wait(), timeout=300
                    )  # 5 min timeout
                except asyncio.TimeoutError:
                    yield error_event("Timed out waiting for answer", recoverable=True)
                    del _pending_answers[session_id]
                    return

                user_answer = _pending_answers[session_id]["answer"]
                del _pending_answers[session_id]

                yield status_update(
                    "narrowing",
                    f"Filtering candidates by {question['field']}...",
                    len(current_state.get("candidates", [])),
                )

                # Resume graph with the user's answer
                await graph.aupdate_state(
                    thread_config,
                    {
                        "user_answer": user_answer,
                        "status": SessionStatus.NARROWING,
                    },
                )

                # Continue running from the interrupt
                async for event in graph.astream(None, config=thread_config):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict):
                            current_state = {**current_state, **node_output}

                    cs = current_state.get("status")
                    if cs == SessionStatus.COMPILING:
                        yield status_update(
                            "compiling",
                            "Compiling final profile...",
                            len(current_state.get("candidates", [])),
                        )

            # Done -- save profile
            final_state = (await graph.aget_state(thread_config)).values
            profile = final_state.get("final_profile")

            if profile:
                await save_profile(session_id, profile)
                await update_session(session_id, status="done")
                yield profile_ready_event(session_id)
            else:
                yield error_event("Failed to compile profile")
                await update_session(session_id, status="failed")

        except Exception as e:
            logger.exception(f"Stream error for session {session_id}")
            yield error_event(str(e))
            await update_session(session_id, status="failed")
        finally:
            client_ip = request.client.host if request.client else "unknown"
            release_session(client_ip)

    return EventSourceResponse(event_generator())


@router.post("/search/{session_id}/answer")
async def submit_answer(session_id: str, req: AnswerRequest):
    """Submit the user's answer to a narrowing question."""
    if session_id not in _pending_answers:
        raise HTTPException(400, "No pending question for this session")

    _pending_answers[session_id]["answer"] = req.answer
    _pending_answers[session_id]["event"].set()

    return {"status": "received"}


@router.get("/search/{session_id}/status")
async def get_status(session_id: str):
    """Get current session status."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.get("/search/{session_id}/profile")
async def get_session_profile(session_id: str):
    """Get the final compiled profile."""
    profile = await get_profile(session_id)
    if not profile:
        raise HTTPException(404, "Profile not ready")
    return profile.model_dump()


@router.delete("/search/{session_id}")
async def cancel_session(session_id: str):
    """Cancel and clean up a session."""
    await delete_session(session_id)
    if session_id in _pending_answers:
        _pending_answers[session_id]["event"].set()
        del _pending_answers[session_id]
    return {"status": "deleted"}


@router.get("/health")
async def health_check():
    """Check Ollama connectivity and model availability."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            has_model = any(settings.ollama_model in name for name in model_names)

            return {
                "status": "healthy" if has_model else "degraded",
                "ollama": "connected",
                "model_loaded": has_model,
                "available_models": model_names,
                "search_engine": "duckduckgo (no API key needed)",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama": f"error: {e}",
            "model_loaded": False,
            "fix": "Run: ollama serve (in a separate terminal)",
        }
