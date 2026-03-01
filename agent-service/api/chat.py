"""Chat API — REST and WebSocket endpoints for the supervisor agent."""

import logging
import uuid

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

logger = logging.getLogger("agent.api.chat")

router = APIRouter(tags=["Chat"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat message."""

    message: str = Field(..., description="User message text")
    thread_id: str | None = Field(
        None, description="Conversation thread ID. Auto-generated if omitted."
    )


class ChatResponse(BaseModel):
    """Agent response payload."""

    thread_id: str = Field(..., description="Conversation thread ID")
    message: str = Field(..., description="Agent reply text")
    tool_calls: list[dict] = Field(
        default_factory=list, description="Tool calls made during processing"
    )


# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------

@router.post("/api/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Send a message to the agent and receive the full response (non-streaming)."""
    graph = request.app.state.supervisor
    thread_id = body.thread_id or str(uuid.uuid4())

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        logger.exception("Graph invocation failed for thread %s", thread_id)
        raise exc

    # Extract the last AI message from the result
    ai_message = ""
    tool_calls: list[dict] = []
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            ai_message = msg.content if isinstance(msg.content, str) else ""
            tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc["args"],
                }
                for tc in (msg.tool_calls or [])
            ]
            break

    return ChatResponse(
        thread_id=thread_id,
        message=ai_message,
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/api/chat/ws")
async def chat_ws(websocket: WebSocket) -> None:
    """Stream agent responses over a WebSocket connection.

    Client sends: ``{"type": "message", "content": "...", "thread_id": "..."}``

    Server streams frames with ``type`` in
    ``token | tool_call | tool_result | done | error``.

    Uses ``astream(stream_mode="updates")`` instead of ``astream_events``
    because the LLM does not support function calling in streaming mode.
    Each node completion emits an update with the full messages produced.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") != "message":
                await websocket.send_json(
                    {"type": "error", "message": "Expected type 'message'"}
                )
                continue

            content = data.get("content", "")
            thread_id = data.get("thread_id") or str(uuid.uuid4())

            graph = websocket.app.state.supervisor

            try:
                async for event in graph.astream(
                    {"messages": [HumanMessage(content=content)]},
                    config={"configurable": {"thread_id": thread_id}},
                    stream_mode="updates",
                ):
                    for node_name, updates in event.items():
                        for msg in updates.get("messages", []):
                            if isinstance(msg, AIMessage):
                                # Send assistant text as a single token block
                                text = msg.content if isinstance(msg.content, str) else ""
                                if text:
                                    await websocket.send_json(
                                        {"type": "token", "content": text}
                                    )
                                # Send tool calls the model wants to make
                                for tc in (msg.tool_calls or []):
                                    await websocket.send_json(
                                        {
                                            "type": "tool_call",
                                            "name": tc["name"],
                                            "args": tc.get("args", {}),
                                        }
                                    )
                            elif isinstance(msg, ToolMessage):
                                await websocket.send_json(
                                    {
                                        "type": "tool_result",
                                        "name": msg.name or "",
                                        "content": str(msg.content),
                                    }
                                )

                await websocket.send_json(
                    {"type": "done", "thread_id": thread_id}
                )

            except Exception as exc:
                logger.exception(
                    "Streaming error for thread %s: %s", thread_id, exc
                )
                await websocket.send_json(
                    {"type": "error", "message": str(exc)}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("Unexpected WebSocket error: %s", exc)
