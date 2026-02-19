"""
Agent runner — executes agent tasks with streaming event tracking.
Used by both FastAPI (background task) and CLI.
"""

import logging
import uuid
from datetime import datetime, timezone

from agents import Runner, ItemHelpers

from . import AgentTask, AgentServices

logger = logging.getLogger(__name__)


async def run_task(
    agent,
    services: AgentServices,
    prompt: str,
    max_turns: int = 30,
    task: AgentTask = None,
    session=None,
    session_store=None,
) -> AgentTask:
    """Run an agent task with streaming event tracking.

    Creates an AgentTask (or uses a pre-created one), runs the agent with
    streaming, and populates the task with step/log data as events arrive.
    Returns the completed task.

    Args:
        agent: The Agent instance to run.
        services: AgentServices context (passed to tools).
        prompt: The user prompt.
        max_turns: Maximum agent iterations (default: 30).
        task: Optional pre-created AgentTask (for FastAPI integration where
              the task must be registered before the coroutine starts).
        session: Optional RedisSession for conversation history persistence.
        session_store: Optional SessionStore for task metadata persistence.
    """
    if task is None:
        task = AgentTask(
            task_id=uuid.uuid4().hex[:12],
            prompt=prompt,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
    services.task = task

    # Discover model_id from agent
    try:
        model = agent.model
        task.model_id = getattr(model, 'model', None) or str(model)
    except Exception:
        task.model_id = "unknown"

    task.log(f"Starting with model: {task.model_id}")
    task.add_step("thinking", f"Planning approach for: {prompt}")

    # Persist initial state
    if session_store:
        try:
            await session_store.update_task(task)
        except Exception as e:
            logger.warning(f"Redis update_task failed: {e}")

    try:
        run_kwargs = dict(
            input=prompt,
            context=services,
            max_turns=max_turns,
        )
        if session:
            run_kwargs["session"] = session

        result = Runner.run_streamed(agent, **run_kwargs)

        async for event in result.stream_events():
            if event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    # raw_item may be a ResponseFunctionToolCall object, not a dict
                    raw = getattr(item, 'raw_item', None)
                    tool_name = (
                        getattr(item, 'name', None)
                        or getattr(raw, 'name', None)
                        or (raw.get('name') if isinstance(raw, dict) else None)
                        or 'unknown'
                    )
                    task.add_step("tool_call", str(tool_name))
                    task.log(f"Tool call: {tool_name}")
                    if tool_name not in task.tools_called:
                        task.tools_called.append(tool_name)
                elif item.type == "tool_call_output_item":
                    output = str(getattr(item, 'output', ''))[:2000]
                    task.add_step("tool_result", output)
                elif item.type == "message_output_item":
                    text = ItemHelpers.text_message_output(item)
                    task.add_step("response", text)
                    task.log(f"Agent response: {text[:300]}")

        task.summary = result.final_output
        task.status = "completed"
        task.log(f"COMPLETED: {task.summary[:300] if task.summary else 'No summary'}")

    except Exception as e:
        task.status = "failed"
        task.error = str(e)[:500]
        task.log(f"FAILED: {str(e)[:400]}")
        logger.error(f"Agent task {task.task_id} failed: {e}", exc_info=True)

    task.ended_at = datetime.now(timezone.utc).isoformat()
    task.log(f"Task ended: {task.status}")

    # Persist final state
    if session_store:
        try:
            await session_store.update_task(task)
        except Exception as e:
            logger.warning(f"Redis final update_task failed: {e}")

    return task
