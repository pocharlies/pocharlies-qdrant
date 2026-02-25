"""Tasks API — list and inspect agent tasks and their logs."""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from state.database import async_session
from state.models import Task, TaskLog

logger = logging.getLogger("agent.api.tasks")

router = APIRouter(prefix="/api/tasks", tags=["Tasks"])


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class TaskLogResponse(BaseModel):
    """Single log entry for a task."""

    id: int
    task_id: UUID
    level: str
    message: str
    data: Any | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class TaskResponse(BaseModel):
    """Task record returned by the API."""

    id: UUID
    thread_id: str
    trigger: str
    trigger_ref: str | None = None
    status: str
    prompt: str
    summary: str | None = None
    tools_used: list[Any] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}

    @classmethod
    def from_task(cls, task: Task) -> "TaskResponse":
        return cls(
            id=task.id,
            thread_id=task.thread_id,
            trigger=task.trigger,
            trigger_ref=task.trigger_ref,
            status=task.status,
            prompt=task.prompt,
            summary=task.summary,
            tools_used=task.tools_used or [],
            started_at=task.started_at,
            ended_at=task.ended_at,
            error=task.error,
            metadata=task.metadata_ or {},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[TaskResponse], summary="List tasks")
async def list_tasks(
    status: str | None = Query(None, description="Filter by task status"),
    limit: int = Query(50, ge=1, le=500, description="Max number of tasks to return"),
) -> list[TaskResponse]:
    """Return tasks ordered by started_at descending, with optional status filter."""
    async with async_session() as session:
        stmt = select(Task).order_by(Task.started_at.desc()).limit(limit)
        if status:
            stmt = stmt.where(Task.status == status)
        result = await session.execute(stmt)
        tasks = result.scalars().all()
    return [TaskResponse.from_task(t) for t in tasks]


@router.get("/{task_id}", response_model=TaskResponse, summary="Get task by ID")
async def get_task(task_id: UUID) -> TaskResponse:
    """Return a single task by its UUID."""
    async with async_session() as session:
        stmt = select(Task).where(Task.id == task_id)
        result = await session.execute(stmt)
        task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse.from_task(task)


@router.get(
    "/{task_id}/logs",
    response_model=list[TaskLogResponse],
    summary="Get task logs",
)
async def get_task_logs(task_id: UUID) -> list[TaskLogResponse]:
    """Return log entries for a task, ordered by created_at ascending."""
    async with async_session() as session:
        # Verify the task exists
        task_result = await session.execute(select(Task).where(Task.id == task_id))
        if task_result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        stmt = (
            select(TaskLog)
            .where(TaskLog.task_id == task_id)
            .order_by(TaskLog.created_at.asc())
        )
        result = await session.execute(stmt)
        logs = result.scalars().all()
    return [TaskLogResponse.model_validate(log) for log in logs]
