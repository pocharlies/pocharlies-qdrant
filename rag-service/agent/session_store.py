"""
Redis-backed store for AgentTask metadata, steps, and logs.

Replaces the in-memory agent_tasks dict with persistent storage.
Tasks are indexed in a sorted set for chronological listing.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from redis.asyncio import Redis

from . import AgentTask

logger = logging.getLogger(__name__)

DEFAULT_TTL = 7 * 24 * 3600  # 7 days


class SessionStore:
    """Redis-backed store for agent task metadata."""

    TASK_PREFIX = "agent:task:"
    TASK_INDEX = "agent:tasks"

    def __init__(self, redis: Redis, ttl: int = DEFAULT_TTL):
        self.redis = redis
        self.ttl = ttl

    # ── Task CRUD ──

    async def create_task(self, task: AgentTask, source: str = "web") -> None:
        key = f"{self.TASK_PREFIX}{task.task_id}"
        started_ts = 0.0
        if task.started_at:
            try:
                started_ts = datetime.fromisoformat(task.started_at).timestamp()
            except (ValueError, TypeError):
                pass

        pipe = self.redis.pipeline()
        pipe.hset(key, mapping={
            "task_id": task.task_id,
            "prompt": task.prompt,
            "status": task.status,
            "started_at": task.started_at or "",
            "ended_at": task.ended_at or "",
            "model_id": task.model_id or "",
            "summary": task.summary or "",
            "error": task.error or "",
            "source": source,
            "tools_called": json.dumps(task.tools_called),
        })
        pipe.expire(key, self.ttl)
        pipe.zadd(self.TASK_INDEX, {task.task_id: started_ts})
        await pipe.execute()

    async def update_task(self, task: AgentTask) -> None:
        key = f"{self.TASK_PREFIX}{task.task_id}"
        pipe = self.redis.pipeline()
        pipe.hset(key, mapping={
            "status": task.status,
            "ended_at": task.ended_at or "",
            "model_id": task.model_id or "",
            "summary": task.summary or "",
            "error": task.error or "",
            "tools_called": json.dumps(task.tools_called),
        })
        pipe.expire(key, self.ttl)
        await pipe.execute()

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        key = f"{self.TASK_PREFIX}{task_id}"
        data = await self.redis.hgetall(key)
        if not data:
            return None
        decoded = {
            (k.decode() if isinstance(k, bytes) else k):
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in data.items()
        }
        decoded["tools_called"] = json.loads(decoded.get("tools_called", "[]"))
        decoded["step_count"] = await self.redis.llen(f"{key}:steps")
        decoded["log_count"] = await self.redis.llen(f"{key}:logs")
        return decoded

    async def list_tasks(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        task_ids = await self.redis.zrevrange(self.TASK_INDEX, offset, offset + limit - 1)
        tasks = []
        for tid_raw in task_ids:
            tid = tid_raw.decode() if isinstance(tid_raw, bytes) else tid_raw
            task_data = await self.get_task(tid)
            if task_data:
                steps = await self.get_steps(tid, last_n=50)
                task_data["steps"] = steps
                tasks.append(task_data)
            else:
                # Stale index entry — task expired from Redis
                await self.redis.zrem(self.TASK_INDEX, tid)
        return tasks

    # ── Steps ──

    async def add_step(self, task_id: str, step: Dict[str, Any]) -> None:
        key = f"{self.TASK_PREFIX}{task_id}:steps"
        pipe = self.redis.pipeline()
        pipe.rpush(key, json.dumps(step, default=str))
        pipe.expire(key, self.ttl)
        await pipe.execute()

    async def get_steps(self, task_id: str, last_n: int = 50) -> List[Dict[str, Any]]:
        key = f"{self.TASK_PREFIX}{task_id}:steps"
        raw = await self.redis.lrange(key, -last_n, -1)
        return [json.loads(item) for item in raw]

    async def get_all_steps(self, task_id: str) -> List[Dict[str, Any]]:
        key = f"{self.TASK_PREFIX}{task_id}:steps"
        raw = await self.redis.lrange(key, 0, -1)
        return [json.loads(item) for item in raw]

    # ── Logs ──

    async def add_log(self, task_id: str, entry: str) -> None:
        key = f"{self.TASK_PREFIX}{task_id}:logs"
        pipe = self.redis.pipeline()
        pipe.rpush(key, entry)
        pipe.expire(key, self.ttl)
        await pipe.execute()

    async def get_logs(self, task_id: str, offset: int = 0, limit: int = 500) -> tuple:
        key = f"{self.TASK_PREFIX}{task_id}:logs"
        total = await self.redis.llen(key)
        raw = await self.redis.lrange(key, offset, offset + limit - 1)
        entries = [(item.decode() if isinstance(item, bytes) else item) for item in raw]
        return entries, total
