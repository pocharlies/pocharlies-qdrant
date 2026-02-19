"""
Redis-backed session for the OpenAI Agents SDK.

Implements SessionABC so the Runner automatically loads conversation
history before each run and stores new items after each run.
This enables multi-turn conversation resume.
"""

import json
import logging
from typing import List, Optional

from redis.asyncio import Redis
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem

logger = logging.getLogger(__name__)

DEFAULT_TTL = 7 * 24 * 3600  # 7 days


class RedisSession(SessionABC):
    """OpenAI Agents SDK Session backed by Redis lists."""

    KEY_PREFIX = "agent:session:"

    def __init__(self, session_id: str, redis: Redis, ttl: int = DEFAULT_TTL):
        self.session_id = session_id
        self.redis = redis
        self.ttl = ttl
        self._key = f"{self.KEY_PREFIX}{session_id}"

    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        raw = await self.redis.lrange(self._key, 0, -1)
        items = [json.loads(item) for item in raw]
        if limit is not None:
            items = items[-limit:]
        return items

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        if not items:
            return
        pipe = self.redis.pipeline()
        for item in items:
            pipe.rpush(self._key, json.dumps(item, default=str))
        pipe.expire(self._key, self.ttl)
        await pipe.execute()

    async def pop_item(self) -> Optional[TResponseInputItem]:
        raw = await self.redis.rpop(self._key)
        if raw is None:
            return None
        return json.loads(raw)

    async def clear_session(self) -> None:
        await self.redis.delete(self._key)
