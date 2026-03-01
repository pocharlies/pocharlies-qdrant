"""
Sync State + Content Hash Dedup (Redis-backed).
Tracks catalog sync operations and deduplicates content to avoid re-embedding unchanged items.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SyncStateStore:
    """Track catalog sync operations in Redis."""

    PREFIX = "catalog:sync"

    def __init__(self, redis_client):
        self.redis = redis_client

    async def create_sync(self, sync_type: str = "full") -> str:
        """Create a new sync record. Returns sync_id."""
        sync_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "sync_id": sync_id,
            "sync_type": sync_type,
            "status": "running",
            "items_processed": 0,
            "items_skipped": 0,
            "errors": "[]",
            "started_at": now,
            "ended_at": "",
            "cursor": "",
        }
        key = f"{self.PREFIX}:{sync_id}"
        await self.redis.hset(key, mapping=data)
        await self.redis.expire(key, 86400 * 7)  # 7 day TTL

        # Track in ordered list for history
        await self.redis.zadd(f"{self.PREFIX}:history", {sync_id: datetime.now(timezone.utc).timestamp()})
        await self.redis.zremrangebyrank(f"{self.PREFIX}:history", 0, -51)  # keep last 50

        logger.info(f"Created sync {sync_id} (type={sync_type})")
        return sync_id

    async def update_sync(self, sync_id: str, **kwargs) -> None:
        """Update sync record fields."""
        key = f"{self.PREFIX}:{sync_id}"
        updates = {}
        for k, v in kwargs.items():
            if isinstance(v, (list, dict)):
                updates[k] = json.dumps(v)
            else:
                updates[k] = str(v) if v is not None else ""
        if updates:
            await self.redis.hset(key, mapping=updates)

    async def complete_sync(self, sync_id: str, cursor: str = "", error: str = "") -> None:
        """Mark sync as completed or failed."""
        now = datetime.now(timezone.utc).isoformat()
        status = "failed" if error else "completed"
        updates = {"status": status, "ended_at": now}
        if cursor:
            updates["cursor"] = cursor
        if error:
            updates["error"] = error
        await self.update_sync(sync_id, **updates)
        logger.info(f"Sync {sync_id} {status}")

    async def get_sync(self, sync_id: str) -> Optional[Dict[str, Any]]:
        """Get sync record."""
        key = f"{self.PREFIX}:{sync_id}"
        data = await self.redis.hgetall(key)
        if not data:
            return None
        return {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in data.items()}

    async def get_last_cursor(self) -> Optional[str]:
        """Get the cursor (timestamp) from the most recent completed sync."""
        sync_ids = await self.redis.zrevrange(f"{self.PREFIX}:history", 0, 10)
        for sid in sync_ids:
            sid_str = sid.decode() if isinstance(sid, bytes) else sid
            sync = await self.get_sync(sid_str)
            if sync and sync.get("status") == "completed" and sync.get("cursor"):
                return sync["cursor"]
        return None

    async def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sync history."""
        sync_ids = await self.redis.zrevrange(f"{self.PREFIX}:history", 0, limit - 1)
        history = []
        for sid in sync_ids:
            sid_str = sid.decode() if isinstance(sid, bytes) else sid
            sync = await self.get_sync(sid_str)
            if sync:
                history.append(sync)
        return history


class ContentHashStore:
    """Track content hashes in Redis to skip re-embedding unchanged items."""

    PREFIX = "catalog:hash"

    def __init__(self, redis_client):
        self.redis = redis_client

    @staticmethod
    def compute_hash(content: str) -> str:
        """SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def has_changed(self, item_key: str, content: str) -> bool:
        """Check if content has changed since last index. Returns True if changed or new."""
        new_hash = self.compute_hash(content)
        redis_key = f"{self.PREFIX}:{item_key}"
        old_hash = await self.redis.get(redis_key)
        if old_hash:
            old_hash = old_hash.decode() if isinstance(old_hash, bytes) else old_hash
        return old_hash != new_hash

    async def set_hash(self, item_key: str, content: str) -> None:
        """Store content hash after successful indexing."""
        content_hash = self.compute_hash(content)
        redis_key = f"{self.PREFIX}:{item_key}"
        await self.redis.set(redis_key, content_hash)

    async def delete_hash(self, item_key: str) -> None:
        """Remove content hash (e.g. on product delete)."""
        await self.redis.delete(f"{self.PREFIX}:{item_key}")
