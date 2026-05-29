import asyncio
import datetime
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from nexus_llm.exceptions import NexusLLMError

logger = logging.getLogger(__name__)


class NoKeysAvailableError(NexusLLMError):
    """Raised when no API keys are available (all exhausted/cooldown)."""


class RouterCore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db
        # Dictionary to hold per-key locks: {key_hash: asyncio.Lock()}
        self._locks: dict[str, asyncio.Lock] = {}
        # In-memory cooldown cache to avoid hitting DB constantly for expired keys
        self._cooldowns: dict[str, float] = {}

    def _get_lock(self, key_hash: str) -> asyncio.Lock:
        if key_hash not in self._locks:
            self._locks[key_hash] = asyncio.Lock()
        return self._locks[key_hash]

    async def get_next_key(self, platform: str) -> dict[str, Any]:
        """
        Retrieves the best available key for a given platform.
        Prioritizes highest `priority`, then oldest `last_used_at`.
        Skips keys currently in cooldown (try_after).
        """
        now = datetime.datetime.now(datetime.UTC).timestamp()

        async with self.db.execute(
            """
            SELECT key_hash, key_value, priority, try_after, last_used_at, meta
            FROM api_keys
            WHERE platform = ?
            ORDER BY priority DESC, last_used_at ASC
            """,
            (platform,),
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            raise NoKeysAvailableError(f"No keys configured for platform: {platform}")

        for row in rows:
            key_hash, key_value, priority, try_after, _, meta = row

            # Check in-memory fast path cooldown
            if key_hash in self._cooldowns and self._cooldowns[key_hash] > now:
                continue

            # DB level cooldown check
            if try_after:
                try_after_ts = datetime.datetime.fromisoformat(try_after).timestamp()
                if try_after_ts > now:
                    self._cooldowns[key_hash] = try_after_ts
                    continue
                else:
                    # Cooldown expired, clear it
                    self._cooldowns.pop(key_hash, None)
                    await self.db.execute(
                        "UPDATE api_keys SET try_after = NULL WHERE key_hash = ?", (key_hash,)
                    )
                    await self.db.commit()

            # If we get here, the key is not in cooldown.
            # Check if the lock is currently held. Skip to next to maximize concurrency.
            lock = self._get_lock(key_hash)
            if lock.locked():
                continue

            return {
                "key_hash": key_hash,
                "key_value": key_value,
                "priority": priority,
                "meta": meta,
            }

        # Second pass: if all unlocked keys are in cooldown, can we wait on a locked key?
        # We return the first non-cooldown key, even if locked.
        for row in rows:
            key_hash, key_value, priority, try_after, _, meta = row
            if key_hash in self._cooldowns and self._cooldowns[key_hash] > now:
                continue

            return {
                "key_hash": key_hash,
                "key_value": key_value,
                "priority": priority,
                "meta": meta,
            }

        raise NoKeysAvailableError(
            f"All keys for {platform} are currently exhausted or in cooldown."
        )

    @asynccontextmanager
    async def use_key(self, key_hash: str) -> AsyncGenerator[None, None]:
        """Context manager to lock a key during usage and update its last_used_at."""
        lock = self._get_lock(key_hash)
        async with lock:
            try:
                yield
            finally:
                await self.mark_key_used(key_hash)

    async def mark_key_exhausted(self, key_hash: str, duration_seconds: float) -> None:
        """Puts a key in cooldown for a specified duration."""
        now = datetime.datetime.now(datetime.UTC)
        try_after = now + datetime.timedelta(seconds=duration_seconds)
        self._cooldowns[key_hash] = try_after.timestamp()

        await self.db.execute(
            "UPDATE api_keys SET try_after = ? WHERE key_hash = ?",
            (try_after.isoformat(), key_hash),
        )
        await self.db.commit()
        logger.warning(f"Key {key_hash} exhausted. Cooldown until {try_after.isoformat()}")

    async def mark_key_used(self, key_hash: str) -> None:
        """Updates the last_used_at timestamp for a key."""
        now = datetime.datetime.now(datetime.UTC)
        await self.db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
            (now.isoformat(), key_hash),
        )
        await self.db.commit()
