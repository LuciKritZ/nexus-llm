from collections.abc import AsyncGenerator

import aiosqlite
import pytest
import pytest_asyncio

from nexus_llm.services.db import init_db


@pytest_asyncio.fixture
async def memory_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Provides a fresh in-memory SQLite connection."""
    async with aiosqlite.connect(":memory:") as db:
        yield db


@pytest.mark.asyncio
async def test_init_db_creates_table(memory_db: aiosqlite.Connection) -> None:
    """Test that init_db creates the api_keys table correctly."""
    await init_db(memory_db)

    query = "SELECT name FROM sqlite_master WHERE type='table' AND name='api_keys';"
    async with memory_db.execute(query) as cursor:
        table = await cursor.fetchone()
        assert table is not None
        assert table[0] == "api_keys"

    # Verify schema columns
    async with memory_db.execute("PRAGMA table_info(api_keys);") as cursor:
        columns = await cursor.fetchall()
        col_names = [col[1] for col in columns]

        expected_cols = [
            "id",
            "platform",
            "key_hash",
            "key_value",
            "priority",
            "try_after",
            "last_used_at",
            "meta",
        ]
        for col in expected_cols:
            assert col in col_names
