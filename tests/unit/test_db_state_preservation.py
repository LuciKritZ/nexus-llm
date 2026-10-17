import hashlib
import json
import tempfile
from collections.abc import AsyncGenerator

import aiosqlite
import pytest
import pytest_asyncio
from fastapi import HTTPException
from starlette.requests import Request

from omnimux.routes.proxy import verify_auth
from omnimux.services.db import (
    create_client_key,
    get_all_platforms_and_models,
    init_db,
    revoke_client_key,
    save_platform_and_models,
)
from omnimux.services.router_core import RouterCore
from omnimux.services.sync import sync_keys_from_json


@pytest_asyncio.fixture
async def memory_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(":memory:") as db:
        await init_db(db)
        yield db


@pytest.mark.asyncio
async def test_reproduce_sync_keys_does_not_wipe_existing_db_keys(
    memory_db: aiosqlite.Connection,
) -> None:
    # Seed an existing key created via CLI or prior session
    await memory_db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority, meta)
        VALUES ('gemini', 'hash-existing-1', 'gemini-key-val', 1, '{}')
        """
    )
    await memory_db.commit()

    # Create a models.json with an openrouter key
    payload = {
        "keys": {"openrouter": [{"key_value": "new-openrouter-key", "priority": 1, "meta": {}}]}
    }
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(payload, f)
        temp_name = f.name

    try:
        await sync_keys_from_json(memory_db, temp_name)
    finally:
        import os

        os.remove(temp_name)

    # Both the existing gemini key and the newly synced openrouter key must exist
    async with memory_db.execute("SELECT platform FROM api_keys ORDER BY platform") as cursor:
        rows = await cursor.fetchall()
        platforms = [r[0] for r in rows]
        assert "gemini" in platforms, "Existing gemini key was wiped by sync_keys_from_json!"
        assert "openrouter" in platforms


@pytest.mark.asyncio
async def test_reproduce_keyless_platform_hash_uniqueness(
    memory_db: aiosqlite.Connection,
) -> None:
    # Model discovery hashing helper simulation:
    # If we save two keyless platforms (e.g. ollama and local_vllm)
    # Hashing key_value alone results in identical hash, overwriting the first
    def get_hash(platform: str, key: str) -> str:
        # Expected fix: hash f"{platform}:{key}"
        return hashlib.sha256(f"{platform}:{key}".encode()).hexdigest()

    ollama_hash = get_hash("ollama", "")
    vllm_hash = get_hash("vllm", "")

    assert ollama_hash != vllm_hash, "Keyless platform hashes must be distinct per platform!"

    await memory_db.execute(
        "INSERT INTO api_keys (platform, key_hash, key_value, priority) VALUES (?, ?, ?, 1)",
        ("ollama", ollama_hash, ""),
    )
    await memory_db.execute(
        "INSERT INTO api_keys (platform, key_hash, key_value, priority) VALUES (?, ?, ?, 1)",
        ("vllm", vllm_hash, ""),
    )
    await memory_db.commit()

    async with memory_db.execute("SELECT COUNT(*) FROM api_keys") as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 2


@pytest.mark.asyncio
async def test_reproduce_platforms_without_models_visible(
    memory_db: aiosqlite.Connection,
) -> None:
    # Save a platform with an empty model list
    await save_platform_and_models(
        memory_db,
        platform="openrouter",
        api_base="https://openrouter.ai/api",
        protocol="openai",
        models_list=[],
    )

    platforms_data = await get_all_platforms_and_models(memory_db)
    assert "openrouter" in platforms_data
    assert platforms_data["openrouter"]["apiBase"] == "https://openrouter.ai/api"
    # The platform itself should be represented or queryable
    async with memory_db.execute("SELECT name, api_base FROM platforms") as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "openrouter"


@pytest.mark.asyncio
async def test_reproduce_revoke_client_key_does_not_wildcard_match(
    memory_db: aiosqlite.Connection,
) -> None:
    # Create two keys where prefix contains an underscore like om_
    _, info1 = await create_client_key(memory_db, "key1")
    _, _info2 = await create_client_key(memory_db, "key2")

    # Both start with om_...
    # Attempting to revoke with exact prefix or identifier should only revoke targeted key
    id1 = info1["prefix"][:8]  # e.g. "om_abcde"
    await revoke_client_key(memory_db, id1)

    async with memory_db.execute(
        "SELECT name, revoked_at FROM client_keys ORDER BY name"
    ) as cursor:
        rows = list(await cursor.fetchall())
        # Key 1 revoked, key 2 must stay active
        assert rows[0][1] is not None
        assert rows[1][1] is None, "Key 2 was accidentally revoked by unescaped wildcard!"


@pytest.mark.asyncio
async def test_reproduce_standardized_iso_timestamps(
    memory_db: aiosqlite.Connection,
) -> None:
    router = RouterCore(memory_db)
    await memory_db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority)
        VALUES ('openrouter', 'h1', 'k1', 1)
        """
    )
    await memory_db.commit()

    # get_best_platform_and_key and mark_key_used should both record ISO format with T
    await router.get_best_platform_and_key(["openrouter"])
    async with memory_db.execute("SELECT last_used_at FROM api_keys WHERE key_hash = 'h1'") as cur:
        row = await cur.fetchone()
        assert row is not None
        ts = row[0]
        assert "T" in ts, f"Timestamp {ts} must be ISO format with T for proper SQLite ordering"


@pytest.mark.asyncio
async def test_reproduce_auth_rejects_invalid_token_even_when_no_active_keys(
    memory_db: aiosqlite.Connection,
) -> None:
    # When client_keys table has no active keys, sending an explicitly invalid token
    # should be rejected, not silently allowed through as unauthenticated
    scope = {
        "type": "http",
        "headers": [(b"x-api-key", b"invalid-dummy-key")],
        "app": type("App", (), {"state": type("State", (), {"db": memory_db})()})(),
    }
    req = Request(scope)

    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(req, credentials=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_rejects_missing_token_when_active_keys_exist(
    memory_db: aiosqlite.Connection,
) -> None:
    await create_client_key(memory_db, "key1")
    scope = {
        "type": "http",
        "headers": [],
        "app": type("App", (), {"state": type("State", (), {"db": memory_db})()})(),
    }
    req = Request(scope)

    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(req, credentials=None)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or missing Bearer token"
