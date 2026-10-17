from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import httpx
import pytest
import pytest_asyncio

from omnimux.services.db import init_db
from omnimux.services.model_discovery import (
    fetch_upstream_models,
    list_configured_models,
    run_add_model_wizard,
)


@pytest_asyncio.fixture
async def memory_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(":memory:") as db:
        await init_db(db)
        yield db


@pytest.mark.asyncio
async def test_fetch_upstream_models_success() -> None:
    fake_response = httpx.Response(
        status_code=200,
        request=httpx.Request("GET", "https://api.test/v1/models"),
        json={"data": [{"id": "m1", "name": "Model 1", "context_length": 64000}, {"id": "m2"}]},
    )
    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=fake_response)):
        models = await fetch_upstream_models("https://api.test", "key-123", "openai")
        assert len(models) == 2
        assert models[0]["id"] == "m1"
        assert models[0]["max_input_tokens"] == 64000


@pytest.mark.asyncio
async def test_fetch_upstream_models_anthropic_and_list() -> None:
    fake_response = httpx.Response(
        status_code=200,
        request=httpx.Request("GET", "https://api.test/v1/models"),
        json=["model-a", "model-b"],
    )
    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=fake_response)):
        models = await fetch_upstream_models("https://api.test/v1", "key-123", "anthropic")
        assert len(models) == 2
        assert models[0]["id"] == "model-a"


@pytest.mark.asyncio
async def test_fetch_upstream_models_error() -> None:
    fake_response = httpx.Response(
        status_code=401,
        request=httpx.Request("GET", "https://api.test/v1/models"),
        text="Unauthorized",
    )
    with (
        patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=fake_response)),
        pytest.raises(RuntimeError),
    ):
        await fetch_upstream_models("https://api.test", "bad-key")


@pytest.mark.asyncio
async def test_run_add_model_wizard_success(memory_db: aiosqlite.Connection) -> None:
    mock_models = [{"id": "m1", "name": "Model 1", "max_input_tokens": 128000}]
    with (
        patch(
            "omnimux.services.model_discovery.fetch_upstream_models",
            AsyncMock(return_value=mock_models),
        ),
        patch("builtins.input", side_effect=["testplat", "1", "https://api.test", "testkey"]),
        patch(
            "omnimux.services.model_discovery.run_checklist",
            MagicMock(return_value=["m1"]),
        ),
    ):
        await run_add_model_wizard(memory_db)

    async with memory_db.execute("SELECT model_id FROM models WHERE platform = 'testplat'") as cur:
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == "m1"


@pytest.mark.asyncio
async def test_run_add_model_wizard_manual_fallback(memory_db: aiosqlite.Connection) -> None:
    with (
        patch(
            "omnimux.services.model_discovery.fetch_upstream_models",
            AsyncMock(side_effect=Exception("Connection refused")),
        ),
        patch(
            "builtins.input",
            side_effect=["errplat", "2", "https://api.test", "key", "manual-model"],
        ),
        patch(
            "omnimux.services.model_discovery.run_checklist",
            MagicMock(return_value=["manual-model"]),
        ),
    ):
        await run_add_model_wizard(memory_db)

    async with memory_db.execute("SELECT model_id FROM models WHERE platform = 'errplat'") as cur:
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == "manual-model"


@pytest.mark.asyncio
async def test_run_add_model_wizard_abort_empty(memory_db: aiosqlite.Connection) -> None:
    with (
        patch(
            "omnimux.services.model_discovery.fetch_upstream_models",
            AsyncMock(return_value=[]),
        ),
        patch("builtins.input", side_effect=["emptyplat", "1", "https://api.test", "key"]),
    ):
        await run_add_model_wizard(memory_db)

    async with memory_db.execute("SELECT COUNT(*) FROM models") as cur:
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == 0


@pytest.mark.asyncio
async def test_list_configured_models(
    memory_db: aiosqlite.Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Empty
    await list_configured_models(memory_db)
    assert "No models configured yet" in capsys.readouterr().out

    # 2. Populated
    from omnimux.services.db import save_platform_and_models

    await save_platform_and_models(
        memory_db, "openrouter", "https://api", "openai", [{"id": "gpt-4"}]
    )
    await list_configured_models(memory_db)
    assert "openrouter/gpt-4" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_fetch_upstream_models_unexpected_json() -> None:
    fake_response = httpx.Response(
        status_code=200,
        request=httpx.Request("GET", "https://api.test/v1/models"),
        json=12345,
    )
    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=fake_response)):
        models = await fetch_upstream_models("https://api.test", "key")
        assert models == []


@pytest.mark.asyncio
async def test_run_add_model_wizard_manual_abort(memory_db: aiosqlite.Connection) -> None:
    with (
        patch(
            "omnimux.services.model_discovery.fetch_upstream_models",
            AsyncMock(side_effect=Exception("Connection refused")),
        ),
        patch(
            "builtins.input",
            side_effect=["errplat", "2", "https://api.test", "key", ""],
        ),
    ):
        await run_add_model_wizard(memory_db)


@pytest.mark.asyncio
async def test_run_add_model_wizard_none_selected(memory_db: aiosqlite.Connection) -> None:
    mock_models = [{"id": "m1", "name": "Model 1"}]
    with (
        patch(
            "omnimux.services.model_discovery.fetch_upstream_models",
            AsyncMock(return_value=mock_models),
        ),
        patch("builtins.input", side_effect=["plat", "1", "https://api.test", "key"]),
        patch(
            "omnimux.services.model_discovery.run_checklist",
            MagicMock(return_value=[]),
        ),
    ):
        await run_add_model_wizard(memory_db)
