from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nexus_llm.services.gatekeeper import Gatekeeper


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock(spec=httpx.AsyncClient)


@pytest.mark.asyncio
async def test_classify_empty_payload(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    result = await gatekeeper.classify({})
    assert result == "simple"


@pytest.mark.asyncio
async def test_classify_heuristic_title_generation(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {
        "messages": [{"role": "user", "content": "Please generate a title for this snippet"}]
    }
    result = await gatekeeper.classify(payload)
    assert result == "simple"


@pytest.mark.asyncio
async def test_classify_heuristic_ide_fingerprint(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    # A massive system prompt triggers complex
    payload = {
        "messages": [{"role": "system", "content": "A" * 4000}, {"role": "user", "content": "Hi"}]
    }
    result = await gatekeeper.classify(payload)
    assert result == "complex"


@pytest.mark.asyncio
async def test_classify_heuristic_huge_context(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": "B" * 9000}]}
    result = await gatekeeper.classify(payload)
    assert result == "complex"


@pytest.mark.asyncio
async def test_classify_heuristic_very_short(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    result = await gatekeeper.classify(payload)
    assert result == "simple"


@pytest.mark.asyncio
async def test_classify_llm_judge_complex(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "How do I build a scalable microservice architecture? " * 10,
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "complex"}
    http_client.post = AsyncMock(return_value=mock_response)

    result = await gatekeeper.classify(payload)
    assert result == "complex"
    http_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_classify_llm_judge_simple(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": "What is the capital of France? " * 10}]}

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "simple task"}
    http_client.post = AsyncMock(return_value=mock_response)

    result = await gatekeeper.classify(payload)
    assert result == "simple"


@pytest.mark.asyncio
async def test_classify_llm_judge_timeout(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": "Normal question " * 20}]}

    http_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

    result = await gatekeeper.classify(payload)
    # Default to complex on timeout
    assert result == "complex"


@pytest.mark.asyncio
async def test_classify_llm_judge_error(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": "Normal question " * 20}]}

    http_client.post = AsyncMock(side_effect=Exception("Generic error"))

    result = await gatekeeper.classify(payload)
    # Default to complex on error
    assert result == "complex"
