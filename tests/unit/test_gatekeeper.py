from unittest.mock import MagicMock

import httpx
import pytest

from nexus_llm.services.gatekeeper import Gatekeeper


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock(spec=httpx.AsyncClient)


@pytest.mark.asyncio
async def test_profile_empty_payload(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    result = await gatekeeper.profile_request({})
    assert result == {"context_length": 0, "has_image": False}


@pytest.mark.asyncio
async def test_profile_context_length(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": "Hello world!"}]}
    result = await gatekeeper.profile_request(payload)
    assert result == {"context_length": 12}
