import typing
from unittest.mock import patch

import pytest

from nexus_llm.config import settings


@pytest.fixture(autouse=True)
def mock_settings_proxy_password() -> typing.Generator[None, None, None]:
    with patch.object(settings, "proxy_password", None):
        yield
