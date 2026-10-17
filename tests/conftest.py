import typing
from unittest.mock import patch

import pytest

from omnimux.config import settings


@pytest.fixture(autouse=True)
def mock_settings_proxy_password() -> typing.Generator[None, None, None]:
    with patch.object(settings, "proxy_password", None):
        yield


@pytest.fixture(autouse=True)
def mock_settings_sqlite_db(
    tmp_path: typing.Any,
) -> typing.Generator[None, None, None]:
    test_db = str(tmp_path / "test_omnimux.db")
    with patch.object(settings, "sqlite_db_path", test_db):
        yield
