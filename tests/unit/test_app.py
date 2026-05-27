import pytest
from fastapi.testclient import TestClient

from nexus_llm.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)

def test_chat_completions_skeleton(client: TestClient) -> None:
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Hi"}]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.json() == {"message": "Skeleton for /v1/chat/completions"}
