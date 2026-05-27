from nexus_llm.models.schemas import ChatCompletionRequest


def test_chat_completion_request_text() -> None:
    payload = {
        "model": "qwen3.5",
        "messages": [
            {"role": "user", "content": "Hello world!"}
        ]
    }
    request = ChatCompletionRequest.model_validate(payload)
    assert request.model == "qwen3.5"
    assert len(request.messages) == 1
    assert request.messages[0].content == "Hello world!"

def test_chat_completion_request_multimodal() -> None:
    payload = {
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,12345"}}
                ]
            }
        ]
    }
    request = ChatCompletionRequest.model_validate(payload)
    content = request.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0].type == "text"
    assert content[1].type == "image_url"
    assert content[1].image_url is not None
    assert content[1].image_url.url == "data:image/jpeg;base64,12345"

def test_chat_completion_extra_fields_preserved() -> None:
    payload = {
        "model": "qwen3.5",
        "messages": [],
        "some_unknown_field": "test_value"
    }
    request = ChatCompletionRequest.model_validate(payload)
    assert request.model_extra is not None
    assert request.model_extra["some_unknown_field"] == "test_value"
