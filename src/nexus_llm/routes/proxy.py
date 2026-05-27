import typing

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from nexus_llm.config import settings
from nexus_llm.models.schemas import ChatCompletionRequest
from nexus_llm.services.compressor import ContextCompressor
from nexus_llm.services.unloader import ModelUnloader

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    fastapi_req: Request, payload: ChatCompletionRequest
) -> StreamingResponse:
    unloader: ModelUnloader = fastapi_req.app.state.unloader
    compressor: ContextCompressor = fastapi_req.app.state.compressor
    http_client: httpx.AsyncClient = fastapi_req.app.state.http_client

    await unloader.unload_if_needed(payload.model)

    for message in payload.messages:
        if isinstance(message.content, str):
            message.content = compressor.compress_if_needed(message.content)
        elif isinstance(message.content, list):
            for part in message.content:
                if part.type == "text" and part.text is not None:
                    part.text = compressor.compress_if_needed(part.text)

    target_url = f"{settings.ollama_base_url}/v1/chat/completions"

    req = http_client.build_request("POST", target_url, json=payload.model_dump(exclude_none=True))
    response = await http_client.send(req, stream=True)

    async def stream_generator() -> typing.AsyncGenerator[bytes, None]:
        async for chunk in response.aiter_bytes():
            yield chunk
        await response.aclose()

    return StreamingResponse(
        stream_generator(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )
