import typing

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from nexus_llm.config import settings
from nexus_llm.models.schemas import ChatCompletionRequest
from nexus_llm.services.cache import ImageCache
from nexus_llm.services.compressor import ContextCompressor
from nexus_llm.services.gemini_client import GeminiClient
from nexus_llm.services.unloader import ModelUnloader

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    fastapi_req: Request, payload: ChatCompletionRequest
) -> StreamingResponse:
    unloader: ModelUnloader = fastapi_req.app.state.unloader
    compressor: ContextCompressor = fastapi_req.app.state.compressor
    http_client: httpx.AsyncClient = fastapi_req.app.state.http_client
    cache: ImageCache = fastapi_req.app.state.cache
    gemini_client: GeminiClient = fastapi_req.app.state.gemini_client

    await unloader.unload_if_needed(payload.model)

    has_images = False
    for message in payload.messages:
        if isinstance(message.content, str):
            message.content = compressor.compress_if_needed(message.content)
        elif isinstance(message.content, list):
            for part in message.content:
                if part.type == "text" and part.text is not None:
                    part.text = compressor.compress_if_needed(part.text)
                elif part.type == "image_url" and part.image_url is not None:
                    has_images = True
                    # Cache the image locally to avoid VRAM overload and save it for inspection
                    # Part format for image_url: "data:image/jpeg;base64,xxxx"
                    url = part.image_url.url
                    if url.startswith("data:"):
                        try:
                            # Split header and base64
                            _, b64_data = url.split(",", 1)
                            # Try to decode base64 and store it
                            import base64

                            raw_bytes = base64.b64decode(b64_data)
                            image_hash = cache.hash_image(b64_data)
                            cache.store(image_hash, raw_bytes)
                        except Exception:
                            # If cache fails or base64 decoding fails,
                            # it's safer to let the external API fail
                            pass

    if has_images:
        # Route to Gemini API
        payload_dict = payload.model_dump(exclude_none=True)
        return StreamingResponse(
            gemini_client.stream_generate_content(payload_dict),
            media_type="text/event-stream",
        )

    # Standard Ollama route
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
