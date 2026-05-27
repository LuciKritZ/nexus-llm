import logging
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
logger = logging.getLogger(__name__)


@router.post("/v1/chat/completions")
async def chat_completions(
    fastapi_req: Request, payload: ChatCompletionRequest
) -> StreamingResponse:
    unloader: ModelUnloader = fastapi_req.app.state.unloader
    compressor: ContextCompressor = fastapi_req.app.state.compressor
    http_client: httpx.AsyncClient = fastapi_req.app.state.http_client
    cache: ImageCache = fastapi_req.app.state.cache
    gemini_client: GeminiClient = fastapi_req.app.state.gemini_client

    if settings.ollama_model:
        payload.model = settings.ollama_model

    await unloader.unload_if_needed(payload.model)

    has_images_in_latest = False
    latest_msg_index = len(payload.messages) - 1
    cached_hashes = {}

    for i, message in enumerate(payload.messages):
        if isinstance(message.content, str):
            message.content = compressor.compress_if_needed(message.content)
        elif isinstance(message.content, list):
            for part in message.content:
                if part.type == "text" and part.text is not None:
                    part.text = compressor.compress_if_needed(part.text)
                elif part.type == "image_url" and part.image_url is not None:
                    if i == latest_msg_index:
                        has_images_in_latest = True
                    # Cache the image locally to avoid VRAM overload and save it for inspection
                    url = part.image_url.url
                    if url.startswith("data:"):
                        try:
                            _, b64_data = url.split(",", 1)
                            import base64

                            raw_bytes = base64.b64decode(b64_data)
                            image_hash = cache.hash_image(b64_data)
                            cache.store(image_hash, raw_bytes)
                            # Track hash locally instead of mutating Pydantic model
                            cached_hashes[id(part)] = image_hash
                        except Exception:
                            pass

    if has_images_in_latest:
        logger.info("Routing request to Gemini (images detected in latest message)")
        # Route to Gemini API
        payload_dict = payload.model_dump(exclude_none=True)
        return StreamingResponse(
            gemini_client.stream_generate_content(payload_dict),
            media_type="text/event-stream",
        )

    # Standard Ollama route
    # Strip any images from older messages by converting them to text placeholders
    for message in payload.messages:
        if isinstance(message.content, list):
            for part in message.content:
                if part.type == "image_url":
                    part.type = "text"
                    image_hash = cached_hashes.get(id(part), "unknown_image")
                    part.text = f"[Image: {image_hash}]"
                    part.image_url = None

    logger.info(f"Routing request to Ollama (model: {payload.model})")
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
