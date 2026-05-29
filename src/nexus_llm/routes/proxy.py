import logging
import typing

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from nexus_llm.config import settings
from nexus_llm.models.schemas import ChatCompletionRequest
from nexus_llm.services.cache import ImageCache
from nexus_llm.services.compressor import ContextCompressor
from nexus_llm.services.gatekeeper import Gatekeeper
from nexus_llm.services.multiplexer import Multiplexer
from nexus_llm.services.unloader import ModelUnloader

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/chat/completions")
async def chat_completions(
    fastapi_req: Request, payload: ChatCompletionRequest
) -> StreamingResponse:
    unloader: ModelUnloader = fastapi_req.app.state.unloader
    compressor: ContextCompressor = fastapi_req.app.state.compressor
    _http_client: httpx.AsyncClient = fastapi_req.app.state.http_client
    cache: ImageCache = fastapi_req.app.state.cache
    gatekeeper: Gatekeeper = fastapi_req.app.state.gatekeeper
    multiplexer: Multiplexer = fastapi_req.app.state.multiplexer

    if settings.ollama_model:
        payload.model = settings.ollama_model

    await unloader.unload_if_needed(payload.model)

    # Image caching and context compression logic
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
                    url = part.image_url.url
                    if url.startswith("data:"):
                        try:
                            _, b64_data = url.split(",", 1)
                            import base64

                            raw_bytes = base64.b64decode(b64_data)
                            image_hash = cache.hash_image(b64_data)
                            cache.store(image_hash, raw_bytes)
                            cached_hashes[id(part)] = image_hash
                        except Exception:
                            pass

                    if i != latest_msg_index:
                        # Strip images from older messages
                        part.type = "text"
                        image_hash = cached_hashes.get(id(part), "unknown_image")
                        part.text = f"[Image: {image_hash}]"
                        part.image_url = None

    payload_dict = payload.model_dump(exclude_none=True)

    # Virtual configuration mapping
    platform = "openrouter"
    target_model = payload.model

    if payload.model == "nexus-auto" or payload.model == "auto":
        complexity = await gatekeeper.classify(payload_dict)
        logger.info(f"Gatekeeper classified request as {complexity}")
        if complexity == "complex":
            platform = "openrouter"
            target_model = "anthropic/claude-3-opus"
        else:
            platform = "openrouter"
            target_model = "google/gemini-2.0-flash-001"

    if has_images_in_latest:
        logger.info("Images detected, forcing gemini vision capability")
        platform = "gemini"
        target_model = settings.gemini_model

    # Use multiplexer for the actual request
    logger.info(f"Routing request to {platform} (model: {target_model})")

    async def sse_wrapper() -> typing.AsyncGenerator[bytes, None]:
        import json

        async for chunk in multiplexer.generate_stream(
            platform=platform,
            model=target_model,
            messages=payload_dict.get("messages", []),
            **payload_dict.get("kwargs", {}),
        ):
            openai_chunk = {
                "id": f"chatcmpl-{platform}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": chunk}}],
            }
            yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        sse_wrapper(),
        media_type="text/event-stream",
    )
