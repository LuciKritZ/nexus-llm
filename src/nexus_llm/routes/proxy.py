import base64
import json
import logging
import typing

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from nexus_llm.config import settings
from nexus_llm.models.schemas import ChatCompletionRequest
from nexus_llm.services.cache import ImageCache
from nexus_llm.services.compressor import ContextCompressor
from nexus_llm.services.gatekeeper import Gatekeeper
from nexus_llm.services.multiplexer import Multiplexer
from nexus_llm.services.unloader import ModelUnloader

router = APIRouter()
logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def verify_auth(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> None:  # noqa: B008
    if settings.proxy_password and (
        not credentials or credentials.credentials != settings.proxy_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
async def chat_completions(
    fastapi_req: Request, payload: ChatCompletionRequest
) -> StreamingResponse:
    unloader: ModelUnloader = fastapi_req.app.state.unloader
    compressor: ContextCompressor = fastapi_req.app.state.compressor
    cache: ImageCache = fastapi_req.app.state.cache
    gatekeeper: Gatekeeper = fastapi_req.app.state.gatekeeper
    multiplexer: Multiplexer = fastapi_req.app.state.multiplexer

    platforms_data = getattr(fastapi_req.app.state, "platforms", {})
    fallback_model = platforms_data.get("system_fallback", {}).get("model")

    if fallback_model and payload.model not in ("auto", "nexus-auto"):
        pass

    await unloader.unload_if_needed(payload.model)

    has_images_in_latest = False
    latest_msg_index = len(payload.messages) - 1
    cached_hashes = {}

    for i, message in enumerate(payload.messages):
        if isinstance(message.content, list):
            for part in message.content:
                if part.type == "image_url" and part.image_url is not None:
                    if i == latest_msg_index:
                        has_images_in_latest = True
                    url = part.image_url.url
                    if url.startswith("data:"):
                        try:
                            _, b64_data = url.split(",", 1)
                            raw_bytes = base64.b64decode(b64_data)
                            image_hash = cache.hash_image(b64_data)
                            cache.store(image_hash, raw_bytes)
                            cached_hashes[id(part)] = image_hash
                        except Exception:
                            pass

                    if i != latest_msg_index:
                        part.type = "text"
                        image_hash = cached_hashes.get(id(part), "unknown_image")
                        part.text = f"[Image: {image_hash}]"
                        part.image_url = None

    payload_dict_for_gatekeeper = payload.model_dump(exclude_none=True)
    profile = await gatekeeper.profile_request(payload_dict_for_gatekeeper)
    context_length = profile.get("context_length", 0)

    candidates = []

    if payload.model in ("auto", "nexus-auto"):
        for key, config in platforms_data.items():
            if not isinstance(config, dict) or key == "system_fallback":
                continue
            max_tokens = config.get("max_input_tokens", 0)
            if max_tokens < context_length:
                continue
            if has_images_in_latest and not config.get("supports_vision"):
                continue
            candidates.append(key)

        if not candidates:
            logger.warning("No candidates found, falling back to system fallback")
            candidates = ["system_fallback"]

        # Target model for compression heuristics (use first candidate)
        target_model_for_compression = (
            candidates[0].split("/", 1)[1] if "/" in candidates[0] else candidates[0]
        )
        if candidates[0] == "system_fallback":
            target_model_for_compression = fallback_model
    else:
        # Explicit model requested
        if payload.model == fallback_model:
            candidates = ["system_fallback"]
            target_model_for_compression = fallback_model
        elif "/" in payload.model:
            candidates = [payload.model]
            target_model_for_compression = payload.model.split("/", 1)[1]
        else:
            found = [k for k in platforms_data if k.endswith(f"/{payload.model}")]
            if found:
                candidates = found
                target_model_for_compression = payload.model
            else:
                candidates = [f"ollama/{payload.model}"]
                target_model_for_compression = payload.model

    payload.messages = compressor.compress_messages(
        payload.messages, target_model_for_compression, has_images_in_latest
    )

    payload_dict = payload.model_dump(exclude_none=True)

    async def sse_wrapper() -> typing.AsyncGenerator[bytes, None]:
        extra_kwargs = {k: v for k, v in payload_dict.items() if k not in ("model", "messages")}
        async for chunk in multiplexer.generate_stream(
            candidate_models=candidates,
            messages=payload_dict.get("messages", []),
            **extra_kwargs,
        ):
            # For SSE response, we can just use a generic platform name since multiplexer handles it
            openai_chunk = {
                "id": "chatcmpl-dynamic",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": chunk}}],
            }
            yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        sse_wrapper(),
        media_type="text/event-stream",
    )
