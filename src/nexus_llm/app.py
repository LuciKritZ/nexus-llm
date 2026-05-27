import typing
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nexus_llm.config import settings
from nexus_llm.exceptions import CacheError, GeminiAPIError, NexusLLMError
from nexus_llm.routes.proxy import router as proxy_router
from nexus_llm.services.cache import ImageCache
from nexus_llm.services.compressor import ContextCompressor
from nexus_llm.services.gemini_client import GeminiClient
from nexus_llm.services.unloader import ModelUnloader


@asynccontextmanager
async def lifespan(app: FastAPI) -> typing.AsyncGenerator[None, None]:
    client = httpx.AsyncClient(timeout=300.0)
    unloader = ModelUnloader(ollama_url=settings.ollama_base_url, http_client=client)
    compressor = ContextCompressor()
    cache = ImageCache()
    gemini_client = GeminiClient(client)

    app.state.http_client = client
    app.state.unloader = unloader
    app.state.compressor = compressor
    app.state.cache = cache
    app.state.gemini_client = gemini_client

    yield

    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="nexus-llm", lifespan=lifespan)
    app.include_router(proxy_router)

    @app.exception_handler(GeminiAPIError)
    async def gemini_exception_handler(request: Request, exc: GeminiAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc)},
        )

    @app.exception_handler(CacheError)
    async def cache_exception_handler(request: Request, exc: CacheError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    @app.exception_handler(NexusLLMError)
    async def nexus_exception_handler(request: Request, exc: NexusLLMError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    return app
