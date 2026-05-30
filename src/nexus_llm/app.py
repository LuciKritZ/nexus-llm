import typing
from contextlib import asynccontextmanager

import aiosqlite
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nexus_llm.config import settings
from nexus_llm.exceptions import CacheError, GeminiAPIError, NexusLLMError
from nexus_llm.routes.proxy import router as proxy_router
from nexus_llm.services.cache import ImageCache
from nexus_llm.services.compressor import ContextCompressor
from nexus_llm.services.db import init_db
from nexus_llm.services.gatekeeper import Gatekeeper
from nexus_llm.services.gemini_client import GeminiClient
from nexus_llm.services.multiplexer import Multiplexer
from nexus_llm.services.router_core import RouterCore
from nexus_llm.services.sync import sync_keys_from_json
from nexus_llm.services.unloader import ModelUnloader


@asynccontextmanager
async def lifespan(app: FastAPI) -> typing.AsyncGenerator[None, None]:
    import json
    from pathlib import Path

    platforms_data = {}
    platforms_path = Path("platforms.json")
    if platforms_path.exists():
        try:
            with open(platforms_path, encoding="utf-8") as f:
                platforms_data = json.load(f)
        except Exception:  # pragma: no cover
            pass

    client = httpx.AsyncClient(timeout=300.0)
    unloader = ModelUnloader(http_client=client, platforms_data=platforms_data)
    compressor = ContextCompressor()
    cache = ImageCache()
    gemini_client = GeminiClient(client=client)

    db = await aiosqlite.connect(settings.sqlite_db_path)
    await init_db(db)
    await sync_keys_from_json(db, settings.keys_json_path)

    router_core = RouterCore(db)
    gatekeeper = Gatekeeper(client, platforms_data)
    multiplexer = Multiplexer(router_core, client, platforms_data)

    app.state.http_client = client
    app.state.unloader = unloader
    app.state.compressor = compressor
    app.state.cache = cache
    app.state.gemini_client = gemini_client
    app.state.db = db
    app.state.router_core = router_core
    app.state.gatekeeper = gatekeeper
    app.state.multiplexer = multiplexer
    app.state.platforms = platforms_data

    yield

    await client.aclose()
    await db.close()


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
