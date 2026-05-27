import typing
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from nexus_llm.config import settings
from nexus_llm.routes.proxy import router as proxy_router
from nexus_llm.services.compressor import ContextCompressor
from nexus_llm.services.unloader import ModelUnloader


@asynccontextmanager
async def lifespan(app: FastAPI) -> typing.AsyncGenerator[None, None]:
    client = httpx.AsyncClient()
    unloader = ModelUnloader(ollama_url=settings.ollama_base_url, http_client=client)
    compressor = ContextCompressor()

    app.state.http_client = client
    app.state.unloader = unloader
    app.state.compressor = compressor

    yield

    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="nexus-llm", lifespan=lifespan)
    app.include_router(proxy_router)
    return app
