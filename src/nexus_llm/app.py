from fastapi import FastAPI

from nexus_llm.routes.proxy import router as proxy_router


def create_app() -> FastAPI:
    app = FastAPI(title="nexus-llm")
    app.include_router(proxy_router)
    return app
