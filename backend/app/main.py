from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config.settings import get_settings
from app.core.container import build_container
from app.persistence.database import (
    build_engine,
    build_session_factory,
    initialize_database,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = build_engine(settings.database_url)
    initialize_database(engine)
    session_factory = build_session_factory(engine)
    try:
        container = build_container(settings, session_factory)
        container.indexing.recover_interrupted_tasks()
        app.state.container = container
        yield
    finally:
        engine.dispose()


app = FastAPI(
    title="Roborock RAG API",
    version="0.1.0",
    lifespan=lifespan,
)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix=settings.api_prefix)

