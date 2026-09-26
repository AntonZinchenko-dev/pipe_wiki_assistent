"""Точка входа FastAPI.

Здесь собирается ровно один экземпляр каждого долгоживущего объекта: HTTP-клиент,
хранилище, реестр провайдеров, писатель трейсов. Новый HTTP-клиент на запрос —
это новое TLS-рукопожатие каждый раз и исчерпание портов под нагрузкой, поэтому
клиент создаётся при старте и закрывается при остановке (раздел 2).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .alerts import AlertMonitor
from .api.routes import router
from .config import get_settings
from .pipeline import Pipeline
from .providers import build_registry
from .rag.store import Store
from .ratelimit import RateLimiter
from .trace import TraceWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("pipewiki")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        # Общий таймаут задаётся на каждом вызове отдельно: у эмбеддингов и у
        # стрима генерации разные требования, и один общий таймаут неизбежно
        # либо рвёт законный длинный поток, либо разрешает зависшему висеть.
        timeout=httpx.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s),
    )
    store = Store(settings.db_path)
    providers = build_registry(settings, client)
    traces = TraceWriter(settings.trace_dir)

    app.state.settings = settings
    app.state.client = client
    app.state.store = store
    app.state.providers = providers
    app.state.limiter = RateLimiter(rpm=settings.rate_limit_rpm, tpm=settings.rate_limit_tpm)
    app.state.alerts = AlertMonitor(settings)
    app.state.pipeline = Pipeline(
        settings=settings, store=store, providers=providers, traces=traces,
        http=client, alerts=app.state.alerts,
    )
    # Версия берётся у ТОГО ЖЕ объекта, что отвечает, а не из константы
    # модуля: иначе при смене настройки здоровье показывало бы одну версию,
    # а работала бы другая.
    app.state.pipeline_prompt_version = app.state.pipeline.prompt_version

    stats = store.stats()
    log.info(
        "старт: документов %s, чанков %s, промпт %s, цепочка провайдеров %s",
        stats["documents"], stats["chunks"], app.state.pipeline_prompt_version,
        providers.chain,
    )
    if not stats["chunks"]:
        log.warning("индекс пуст — выполните `python scripts/ingest.py`")

    try:
        yield
    finally:
        # Сначала провайдеры: у них могут быть свои HTTP-клиенты, о которых
        # здесь не знают (например, со своим списком доверенных сертификатов).
        await providers.aclose()
        await client.aclose()
        store.close()
        log.info("остановка: клиенты и хранилище закрыты")


app = FastAPI(
    title="PIPEWIKI — ассистент по внутренней вики",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
