"""HTTP-слой: вики, поиск, чат потоком, здоровье.

Ключ провайдера живёт только здесь, на сервере, и в браузер не уходит никогда:
фронт общается с моделью исключительно через эти эндпоинты (раздел 2).
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..access import RIGHT_AGENT, RIGHT_LIVE_READ, identify
from ..pipeline import Event, Pipeline
from ..rag.history import Turn
from ..rag.result import OpenTable
from ..ratelimit import client_key
from ..rag.context import estimate_tokens
from ..rag.embed import embed_query
from ..rag.search import hybrid_search
from ..rag.store import IndexMismatch

router = APIRouter(prefix="/api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class HistoryTurn(BaseModel):
    """Одна реплика прошлой переписки, присланная браузером.

    ПЕРЕПИСКУ ХРАНИТ КЛИЕНТ, А НЕ МЫ. Сервер получает её в запросе,
    пользуется и забывает: ни таблицы, ни чистки старого, ни разговора о
    том, сколько мы храним переписку сотрудников.

    Обратная сторона ровно одна, и её надо держать в голове: это поле
    запроса, а не наша база. Прислать сюда можно что угодно, включая
    реплику «ассистента», которой он не говорил. Поэтому здесь стоят
    потолки на разбор, дальше конвейер режет по бюджету токенов, а текст
    проходит тот же санитайзер, что документы. Верить содержимому нельзя —
    источником для цитаты переписка не становится никогда.
    """

    # Ролей ровно две. Строка без ограничения открыла бы дорогу роли
    # «system»: реплика с такой ролью читается моделью как указание, а
    # пришла она из браузера.
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class OpenTableRef(BaseModel):
    """Таблица, которую человек УЖЕ видит, и метка её следующей страницы.

    Курсор держит тот, кто листает, — браузер. Сервер диалогов не помнит,
    поэтому на «следующие пять» агенту нечем было понять, где он
    остановился: он запрашивал первые десять заново.

    Содержимому здесь не верим ровно так же, как переписке. Метка приходит
    из браузера и разбирается тем же разбором, что и метка от инструмента:
    испорченная — отказ, числа в ней проходят проверку порогов. Показать
    по такой метке можно только то, что человеку и так разрешено видеть.
    """

    handle: str = Field(default="", max_length=16)
    title: str = Field(default="", max_length=200)
    offset: int = Field(default=0, ge=0, le=100_000)
    shown: int = Field(default=0, ge=0, le=1000)
    total_found: int = Field(default=0, ge=0, le=1_000_000)
    next_cursor: str = Field(default="", max_length=120)


class ChatRequestBody(AskRequest):
    """Вопрос плюс выбор модели.

    Выбор ПРОВЕРЯЕТСЯ по списку, который сервер собрал сам, и делать иначе
    нельзя: имя модели приходит из браузера и уходит прямиком в чужой API.
    Непроверенная строка отсюда — это чужой запрос, отправленный за наш счёт
    и с нашим ключом.

    Пустые значения означают «как настроено», и это не то же самое, что
    отсутствие проверки: пустоту мы дальше не передаём вовсе.
    """

    provider: str = Field(default="", max_length=40)
    model: str = Field(default="", max_length=120)
    # Агентский режим: модель сама решает, дособрать ли контекст. Включается
    # только если разрешён на сервере — просьба из браузера сама по себе
    # права не даёт.
    agent: bool = False
    # Прошлая переписка этого же диалога. Потолок на ДЛИНУ СПИСКА нужен
    # здесь, до разбора: бюджет токенов в конвейере обрежет лишнее по весу,
    # но разбирать двадцать тысяч реплик, чтобы выкинуть девятнадцать
    # тысяч девятьсот, — значит оплатить работу впустую.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=40)
    # Памятка по свёрнутой части разговора. Её считает сервер и отдаёт
    # событием `memory`, а хранит и присылает обратно браузер — сервер
    # диалогов не помнит. Потолок тот же по смыслу, что у реплик: это
    # поле запроса, а не наша база.
    summary: str = Field(default="", max_length=4000)
    # Таблицы, показанные в этом же диалоге. Потолок маленький нарочно:
    # смысл имеют последние, а не все за день.
    open_tables: list[OpenTableRef] = Field(default_factory=list, max_length=4)


@router.get("/live")
async def live() -> dict:
    """Жив ли ПРОЦЕСС. Модель здесь не спрашивается, и это принципиально.

    Живость и готовность — разные вопросы, и путать их дорого. Оркестратор
    по живости ПЕРЕЗАПУСКАЕТ. Если сюда добавить проверку модели, то падение
    чужого сервиса — облака, локального рантайма, сети — будет выглядеть как
    смерть нашего процесса, и нас начнут перезапускать по кругу ровно в тот
    момент, когда мы исправно работаем и отдаём понятные ошибки. Перезапуск
    при этом ничего не чинит: модель от него не поднимется.

    Поэтому здесь буквально «интерпретатор отвечает».
    """
    return {"ok": True}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict:
    """Можно ли слать сюда трафик. Здесь модель спрашивается ОБЯЗАТЕЛЬНО.

    По готовности балансировщик СНИМАЕТ НАГРУЗКУ, не перезапуская. Это
    правильная реакция на «модель недоступна» и на «индекс пуст»: сервис
    жив, но отвечать ему нечем, и присылать людей к нему незачем.

    Отдаём 503, а не 200 с полем `ok: false`. Балансировщики читают код
    ответа, а не тело; двухсотка с честным телом внутри для них означает
    «всё хорошо, шлите людей».
    """
    app = request.app.state
    providers = await app.providers.health()
    chunks = app.store.stats().get("chunks", 0)
    model_ok = any(item.get("ok") for item in providers)
    ok = bool(model_ok and chunks)
    if not ok:
        response.status_code = 503
    return {
        "ok": ok,
        # Причина словами: по коду 503 видно, что не готов, и не видно чем.
        "why": "" if ok else (
            "нет доступного провайдера модели" if not model_ok else "индекс пуст"
        ),
        "providers": providers,
        "chunks": chunks,
    }


@router.get("/health")
async def health(request: Request) -> dict:
    app = request.app.state
    return {
        "ok": True,
        "index": app.store.stats(),
        "providers": await app.providers.health(),
        "prompt_version": app.pipeline_prompt_version,
        "settings": {
            "chat_model": app.settings.ollama_chat_model,
            "embed_model": app.settings.ollama_embed_model,
            "similarity_floor": app.settings.similarity_floor,
            "context_token_budget": app.settings.context_token_budget,
            "temperature": app.settings.temperature,
        },
    }


@router.get("/models")
async def models(request: Request) -> dict:
    """Чем можно отвечать прямо сейчас.

    Список опрашивается у провайдеров, а не берётся из настроек: настройки
    говорят, что мы написали, опрос — что есть на самом деле. Разошлись они
    ровно один раз, и отказ «нет такой модели» тогда выглядел как поломка
    облака.
    """
    app = request.app.state
    return {
        "models": await app.providers.list_models(),
        "default": {
            "provider": app.providers.chain[0],
            "model": app.providers.answering_model,
        },
        # Доступен ли агентский режим. Интерфейс по этому полю решает,
        # показывать ли переключатель: кнопка, которая всегда возвращает
        # ошибку, хуже отсутствующей.
        "agent": {
            "available": app.settings.agent_enabled,
            "max_steps": app.settings.agent_max_steps,
            "decision": app.settings.agent_decision,
            # Настроен ли сервис живых данных. Без этого поля снаружи не
            # видно, предлагались ли модели инструменты вообще, — и
            # «агент не сходил в сервис» выглядит одинаково при
            # ненастроенном адресе и при отказе модели.
            "live_data": bool(app.settings.fatigue_api_url),
        },
    }


@router.get("/me")
async def me(request: Request) -> dict:
    """Кто спрашивает и что ему можно.

    Интерфейс по этому решает, показывать ли переключатель агента: кнопка,
    которая всегда возвращает отказ, хуже отсутствующей.

    Список ЗАКРЫТЫХ проектов наружу не отдаём — только их число. Перечень
    того, чего человеку нельзя, сам по себе сведение о том, что в системе
    есть: «вам закрыт проект SURVEYD» сообщает о существовании SURVEYD.
    """
    state = request.app.state
    from ..access import denied_projects

    user = identify(request.headers, state.settings)
    closed = denied_projects(user, state.settings)
    return {
        "name": user.name,
        "roles": sorted(user.roles),
        "rights": {
            "agent": user.may(RIGHT_AGENT),
            "live_data": user.may(RIGHT_LIVE_READ),
        },
        "closed_projects": len(closed),
        "auth_mode": state.settings.auth_mode,
    }


@router.get("/alerts")
async def alerts(request: Request) -> dict:
    """Текущее состояние порогов и история переходов.

    История здесь не для красоты: текущее значение отвечает на вопрос «что
    сейчас», а дежурному утром нужен ответ на вопрос «что было ночью».
    """
    monitor = getattr(request.app.state, "alerts", None)
    if monitor is None:
        return {"level": "ok", "metrics": [], "events": [], "note": "монитор не поднят"}
    return monitor.report()


@router.get("/prompts")
async def prompts(request: Request) -> dict:
    """Какие версии промпта есть и какая работает сейчас.

    Нужно затем, что «откат без выкладки» бессмысленен, если не видно, на
    что откатываться. `drifted` — признак того, что текст версии правили на
    месте: тогда версия под этим именем больше не та, что была, и откат к
    ней вернёт не то, что ожидают.
    """
    from ..rag.prompt import catalogue

    return {
        "active": request.app.state.pipeline_prompt_version,
        "setting": "PW_PROMPT_NAME",
        "versions": catalogue(),
    }


@router.get("/documents")
async def documents(request: Request) -> dict:
    from ..access import denied_projects

    state = request.app.state
    closed = denied_projects(identify(request.headers, state.settings), state.settings)
    return {
        "documents": [
            document for document in state.store.list_documents()
            if document.get("project") not in closed
        ]
    }


@router.get("/documents/{doc_id}")
async def document(doc_id: str, request: Request) -> dict:
    from ..access import denied_projects

    state = request.app.state
    found = state.store.document(doc_id)
    closed = denied_projects(identify(request.headers, state.settings), state.settings)
    # 404, а не 403, и это осознанно: «нет доступа» сообщает о том, что
    # документ существует. Для закрытого проекта сам факт наличия документа
    # — уже сведение, которого человеку знать не положено.
    if found is None or found.get("project") in closed:
        raise HTTPException(status_code=404, detail=f"документ {doc_id} не найден")
    return found


@router.post("/search")
async def search(payload: AskRequest, request: Request) -> dict:
    """Отладочный поиск без генерации.

    Нужен не для интерфейса, а для работы руками: именно этим эндпоинтом
    проверяют пары с отрицанием, калибруют порог и смотрят, что вообще
    поднимается по запросу (раздел 3 и 4).
    """
    state = request.app.state
    try:
        state.store.assert_compatible(
            embed_model=state.settings.ollama_embed_model,
            embed_dim=state.settings.ollama_embed_dim,
        )
    except IndexMismatch as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    # Отладочный поиск тоже уважает права.
    #
    # Это не формальность: эндпоинт отдаёт куски текста напрямую, и без
    # проверки он становится обходным путём к тому, что закрыто в основном
    # поиске. Ограничение действует на ДАННЫЕ, а не на конкретный вход —
    # это дословно пункт 4.2.3.5 регламента.
    from ..access import denied_projects

    user = identify(request.headers, state.settings)
    vector = await embed_query(state.providers.embedder, payload.question)
    result = await hybrid_search(
        store=state.store,
        query=payload.question,
        query_vector=vector,
        top_k=state.settings.search_top_k,
        floor=state.settings.similarity_floor,
        rrf_k=state.settings.rrf_k,
        denied_projects=denied_projects(user, state.settings),
        # Вес ключевого списка обязан быть и здесь. Этот эндпоинт — отладочный,
        # по нему смотрят, ЧТО нашлось, и молчаливое расхождение с продовым
        # слиянием превращает его из инструмента диагностики в источник ложных
        # выводов: человек видит один порядок, а модель получает другой.
        keyword_weight=state.settings.keyword_weight,
    )
    return {
        "question": payload.question,
        "best_cosine": round(result.best_vector_score, 4),
        "floor": result.floor,
        "passed_floor": result.passed_floor,
        "passed_by_keyword": result.passed_by_keyword,
        "duplicates_dropped": result.dropped_duplicates,
        "hits": [
            {
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "heading_path": hit.chunk.heading_path,
                "found_by": hit.found_by,
                "cosine": round(hit.vector_score, 4) if hit.vector_score is not None else None,
                "bm25": round(hit.keyword_score, 4) if hit.keyword_score is not None else None,
                "vector_rank": hit.vector_rank,
                "keyword_rank": hit.keyword_rank,
                "rrf": round(hit.fused_score, 5),
                "preview": hit.chunk.body[:280],
            }
            for hit in result.hits
        ],
    }


@router.post("/chat")
async def chat(payload: ChatRequestBody, request: Request) -> StreamingResponse:
    state = request.app.state

    # Выбор модели сверяем со списком, собранным сервером. Неизвестную пару
    # не подменяем молча на настроенную: человек выбрал конкретную модель,
    # и тихая подмена дала бы ему ответ другой модели под видом выбранной.
    chosen_provider, chosen_model = payload.provider.strip(), payload.model.strip()
    if chosen_provider or chosen_model:
        allowed = await state.providers.list_models()
        pairs = {(item["provider"], item["model"]) for item in allowed}
        if (chosen_provider, chosen_model) not in pairs:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"модель {chosen_model or '—'} у провайдера "
                    f"{chosen_provider or '—'} сейчас недоступна"
                ),
            )

    user = identify(request.headers, state.settings)

    if payload.agent and not state.settings.agent_enabled:
        raise HTTPException(
            status_code=400,
            detail="агентский режим выключен на сервере (PW_AGENT_ENABLED)",
        )
    if payload.agent and not user.may(RIGHT_AGENT):
        # Отказ, а не молчаливый обычный ответ. Тихо сделать не то, о чём
        # попросили, — худший вариант: человек считает, что агент работал,
        # и делает выводы о его пользе по прогону, которого не было.
        raise HTTPException(status_code=403, detail="нет прав на агентский режим")

    key = client_key(
        request.headers,
        request.client.host if request.client else None,
        trust_proxy=state.settings.trust_proxy_headers,
    )

    decision = state.limiter.check(
        key,
        estimated_tokens=estimate_tokens(payload.question) + state.settings.context_token_budget,
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"{decision.reason}: попробуйте через {decision.retry_after_s} с",
            headers={"Retry-After": str(decision.retry_after_s)},
        )

    pipeline: Pipeline = state.pipeline

    async def body():
        try:
            async for event in pipeline.stream_answer(
                payload.question,
                provider=chosen_provider,
                model=chosen_model,
                agent=payload.agent,
                user=user,
                history=[
                    Turn(role=turn.role, content=turn.content)
                    for turn in payload.history
                ],
                summary=payload.summary,
                open_tables=[
                    OpenTable(
                        handle=table.handle,
                        title=table.title,
                        offset=table.offset,
                        shown=table.shown,
                        total_found=table.total_found,
                        next_cursor=table.next_cursor,
                    )
                    for table in payload.open_tables
                ],
            ):
                yield event.encode()
        except asyncio.CancelledError:
            # Клиент закрыл соединение (кнопка «Стоп» на фронте). Пробрасываем
            # отмену дальше: она должна дойти до провайдера и остановить
            # генерацию, иначе модель продолжит писать в никуда, а мы —
            # платить за это (раздел 5).
            raise
        except Exception as error:  # последняя линия: не оставляем висящий поток
            yield Event("error", {
                "code": "internal",
                "message": "Внутренняя ошибка сервера.",
                "detail": str(error)[:300],
            }).encode()
            yield Event("done", {"finish_reason": "error", "stop_reason": "internal"}).encode()

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Иначе буферизующий прокси соберёт весь поток и отдаст целиком —
            # классическая причина «стриминг работает локально, ломается в
            # проде».
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
