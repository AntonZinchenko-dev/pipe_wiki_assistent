"""Оркестратор ответа: поиск -> контекст -> генерация -> проверка -> поток.

Это цепочка, а не агент. Шаги известны заранее и не зависят от промежуточных
результатов, поэтому агент здесь добавил бы недетерминированность, задержку и
стоимость, не добавив ничего (раздел 0). Когда появится ветвление — например,
модель сама решает, переформулировать ли запрос и искать ли повторно, — вот
тогда и появится агент, отдельным осознанным шагом.

Поток наружу — типизированные события, а не голый текст. Фронт получает
отдельно: метаданные с тем, что нашлось (для блока «как получен ответ»),
чистый текст ответа по мере генерации, финальную структуру с проверенными
цитатами и финал с причиной завершения.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator

from .access import ANONYMOUS, RIGHT_AGENT, RIGHT_LIVE_READ, User, denied_projects
from .agent import Agent, AgentOutcome, AgentProgress
from .alerts import AlertMonitor
from .config import Settings, budget_problems
from .providers import ChatRequest, ProviderError, ProviderRegistry, ProviderUnavailable
from .providers.base import FinishReason
from .rag import answer as answer_mod
from .rag.answer import AnswerEnvelope, AnswerStatus, StreamingAnswerParser
from .rag.context import build_context, estimate_tokens
from .rag.embed import embed_query
from .rag.history import Turn, as_block, needs_history, summarize, support_query
from .rag.history import trim as history_trim
from .rag.prompt import PromptSpec, active_prompt, build_user_message
from .rag.result import TABLE_RULES, OpenTable
from .rag.search import Query, hybrid_search_many
from .rag.live import LiveApi
from .rag.store import IndexMismatch, Store
from .rag.tools import Toolbox
from .trace import Trace, TraceWriter

MAX_QUESTION_CHARS = 1000


@dataclass(slots=True)
class Event:
    """Событие потока. `name` уходит в SSE как event:, `data` — как JSON."""

    name: str
    data: dict

    def encode(self) -> str:
        return f"event: {self.name}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


class Pipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        store: Store,
        providers: ProviderRegistry,
        traces: TraceWriter,
        http=None,
        alerts: AlertMonitor | None = None,
        prompt: PromptSpec | None = None,
    ) -> None:
        self._s = settings
        # Версия промпта выбирается ОДИН раз, на сборке конвейера, и дальше
        # не меняется. Читать настройку на каждом запросе значило бы, что
        # два запроса одного пользователя могут уехать по разным промптам —
        # и в журнале это будет выглядеть как невоспроизводимость модели.
        self._prompt = prompt or active_prompt(settings)
        # ПРОВЕРКА БЮДЖЕТА — ЗДЕСЬ, А НЕ НА СТАРТЕ СЕРВЕРА.
        #
        # Конвейер — единственное место, через которое проходят оба пути:
        # и сервер, и прогоны. Поставить проверку в запуск приложения значило
        # бы, что прогон её не делает, — а перебор бюджета портит именно
        # измерение, молча обрезая контекст.
        #
        # Отказ, а не предупреждение. Это детерминированная ошибка настройки,
        # видная до первого запроса: сервис, который поднялся с заведомо
        # обрезанным контекстом, выдаёт не ошибку, а тихо ухудшённые ответы —
        # худший из возможных исходов, потому что его никто не заметит.
        problems = budget_problems(
            settings, system_tokens=estimate_tokens(self._prompt.system)
        )
        if problems:
            raise RuntimeError("; ".join(problems))
        self._store = store
        self._providers = providers
        self._traces = traces
        self._alerts = alerts
        # Агент собирается всегда, включается — по запросу. Собрать его
        # дёшево (это две ссылки), а вот развилка «есть объект или нет»
        # разошлась бы по всему методу и превратила один понятный `if` в
        # четыре разбросанных.
        self._agent = Agent(
            settings=settings,
            toolbox=Toolbox(
                store=store,
                settings=settings,
                embedder=providers.embedder,
                # Клиент живых данных ходит ОБЩИМ http-клиентом приложения:
                # это внутренний сервис, никаких особых сертификатов ему не
                # нужно, а свой клиент означал бы ещё один пул соединений,
                # который надо не забыть закрыть.
                live=LiveApi(settings, http),
            ),
            providers=providers,
        )

    @property
    def prompt_version(self) -> str:
        """Версия промпта, которой этот конвейер отвечает.

        Спрашивать надо у него, а не у константы модуля: с появлением
        выбора версии константа отвечает на вопрос «что в коде», а нужен
        ответ на вопрос «чем отвечают».
        """
        return self._prompt.version

    async def stream_answer(
        self,
        question: str,
        *,
        provider: str = "",
        model: str = "",
        agent: bool = False,
        user: User | None = None,
        history: list[Turn] | None = None,
        summary: str = "",
        open_tables: list[OpenTable] | None = None,
    ) -> AsyncIterator[Event]:
        question = question.strip()[:MAX_QUESTION_CHARS]
        trace = self._traces.start(question)
        trace.prompt_version = self._prompt.version
        # Это НАМЕРЕНИЕ, а не факт: кого собирались спросить. Настоящий
        # ответчик станет известен, когда отдаст первый кусок, и тогда обе
        # строки перепишутся.
        trace.provider = provider or self._providers.chain[0]
        trace.model = model or self._providers.answering_model

        try:
            async for event in self._run(
                question, trace, provider=provider, model=model, agent=agent,
                user=user or ANONYMOUS,
                # Переписка идёт дальше ЦЕЛИКОМ, необрезанной.
                #
                # Обрезка делается внутри, и это не перестановка строк.
                # Свернуть в памятку надо всё, что пришло, — иначе
                # обрезанный хвост исчезнет, так и не попав в памятку, то
                # есть именно то, ради чего памятка и заведена.
                history=history or [],
                summary=summary,
                open_tables=open_tables,
            ):
                yield event
        except asyncio.CancelledError:
            # Клиент отключился или нажал «Стоп». Это не ошибка, но причина
            # остановки должна быть названа и попасть в отчёт (раздел 7).
            trace.stop_reason = "client_disconnect"
            trace.status = trace.status or "cancelled"
            raise
        except GeneratorExit:
            # То же самое, но другим путём. Когда отмена застаёт сервер на
            # отправке данных клиенту, до нас доходит не CancelledError, а
            # закрытие генератора — и прежний код такую отмену записывал как
            # «ничего не произошло»: статус пустой, причина пустая. Часть
            # отмен помечалась правильно, часть нет, и отчёт по причинам
            # остановки становился бессмысленным именно потому, что выглядел
            # правдоподобно.
            trace.stop_reason = trace.stop_reason or "client_disconnect"
            trace.status = trace.status or "cancelled"
            raise
        except Exception as error:  # noqa: BLE001 — ловим намеренно всё
            # Неожиданное исключение обязано быть видно в трейсе.
            #
            # Без этой ветки упавший запрос попадал в журнал с пустым статусом
            # и был неотличим от успешного, у которого статус просто не
            # выставили. Разбирать по такому трейсу жалобу невозможно — а он
            # ровно для этого и написан.
            trace.status = "internal_error"
            trace.stop_reason = "exception"
            trace.error = f"{error.__class__.__name__}: {error}"[:300]
            raise
        finally:
            self._traces.finish(trace)
            # Монитор узнаёт о запросе ЗДЕСЬ, а не в обработчике HTTP: сюда
            # доходят и отменённые, и упавшие запросы. Считать долю отказов
            # по одним успешным — это считать её по тем, кого не спрашивали.
            if self._alerts is not None:
                self._alerts.observe(trace)

    async def _run(
        self,
        question: str,
        trace: Trace,
        *,
        provider: str = "",
        model: str = "",
        agent: bool = False,
        user: User = ANONYMOUS,
        history: list[Turn] | None = None,
        summary: str = "",
        open_tables: list[OpenTable] | None = None,
    ) -> AsyncIterator[Event]:
        if not question:
            trace.status = "bad_request"
            trace.stop_reason = "empty_question"
            yield Event("error", {"code": "empty_question", "message": "Пустой вопрос."})
            return

        # 1. Индекс построен той же моделью, что настроена сейчас?
        try:
            self._store.assert_compatible(
                embed_model=self._s.ollama_embed_model, embed_dim=self._s.ollama_embed_dim
            )
        except IndexMismatch as error:
            trace.status = "index_mismatch"
            trace.stop_reason = "index_mismatch"
            yield Event("error", {"code": "index_mismatch", "message": str(error)})
            return

        # 1.5. Чем искать: вопросом человека, а при обрывке — ещё и склейкой.
        #
        # Вопрос человека НЕ ПОДМЕНЯЕТСЯ нигде: ни в поиске, ни у агента, ни у
        # отвечающей модели. Склейка — второй запрос поиска, а не замена
        # первого, и до текста ответа она не доходит вовсе.
        #
        # Это важнее, чем кажется. Пока вопрос подменялся, человек видел на
        # экране своё «а сколько их?», а система работала с чужой
        # формулировкой — и разобрать «почему ассистент переиначил вопрос»
        # было нечем.
        arrived = list(history or [])
        # Что увидит модель: хвост, влезающий в бюджет токенов. Ограничение
        # в схеме запроса — защита разбора, бюджет токенов — правило
        # системы, и жить ему здесь.
        turns = history_trim(
            arrived,
            max_turns=self._s.history_max_turns,
            token_budget=self._s.history_token_budget,
        )
        # ВОПРОС НЕ ПЕРЕПИСЫВАЕТСЯ. Обрывок подпирается ВТОРЫМ ЗАПРОСОМ.
        #
        # Раньше здесь стоял отдельный вызов модели: она переписывала вопрос,
        # подставляя предмет из переписки. Механизм стоил секунд на каждом
        # вопросе и регулярно врал — «дай топ 10 труб парка» уезжало в поиск
        # как вопрос про кэш телеметрии, и ответ приходил про кэш, уверенный
        # и со ссылками. Заметить это человек не мог: на экране его вопрос, а
        # искали другим. Вокруг механизма пришлось поставить забор из
        # проверок, а забор — это признание, что механизму нельзя верить.
        #
        # Теперь решение принимает арифметика. Самостоятельный вопрос ищется
        # сам по себе, обрывок («а 10?», «следующие») — двумя запросами:
        # собой и склейкой с прошлым вопросом человека. Сливаются они той же
        # формулой RRF, которой мы уже сливаем векторный поиск с ключевым:
        # она написана, измерена и работает. Если склейка притащила чужую
        # тему, согласия между списками не будет, и чужое утонет.
        #
        # Ни одной генерации, нечему соврать, и на секунду быстрее.
        support = ""
        if (turns or summary) and self._s.history_rewrite != "off":
            if needs_history(question):
                support = support_query(question, turns, summary)

        # 2. Эмбеддинги запросов. Их один или два — второго вызова модели нет.
        wanted = [question] + ([support] if support else [])
        try:
            with trace.span("embed_query", model=self._s.ollama_embed_model, queries=len(wanted)):
                vectors = [
                    await embed_query(self._providers.embedder, text) for text in wanted
                ]
        except (ProviderError, ProviderUnavailable) as error:
            trace.status = "embed_unavailable"
            trace.stop_reason = "provider_unavailable"
            yield Event("error", {
                "code": "embed_unavailable",
                "message": "Поиск сейчас недоступен: модель эмбеддингов не отвечает.",
                "detail": str(error)[:300],
                "retry_after_s": error.retry_after_s or 20,
            })
            return

        # КОГДА СКЛЕЙКА ЕСТЬ — ГЛАВНАЯ ОНА, А НЕ ВОПРОС.
        #
        # Сначала было наоборот: вопрос с весом 1.0, склейка 0.6, из
        # соображения «склейка тянет прошлую тему, пусть только помогает».
        # Замер это опроверг, и поучительно, как именно.
        #
        # Вопрос «а почему нельзя в проде?»: склейка нашла нужный документ с
        # близостью 0.72 — то есть сработала идеально. А в выдаче остались
        # онбординг, инклинометрия и FAQ, потому что голый вопрос ранжировал
        # их с весом 1.0 + 0.45, а склейка свой документ — с 0.6 + 0.27.
        # Слияние честно сложило числа и выбрало мусор.
        #
        # Ошибка была в рассуждении, а не в константе. Склейку мы строим
        # ТОЛЬКО тогда, когда `needs_history` уже установил: вопрос сам по
        # себе не значит ничего. Давать главный вес заведомо недостаточному
        # запросу — значит доверять тому, о чём только что решили, что ему
        # доверять нельзя.
        #
        # Голый вопрос остаётся вторым запросом, а не выбрасывается: в нём
        # есть слова, которых в прошлом вопросе не было («включить», «в
        # проде»), и они уточняют, какое именно место документа нужно.
        if support:
            queries = [
                Query(text=support, vector=vectors[1], weight=1.0),
                Query(text=question, vector=vectors[0], weight=0.5),
            ]
        else:
            queries = [Query(text=question, vector=vectors[0], weight=1.0)]

        # Агент получает склейку, если она есть: ему надо понять, о чём речь,
        # чтобы выбрать инструмент, а по «а 10?» он не выберет ничего.
        agent_question = support or question

        # 3. Гибридный поиск — уже с правами этого человека.
        #
        # Ограничения считаются ОДИН раз и уходят в поиск, а не применяются
        # к результату. Фрагмент, который человеку не положен, не должен
        # попасть в контекст вообще: спрятать его в интерфейсе мало, модель
        # всё равно прочитает текст и перескажет его своими словами. Утечка
        # произойдёт через ответ, при пустом списке источников на экране, — и
        # выглядеть это будет как нормальная работа.
        closed = denied_projects(user, self._s)
        with trace.span("search", top_k=self._s.search_top_k) as span:
            result = await hybrid_search_many(
                store=self._store,
                queries=queries,
                top_k=self._s.search_top_k,
                floor=self._s.similarity_floor,
                rrf_k=self._s.rrf_k,
                keyword_weight=self._s.keyword_weight,
                denied_projects=closed,
            )
            span.attributes.update(
                hits=len(result.hits),
                # Число закрытых проектов, а не их имена: перечень закрытого
                # — сам по себе сведение о том, что в системе есть, и в
                # общий журнал ему не надо.
                closed_projects=len(closed),
                best_cosine=round(result.best_vector_score, 4),
                passed_floor=result.passed_floor,
                passed_by_keyword=result.passed_by_keyword,
                duplicates_dropped=result.dropped_duplicates,
            )

        # 4. Агентский шаг: модель смотрит на найденное и решает, добирать ли.
        #
        # Стоит он ДО проверки порога, а не после, и это существенно. Именно
        # пустой результат — главный случай, ради которого шаг и заведён:
        # вопрос задан не теми словами, что документ. Поставить агента после
        # отказа значило бы отрезать его ровно там, где он нужнее всего.
        hits = list(result.hits)
        outcome: AgentOutcome | None = None
        if agent and self._s.agent_enabled and user.may(RIGHT_AGENT):
            with trace.span("agent", max_steps=self._s.agent_max_steps) as span:
                async for item in self._agent.stream(
                    # Агенту вопрос идёт КАК НАПИСАН, а недостающее он
                    # берёт из переписки: она уходит ему тем же блоком, что и
                    # отвечающей модели. Подставлять ему переписанный вопрос
                    # значило бы решать за него то, что он решает сам, — и
                    # ошибаться в этом молча.
                    agent_question, result, trace=trace, provider=provider, model=model,
                    # Право на живые данные проверяется ОТ ИМЕНИ ЧЕЛОВЕКА, а
                    # не от имени сервиса. У сервиса токен есть всегда — если
                    # спрашивать его, то ответ будет «можно» вообще для всех,
                    # и проверка выродится в украшение.
                    live_allowed=user.may(RIGHT_LIVE_READ),
                    # Курсоры таблиц, которые человек уже видит. Держит их
                    # браузер: сервер диалогов не помнит, а «следующие пять»
                    # без метки страницы — догадка.
                    open_tables=open_tables,
                ):
                    # Чем агент занят — сразу, а не постфактум. До этого
                    # человек несколько секунд смотрел в пустой экран и не
                    # знал, думает система или зависла.
                    if isinstance(item, AgentProgress):
                        yield Event("step", {"text": item.text})
                        continue
                    outcome = item

                assert outcome is not None
                hits = outcome.hits
                # ТАБЛИЦЫ УХОДЯТ ПРЯМО В БРАУЗЕР, минуя модель.
                #
                # Событие отправляется ДО генерации: пользователь видит
                # данные, пока модель ещё думает, что про них сказать. Это
                # не только приятнее — это честнее: таблица пришла из
                # сервиса и верна независимо от того, что напишет модель
                # и допишет ли вообще.
                for data in outcome.datasets:
                    yield Event("dataset", data.to_dict())
                span.attributes.update(
                    datasets=len(outcome.datasets),
                    steps=len(outcome.steps),
                    used_tools=outcome.used_tools,
                    hit_limit=outcome.hit_limit,
                    trail_cut=outcome.trail_cut,
                    # Чем решали и что модель ответила вместо вызова. Без
                    # этих двух полей «агент ничего не сделал» выглядит
                    # одинаково при решении модели и при её неумении, и
                    # чинить приходится наугад.
                    mechanism=outcome.mechanism,
                    declined_with=outcome.declined_with[:200],
                )

        # 5. Ничего выше порога — отвечаем отказом БЕЗ вызова модели.
        # Это не оптимизация, а честность: подсунуть модели слабо релевантные
        # куски и попросить ответить — значит заказать галлюцинацию (раздел 3).
        if result.empty:
            # Агент видел и то, что не прошло порог, — это полезная для него
            # информация («такими словами не находится»). Но в КОНТЕКСТ эти
            # фрагменты попасть не должны: порог их уже отверг, и протащить
            # их обратно через агента значило бы отменить порог, ничего об
            # этом не сказав.
            below = {hit.chunk.chunk_id for hit in result.hits}
            hits = [hit for hit in hits if hit.chunk.chunk_id not in below]

        if not hits:
            envelope = AnswerEnvelope(
                status=AnswerStatus.NO_CONTEXT,
                answer=(
                    "В вики нет фрагмента, который отвечал бы на этот вопрос. "
                    f"Лучшая найденная близость {result.best_vector_score:.2f} ниже "
                    f"порога {result.floor:.2f}."
                ),
            )
            trace.status = envelope.status.value
            trace.stop_reason = "below_similarity_floor"
            yield Event("meta", self._meta(
                trace, result, fragments=[], agent=outcome,
                turns=len(turns), rewritten=support,
            ))
            yield Event("answer", envelope.to_dict())
            yield Event("done", self._done(trace, FinishReason.STOP))
            return

        # 6. Контекст по бюджету токенов.
        with trace.span("build_context") as span:
            context = build_context(
                hits,
                token_budget=self._s.context_token_budget,
                max_fragments=self._s.context_max_fragments,
            )
            span.attributes.update(
                fragments=len(context.fragments),
                estimated_tokens=context.estimated_tokens,
                skipped_by_budget=context.skipped_by_budget,
            )

        yield Event("meta", self._meta(
            trace, result, fragments=context.fragments, agent=outcome,
            turns=len(turns), rewritten=support,
        ))

        # 6. Генерация со схемой, поток с инкрементальным разбором.
        request = ChatRequest(
            system=self._prompt.system,
            user=build_user_message(
                question, context.prompt_block, as_block(summary, turns), support,
                # Правила о таблицах — только когда таблицы есть. Инструкция
                # про то, чего нет в контексте, это чистый шум в промпте.
                TABLE_RULES if (outcome and outcome.datasets) else "",
            ),
            temperature=self._s.temperature,
            max_tokens=self._s.max_answer_tokens,
            seed=self._s.seed,
            json_schema=self._prompt.schema,
            model=model,
            # РАЗМЫШЛЕНИЯ ВЫКЛЮЧЕНЫ, и это не экономия, а условие работы.
            #
            # У думающей модели рассуждения идут ДО ответа и тратят тот же
            # лимит токенов. На `nemotron` первый же вопрос кончился так:
            # «обрезано по лимиту токенов; структура не разобрана:
            # JSONDecodeError: Expecting value: line 1 column 1 (char 0)».
            # «Char 0» здесь — главная улика: модель не отдала ни одного
            # байта JSON, весь лимит ушёл на размышления, и ответ даже не
            # начался.
            #
            # Рассуждать вслух этой модели тут и незачем: форму ответа
            # задаёт схема, а не размышление о том, какой она должна быть.
            # Агент и переписыватель вопроса давно зовутся с think=False —
            # основной ответ оставался последним местом, где переключатель
            # не выставляли вовсе.
            #
            # Модель, которая переключателя не понимает, обслуживается
            # адаптером: он снимает поле и повторяет запрос.
            think=False,
        )

        parser = StreamingAnswerParser()
        streamed: list[str] = []
        finish_reason = FinishReason.ERROR
        # Сколько модель наразмышляла до ответа.
        #
        # Раньше размышления просто выбрасывались, и обрыв по лимиту
        # токенов выглядел загадкой: ответ короткий, а лимит кончился.
        # Причина в том, что размышления тратят ТОТ ЖЕ лимит, и без этого
        # числа её не видно — ни в интерфейсе, ни в журнале.
        thought = 0

        def answered_by(name: str, used_model: str) -> None:
            """Кто на самом деле отвечает. Вызывается на первом куске потока.

            До этой правки в трейс писалась модель из настроек локального
            провайдера — независимо от того, кто отвечал. Пока провайдер был
            один, это совпадало. С двумя совпадать перестало, и сберовские
            ответы оказались подписаны именем `qwen`. Журнал, который врёт об
            авторе ответа, бесполезен ровно тогда, когда он нужен: при
            разборе жалобы через неделю.
            """
            trace.provider = name
            trace.model = used_model
            span.attributes["model"] = used_model
            span.attributes["provider"] = name

        with trace.span("generate", model=trace.model) as span:
            try:
                async for chunk in self._providers.stream_chat(
                    request, prefer=provider, on_provider=answered_by
                ):
                    if chunk.thinking:
                        thought += len(chunk.thinking)
                    if chunk.text:
                        trace.mark_first_token()
                        delta = parser.feed(chunk.text)
                        if delta:
                            streamed.append(delta)
                            yield Event("delta", {"text": delta})
                    if chunk.done:
                        finish_reason = chunk.finish_reason or FinishReason.STOP
                        if chunk.usage:
                            trace.prompt_tokens = chunk.usage.prompt_tokens
                            trace.completion_tokens = chunk.usage.completion_tokens
                            trace.cost_rub = chunk.usage.cost_rub
            except ProviderUnavailable as error:
                trace.status = "provider_unavailable"
                trace.stop_reason = "provider_unavailable"
                span.attributes["error"] = str(error)[:200]
                yield Event("error", {
                    "code": "provider_unavailable",
                    "message": "Модель сейчас недоступна. Попробуйте через полминуты.",
                    "detail": str(error)[:300],
                    "retry_after_s": 30,
                })
                return
            except ProviderError as error:
                trace.status = "provider_error"
                trace.stop_reason = "provider_error"
                span.attributes["error"] = str(error)[:200]
                # Если текст уже шёл — отдаём то, что успело прийти: человек
                # это уже прочитал, и стирать прочитанное хуже, чем показать
                # обрыв (раздел 5).
                if streamed:
                    yield Event("answer", AnswerEnvelope(
                        status=AnswerStatus.ANSWERED,
                        answer="".join(streamed),
                        schema_valid=False,
                        schema_error="поток прерван, структура не получена",
                    ).to_dict())
                    yield Event("done", self._done(trace, FinishReason.ERROR))
                else:
                    yield Event("error", {
                        "code": "provider_error",
                        "message": "Модель ответила ошибкой.",
                        "detail": str(error)[:300],
                        "retry_after_s": error.retry_after_s,
                    })
                return

        # 7. Валидация схемы и дословная проверка цитат.
        with trace.span("validate") as span:
            span.attributes["thinking_chars"] = thought
            envelope = answer_mod.finalize(
                parser.raw, context.fragments, streamed_text="".join(streamed),
                # Вопрос нужен детектору выдуманных идентификаторов: человек
                # вправе спросить «а что с PP-0007», и эхо его вопроса в
                # ответе — не выдумка.
                question=question,
            )
            span.attributes.update(
                unknown_refs=len(envelope.unknown_refs),
                mismatched_refs=len(envelope.mismatched_refs),
            )
            if finish_reason is FinishReason.LENGTH:
                # Ответ обрезан по лимиту токенов. Без этой пометки он выглядит
                # для пользователя как законченный (раздел 5).
                #
                # Условие «только если статус answered» здесь стояло раньше и
                # было ошибкой: при обрыве JSON не закрывается, разбор падает,
                # и статус к этому моменту уже НЕ answered — то есть пометка не
                # ставилась именно в том случае, для которого она нужна больше
                # всего. Причина обрыва важна всегда, независимо от того, что
                # получилось разобрать.
                envelope.schema_valid = False
                # Если модель размышляла — говорим об этом ПЕРВЫМ делом.
                # «Обрыв по лимиту» при коротком ответе выглядит как наша
                # поломка, пока не видно, что лимит съели размышления.
                thinking_note = (
                    f" (на размышления ушло {thought} знаков — они тратят тот же "
                    f"лимит, что и ответ)"
                    if thought
                    else ""
                )
                envelope.schema_error = (
                    f"обрыв по лимиту токенов{thinking_note}; {envelope.schema_error}"
                    if envelope.schema_error
                    else "ответ обрезан по лимиту токенов"
                )
            span.attributes.update(
                status=envelope.status.value,
                citations=len(envelope.citations),
                citations_failed=envelope.citations_failed,
                citations_dropped=envelope.citations_dropped,
                schema_valid=envelope.schema_valid,
            )

        trace.status = envelope.status.value
        trace.citations_failed = envelope.citations_failed
        trace.citations_dropped = envelope.citations_dropped
        trace.finish_reason = finish_reason.value
        trace.stop_reason = trace.stop_reason or finish_reason.value

        yield Event("answer", envelope.to_dict())

        # Сворачивание переписки в памятку — ПОСЛЕ ответа, а не до.
        #
        # Это отдельный вызов модели, и на локальной он стоит секунды.
        # Поставить его перед генерацией значило бы, что каждый пятый
        # вопрос человек ждёт вдвое дольше без всякой видимой причины.
        # После ответа те же секунды проходят незаметно: текст на экране
        # уже есть, поток просто ещё не закрылся.
        async for event in self._fold(arrived, summary, trace, provider, model):
            yield event

        yield Event("done", self._done(trace, finish_reason))

    async def _fold(self, arrived, summary, trace, provider, model):
        """Свернуть накопившуюся переписку в памятку и отдать её клиенту.

        ХРАНИТ ПАМЯТКУ КЛИЕНТ. Сервер её считает и отдаёт отдельным
        событием; браузер кладёт её в свою базу и присылает обратно
        следующим запросом вместо свёрнутых реплик. Сервер при этом
        остаётся без состояния — тем же, чем был до появления памяти.

        Зачем вообще сворачивать, если есть бюджет токенов. Бюджет
        отвечает на вопрос «сколько влезет», а не «что важно»: когда
        разговор перерастает бюджет, хвост отрезается, и первый вопрос — в
        котором обычно и задан предмет разговора — исчезает первым. Со
        стороны это выглядит так, будто ассистент вдруг поглупел.

        СЧИТАЕМ ПО ВЕСУ, А НЕ ПО ЧИСЛУ РЕПЛИК.

        Раньше свёртка запускалась на каждой десятой реплике. Десять —
        число с потолка, и оно меряет не то: десять коротких «ага» весят
        меньше одного вопроса с таблицей, а пять длинных разборов уже не
        влезают в бюджет. Порог по числу реплик срабатывает то слишком
        рано — и мы платим за вызов модели впустую, — то слишком поздно,
        и тогда хвост уже отрезан, то есть свёртывать нечего: предмет
        разговора выпал раньше, чем до него дошли руки.

        Бюджет токенов рядом, он уже считается в `trim`, и мерит он ровно
        то, из-за чего свёртка вообще нужна. Свёртываем, когда переписка
        перерастает бюджет — тогда и ровно тогда."""
        if self._s.history_fold_at <= 0 or not arrived:
            return

        # Порог по числу реплик остаётся нижней границей: он не даёт звать
        # модель на каждой второй реплике в разговоре из длинных ответов.
        if len(arrived) < self._s.history_fold_at:
            return
        weight = estimate_tokens("\n".join(turn.content for turn in arrived))
        if weight < self._s.history_token_budget:
            return
        with trace.span("fold_history", turns=len(arrived)) as span:
            folded = await summarize(
                summary, arrived,
                providers=self._providers, settings=self._s,
                provider=provider, model=model,
            )
            span.attributes.update(ok=bool(folded))
        if folded:
            # Не свернулось — не событие. Клиент оставит прежнюю памятку и
            # продолжит слать сырой хвост, то есть будет работать ровно
            # так, как работал до сворачивания.
            yield Event("memory", {"summary": folded, "folded_turns": len(arrived)})

    # ------------------------------------------------------------- сборка событий

    def _meta(
        self, trace: Trace, result, *, fragments,
        agent: AgentOutcome | None = None,
        turns: int = 0, rewritten: str = "",
    ) -> dict:
        """Метаданные для блока «как получен ответ».

        Отдаём и то, что попало в контекст, и то, что нашлось но не попало:
        именно на этой разнице потом видно, порог отсёк нужное или бюджет.

        Числа в `retrieval` описывают ПЕРВЫЙ поиск и только его. Смешивать
        их с добытым агентом нельзя: «лучшая близость» — это свойство
        одного запроса, и усреднять её по трём разным запросам значит
        получить величину, которая не значит ничего. Что добавил агент,
        видно отдельно, в `agent`.
        """
        return {
            "trace_id": trace.trace_id,
            "prompt_version": trace.prompt_version,
            "provider": trace.provider,
            "model": trace.model,
            # Что система знала о прошлом разговоре и чем в итоге искала.
            #
            # Второе поле важнее первого. Переписанный вопрос — это то, по
            # чему на самом деле шёл поиск, и без него «почему нашлось не
            # то» превращается в гадание: человек видит свой вопрос, а
            # искали другим. Пустая строка означает «искали ровно тем, что
            # человек написал», и это отдельный честный случай, а не
            # отсутствие данных.
            "history": {"turns": turns, "search_question": rewritten},
            "embed_model": self._s.ollama_embed_model,
            # Что делал агент. `null` — агент не участвовал вовсе, и это
            # НЕ то же самое, что «участвовал и ничего не сделал»: первое
            # означает обычную цепочку, второе — что модель посмотрела и
            # решила, что хватает. Различать их обязательно, иначе по
            # отчёту нельзя понять, работал ли режим вообще.
            "agent": None if agent is None else {
                "used_tools": agent.used_tools,
                "hit_limit": agent.hit_limit,
                # Предупреждать пользователя стоит только по второму полю:
                # шаги кончились И последний из них что-то принёс. Упёрлись
                # в предел после пустого шага — терять было нечего.
                "trail_cut": agent.trail_cut,
                "mechanism": agent.mechanism,
                # 600, а не 200: когда механизм — «схема (обрыв)», разбирать
                # придётся именно хвост бланка, а на 200 символах он до
                # отчёта не доезжает.
                "declined_with": agent.declined_with[:600],
                "steps": [
                    {
                        "number": step.number,
                        "tool": step.tool,
                        "arguments": step.arguments,
                        "result": step.result,
                        "added": step.added,
                        "ok": step.ok,
                    }
                    for step in agent.steps
                ],
            },
            "retrieval": {
                "best_cosine": round(result.best_vector_score, 4),
                "floor": result.floor,
                "passed_floor": result.passed_floor,
                "passed_by_keyword": result.passed_by_keyword,
                "hits_total": len(result.hits),
                "duplicates_dropped": result.dropped_duplicates,
            },
            "fragments": [
                {
                    "number": fragment.number,
                    # Живые данные, а не кусок документа. Признак —
                    # отрицательный номер куска: настоящие куски нумеруются
                    # базой с единицы. Интерфейсу это нужно, чтобы не вести
                    # по такой ссылке в документ, которого нет, а человеку —
                    # чтобы видеть: это снимок на момент запроса, а не
                    # цитата из регламента. Число из API устаревает, строчка
                    # регламента — нет.
                    "live": fragment.live,
                    # Номер куска в базе. Нужен интерфейсу, чтобы ссылка [4]
                    # вела не просто в документ, а в ТО САМОЕ место. Документ
                    # на сорок фрагментов без этого означает «ищи сам», и
                    # проверка цитаты, ради которой всё затевалось, перестаёт
                    # быть проверкой: никто не листает.
                    "chunk_id": fragment.chunk_id,
                    "doc_id": fragment.doc_id,
                    "doc_title": fragment.doc_title,
                    "heading_path": fragment.heading_path,
                    "page_from": fragment.page_from,
                    "page_to": fragment.page_to,
                    "source_label": fragment.source_label,
                    "found_by": fragment.found_by,
                    "cosine": round(fragment.vector_score, 4) if fragment.vector_score else None,
                    "rrf": round(fragment.fused_score, 5),
                    "doc_status": fragment.doc_status,
                    "doc_version": fragment.doc_version,
                    "body": fragment.body,
                }
                for fragment in fragments
            ],
        }

    def _done(self, trace: Trace, finish_reason: FinishReason) -> dict:
        return {
            "finish_reason": finish_reason.value,
            "stop_reason": trace.stop_reason or finish_reason.value,
            # Кто ответил НА САМОМ ДЕЛЕ.
            #
            # В событии `meta` стоит намерение: оно уходит до генерации, когда
            # ответчик ещё не известен. Если основной провайдер откажет и
            # вступит резерв, эти две пары разойдутся — и человек увидит, что
            # отвечала не та модель, которую он выбрал. Прятать такое нельзя:
            # именно из-за молчаливой подмены потом не сходятся замеры.
            "provider": trace.provider,
            "model": trace.model,
            "usage": {
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "cost_rub": trace.cost_rub,
            },
            "timing": {
                "ttft_ms": round(trace.ttft_ms, 1) if trace.ttft_ms is not None else None,
            },
            "trace_id": trace.trace_id,
        }
