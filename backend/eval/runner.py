"""Прогон золотого набора и сохранение результата целиком.

Два принципа, оба из раздела 4 чек-листа.

Первый: прогон идёт через ТОТ ЖЕ код, который обслуживает пользователей.
Не через «примерно такой же пайплайн для тестов» — иначе мы измеряем свою
копию, а не систему. Поэтому режим ответа вызывает `Pipeline.stream_answer`,
то есть ровно то, что дергает фронтенд.

Второй: сохраняется не средняя цифра, а прогон целиком — конфигурация и
результат по каждому вопросу. Средние скрывают структуру: «метрика не
изменилась» может означать «два вопроса починились, три сломались», и это
два, а не ноль изменений в поведении, о которых надо знать.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from app.config import Settings
from app.pipeline import Pipeline
from app.rag.embed import embed_query
from app.rag.history import Turn, needs_history, support_query
from app.rag.search import Query, hybrid_search_many
from app.rag.store import Store

from .dataset import Question
from .metrics import CORPUS_LANGUAGE, answer_contains, answer_language, mean, score_retrieval

Mode = Literal["search", "answer"]


@dataclass(slots=True)
class RunConfig:
    """Снимок всего, что влияет на результат.

    Без него прогон нельзя ни сравнить, ни воспроизвести: цифра 0.83 сама по
    себе ничего не значит, если неизвестно, при каком пороге и на какой модели
    она получена.
    """

    label: str
    mode: str
    chat_model: str
    embed_model: str
    prompt_version: str
    # Отпечаток кода, влияющего на ответ. Без него «конфигурации совпадают»
    # означает всего лишь «настройки совпадают», и сравнение двух версий кода
    # выглядит как замер разброса. См. app/version.py.
    code_version: str
    # Отпечаток разметки золотого набора — линейки, которой всё измерено.
    # Смена линейки делает два прогона несравнимыми, и это должно быть видно.
    dataset_version: str
    similarity_floor: float
    search_top_k: int
    context_max_fragments: int
    context_token_budget: int
    temperature: float
    rrf_k: int
    index_meta: dict
    index_chunks: int
    # Та же линейка, разобранная на две половины: разметку поиска пересчитать
    # задним числом нельзя, разметку ответа — можно. См. dataset.label_versions.
    labels_retrieval: str = ""
    labels_answer: str = ""
    # Прибор тоже часть конфигурации: смена модели судьи меняет judge_ok, и
    # это не изменение системы.
    judge_model: str = ""
    judge_think: bool | None = None
    judge_max_tokens: int = 0
    # Вес ключевого списка при слиянии. Обязан быть в отпечатке: без него два
    # прогона с разным весом выглядят как одна конфигурация, и рост метрики
    # читается как улучшение системы, хотя изменилась настройка. Ровно на этом
    # мы уже попадались с отпечатком кода и с моделью судьи.
    keyword_weight: float = 1.0
    # Цепочка провайдеров целиком: кто отвечал и кто стоял в резерве.
    #
    # Модель записана в `chat_model`, но её мало: «GigaChat-3-Lightning» не
    # говорит, был ли за ним локальный резерв и мог ли он вступить. Два
    # прогона с одной моделью и разным резервом — разные системы по
    # устойчивости, и различать их надо в шапке, а не по памяти.
    provider_chain: str = ""
    # Работал ли агентский шаг и сколько ему было разрешено шагов.
    #
    # Это ДРУГАЯ система, а не настройка вроде порога: меняется число
    # вызовов модели, расход токенов и состав контекста. Без этих двух
    # полей прогон с агентом и прогон без него легли бы в журнал под
    # одинаковой шапкой, и `compare` объявил бы разницу между ними
    # разбросом. Ровно на этом мы уже попадались — с отпечатком кода, с
    # моделью судьи и с весом ключевого списка.
    agent: bool = False
    agent_max_steps: int = 0
    # Ограничения по проектам. Прогон с закрытым проектом измеряет ДРУГУЮ
    # систему: часть корпуса просто не участвует в поиске, и метрики
    # поиска падают не потому, что он стал хуже. Без этого поля два таких
    # прогона легли бы в журнал под одинаковой шапкой.
    restricted_projects: str = ""
    started_at: float = field(default_factory=time.time)
    note: str = ""


@dataclass(slots=True)
class RowResult:
    question_id: str
    question: str
    type: str
    difficulty: str
    answerable: bool
    critical: bool
    retrieved_docs: list[str]
    retrieval: dict
    # Часть набора: настроечная или отложенная. Хранится в строке прогона,
    # чтобы разрез считался из файла и не зависел от того, не поменялся ли
    # набор с тех пор.
    split: str = "tune"
    # Поля режима ответа заполняются только в mode="answer"
    status: str = ""
    status_ok: bool | None = None
    answer: str = ""
    citations_total: int = 0
    citations_failed: int = 0
    # Непрошедшие ссылки целиком: номер фрагмента, причина и то, что указала
    # модель. Без этого «цитат не прошло: 16» — число без диагноза: чтобы
    # понять, промахивается модель мимо места или указывает верно, а нашего
    # поиска не хватает, нужен сам текст опоры. Я это уже проходил с судьёй.
    citations_bad: list[dict] = field(default_factory=list)
    # Ссылки, приложенные к ОТКАЗУ. Отдельный дефект, а не разновидность
    # непрошедшей цитаты: модель говорит «сведений нет» и одновременно
    # показывает пальцем на фрагменты. Это противоречие внутри одного ответа,
    # и лечится оно промптом, а не проверкой цитат.
    citations_on_refusal: int = 0
    # Ссылки, прошедшие после починки сервером: оборванное слово, склейка двух
    # настоящих мест, перепутанный номер. Дефект модели, вылеченный нами, — и
    # потому его надо считать отдельно, а не растворять в успехе.
    citations_repaired: int = 0
    schema_valid: bool | None = None
    # Дешёвая проверка правильности: есть ли обязательный факт в тексте
    # ответа. None — проверять нечем (нет must_contain или вопрос неотвечаемый).
    answer_contains: bool | None = None
    # Язык ответа: "ru", "zh", "en" или "" («букв нет, определять нечего»).
    #
    # Считается по буквам из сохранённого текста, поэтому досчитывается по
    # старым прогонам командой `recheck`. Нужен затем, что `answer_contains`
    # падает и когда факт не найден, и когда ответ на другом языке, — а это
    # две разные болезни. См. metrics.answer_language.
    answer_language: str = ""
    judge_verdict: bool | None = None
    judge_reason: str = ""
    # На какую выдержку опёрся судья: номер и заголовок. Хранится всегда, не
    # только при неудаче: по этому видно, ЧЕМ судья обосновал вердикт, а без
    # этого разбирать расхождения с человеком нечем.
    judge_fragment: int = 0
    judge_fragment_label: str = ""
    # Сырой ответ судьи — ТОЛЬКО когда вердикт не получился. Без него
    # «вердикт не разобран» нечем диагностировать: сам ответ уже потерян, и
    # остаётся гадать. Стенд обязан записывать достаточно, чтобы объяснить
    # собственную поломку; в успешном случае сырой текст не нужен и файл не
    # раздувает.
    judge_raw: str = ""
    # ------------------------------------------------- агентский режим
    #
    # ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ И ПОЧЕМУ НЕ ТЕКСТ.
    #
    # На вопросе «дай топ 5 труб» ответом служит ТАБЛИЦА, а не абзац. Всё,
    # что мы чинили в агенте за день — лишняя страница, выдуманный сервис,
    # ссылка именем документа, «я вижу только первую строку», — не ловится
    # проверкой подстрок в тексте: текст при этих поломках выглядит
    # осмысленно. Ловится это счётом таблиц, сверкой строк с сервисом и
    # детекторами выдуманных обозначений.
    #
    # Отсюда отдельные поля, а не переиспользование `answer_contains`: там
    # мерится пересказ, здесь — данные.
    tables: int = 0
    # Сколько таблиц ожидалось. None — вопрос не про таблицы.
    tables_ok: bool | None = None
    tools_used: list[str] = field(default_factory=list)
    tools_ok: bool | None = None
    # Обозначения, которых сервис не отдавал, и числа, разошедшиеся с ним.
    # Детекторы уже написаны и работают в проде — здесь они становятся
    # метрикой, а не только предупреждением на экране.
    invented_refs: list[str] = field(default_factory=list)
    mismatched_refs: list[str] = field(default_factory=list)
    live_clean: bool | None = None

    latency_ms: float = 0.0
    # Ошибка на этом вопросе. Прогон её переживает: одна упавшая модель не
    # должна уносить с собой семьдесят посчитанных вопросов.
    error: str = ""


async def run_search(
    questions: list[Question],
    *,
    store: Store,
    provider,
    settings: Settings,
) -> list[RowResult]:
    """Только поиск: без генерации, значит быстро и без недетерминированности.

    Такой прогон стоит делать основным рабочим инструментом. Он занимает
    секунды, ничего не стоит и отвечает на вопрос, который надо задавать
    первым: доехал ли нужный текст до контекста. Если нет — крутить промпт
    бессмысленно.
    """
    rows: list[RowResult] = []

    for question in questions:
        started = time.monotonic()
        # Переписка влияет на ПОИСК, а не только на ответ: обрывок
        # подпирается вторым запросом. Гонять режим поиска без неё значило
        # бы мерить систему, которой нет в проде.
        turns = [Turn(role=t["role"], content=t["content"]) for t in question.history]
        support = (
            support_query(question.question, turns)
            if turns and needs_history(question.question)
            else ""
        )
        wanted = [question.question] + ([support] if support else [])
        vectors = [await embed_query(provider, text) for text in wanted]
        # Веса те же, что в проде: склейка главная, голый вопрос уточняет.
        # Разойтись им нельзя — иначе замер меряет не ту систему.
        if support:
            queries = [Query(support, vectors[1], 1.0), Query(question.question, vectors[0], 0.5)]
        else:
            queries = [Query(question.question, vectors[0], 1.0)]

        result = await hybrid_search_many(
            store=store,
            queries=queries,
            top_k=settings.search_top_k,
            floor=settings.similarity_floor,
            rrf_k=settings.rrf_k,
            keyword_weight=settings.keyword_weight,
        )

        # Контекст собираем ровно так же, как в проде: с порогом, бюджетом и
        # дедупликацией. Иначе метрика «нужный абзац дошёл до модели» измеряла
        # бы не то, что реально доходит.
        from app.rag.context import build_context

        context = (
            build_context(
                result.hits,
                token_budget=settings.context_token_budget,
                max_fragments=settings.context_max_fragments,
            )
            if result.passed_floor
            else None
        )

        retrieved_docs: list[str] = []
        for hit in result.hits:
            if hit.chunk.doc_id not in retrieved_docs:
                retrieved_docs.append(hit.chunk.doc_id)

        context_text = (
            "\n".join(fragment.body for fragment in context.fragments) if context else ""
        )
        # Позиция нужного чанка и разрыв считаются по ПОЛНОЙ выдаче поиска, а
        # не по контексту: контекст уже обрезан порогом и бюджетом, и по нему
        # нельзя увидеть, на каком месте стоял правильный фрагмент.
        score = score_retrieval(
            retrieved_docs=retrieved_docs,
            relevant_docs=set(question.docs),
            context_text=context_text,
            must_contain=question.must_contain,
            best_cosine=result.best_vector_score,
            passed_floor=result.passed_floor,
            chunk_texts=[hit.chunk.body for hit in result.hits],
            cosines=[hit.vector_score for hit in result.hits if hit.vector_score is not None],
        )

        rows.append(
            RowResult(
                question_id=question.id,
                question=question.question,
                type=question.type,
                difficulty=question.difficulty,
                answerable=question.answerable,
                critical=question.critical,
                split=question.split,
                retrieved_docs=retrieved_docs[:10],
                retrieval=score.as_dict(),
                latency_ms=round((time.monotonic() - started) * 1000, 1),
            )
        )

    return rows


async def run_answer(
    questions: list[Question],
    *,
    pipeline: Pipeline,
    judge=None,
    on_progress=None,
    agent: bool = False,
) -> list[RowResult]:
    """Полный прогон с генерацией: медленно, дорого, недетерминированно.

    Поэтому он не заменяет поисковый прогон, а дополняет его. И поэтому же
    именно здесь нужен замер разброса: одна и та же система на одном и том же
    наборе даёт разные цифры от прогона к прогону.

    Про устойчивость отдельно. Первая версия этой функции падала целиком, если
    провайдер отвечал ошибкой на одном вопросе, — и уносила с собой все
    посчитанные до него. На семидесяти семи вопросах через две модели это
    неприемлемо: измерительный стенд обязан быть устойчивее того, что он
    измеряет. Теперь ошибка на вопросе становится строкой прогона со статусом
    error, а `on_progress` позволяет сохранять частичный результат, чтобы
    падение не стоило всей работы.
    """
    rows: list[RowResult] = []

    for number, question in enumerate(questions, start=1):
        started = time.monotonic()
        meta: dict = {}
        answer: dict = {}
        datasets: list[dict] = []
        error: dict = {}
        failure = ""

        try:
            async for event in pipeline.stream_answer(
                question.question,
                agent=agent,
                history=[
                    Turn(role=turn["role"], content=turn["content"])
                    for turn in question.history
                ],
            ):
                if event.name == "meta":
                    meta = event.data
                elif event.name == "answer":
                    answer = event.data
                elif event.name == "dataset":
                    # Таблицы копим ЦЕЛИКОМ, а не считаем на лету: по ним
                    # потом сверяются строки с тем, что отдал сервис, и
                    # выбрасывать данные ради счётчика было бы обидно.
                    datasets.append(event.data)
                elif event.name == "error":
                    error = event.data
        except Exception as exception:  # noqa: BLE001 — тут ловим намеренно всё
            failure = f"{exception.__class__.__name__}: {exception}"[:300]

        status = str(answer.get("status") or ("error" if (error or failure) else ""))
        fragments = meta.get("fragments") or []
        retrieved_docs: list[str] = []
        for fragment in fragments:
            if fragment["doc_id"] not in retrieved_docs:
                retrieved_docs.append(fragment["doc_id"])

        retrieval = meta.get("retrieval") or {}
        context_text = "\n".join(fragment.get("body", "") for fragment in fragments)
        score = score_retrieval(
            retrieved_docs=retrieved_docs,
            relevant_docs=set(question.docs),
            context_text=context_text,
            must_contain=question.must_contain,
            best_cosine=float(retrieval.get("best_cosine") or 0.0),
            passed_floor=bool(retrieval.get("passed_floor")),
            # В режиме ответа доступен только контекст — то есть позиция
            # фрагмента среди того, что реально увидела модель.
            chunk_texts=[fragment.get("body", "") for fragment in fragments],
            cosines=[
                fragment["cosine"] for fragment in fragments if fragment.get("cosine") is not None
            ],
        )

        row = RowResult(
            question_id=question.id,
            question=question.question,
            type=question.type,
            difficulty=question.difficulty,
            answerable=question.answerable,
            critical=question.critical,
            split=question.split,
            retrieved_docs=retrieved_docs[:10],
            retrieval=score.as_dict(),
            status=status,
            status_ok=status in question.expected_status,
            answer=str(answer.get("answer", ""))[:2000],
            citations_total=len(answer.get("citations") or []),
            citations_failed=int(answer.get("citations_failed") or 0),
            citations_repaired=int(answer.get("citations_repaired") or 0),
            citations_bad=[
                {
                    "fragment": citation.get("fragment"),
                    "reason": str(citation.get("reason", ""))[:120],
                    "handle": str(citation.get("handle", ""))[:160],
                }
                for citation in (answer.get("citations") or [])
                if not citation.get("ok")
            ][:4],
            # Ссылки при отказе теперь снимает сервер, и в `citations` их уже
            # нет — поэтому считаем по числу снятых. Второе слагаемое оставлено
            # для прогонов, сделанных до этой правки: там ссылки лежат в
            # массиве, и без него старые прогоны начали бы показывать ноль там,
            # где дефект был, — то есть сравнение «до/после» соврало бы в нашу
            # пользу.
            tables=len(datasets),
            tables_ok=(
                None if question.expect_tables is None
                else len(datasets) == question.expect_tables
            ),
            tools_used=[step["tool"] for step in ((meta.get("agent") or {}).get("steps") or [])],
            tools_ok=(
                None if not question.expect_tools
                else set(question.expect_tools)
                <= {step["tool"] for step in ((meta.get("agent") or {}).get("steps") or [])}
            ),
            invented_refs=list(answer.get("unknown_refs") or []),
            mismatched_refs=list(answer.get("mismatched_refs") or []),
            # «Чисто» означает: ни одного обозначения, которого сервис не
            # отдавал, и ни одного числа, разошедшегося с ним. Проверяется
            # только там, где таблицы вообще ожидались: у вопроса по
            # документам выдумывать нечего.
            live_clean=(
                None if question.expect_tables is None
                else not (answer.get("unknown_refs") or answer.get("mismatched_refs"))
            ),
            citations_on_refusal=(
                int(answer.get("citations_dropped") or 0)
                + (
                    len(answer.get("citations") or [])
                    if status in {"not_found", "no_context"}
                    else 0
                )
            ),
            schema_valid=answer.get("schema_valid"),
            answer_contains=(
                answer_contains(str(answer.get("answer", "")), question.answer_needles)
                if question.answerable
                else None
            ),
            # Язык считается по ВСЕМ вопросам, включая неотвечаемые.
            #
            # Именно на них он и нужен больше всего: отказ на китайском —
            # формально правильное поведение, которое пользователь вики не
            # прочтёт. Ограничить проверку отвечаемыми значило бы не видеть
            # худшую половину случаев.
            answer_language=answer_language(str(answer.get("answer", ""))),
            latency_ms=round((time.monotonic() - started) * 1000, 1),
            error=failure or str(error.get("detail") or error.get("message") or "")[:300],
        )

        # Судья вызывается только там, где он осмыслен: система дала ответ на
        # отвечаемый вопрос. Судить отказ по неотвечаемому вопросу нечего —
        # правильность отказа проверяется статусом, без модели и без затрат.
        if judge is not None and judgeable(row):
            try:
                verdict = await judge.verdict(question=question, answer=row.answer)
                row.judge_verdict = verdict.correct
                row.judge_reason = verdict.reason or "без причины"
                row.judge_fragment = verdict.fragment
                row.judge_fragment_label = verdict.fragment_label[:160]
                row.judge_raw = "" if verdict.correct is not None else verdict.raw[:400]
            except Exception as exception:  # noqa: BLE001
                row.judge_reason = f"судья упал: {exception.__class__.__name__}"
                row.error = row.error or f"{exception.__class__.__name__}: {exception}"[:300]

        rows.append(row)
        if on_progress is not None:
            on_progress(number, len(questions), rows)

    return rows


def judgeable(row) -> bool:
    """Можно ли отдавать строку судье — или человеку на разметку.

    Раньше условие было написано прямо в трёх местах и выглядело так:

        if not row.answerable or row.status != "answered":
            continue

    Оно правильно для нашей системы и МОЛЧА вырезает чужую. У образца на
    LlamaIndex поле `status` пустое — сознательно: своей схемы ответа у него
    нет, ярлык «ответил / не найдено» он не выставляет, и писать туда
    что-нибудь было бы выдумкой. В результате `eval.py judge alt1` печатал «к
    оценке 0 ответов» и возвращал успех: единственный прибор, который смотрит
    на СМЫСЛ ответа, не оценил ни одной строки второй реализации, и никто об
    этом не узнал.

    Ошибка ровно того класса, ради которого затевался аудит: инструмент не
    измерил и не пожаловался. И ровно та же подмена, что и с нулём вместо
    прочерка, только с другой стороны: пустая строка прочиталась как «система
    не ответила», хотя означает «статус не измерялся».

    Поэтому решение принимается по трём признакам, а не по одному:

    - вопрос отвечаемый — иначе судить нечего, правильность отказа
      проверяется статусом, без модели и без затрат;
    - на вопросе не было ошибки — судить обломок прогона бессмысленно;
    - система что-то ответила: либо статус прямо говорит `answered`, либо
      статуса нет вообще, но текст ответа есть.

    Последний пункт — главный, и он же страхует нашу собственную систему:
    вырожденный случай, когда пайплайн не отдал ни ответа, ни ошибки, даёт
    пустой статус И пустой текст, и такая строка судье по-прежнему не идёт.

    Принимает и `RowResult`, и словарь из файла прогона. Это не всеядность
    ради удобства. Строка прогона физически существует в двух видах — объектом
    в памяти и словарём на диске, — и разные команды стенда работают с разными:
    судейство с объектами, ручная разметка и замер чувствительности со
    словарями. Два условия допуска для двух представлений ОДНОЙ строки
    разойдутся ровно так же, как разошлись четыре рукописные копии до этого, —
    и разошлись бы молча.
    """
    if isinstance(row, dict):
        answerable = bool(row.get("answerable"))
        error = str(row.get("error") or "")
        status = str(row.get("status") or "")
        answer = str(row.get("answer") or "")
    else:
        answerable = row.answerable
        error = row.error
        status = row.status
        answer = row.answer

    if not answerable:
        return False
    if error:
        return False
    if status:
        return status == "answered"
    return bool(answer.strip())


async def judge_rows(
    rows: list[RowResult],
    *,
    questions: list[Question],
    judge,
    on_progress=None,
) -> int:
    """Второй проход: судья по УЖЕ сохранённым ответам.

    Разделение на два прохода — не удобство, а требование железа. На машине с
    одной видеокартой отвечающая модель и судья одновременно не помещаются:
    попытка поднять вторую рядом с первой уронила llama-server с ошибкой CUDA.
    Если этапы разнесены, в памяти всегда одна модель.

    Проход идемпотентный: уже вынесенные вердикты не пересчитываются, поэтому
    прерванное судейство можно просто запустить снова.

    «Уже оценено» определяется не по вердикту, а по наличию ПРИЧИНЫ. Вердикт
    бывает пустым в двух совершенно разных случаях: строку ещё не судили, и
    строку судили, но судья не смог. Если различать их только по вердикту,
    второй случай будет перезапускаться на каждом проходе вечно — а он, как
    правило, воспроизводится.
    """
    by_id = {question.id: question for question in questions}
    judged = 0

    for number, row in enumerate(rows, start=1):
        if row.judge_verdict is not None or row.judge_reason:
            continue
        if not judgeable(row):
            continue
        question = by_id.get(row.question_id)
        if question is None:
            continue

        try:
            verdict = await judge.verdict(question=question, answer=row.answer)
            row.judge_verdict = verdict.correct
            row.judge_reason = verdict.reason or "без причины"
            row.judge_fragment = verdict.fragment
            row.judge_fragment_label = verdict.fragment_label[:160]
            row.judge_raw = "" if verdict.correct is not None else verdict.raw[:400]
            if verdict.correct is not None:
                judged += 1
        except Exception as exception:  # noqa: BLE001
            row.judge_reason = f"судья упал: {exception.__class__.__name__}"
            row.error = row.error or f"{exception.__class__.__name__}: {exception}"[:300]

        if on_progress is not None:
            on_progress(number, len(rows), rows)

    return judged


def _rate(rows, field_name: str) -> float | None:
    """Доля True среди тех строк, где проверка вообще проводилась.

    None вместо нуля, когда проверять было нечего. Ноль здесь читался бы
    как «всё плохо», а означал бы «в этом подмножестве нет таких вопросов»
    — ровно та разница, из-за которой можно полдня чинить несуществующую
    беду.
    """
    values = [
        getattr(row, field_name) for row in rows if getattr(row, field_name) is not None
    ]
    return mean([float(value) for value in values]) if values else None


def aggregate(rows: list[RowResult]) -> dict:
    """Сводка: общая, по типам, по сложности, по критичности."""

    def block(subset: list[RowResult]) -> dict:
        if not subset:
            return {}

        def over_answerable(values: list[float | None]) -> float | None:
            """Метрика поиска считается только по отвечаемым вопросам.

            Если отвечаемых в подмножестве нет (блок неотвечаемых, тип
            unanswerable), мерить нечего — и это None, а не ноль. Ноль в
            таблице читался как полный провал поиска на этом типе: строка
            `unanswerable 10 0.000 0.000 0.000` выглядела катастрофой там, где
            просто нет предмета измерения.
            """
            # None среди значений — это «у вопроса не было ожидаемых
            # документов», и такие вопросы из среднего выпадают. Иначе
            # двенадцать вопросов про живые данные утягивали общий recall@5
            # с 0.977 до 0.928: метрика поиска отчитывалась о провале там,
            # где поиска не было.
            measured = [value for value in values if value is not None]
            return mean(measured) if measured else None
        answered = [row for row in subset if row.status]
        judged = [row for row in subset if row.judge_verdict is not None]
        # Строки, которые судья пытался оценить и не смог. Это не «неверно»,
        # это неисправность прибора, и считать её надо отдельно: если таких
        # много, судейской цифре нельзя верить вообще.
        unresolved = [
            row
            for row in subset
            if row.judge_verdict is None and row.judge_reason and row.answerable
        ]
        # Отдельно — вердикты, отвергнутые из-за ссылки на несуществующую
        # выдержку. Это не «прибор не сработал», это «прибор указал в пустоту»,
        # и цифра должна быть на виду: она характеризует судью, а не систему.
        invented = [
            row
            for row in subset
            if row.judge_verdict is None and "которой нет" in row.judge_reason
        ]
        checkable = [row for row in subset if row.answer_contains is not None]
        # Строки, у которых язык вообще определился. Пустой ответ и поисковый
        # прогон сюда не входят: у них языка нет, а не «язык неправильный».
        with_language = [row for row in subset if row.answer_language]
        return {
            "n": len(subset),
            "recall@5": over_answerable([row.retrieval["recall@5"] for row in subset if row.answerable]),
            "mrr": over_answerable([row.retrieval["mrr"] for row in subset if row.answerable]),
            "ndcg@10": over_answerable([row.retrieval["ndcg@10"] for row in subset if row.answerable]),
            # Метрика на уровне чанков: в отличие от recall по документам она
            # не насыщается на маленьком корпусе и видит работу реранкера.
            "chunk_mrr": over_answerable(
                [row.retrieval.get("chunk_mrr", 0.0) for row in subset if row.answerable]
            ),
            "chunk_top1": over_answerable(
                [
                    # None по той же причине, что и у chunk_mrr: без
                    # ожидаемой подстроки «нужный чанк не первый» значит
                    # только, что нужного чанка не было задано.
                    None
                    if row.retrieval.get("chunk_mrr") is None
                    else float(row.retrieval.get("chunk_rank") == 1)
                    for row in subset
                    if row.answerable
                ]
            ),
            "context_hit": over_answerable(
                [float(row.retrieval["context_hit"]) for row in subset if row.answerable]
            ),
            # None, а не 0.0: у поискового прогона статусов нет вообще, и ноль
            # печатался как измеренная величина — «правильные отказы 0.000 из
            # 10», то есть «система не отвечает». Единообразие с остальными
            # необязательными метриками.
            "status_ok": (
                mean([float(bool(row.status_ok)) for row in answered]) if answered else None
            ),
            # Ключевой факт в тексте ответа. Считается по всем отвечаемым
            # вопросам, включая те, где система отказалась: отказ там, где
            # ответ есть, — это провал ответа, а не отсутствие измерения.
            "answer_contains": (
                mean([float(bool(row.answer_contains)) for row in checkable])
                if checkable
                else None
            ),
            # Доля ответов на языке корпуса. Считается по ВСЕМ строкам, где
            # язык определился, — и по отвечаемым, и по неотвечаемым: отказ на
            # чужом языке остаётся ответом, который пользователь не прочтёт.
            #
            # None при поисковом прогоне: там ответов нет вообще, и ноль
            # читался бы как «вся система отвечает не на том языке».
            "language_ok": (
                mean(
                    [float(row.answer_language == CORPUS_LANGUAGE) for row in with_language]
                )
                if with_language
                else None
            ),
            # Сколько и на каких языках. Одной доли мало: «11 % не на том
            # языке» не говорит, английский это (читаемо) или китайский (нет),
            # а лечится это разными строчками промпта.
            "languages": {
                language: sum(1 for row in with_language if row.answer_language == language)
                for language in sorted({row.answer_language for row in with_language})
            },
            # None, а не 0.0: «судья не запускался» и «судья всё забраковал» —
            # это разные новости, и ноль вместо пустоты читается как катастрофа.
            "judge_ok": (
                mean([float(bool(row.judge_verdict)) for row in judged]) if judged else None
            ),
            "judged": len(judged),
            "judge_unresolved": len(unresolved),
            "judge_bad_reference": len(invented),
            # ДОЛЯ, А НЕ ТОЛЬКО ШТУКИ, и это не косметика.
            #
            # `compare` сравнивает метрики из сводки, а счётчик штук в неё не
            # входит. Из-за этого правка, которая чинила ровно цитаты, в
            # сравнении выглядела как «починилось 0»: свою работу она
            # сделала (неудачных ссылок стало с восьми пять), но увидеть это
            # можно было только глазами, листая два отчёта рядом.
            #
            # Правило общее: то, ради чего правят, обязано быть в сводке.
            # Иначе мерим не то, что чиним.
            # Агентские метрики. None, когда в подмножестве нет ни одного
            # вопроса про живые данные: ноль читался бы как «всё плохо».
            "tables_ok": _rate(subset, "tables_ok"),
            "tools_ok": _rate(subset, "tools_ok"),
            "live_clean": _rate(subset, "live_clean"),
            "citations_ok": (
                1.0 - sum(row.citations_failed for row in subset) / total_citations
                if (total_citations := sum(row.citations_total for row in subset))
                else None
            ),
            "citations_failed": sum(row.citations_failed for row in subset),
            "citations_repaired": sum(row.citations_repaired for row in subset),
            "citations_on_refusal": sum(row.citations_on_refusal for row in subset),
            # Упавшие вопросы обязаны быть видны в сводке. Без счётчика двадцать
            # ошибок провайдера выглядели в compare как обычные регрессии:
            # пустой контекст, context_hit False — не отличить от «поиск стал
            # хуже».
            "errors": sum(1 for row in subset if row.error),
            "latency_ms_avg": mean([row.latency_ms for row in subset]),
        }

    by_type: dict[str, dict] = {}
    for row in rows:
        by_type.setdefault(row.type, []).append(row)  # type: ignore[arg-type]
    by_difficulty: dict[str, dict] = {}
    for row in rows:
        by_difficulty.setdefault(row.difficulty, []).append(row)  # type: ignore[arg-type]

    return {
        "overall": block(rows),
        "answerable": block([row for row in rows if row.answerable]),
        "unanswerable": block([row for row in rows if not row.answerable]),
        "critical": block([row for row in rows if row.critical]),
        "by_type": {key: block(value) for key, value in sorted(by_type.items())},  # type: ignore[arg-type]
        "by_difficulty": {
            key: block(value) for key, value in sorted(by_difficulty.items())  # type: ignore[arg-type]
        },
        # РАЗРЕЗ ПО ЧАСТЯМ НАБОРА — в каждом прогоне, без просьбы.
        #
        # Настройки выбираются только на настроечной части, а честная цифра —
        # на отложенной. Если разрез считать «когда понадобится», он не будет
        # посчитан никогда: смотреть на него неприятно ровно в тот момент,
        # когда он важен.
        "by_split": {
            key: block([row for row in rows if row.split == key])
            for key in ("tune", "holdout")
        },
    }


def save_run(
    *, config: RunConfig, rows: list[RowResult], directory: Path
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(config.started_at))
    path = directory / f"{stamp}-{config.mode}-{config.label}.json"
    payload = {
        "config": asdict(config),
        "aggregate": aggregate(rows),
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_run(run: dict) -> list[RowResult]:
    """Читает строки сохранённого прогона, терпимо к чужим полям.

    Файл прогона — это ФОРМАТ, а не внутренняя структура: прогоны месячной
    давности надо уметь открывать сегодняшним кодом. Значит переименование поля
    ломает не текущий прогон, а все прошлые, и `RowResult(**row)` падает на
    ключе, которого в классе больше нет.

    Мы на этом и поймались: поле `judge_quote` стало `judge_fragment`, и
    прогон, записанный двадцатью минутами раньше, перестал читаться вообще.

    Поэтому: неизвестные поля отбрасываются, отсутствующие берутся по
    умолчанию. Терпимость тут не лень, а требование к формату исторических
    записей — измерение, которое нельзя перечитать, перестаёт быть измерением.
    Отброшенные ключи печатаются один раз: молча терять данные нельзя, иначе
    однажды потеряется то, что было нужно.
    """
    known = set(RowResult.__dataclass_fields__)
    dropped: set[str] = set()
    rows: list[RowResult] = []

    for row in run["rows"]:
        extra = set(row) - known
        dropped |= extra
        rows.append(RowResult(**{key: value for key, value in row.items() if key in known}))

    if dropped:
        print(
            f"  прогон записан другой версией стенда, поля не читаются: "
            f"{', '.join(sorted(dropped))}"
        )
    return rows


def write_run(path: Path, *, config: dict, rows: list[RowResult]) -> None:
    """Перезаписывает прогон с пересчётом агрегатов.

    Нужна, чтобы второй проход (судейство) дописывал вердикты в тот же файл, а
    не создавал второй прогон: прогон — это одно измерение одной конфигурации,
    и разваливать его на два файла значит потерять возможность сравнивать.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "aggregate": aggregate(rows),
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
