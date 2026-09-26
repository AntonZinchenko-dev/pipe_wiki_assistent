"""Инструменты, которые модель имеет право вызвать сама.

Зачем это вообще появилось. Наш конвейер — цепочка: ищем один раз, отдаём
найденное модели, получаем ответ. Шаги известны заранее, и пока вопрос
похож на документ, этого хватает. Ломается цепочка в двух случаях, и оба
видны в замерах.

Первый: вопрос задан не теми словами, что документ. «Как отключить кэш» и
«параметр ttl_cache» — про одно и то же, но ищутся по-разному. Цепочка
ошибается один раз и навсегда: второго шанса у неё нет.

Второй: найденный фрагмент ссылается на соседний. «Порядок описан в разделе
4.2» — и всё, дальше в контексте пусто. Человек в такой ситуации листает
документ; цепочка не умеет.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ПЕРЕПИСЫВАНИЯ ЗАПРОСА, КОТОРОЕ МЫ УЖЕ МЕРИЛИ И
ВЫКИНУЛИ. Тогда переписывание было БЕЗУСЛОВНЫМ: переписывался каждый
вопрос, в том числе те, что и так находились. Выигрыш на плохих вопросах
съедался проигрышем на хороших, и в сумме стало хуже. Здесь решение
принимается ПОСЛЕ первого поиска и по его результату: модель видит, что
нашлось, и ищет ещё только если нашлось плохо. Это другой эксперимент, и
мерить его надо заново — предыдущий результат его не предсказывает.

ЧТО ЗДЕСЬ СОЗНАТЕЛЬНО НЕ СДЕЛАНО.

Инструменты только ЧИТАЮТ. Ни одного, который меняет данные, отправляет
письмо или ходит в сеть. Модель управляется текстом из документов, а
документ в вики может положить любой сотрудник; инструмент с побочным
эффектом превращает «положил документ» в «выполнил команду на сервере».
Список инструментов — это список прав, и он должен начинаться с минимума.

Результат инструмента возвращается модели как ТЕКСТ ИЗ ДОКУМЕНТОВ, то есть
как недоверенные данные: он проходит ту же чистку, что и контекст. Иначе мы
бы закрыли парадную дверь и оставили открытой чёрную — инъекция через
результат поиска работает ровно так же, как через контекст.

Полные тексты в диалог агента не уходят. Модель видит короткие выжимки и
решает по ним; сами фрагменты копятся отдельно и попадают в контекст один
раз, в конце, через обычный `build_context` с его бюджетом токенов. Иначе
за три шага мы бы оплатили один и тот же текст трижды.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from .context import sanitize_document_text
from .embed import embed_query
from .live import LiveApi
from .result import Dataset
from .search import Hit, hybrid_search
from .store import Store

# Сколько строк выжимки показываем модели на один вызов. Больше не нужно:
# решение «искать ещё или хватит» принимается по верхушке списка, а каждая
# лишняя строка — это токены в КАЖДОМ следующем шаге, потому что история
# отправляется целиком.
PREVIEW_LINES = 6
PREVIEW_CHARS = 180

# Потолок на один вызов `read_section`: документ целиком в контекст не
# влезет, а молча обрезать до бюджета — значит отдать случайную половину.
MAX_SECTION_CHUNKS = 6


# Имена живых инструментов берутся ИЗ САМИХ СПЕЦИФИКАЦИЙ.
#
# Рядом лежал набор из двух имён, и при добавлении третьего инструмента его
# забыли бы там дописать: инструмент показался бы в списке, модель бы его
# позвала, а проверка прав не узнала бы имени и вернула «такого нет».
# Ошибка выглядела бы как галлюцинация модели.
LIVE_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "live_pipe",
            "description": (
                "Текущие данные по ОДНОЙ трубе из FATIGUE-API: выработка ресурса, "
                "циклы, критическое сечение. Вызывай, когда в вопросе назван "
                "конкретный идентификатор трубы (например PP-0035) и нужно её "
                "нынешнее состояние. Документы этих чисел не содержат."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pipe_id": {
                        "type": "string",
                        "description": "Идентификатор трубы, например PP-0035",
                    }
                },
                "required": ["pipe_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "live_fleet",
            "description": (
                "Трубы парка по выработке ресурса, от самой изношенной. Два "
                "разных вопроса, и параметры для них разные. «Какие трубы за "
                "порогом» — передай min_damage долей от нуля до единицы и бери "
                "порог ИЗ ДОКУМЕНТОВ, а не из головы: при пороге аварии 80 "
                "процентов передай 0.8. «Топ N самых изношенных», «пять худших», "
                "«покажи десять труб» — передай top_n и НЕ ПРИДУМЫВАЙ порог: "
                "подобранный на глаз порог вернёт не то число труб, о котором "
                "спрашивали. «Следующие пять», «ещё», «дальше» — передай ТОЛЬКО "
                "cursor, взятый из прошлого ответа этого инструмента, как есть. "
                "Ничего не вычисляй и не добавляй к нему других параметров: "
                "метка уже хранит и порог, и размер страницы, и место."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_damage": {
                        "type": "number",
                        "description": "Порог: доля выработки от 0 до 1, например 0.8",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Сколько самых изношенных труб вернуть, например 5",
                    },
                    "cursor": {
                        "type": "string",
                        "description": (
                            "Метка следующей страницы из прошлого ответа. "
                            "Передавай строку дословно"
                        ),
                    },
                },
                # Обязательных нет: без параметров это «покажи парк по
                # убыванию выработки», и это осмысленный вызов. Требовать
                # порог значило бы заставлять модель выдумывать число там,
                # где спрашивали про количество.
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "live_inspections",
            "description": (
                "Инспекции труб из FATIGUE-API: категория осмотра, износ стенки, "
                "дата и просрочка. Это ЕДИНСТВЕННЫЙ источник по списанию: "
                "категория SCRAP — решение инспектора, и выработка ресурса к нему "
                "отношения не имеет. Вопросы «какие трубы под списание», «что "
                "показал осмотр», «у кого просрочена инспекция» — сюда, а не в "
                "live_fleet. Ответить на них по выработке нельзя: правило "
                "«по расчёту труба не списывается никогда» записано в документах. "
                "«Следующие пять», «ещё» — передай ТОЛЬКО cursor из прошлого "
                "ответа, как есть."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Категория осмотра: PREMIUM, CLASS_2, CLASS_3 или "
                            "SCRAP. SCRAP — под списание"
                        ),
                    },
                    "overdue": {
                        "type": "boolean",
                        "description": "Только с просроченной инспекцией",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Сколько строк вернуть, например 5",
                    },
                    "cursor": {
                        "type": "string",
                        "description": (
                            "Метка следующей страницы из прошлого ответа. "
                            "Передавай строку дословно"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "live_wells",
            "description": (
                "Скважины из FATIGUE-API со сводкой по трубам: сколько труб и "
                "какая у худшей из них выработка ресурса. Вопросы «на какой "
                "скважине хуже всего», «сколько труб на W-122», «скважины "
                "Самотлора» — сюда. Про отдельные трубы — live_fleet или "
                "live_pipe. «Следующие пять», «ещё» — передай ТОЛЬКО cursor из "
                "прошлого ответа, как есть."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "Месторождение, например Самотлор",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Сколько скважин вернуть, например 5",
                    },
                    "cursor": {
                        "type": "string",
                        "description": (
                            "Метка следующей страницы из прошлого ответа. "
                            "Передавай строку дословно"
                        ),
                    },
                },
                "required": [],
            },
        },
    }
]


def tool_specs() -> list[dict]:
    """Описание инструментов в нейтральной форме.

    Форма здесь одна для всех провайдеров: `{"type": "function",
    "function": {...}}`. Перевод в то, что понимает конкретный сервис,
    делает адаптер провайдера — ровно как с формой схемы ответа. Если бы
    перевод жил здесь, каждый новый провайдер означал бы правку агента.

    Описания написаны для МОДЕЛИ, а не для человека: в них сказано, когда
    инструмент уместен, а когда нет. Описание вида «ищет в вики» приводит к
    тому, что модель вызывает поиск на каждый вопрос, включая те, что уже
    нашлись.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_wiki",
                "description": (
                    "Повторный поиск по вики другими словами. Вызывай ТОЛЬКО если "
                    "среди уже найденных фрагментов нет ответа на вопрос. "
                    "Формулируй запрос терминами документации, а не словами вопроса: "
                    "например вместо «как отключить кэш» — «параметр времени жизни кэша». "
                    "Если ответ уже есть в найденном, не вызывай ничего."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос терминами документации",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_section",
                "description": (
                    "Дочитать соседний раздел документа. Вызывай, если найденный "
                    "фрагмент ссылается на другой раздел этого же документа "
                    "(«см. раздел…», «порядок описан ниже»), и без него ответа нет. "
                    "doc_id бери из уже найденных фрагментов."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Код документа из найденных фрагментов",
                        },
                        "heading": {
                            "type": "string",
                            "description": "Часть названия раздела, например «4.2» или «Фильтрация»",
                        },
                    },
                    "required": ["doc_id", "heading"],
                },
            },
        },
    ]


@dataclass(slots=True)
class ToolOutcome:
    """Что вернул инструмент.

    `text` уходит модели, `hits` копятся для контекста. Это РАЗНЫЕ вещи, и
    разделение здесь не случайное: модель принимает решение по выжимке, а
    отвечает по полному тексту, собранному один раз в конце.
    """

    text: str
    hits: list[Hit] = field(default_factory=list)
    ok: bool = True
    # Таблица целиком — для БРАУЗЕРА, мимо модели.
    #
    # Ни в `text`, ни в `hits` её строк нет: модель получает выжимку. Это
    # и есть весь смысл переделки — то, что не сгенерировано, не может
    # быть сгенерировано неправильно.
    dataset: "Dataset | None" = None


LIVE_TOOL_NAMES = frozenset(spec["function"]["name"] for spec in LIVE_SPECS)


class Toolbox:
    def __init__(
        self, *, store: Store, settings: Settings, embedder, live: LiveApi | None = None
    ) -> None:
        self._store = store
        self._s = settings
        self._embedder = embedder
        self._live = live

    def specs(self, *, live_allowed: bool = True) -> list[dict]:
        """Инструменты, которые модель видит СЕЙЧАС.

        Список собирается по состоянию системы, а не берётся константой.
        Показать инструмент, который не настроен, — худший вариант из
        возможных: модель попробует его, получит ошибку, попробует ещё раз
        и потратит на это все разрешённые шаги. Отсутствующий инструмент
        честнее сломанного.
        """
        specs = tool_specs()
        # Два условия, и оба обязательны: сервис настроен И человеку можно
        # читать боевые данные. Первое — про систему, второе — про права
        # конкретного человека, и подменять одно другим нельзя.
        if self._live is not None and self._live.configured and live_allowed:
            specs = specs + LIVE_SPECS
        return specs

    async def run(
        self, name: str, arguments: dict, *, live_allowed: bool = True
    ) -> ToolOutcome:
        """Выполнить инструмент по имени.

        Неизвестное имя — не исключение, а обычный ответ инструмента с
        текстом ошибки. Модель придумывает несуществующие инструменты
        регулярно, и падать на этом нельзя: она должна прочитать «такого
        нет» и продолжить. Исключение здесь стоило бы пользователю всего
        ответа из-за одной галлюцинации в названии.
        """
        if name == "search_wiki":
            return await self._search(str(arguments.get("query") or "").strip())
        if name == "read_section":
            return self._read_section(
                str(arguments.get("doc_id") or "").strip(),
                str(arguments.get("heading") or "").strip(),
            )
        if name in LIVE_TOOL_NAMES:
            # Право проверяется ЗДЕСЬ ТОЖЕ, а не только при показе списка.
            #
            # Скрытый инструмент — это не запрещённый инструмент: модель
            # может назвать его по памяти о прошлых диалогах или потому,
            # что имя мелькнуло в документе. Проверка при показе — удобство,
            # проверка при вызове — собственно ограничение. Оставить только
            # первую значит защищаться от вежливой модели.
            if not live_allowed:
                return ToolOutcome(
                    text="Нет прав на чтение боевых данных (роль data-reader по РЛ-4.2.3).",
                    ok=False,
                )
            return await self._live_call(name, arguments)

        available = ", ".join(
            spec["function"]["name"] for spec in self.specs(live_allowed=live_allowed)
        )
        return ToolOutcome(
            text=f"Инструмента «{name}» нет. Доступны: {available}.",
            ok=False,
        )

    async def _live_call(self, name: str, arguments: dict) -> ToolOutcome:
        """Поход в FATIGUE-API за текущими числами.

        Модель называет ОПЕРАЦИЮ и передаёт параметр; адрес, путь и токен
        собирает клиент. Инструмента вида «сходи по этому URL» здесь нет и
        быть не должно: адрес, пришедший от модели, — это адрес, пришедший
        из документа вики, а документ пишут люди. См. rag/live.py.
        """
        if self._live is None or not self._live.configured:
            return ToolOutcome(
                text="Сервис живых данных не настроен, текущих чисел нет.", ok=False
            )

        if name == "live_pipe":
            result = await self._live.pipe(str(arguments.get("pipe_id") or ""))
        elif name == "live_inspections":
            result = await self._live.inspections(
                arguments.get("category"),
                arguments.get("overdue"),
                arguments.get("top_n"),
                arguments.get("cursor"),
            )
        elif name == "live_wells":
            result = await self._live.wells(
                arguments.get("field"),
                arguments.get("top_n"),
                arguments.get("cursor"),
            )
        else:
            result = await self._live.fleet(
                arguments.get("min_damage"),
                arguments.get("top_n"),
                arguments.get("offset"),
                arguments.get("cursor"),
            )

        return ToolOutcome(
            text=result.text, hits=result.hits, ok=result.ok, dataset=result.dataset
        )

    async def _search(self, query: str) -> ToolOutcome:
        if not query:
            return ToolOutcome(text="Пустой запрос. Укажи, что искать.", ok=False)

        vector = await embed_query(self._embedder, query)
        result = await hybrid_search(
            store=self._store,
            query=query,
            query_vector=vector,
            top_k=self._s.search_top_k,
            floor=self._s.similarity_floor,
            rrf_k=self._s.rrf_k,
            keyword_weight=self._s.keyword_weight,
        )

        if result.empty:
            # Говорим ПОЧЕМУ пусто, с числами. «Ничего не найдено» модель
            # читает как «попробуй ещё раз теми же словами»; «лучшая
            # близость 0.31 при пороге 0.45» — как «таких документов нет».
            return ToolOutcome(
                text=(
                    f"По запросу «{query}» ничего выше порога. "
                    f"Лучшая близость {result.best_vector_score:.2f} при пороге "
                    f"{result.floor:.2f}. Скорее всего, в вики этого нет."
                ),
                ok=True,
            )

        return ToolOutcome(text=preview(result.hits, f"По запросу «{query}»"), hits=result.hits)

    def _read_section(self, doc_id: str, heading: str) -> ToolOutcome:
        document = self._store.document(doc_id)
        if document is None:
            return ToolOutcome(text=f"Документа {doc_id} нет.", ok=False)

        needle = heading.casefold()
        matched = [
            chunk
            for chunk in document["chunks"]
            if needle in str(chunk["heading_path"]).casefold()
        ]
        if not matched:
            # Отказ с подсказкой: перечисляем, какие разделы есть. Голое
            # «не найдено» заставляет модель гадать, а гадает она долго и
            # за наши токены.
            names = sorted({str(chunk["heading_path"]) for chunk in document["chunks"]})
            listing = "; ".join(names[:12])
            return ToolOutcome(
                text=f"В {doc_id} нет раздела «{heading}». Есть: {listing}",
                ok=False,
            )

        chunk_ids = [int(chunk["chunk_id"]) for chunk in matched[:MAX_SECTION_CHUNKS]]
        stored = self._store.load_chunks(chunk_ids)
        hits = [
            Hit(
                chunk=stored[chunk_id],
                vector_score=None,
                keyword_score=None,
                vector_rank=None,
                keyword_rank=None,
                # Ноль, а не выдуманный балл. Этот фрагмент попал в контекст
                # не потому, что победил в поиске, и приписывать ему чужую
                # шкалу значило бы врать в отчёте «как получен ответ».
                fused_score=0.0,
                origin="запрошен моделью",
            )
            for chunk_id in chunk_ids
            if chunk_id in stored
        ]
        return ToolOutcome(text=preview(hits, f"Раздел «{heading}» из {doc_id}"), hits=hits)


def preview(hits: list[Hit], title: str) -> str:
    """Короткая выжимка для модели.

    Текст документов проходит ту же чистку, что и контекст. Результат
    инструмента приезжает обратно в диалог и читается моделью как её
    собственная память — то есть это ровно такой же канал управления, как
    сам контекст, и закрывать надо оба. Закрыть один и оставить второй —
    это запереть парадную дверь и уйти через чёрную.
    """
    lines = [f"{title} найдено {len(hits)} фрагментов:"]
    for hit in hits[:PREVIEW_LINES]:
        body = sanitize_document_text(hit.chunk.body).replace("\n", " ")
        score = f"{hit.vector_score:.2f}" if hit.vector_score is not None else "—"
        lines.append(
            f"- [{hit.chunk.doc_id} · {hit.chunk.heading_path}] близость {score}: "
            f"{body[:PREVIEW_CHARS]}"
        )
    if len(hits) > PREVIEW_LINES:
        lines.append(f"…и ещё {len(hits) - PREVIEW_LINES}")
    return "\n".join(lines)
