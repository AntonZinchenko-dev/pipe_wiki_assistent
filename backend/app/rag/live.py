"""Живые данные из FATIGUE-API: правила в вики, числа здесь.

Зачем. До сих пор ассистент умел ровно одно — пересказывать документы. На
вопрос «какие трубы сейчас за порогом аварии» он ответить не мог в принципе,
и не из-за качества модели: порог написан в регламенте, а текущая выработка
не написана нигде. Двух половин ответа никогда не было в одном месте.

Теперь они есть: порог берётся поиском по вики, числа — вызовом сервиса.

ГЛАВНОЕ РЕШЕНИЕ ЭТОГО ФАЙЛА: МОДЕЛЬ ВЫБИРАЕТ ОПЕРАЦИЮ, А НЕ АДРЕС

Соблазн выглядит так: дать инструмент `http_get(url)` и пусть модель сама
складывает путь. Один инструмент вместо пяти, и новые эндпоинты не требуют
правок. Так делать нельзя, и причина не в аккуратности.

Адрес, пришедший от модели, — это адрес, пришедший из документов вики:
модель управляется текстом, а текст пишут люди, у которых есть доступ к
вики. Инструмент, берущий URL снаружи, отправит запрос куда угодно, откуда
дотянется НАШ сервер: во внутреннюю админку, в служебные адреса облака, в
соседний сервис без аутентификации. Это классическая дыра (SSRF), и она
открывается ровно одной подходящей строчкой в документе.

Здесь модель выбирает из двух операций и передаёт параметры. Хост, путь,
заголовки и токен собирает наш код. Худшее, что может сделать испорченный
документ, — заставить нас спросить паспорт несуществующей трубы.

ВТОРОЕ РЕШЕНИЕ: ЖИВЫЕ ДАННЫЕ ВХОДЯТ В КОНТЕКСТ ОБЫЧНЫМ ФРАГМЕНТОМ

Не отдельным блоком с особыми правилами, а фрагментом с номером — таким же,
как кусок документа. Тогда вся машина проверки цитат работает без единой
правки: модель обязана сослаться на номер, сервер найдёт указанное место в
теле фрагмента и вырежет цитату сам. Числа из API оказываются проверяемыми
ровно так же, как предложения из регламента.

Если бы живые данные шли мимо этой машины, получилось бы худшее из
возможного: половина ответа проверяется, половина нет, и на экране они
неотличимы.

ТРЕТЬЕ: ЖИВОЙ ФРАГМЕНТ ПОМЕЧЕН ВРЕМЕНЕМ

Цитата из документа верна, пока документ не переписали. Число из API верно
на момент снятия — и через час может быть другим. Поэтому в шапке живого
фрагмента стоит время запроса: и модель, и человек должны видеть, что это
снимок, а не вечная истина.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import httpx

from ..config import Settings
from .result import MAX_ROWS_FOR_VIEW, Column, Dataset, ToolError
from .search import Hit
from .store import StoredChunk

# Сколько труб обходим в агрегате.
#
# В контракте записано отдельным пунктом: `/pipes` отдаёт только
# идентичность, поэтому выработку по парку потребитель собирает сам, вызывая
# `/passport` на каждую трубу. Мы делаем ровно это — и ограничиваем число
# вызовов, потому что у сервисного токена лимит 600 запросов в минуту, а
# обход всего боевого парка занимает около сорока секунд.
FLEET_SCAN_LIMIT = 60
# Насколько живой снимок считается свежим. Повторный вопрос через минуту не
# должен заново обходить весь парк.
CACHE_TTL_S = 45.0
# Размер страницы, когда листают, но размер не назвали.
#
# Пять — не круглое число ради красоты: именно столько просят в живой
# речи («следующие пять», «топ пять»), и именно столько влезает в экран
# панели без прокрутки.
DEFAULT_PAGE_ROWS = 5


@dataclass(slots=True)
class LiveResult:
    """Что вернул сервис.

    Три разных получателя, три разных вида одних и тех же данных:

      `dataset` — ТАБЛИЦА целиком, она уходит прямо в браузер;
      `text`    — выжимка для агента, по ней он решает, хватит ли;
      `hits`    — фрагмент контекста, в нём та же выжимка для отвечающей
                  модели, чтобы работала проверка цитат.

    Строк таблицы нет ни в `text`, ни в `hits`, и это главное изменение:
    модель их не видит, значит и переписать неправильно не может.
    """

    text: str
    hits: list[Hit]
    ok: bool = True
    dataset: Dataset | None = None


class LiveApi:
    """Клиент FATIGUE-API. Наружу торчат операции, внутрь — адреса."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._s = settings
        self._client = client
        self._cache: dict[str, tuple[float, LiveResult]] = {}
        # Счётчик номеров живых фрагментов. Отрицательные, чтобы никогда не
        # столкнуться с настоящими кусками из базы: по `chunk_id` идёт отсев
        # повторов, и совпадение номера тихо выбросило бы живой фрагмент.
        self._next_id = -1

    @property
    def configured(self) -> bool:
        # Адреса мало: без http-клиента «настроен» означало бы «настроен, но
        # упадёт на первом вызове». Инструмент, который показывается и не
        # работает, хуже отсутствующего.
        return bool(self._s.fatigue_api_url and self._client is not None)

    async def pipe(self, pipe_id: str) -> LiveResult:
        """Паспорт одной трубы."""
        pipe_id = _clean_id(pipe_id)
        if not pipe_id:
            return LiveResult(text="Не указан идентификатор трубы.", hits=[], ok=False)

        status, payload = await self._get(f"/api/v1/pipes/{pipe_id}/passport")
        if status >= 400:
            return self._failed(f"Паспорт трубы {pipe_id}", "pipe.passport",
                                _as_error(status, payload, pipe_id))

        data = Dataset(
            kind="pipe.passport",
            title=f"Паспорт трубы {pipe_id}",
            columns=PIPE_COLUMNS,
            rows=[_pipe_row(payload)],
            total_found=1,
            taken_at=_stamp(),
        )
        return self._delivered(data)

    async def fleet(self, min_damage=None, top_n=None, offset=None, cursor=None) -> LiveResult:
        """Трубы парка: по порогу выработки, по количеству — или и то и то.

        Порог сюда приходит от модели, а модель берёт его из регламента.
        Зашивать 0.8 в код нельзя: пороги пересматривают, и зашитое число
        разошлось бы с документом молча — ровно та ошибка, которую регламент
        сам и описывает («пороги из постмортемов до 2026 года
        недействительны»).

        `top_n` ДОБАВЛЕН ПО ЖИВОМУ ПРОМАХУ, и промах поучительный. На вопрос
        «топ 5 изношенных труб парка» инструмент умел только порог, поэтому
        модель подобрала порог сама — 95 процентов, — за него прошла ОДНА
        труба, и в ответ поехало «топ-5: одна труба». Ни модель, ни проверка
        цитат тут ни при чём: обе отработали честно на том, что получили.

        Ошибка была в наборе инструментов. «Сколько труб за порогом» и
        «покажи N худших» — разные вопросы, и второй через первый не
        выражается: чтобы получить ровно пять, порог надо угадать, а угадать
        его нельзя, не зная ответа. Инструмент, который вынуждает модель
        угадывать, производит правдоподобные числа из ниоткуда — и виноватой
        потом выглядит модель.

        Порог теперь необязателен: без него берётся ноль, то есть весь
        обойденный парк, отсортированный по убыванию выработки.
        """
        # Метка страницы, если она пришла, задаёт ВСЁ: и порог, и размер, и
        # начало. Иначе «следующие пять» с курсором, но без порога, вернули
        # бы следующие пять из другой выборки — и это выглядело бы как
        # продолжение, не будучи им.
        if cursor:
            parsed = _read_cursor(str(cursor))
            if parsed is None:
                return LiveResult(text="Метка страницы испорчена.", hits=[], ok=False)
            min_damage, top_n, offset = parsed

        threshold, error = _as_threshold(min_damage)
        if error:
            return LiveResult(text=error, hits=[], ok=False)
        limit, error = _as_limit(top_n)
        if error:
            return LiveResult(text=error, hits=[], ok=False)
        start, error = _as_offset(offset)
        if error:
            return LiveResult(text=error, hits=[], ok=False)
        # ЛИСТАЮТ — ЗНАЧИТ СТРАНИЦАМИ, и у страницы есть размер.
        #
        # Из живого прогона: на «следующие 5» модель передала смещение и
        # забыла размер. Получилось «начиная с одиннадцатой» — двадцать
        # девять строк вместо пяти. Дальше она попыталась пересказать из
        # них пятёрку и выдумала четыре трубы.
        #
        # Формально она не ошиблась: без размера «покажи с одиннадцатой»
        # и правда значит «и до конца». Но это тот случай, когда
        # формально верное поведение по умолчанию порождает неверный
        # результат, — и значит умолчание выбрано неправильно.
        if start and not limit:
            limit = DEFAULT_PAGE_ROWS

        key = f"fleet:{threshold:.4f}:{limit}:{start}"
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_TTL_S:
            return cached[1]

        title = _fleet_title(threshold, limit, start)
        status, payload = await self._get("/api/v1/pipes")
        if status >= 400:
            return self._failed(title, "pipes.fleet", _as_error(status, payload, ""))

        ids = [str(row.get("pipe_id")) for row in (payload.get("pipes") or [])]
        ids = ids[:FLEET_SCAN_LIMIT]

        rows: list[dict] = []
        skipped: list[str] = []
        for pipe_id in ids:
            code, one = await self._get(f"/api/v1/pipes/{pipe_id}/passport")
            if code >= 400:
                # Одна труба с устаревшим расчётом не должна отменять обзор
                # парка. Но и молчать про неё нельзя: «две трубы за порогом»
                # и «две трубы за порогом, а по одной расчёт устарел» — это
                # разные ответы, и второй честный.
                skipped.append(f"{pipe_id} ({_error_code(one) or code})")
                continue
            if float(one.get("miner_damage_fraction") or 0.0) >= threshold:
                rows.append(one)

        rows.sort(key=lambda row: row.get("miner_damage_fraction", 0), reverse=True)
        matched = len(rows)
        # Режем ПОСЛЕ сортировки: «топ пять» это пять худших, а не первые
        # пять, попавшиеся при обходе. Смещение — оттуда же: «следующие
        # пять» это шестая-десятая по износу, а не по порядку обхода.
        window = rows[start:start + limit] if limit else rows[start:]

        data = Dataset(
            kind="pipes.fleet",
            title=title,
            columns=FLEET_COLUMNS,
            rows=[_pipe_row(row) for row in window],
            total_found=matched,
            scanned=len(ids),
            offset=start,
            truncated=start + len(window) < matched,
            hint=_fleet_hint(matched, len(window), threshold, limit, start),
            taken_at=_stamp(),
            skipped=skipped[:10],
            next_cursor=(
                _make_cursor(threshold, limit or DEFAULT_PAGE_ROWS, start + len(window))
                if start + len(window) < matched
                else ""
            ),
        )
        result = self._delivered(data)
        self._cache[key] = (time.monotonic(), result)
        return result

    async def inspections(
        self, category=None, overdue=None, top_n=None, cursor=None
    ) -> LiveResult:
        """Инспекции по парку: категория, износ стенки, просрочка.

        ЗАЧЕМ ОТДЕЛЬНЫЙ ИНСТРУМЕНТ, А НЕ ПОЛЕ В ОБЗОРЕ ПАРКА.

        Списание и выработка ресурса — разные вещи, и это главное правило
        домена: по расчёту труба не списывается никогда, решение принимает
        инспектор по физическому осмотру. Склеив их в одну таблицу, мы бы
        предложили модели ровно то сравнение, которого регламент запрещает,
        и получили бы ответ «под списание PP-0035, у неё 97 % ресурса» —
        правдоподобный и неверный.

        СТРАНИЦЫ СЧИТАЕТ СЕРВИС, А НЕ МЫ. У обзора парка выбора не было:
        `/pipes` не отдаёт выработку, приходится обойти весь парк и резать
        у себя. Здесь `limit` и `offset` есть в контракте — значит лишние
        строки не поедут по сети вовсе.
        """
        kind, limit, start, error = _inspection_page(category, overdue, top_n, cursor)
        if error:
            return LiveResult(text=error, hits=[], ok=False)
        wanted, only_overdue = kind

        query = f"?limit={limit}&offset={start}"
        if wanted:
            query += f"&category={wanted}"
        if only_overdue is not None:
            query += f"&overdue={'true' if only_overdue else 'false'}"

        title = _inspection_title(wanted, only_overdue, limit, start)
        status, payload = await self._get("/api/v1/inspections" + query)
        if status >= 400:
            return self._failed(title, "pipes.inspections", _as_error(status, payload, ""))

        rows = [_inspection_row(row) for row in (payload.get("inspections") or [])]
        total = int(payload.get("total") or len(rows))
        shown = start + len(rows)

        data = Dataset(
            kind="pipes.inspections",
            title=title,
            columns=INSPECTION_COLUMNS,
            rows=rows,
            total_found=total,
            # «Просмотрено» здесь НЕ ставим, и это не забывчивость.
            #
            # У обзора парка оно означает «мы обошли сорок труб, чтобы
            # отобрать пять» — работа, которую сделали мы. Здесь фильтрует
            # сервис, и написать «просмотрено 5» значило бы отчитаться о
            # проделанной работе, которой не было, а заодно намекнуть
            # модели, что в парке пять труб.
            scanned=0,
            offset=start,
            truncated=shown < total,
            hint=_page_hint(total, len(rows), limit, start, "инспекций"),
            taken_at=_stamp(),
            next_cursor=(
                _make_inspection_cursor(wanted, only_overdue, limit, shown)
                if shown < total
                else ""
            ),
        )
        return self._delivered(data)

    async def wells(self, field=None, top_n=None, cursor=None) -> LiveResult:
        """Скважины со сводкой по трубам, от самой тяжёлой.

        Сводка здесь есть, хотя у обзора парка её нет, и это не
        непоследовательность: в `/pipes` агрегата нет по контракту, а
        скважина без состава труб и максимальной выработки не имеет смысла
        как сущность. Различие записано в документе, и инструмент его
        повторяет, а не сглаживает.
        """
        wanted = str(field or "").strip()
        limit, start, error = _wells_page(top_n, cursor)
        if error:
            return LiveResult(text=error, hits=[], ok=False)
        if cursor:
            wanted = _read_wells_cursor(str(cursor))[0]

        title = _wells_title(wanted, limit, start)
        query = f"?field={wanted}" if wanted else ""
        status, payload = await self._get("/api/v1/wells" + query)
        if status >= 400:
            return self._failed(title, "wells.fleet", _as_error(status, payload, ""))

        rows = payload.get("wells") or []
        total = int(payload.get("total") or len(rows))
        window = rows[start : start + limit] if limit else rows[start:]
        shown = start + len(window)

        data = Dataset(
            kind="wells.fleet",
            title=title,
            columns=WELL_COLUMNS,
            rows=[_well_row(row) for row in window],
            total_found=total,
            scanned=total,
            offset=start,
            truncated=shown < total,
            hint=_page_hint(total, len(window), limit, start, "скважин"),
            taken_at=_stamp(),
            next_cursor=(
                _make_wells_cursor(wanted, limit or DEFAULT_PAGE_ROWS, shown)
                if shown < total
                else ""
            ),
        )
        return self._delivered(data)

    # ------------------------------------------------------------- внутреннее

    async def _get(self, path: str) -> tuple[int, dict]:
        """Один запрос к сервису.

        Путь собирается ЗДЕСЬ из константы и проверенного идентификатора.
        Снаружи в эту функцию не попадает ни хост, ни схема — только наш
        собственный путь.
        """
        headers = {"Accept": "application/json"}
        if self._s.fatigue_api_token:
            # Токен живёт на сервере и в диалог с моделью не попадает никогда:
            # всё, что уходит модели, — это разобранный ответ.
            headers["Authorization"] = f"Bearer {self._s.fatigue_api_token}"

        try:
            response = await self._client.get(
                self._s.fatigue_api_url.rstrip("/") + path,
                headers=headers,
                timeout=httpx.Timeout(
                    self._s.fatigue_api_timeout_s, connect=self._s.connect_timeout_s
                ),
            )
        except httpx.HTTPError as error:
            return 599, {"error": {"message": str(error)[:200]}}

        try:
            payload = response.json()
        except ValueError:
            return response.status_code, {"error": {"message": "ответ не JSON"}}
        return response.status_code, payload if isinstance(payload, dict) else {"data": payload}

    def _delivered(self, data: Dataset) -> LiveResult:
        """Три получателя — три разных вида одних данных.

        Браузер получает таблицу целиком, агент — выжимку с меткой
        следующей страницы, отвечающая модель — выжимку без метки.
        Метка агенту нужна, а во фрагменте была бы приглашением
        пересказать её пользователю.
        """
        return LiveResult(
            text=data.agent_note(),
            hits=[self._fragment(f"{data.title} · таблица {data.handle}", data.digest())],
            ok=data.error is None,
            dataset=data,
        )

    def _failed(self, title: str, kind: str, error: ToolError) -> LiveResult:
        """Отказ — такой же результат, просто без строк.

        Он тоже попадает в контекст фрагментом: модель должна иметь
        возможность сослаться на него и объяснить человеку, что случилось.
        Раньше отказ уходил только в текст агенту, а отвечающая модель про
        него не знала вовсе — и писала «в документации нет ответа» там, где
        честный ответ был «сервис не отвечает».
        """
        data = Dataset(kind=kind, title=title, error=error, taken_at=_stamp())
        return self._delivered(data)

    def _fragment(self, title: str, body: str) -> Hit:
        """Живые данные в виде обычного фрагмента контекста.

        Дальше с ним работает та же машина, что и с кусками документов:
        нумерация, бюджет токенов, проверка цитат. Отдельного пути для живых
        данных нет сознательно — отдельный путь означал бы, что половина
        ответа проверяется, а половина нет.
        """
        chunk_id = self._next_id
        self._next_id -= 1
        stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
        return Hit(
            chunk=StoredChunk(
                chunk_id=chunk_id,
                doc_id="FATIGUE-API",
                heading_path=f"{title} · снято {stamp}",
                body=body,
                text=body,
                page_from=0,
                page_to=0,
                doc_title="FATIGUE-API, живые данные",
                doc_project="FATIGUE-API",
                doc_version="1.1.0",
                doc_status="живые данные",
                doc_updated=stamp,
            ),
            vector_score=None,
            keyword_score=None,
            vector_rank=None,
            keyword_rank=None,
            fused_score=0.0,
            origin="живые данные сервиса",
        )


def _clean_id(value: str) -> str:
    """Идентификатор трубы — и ничего кроме.

    Строка отсюда попадает в путь запроса, поэтому пропускаем только буквы,
    цифры и дефис. Это не защита от глупой модели, а защита от документа,
    в котором кто-то написал `../../admin`: подставить такое в путь значит
    отправить запрос не туда, куда мы собирались.
    """
    allowed = "".join(char for char in str(value).strip().upper() if char.isalnum() or char == "-")
    return allowed[:32]


def _error_code(payload: dict) -> str:
    error = payload.get("error")
    return str(error.get("code", "")) if isinstance(error, dict) else ""


def _as_error(status: int, payload: dict, pipe_id: str) -> ToolError:
    """Отказ сервиса в разобранном виде.

    Код обязателен: в вики есть таблица кодов, и модель может найти по нему
    инструкцию. Голое «сервис ответил ошибкой» такой возможности не даёт —
    и превращает штатную ситуацию в тупик.

    `retriable` тут важнее текста. «Расчёт устарел, повторите через минуту»
    и «такой трубы нет, повтор не поможет» — разные ответы человеку, а до
    этой правки оба выглядели как «произошла ошибка».
    """
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    code = str(error.get("code") or "") or str(status)
    message = str(error.get("message") or payload.get("error") or "")[:200]

    if code == "E-1108":
        return ToolError(
            code=code,
            message=f"Трубы {pipe_id} нет в парке текущей кампании.",
            retriable=False,
            hint="Проверьте идентификатор трубы: возможно, она уже списана.",
        )
    if code == "E-1042":
        return ToolError(
            code=code,
            message=f"Расчёт по трубе {pipe_id} устарел: изменилась версия съёмки.",
            retriable=True,
            retry_after_s=60,
            hint="Текущего значения выработки сейчас нет, оно появится после пересчёта.",
        )
    if status == 599:
        return ToolError(
            code="network",
            message=f"Сервис FATIGUE-API не отвечает: {message}",
            retriable=True,
            retry_after_s=30,
        )
    if status == 429:
        return ToolError(
            code=code,
            message="Слишком много запросов к FATIGUE-API.",
            retriable=True,
            retry_after_s=60,
        )
    return ToolError(
        code=code,
        message=message or f"FATIGUE-API вернул ошибку {status}.",
        # 5xx — это их сторона и обычно временно; 4xx — наш запрос, и
        # повтор того же запроса даст тот же отказ.
        retriable=status >= 500,
        retry_after_s=30 if status >= 500 else None,
    )


FLEET_COLUMNS = [
    Column(key="pipe_id", title="Труба"),
    Column(key="damage_percent", title="Выработка", unit="%", kind="number"),
    Column(key="cycles", title="Циклов", kind="number"),
    Column(key="critical_m", title="Критическое сечение", unit="м", kind="number"),
    Column(key="well", title="Скважина"),
]

# Паспорт — та же таблица, только подробнее и в одну строку.
PIPE_COLUMNS = FLEET_COLUMNS + [
    Column(key="steel", title="Марка стали"),
    Column(key="diameter_mm", title="Диаметр", unit="мм", kind="number"),
    Column(key="length_m", title="Длина", unit="м", kind="number"),
    Column(key="survey", title="Версия съёмки"),
    Column(key="updated", title="Обновлено"),
]


def _pipe_row(pipe: dict) -> dict:
    """Ответ сервиса в строку таблицы: имена наши, значения его.

    Переименование здесь не косметика. `miner_damage_fraction` — имя из
    чужого контракта, и если протащить его до интерфейса, то переименование
    поля у сервиса придётся править в вёрстке. Граница между чужими именами
    и нашими проходит ровно в этом месте.
    """
    damage = float(pipe.get("miner_damage_fraction") or 0.0)
    return {
        "pipe_id": str(pipe.get("pipe_id") or ""),
        "damage_percent": round(damage * 100, 1),
        "cycles": pipe.get("cycles_total"),
        "critical_m": pipe.get("critical_local_position_m"),
        "well": str(pipe.get("well_id") or ""),
        "steel": pipe.get("steel_grade"),
        "diameter_mm": pipe.get("outer_diameter_mm"),
        "length_m": pipe.get("length_m"),
        "survey": pipe.get("survey_version_id"),
        "updated": pipe.get("updated_at"),
    }


def _stamp() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def _fleet_title(threshold: float, limit: int, start: int = 0) -> str:
    """Заголовок обязан соответствовать содержимому.

    «Топ-5» на строках с шестой по десятую — заголовок, который врёт, и
    заметит это только тот, кто пересчитает строки руками.
    """
    if start and limit:
        return f"Трубы с {start + 1}-й по {start + limit}-ю по выработке ресурса"
    if start:
        return f"Трубы парка начиная с {start + 1}-й по выработке ресурса"
    if limit:
        return f"Топ-{limit} труб по выработке ресурса"
    if threshold > 0:
        return f"Трубы парка с выработкой ресурса от {threshold:.0%}"
    return "Трубы парка по выработке ресурса"


def _fleet_hint(
    matched: int, shown: int, threshold: float, limit: int, start: int = 0
) -> str:
    """Подсказка человеку и модели: что делать с тем, что нашлось.

    Нужна ровно в двух случаях — когда показано не всё и когда не нашлось
    ничего. В остальных молчим: подсказка на каждый ответ перестаёт
    читаться за неделю.

    ПОДСКАЗКА НАЗЫВАЕТ ДЕЙСТВИЕ, а не сообщает факт. «Показано 5 из 39» —
    это факт, из которого непонятно, что делать; «попросите следующие
    пять» — действие. Первая формулировка у нас уже была, и модель на
    просьбу «следующие 5» изобретала обход вместо того, чтобы взять
    следующую страницу.
    """
    if matched == 0:
        within = f" с выработкой от {threshold:.0%}" if threshold > 0 else ""
        return f"Ни одной трубы{within} не нашлось. Попробуйте меньший порог."

    rest = matched - start - shown
    if rest > 0:
        step = limit or shown or 5
        return (
            f"Показаны строки {start + 1}–{start + shown} из {matched}. "
            f"Ниже по списку ещё {rest}: попросите «следующие {step}»."
        )
    if start:
        return f"Это конец списка: всего {matched} труб."
    return ""


def _as_offset(value) -> tuple[int, str]:
    """С какой записи начинать. Ноль — с начала.

    Верхней границы нет и не нужно: смещение за пределы списка даёт
    пустую страницу, а не ошибку, и подсказка честно скажет, что записей
    столько-то. Ошибка тут была бы формализмом — человек просто пролистал
    дальше конца.
    """
    if value in (None, ""):
        return 0, ""
    try:
        start = int(float(value))
    except (TypeError, ValueError):
        return 0, "Смещение должно быть целым числом."
    if start < 0:
        return 0, "Смещение не может быть отрицательным."
    return start, ""


def _as_threshold(value) -> tuple[float, str]:
    """Порог выработки. Пусто — значит ноль, то есть без порога."""
    if value in (None, ""):
        return 0.0, ""
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.0, "Порог должен быть числом от 0 до 1."
    if not 0.0 <= threshold <= 1.0:
        return 0.0, f"Порог {threshold} вне диапазона: доля ресурса это число от 0 до 1."
    return threshold, ""


def _as_limit(value) -> tuple[int, str]:
    """Сколько строк показать. Ноль — сколько найдётся.

    Верхняя граница своя, а не «сколько попросили»: модель регулярно
    просит «топ 100» на вопрос про пятёрку.

    ГРАНИЦА ПОДНЯТА с двенадцати строк. Двенадцать были ограничением
    КОНТЕКСТА: каждая строка стоила токенов и вытесняла документы. Теперь
    строки в контекст не попадают вовсе, и ограничивать показ прежним
    числом значило бы платить старую цену, ничего за неё не получая.
    """
    if value in (None, ""):
        return 0, ""
    try:
        limit = int(float(value))
    except (TypeError, ValueError):
        return 0, "Количество труб должно быть целым числом."
    if limit < 1:
        return 0, "Количество труб должно быть больше нуля."
    return min(limit, MAX_ROWS_FOR_VIEW), ""


def _make_cursor(threshold: float, limit: int, start: int) -> str:
    """Метка страницы: всё, что нужно, чтобы повторить запрос со сдвигом.

    Читаемая, а не зашифрованная: её видно в журнале и в блоке «что делал
    агент», и разбирать жалобу по ней можно глазами. Прятать тут нечего —
    ни адресов, ни прав, только параметры выборки.
    """
    return f"fleet|{threshold:.4f}|{limit}|{start}"


def _read_cursor(value: str) -> tuple[float, int, int] | None:
    """Разбор метки. Испорченная — отказ, а не молчаливый первый экран.

    Молчаливый откат на начало был бы худшим поведением: человек просит
    следующую страницу, получает первую и думает, что дальше ничего нет.
    """
    parts = value.split("|")
    if len(parts) != 4 or parts[0] != "fleet":
        return None
    try:
        return float(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


# --------------------------------------------------------- инспекции и скважины

INSPECTION_COLUMNS = [
    Column(key="pipe_id", title="Труба"),
    Column(key="category", title="Категория"),
    Column(key="wear_percent", title="Износ стенки", unit="%", kind="number"),
    Column(key="inspected_at", title="Инспекция"),
    Column(key="overdue", title="Просрочена"),
    Column(key="base", title="База"),
]

WELL_COLUMNS = [
    Column(key="well_id", title="Скважина"),
    Column(key="field", title="Месторождение"),
    Column(key="pipe_count", title="Труб", kind="number"),
    Column(key="max_damage_percent", title="Макс. выработка", unit="%", kind="number"),
]

CATEGORIES = ("PREMIUM", "CLASS_2", "CLASS_3", "SCRAP")


def _inspection_row(row: dict) -> dict:
    """Ответ сервиса в строку таблицы. Имена наши, значения его.

    `overdue` печатаем словом, а не галочкой: столбец из `true` и `false`
    человек читает, переводя каждую строку, а «просрочена» читается сразу.
    Пустая клетка вместо «нет» — намеренно: глаз ищет заполненные.
    """
    return {
        "pipe_id": str(row.get("pipe_id") or ""),
        "category": str(row.get("category") or ""),
        "wear_percent": row.get("wall_wear_percent"),
        "inspected_at": str(row.get("inspected_at") or "")[:10],
        "overdue": "просрочена" if row.get("overdue") else "",
        "base": str(row.get("base") or ""),
    }


def _well_row(row: dict) -> dict:
    damage = float(row.get("max_damage_fraction") or 0.0)
    return {
        "well_id": str(row.get("well_id") or ""),
        "field": str(row.get("field") or ""),
        "pipe_count": row.get("pipe_count"),
        "max_damage_percent": round(damage * 100, 1),
    }


def _inspection_title(category: str, overdue: bool | None, limit: int, start: int) -> str:
    """Заголовок обязан соответствовать содержимому — как и у обзора парка."""
    what = "Инспекции"
    if category:
        what = f"Трубы с категорией {category}"
    if overdue is True:
        what = f"{what}, просроченные" if category else "Трубы с просроченной инспекцией"
    elif overdue is False:
        what = f"{what}, инспекция в срок"
    if start:
        return f"{what}: строки с {start + 1}-й по {start + limit}-ю"
    if limit and not category and overdue is None:
        return f"{what}: последние {limit}"
    return what


def _wells_title(field: str, limit: int, start: int) -> str:
    where = f" на месторождении {field}" if field else ""
    if start:
        return f"Скважины{where}: с {start + 1}-й по {start + limit}-ю по выработке"
    if limit:
        return f"Топ-{limit} скважин{where} по максимальной выработке труб"
    return f"Скважины{where} по максимальной выработке труб"


def _page_hint(total: int, shown: int, limit: int, start: int, what: str) -> str:
    """Та же подсказка, что у обзора парка, и по той же причине.

    Называет ДЕЙСТВИЕ, а не факт: «показано 5 из 39» не говорит, что
    делать дальше, и модель на просьбу «ещё» изобретает обход вместо того,
    чтобы взять следующую страницу.
    """
    if total == 0:
        return "Ни одной записи не нашлось. Попробуйте снять фильтр."
    rest = total - start - shown
    if rest > 0:
        step = limit or shown or DEFAULT_PAGE_ROWS
        return (
            f"Показаны строки {start + 1}–{start + shown} из {total}. "
            f"Ниже по списку ещё {rest}: попросите «следующие {step}»."
        )
    if start:
        return f"Это конец списка: всего {total} {what}."
    return ""


def _as_category(value) -> tuple[str, str]:
    """Категория инспекции. Опечатка — отказ, а не пустая выборка.

    Тихо вернуть ноль строк на «SCRAPP» значило бы сообщить «таких труб
    нет», и это неправда: разница между «нет записей» и «я не понял
    фильтр» здесь стоит неверного ответа человеку.
    """
    if value in (None, ""):
        return "", ""
    text = str(value).strip().upper()
    if text not in CATEGORIES:
        return "", f"Неизвестная категория «{value}». Бывают: {', '.join(CATEGORIES)}."
    return text, ""


def _as_flag(value) -> tuple[bool | None, str]:
    """Трёхзначный флаг: да, нет, не фильтровать.

    Именно трёхзначный, а не булев. `False` и «не задано» — разные
    запросы: первый просит трубы, у которых инспекция в срок, второй — все
    подряд. Свести их в одно значит потерять половину вопросов.
    """
    if value in (None, ""):
        return None, ""
    if isinstance(value, bool):
        return value, ""
    text = str(value).strip().lower()
    if text in {"true", "да", "1", "yes"}:
        return True, ""
    if text in {"false", "нет", "0", "no"}:
        return False, ""
    return None, f"Не понял значение «{value}»: ожидается «да» или «нет»."


def _make_inspection_cursor(
    category: str, overdue: bool | None, limit: int, start: int
) -> str:
    flag = "" if overdue is None else ("1" if overdue else "0")
    return f"inspect|{category}|{flag}|{limit}|{start}"


def _read_inspection_cursor(value: str):
    parts = value.split("|")
    if len(parts) != 5 or parts[0] != "inspect":
        return None
    flag = {"": None, "1": True, "0": False}.get(parts[2], "bad")
    if flag == "bad":
        return None
    try:
        return parts[1], flag, int(parts[3]), int(parts[4])
    except ValueError:
        return None


def _make_wells_cursor(field: str, limit: int, start: int) -> str:
    return f"wells|{field}|{limit}|{start}"


def _read_wells_cursor(value: str):
    parts = value.split("|")
    if len(parts) != 4 or parts[0] != "wells":
        return "", 0, 0
    try:
        return parts[1], int(parts[2]), int(parts[3])
    except ValueError:
        return "", 0, 0


def _inspection_page(category, overdue, top_n, cursor):
    """Разбор параметров страницы инспекций в одном месте.

    Метка страницы ПЕРЕБИВАЕТ остальные параметры, а не смешивается с
    ними. Модель, передавшая и cursor, и category, почти наверняка
    подставила категорию из вопроса, а метка уже хранит ту, по которой шла
    выборка; смешать их значит молча сменить фильтр на середине списка.
    """
    if cursor:
        parsed = _read_inspection_cursor(str(cursor))
        if parsed is None:
            return ("", None), 0, 0, "Метка страницы испорчена."
        wanted, flag, limit, start = parsed
        return (wanted, flag), limit, start, ""

    wanted, error = _as_category(category)
    if error:
        return ("", None), 0, 0, error
    flag, error = _as_flag(overdue)
    if error:
        return ("", None), 0, 0, error
    limit, error = _as_limit(top_n) if top_n not in (None, "") else (DEFAULT_PAGE_ROWS, "")
    if error:
        return ("", None), 0, 0, error
    return (wanted, flag), limit, 0, ""


def _wells_page(top_n, cursor):
    if cursor:
        _, limit, start = _read_wells_cursor(str(cursor))
        if not limit:
            return 0, 0, "Метка страницы испорчена."
        return limit, start, ""
    if top_n in (None, ""):
        return DEFAULT_PAGE_ROWS, 0, ""
    limit, error = _as_limit(top_n)
    return limit, 0, error
