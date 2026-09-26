"""Поддельный FATIGUE-API — источник ЖИВЫХ данных для ассистента.

Зачем он нужен. Ассистент до сих пор умел одно: пересказывать документы. Это
половина настоящей задачи. Вторая половина выглядит так: «какие трубы сейчас
за порогом аварии» — порог написан в регламенте, а сколько у какой трубы
сейчас, не знает ни один документ. Ни поиск по вики, ни модель сама по себе
на такой вопрос ответить не могут в принципе: нужного числа нет нигде в
тексте.

Поэтому сервис. Он подделка, но подделка ТОЧНАЯ: эндпоинты, имена полей,
коды ошибок и лимиты взяты из документа «FATIGUE-API 1.1.0 — контракт,
эндпоинты, коды ошибок». Это не педантизм. Если сервис отвечает не так, как
написано в вики, то ассистент, который читает вики и ходит в сервис, будет
получать противоречие — и мы будем чинить несуществующую ошибку в агенте.

ДАННЫЕ ДЕТЕРМИНИРОВАННЫЕ. Парк генерируется от фиксированного зерна, то есть
при каждом запуске одинаков. Без этого прогон золотого набора стал бы
бессмысленным: ответ менялся бы от запуска к запуску, и отличить «модель
стала хуже» от «данные другие» было бы нельзя.

ЧТО СОЗНАТЕЛЬНО ВОСПРОИЗВЕДЕНО ИЗ КОНТРАКТА, ХОТЯ И НЕУДОБНО:

`GET /pipes` отдаёт только идентичность, без выработки ресурса. В документе
это записано отдельным пунктом как известное ограничение: чтобы показать
реестр с ресурсом, потребителю приходится звать `/passport` на каждую трубу.
Соблазн «добавить агрегат, так удобнее» здесь надо давить: вся ценность
стенда в том, что он врёт ровно так же, как боевой сервис.

ПОЧЕМУ ФАЙЛ НЕ НАЗЫВАЕТСЯ app.py

Сначала назывался — и это была ошибка. В соседней папке `backend` лежит
пакет с тем же именем, а `uvicorn app:api` ищет модуль ПО ПУТИ ИМПОРТА, а не
в той папке, откуда запускают. Стоит оказаться в другом каталоге — и uvicorn
поднимает чужой `app`, тот самый пакет бэкенда, и честно сообщает, что
атрибута `api` там нет. Ошибка выглядит как поломка стенда, а на самом деле
это столкновение имён.

Правило простое: файл, который запускают по имени модуля, не должен
называться так же, как пакет в соседнем проекте.

ЧЕМ ИГРАТЬ НА СТЕНДЕ

Три трубы отвечают отказом всегда — обработку ошибок проверяют на ошибках:

    PP-0007  E-1042  расчёт устарел, повтор через минуту
    PP-0013  E-5001  воркер недоступен, повтор с задержкой
    PP-0021  медленный ответ, если задан PW_MOCK_SLOW_SECONDS

Ещё два поведения включаются переменными окружения и по умолчанию выключены,
потому что обход парка зовёт /passport сорок раз и стенд, который сам себе
устраивает таймауты, мешает проверять всё остальное:

    PW_MOCK_SLOW_SECONDS=3   задержка «медленной» трубы
    PW_MOCK_RATE_LIMIT=30    лимит запросов за 10 секунд, дальше E-2301

Что стенд изображает прямо сейчас, видно в `/health` — поле `rigged`.

ЗАПУСК

    python fatigue_api.py

Именно так, а не через `-m uvicorn`: при прямом запуске Python берёт файл по
имени файла, и вопрос «какой каталог текущий» просто не возникает. Порт
можно поменять переменной PW_MOCK_PORT.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

api = FastAPI(title="FATIGUE-API (стенд)", version="1.2.0")

# Зерно фиксировано: парк обязан быть одинаковым при каждом запуске, иначе
# золотой набор нечем мерить.
SEED = 20260922
# В боевом контуре парк 400 труб; здесь 40, чтобы обход всего парка через
# `/passport` укладывался в секунды. Ограничение контракта при этом
# сохранено — агрегата в `/pipes` по-прежнему нет.
FLEET_SIZE = 40
SURVEY_VERSIONS = ["SV-2026-07", "SV-2026-08", "SV-2026-09"]
# ТРУБЫ-ИСПЫТАТЕЛЬНЫЕ СТЕНДЫ.
#
# Обработку ошибок проверяют на ошибках, а не на успехах. Каждая из этих
# труб отвечает своим кодом ВСЕГДА, и это часть стенда, а не случайность:
# по ним пишутся тесты, и они же нужны, чтобы посмотреть глазами, как
# ассистент объясняет человеку отказ сервиса.
#
# Коды подобраны так, чтобы покрыть все три поведения клиента: не повторять
# (E-1108), повторить позже (E-1042), повторять с задержкой (E-5001).
STALE_PIPE = "PP-0007"       # E-1042: расчёт устарел, повтор через минуту
BROKEN_PIPE = "PP-0013"      # E-5001: воркер недоступен, повтор с задержкой
SLOW_PIPE = "PP-0021"        # отвечает медленно: проверка таймаутов клиента

# Задержка «медленной» трубы и лимит запросов включаются переменными
# окружения. По умолчанию выключены: обход всего парка зовёт /passport сорок
# раз, и стенд, который сам себе устраивает таймауты и отказы, мешает
# проверять всё остальное.
SLOW_SECONDS = float(os.environ.get("PW_MOCK_SLOW_SECONDS", "0"))
RATE_LIMIT = int(os.environ.get("PW_MOCK_RATE_LIMIT", "0"))
RATE_WINDOW_S = 10.0

CATEGORIES = ["PREMIUM", "CLASS_2", "CLASS_3", "SCRAP"]


def _build_fleet() -> dict[str, dict]:
    rng = random.Random(SEED)
    started = datetime(2026, 9, 1, tzinfo=timezone.utc)
    fleet: dict[str, dict] = {}

    for number in range(1, FLEET_SIZE + 1):
        pipe_id = f"PP-{number:04d}"
        # Распределение подобрано так, чтобы в парке были все три состояния
        # по регламенту РЛ-9.2: ниже 50 %, между 50 и 80, выше 80. Стенд, на
        # котором все трубы в норме, не проверяет ничего.
        damage = round(min(0.98, abs(rng.gauss(0.45, 0.26))), 4)
        length_m = rng.choice([9.1, 9.4, 9.6])
        fleet[pipe_id] = {
            "pipe_id": pipe_id,
            "well_id": f"W-{rng.randint(100, 140)}",
            "steel_grade": rng.choice(["S-135", "G-105", "E-75"]),
            "outer_diameter_mm": rng.choice([127.0, 139.7]),
            "length_m": length_m,
            "miner_damage_fraction": damage,
            "cycles_total": int(damage * rng.uniform(1.4e6, 2.1e6)),
            "critical_local_position_m": round(rng.uniform(0.3, length_m - 0.3), 2),
            "survey_version_id": rng.choice(SURVEY_VERSIONS),
            "calculation_id": f"CALC-{rng.randint(9000, 9999)}",
            "updated_at": (started + timedelta(hours=rng.randint(0, 480))).isoformat(),
            # В контракте прямо сказано: на текущей кампании ноль у всего парка.
            "lifecycle_event_count": 0,
        }
    return fleet


def _build_inspections() -> dict[str, dict]:
    """Последняя инспекция по каждой трубе.

    Категорию проставляет ЧЕЛОВЕК, а не расчёт — это прямая цитата из
    INSPECT-MOBILE, и здесь она соблюдена: категория не выводится из
    `miner_damage_fraction`. Связь между ними есть, но нестрогая, и именно
    на такой связи проверяется главное правило домена — «по расчёту труба
    не списывается никогда». Труба с выработкой 97 % и категорией PREMIUM
    в парке должна быть, иначе вопрос «какие трубы под списание» проходит
    мимо настоящей логики.

    Просроченные инспекции тоже есть: по регламенту такая труба не
    списывается, а переводится в «инспекция просрочена» и не допускается к
    работе. Без них половину FAQ поддержки нечем проверить.
    """
    rng = random.Random(f"{SEED}-inspect")
    today = datetime(2026, 9, 25, tzinfo=timezone.utc)
    result: dict[str, dict] = {}
    for pipe_id, pipe in FLEET.items():
        wear = pipe["miner_damage_fraction"]
        # Смещаем вероятности к износу, но не делаем из этого правило.
        weights = [max(0.05, 1.0 - wear), 0.35, 0.25 + wear * 0.3, 0.05 + wear * 0.35]
        category = rng.choices(CATEGORIES, weights=weights)[0]
        # ИЗНОС СТЕНКИ ВЫВОДИМ ИЗ КАТЕГОРИИ, А НЕ НАОБОРОТ.
        #
        # В INSPECT-MOBILE границы записаны таблицей: PREMIUM до 20 %,
        # CLASS_2 20…30, CLASS_3 30…37,5, SCRAP свыше 37,5. Пока износ
        # разыгрывался отдельно, в парке заводился «SCRAP при износе 24 %» —
        # и ассистент, читающий ту же таблицу, оказывался неправ на верных
        # данных. Стенд обязан врать так же, как боевой сервис, но
        # противоречить документу он не имеет права: такое противоречие мы
        # будем чинить в агенте, где его нет.
        #
        # А вот связи с выработкой по Майнеру здесь по-прежнему нет, и это
        # тоже из документа: по расчёту труба не списывается никогда. Труба
        # с выработкой 97 % и категорией PREMIUM в парке есть — без неё
        # вопрос «какие трубы под списание» проходит мимо настоящего
        # правила.
        bands = {
            "PREMIUM": (4.0, 19.9),
            "CLASS_2": (20.0, 29.9),
            "CLASS_3": (30.0, 37.4),
            "SCRAP": (37.6, 46.0),
        }
        low, high = bands[category]
        days_ago = rng.randint(5, 260)
        result[pipe_id] = {
            "pipe_id": pipe_id,
            "inspection_id": f"INS-{rng.randint(70000, 79999)}",
            "category": category,
            "wall_wear_percent": round(rng.uniform(low, high), 1),
            "inspected_at": (today - timedelta(days=days_ago)).isoformat(),
            "inspector": rng.choice(["И. Ковалёв", "П. Сомов", "А. Дроздова", "Р. Хайруллин"]),
            "base": rng.choice(["Б-1 Нефтеюганск", "Б-2 Когалым", "Б-4 Ноябрьск"]),
            # Срок — 180 дней по регламенту базы. Просрочка не равна списанию.
            "overdue": days_ago > 180,
        }
    return result


def _build_wells() -> dict[str, dict]:
    """Справочник скважин со сводкой по трубам.

    В версии 1.1.0 сущности не было — только поле `well_id` в паспорте, и
    это стояло в документе отдельным пунктом «чего в API нет». В 1.2.0
    сущность появилась, и документ обновлён вместе со стендом: расхождение
    между ними дороже любого удобства.
    """
    wells: dict[str, dict] = {}
    for pipe in FLEET.values():
        well = wells.setdefault(
            pipe["well_id"],
            {"well_id": pipe["well_id"], "pipe_count": 0, "max_damage_fraction": 0.0,
             "pipes": [], "field": ""},
        )
        well["pipe_count"] += 1
        well["pipes"].append(pipe["pipe_id"])
        well["max_damage_fraction"] = max(
            well["max_damage_fraction"], pipe["miner_damage_fraction"]
        )
    rng = random.Random(f"{SEED}-wells")
    for well in wells.values():
        well["field"] = rng.choice(["Приобское", "Самотлор", "Ватьёганское"])
        well["max_damage_fraction"] = round(well["max_damage_fraction"], 4)
        well["pipes"].sort()
    return wells


def _build_surveys() -> dict[str, dict]:
    """Версии инклинометрической съёмки.

    Сама инклинометрия наружу по-прежнему не выставлена — отдаются только
    метаданные версий. Это важно оставить как есть: на версиях съёмки
    построен постмортем 2025-03, и вопрос «почему пересчитали ресурс»
    отвечается именно датой уточнения, а не профилем ствола.
    """
    dates = {"SV-2026-07": (7, 14), "SV-2026-08": (8, 11), "SV-2026-09": (9, 8)}
    return {
        version: {
            "survey_version_id": version,
            "recorded_at": datetime(2026, month, day, tzinfo=timezone.utc).isoformat(),
            "supersedes": previous,
            "reason": "уточнение по данным гироскопа" if previous else "первичная съёмка",
            "pipes_recalculated": sum(
                1 for pipe in FLEET.values() if pipe["survey_version_id"] == version
            ),
        }
        for (version, (month, day)), previous in zip(
            dates.items(), [None, "SV-2026-07", "SV-2026-08"]
        )
    }


FLEET = _build_fleet()
INSPECTIONS = _build_inspections()
WELLS = _build_wells()
SURVEYS = _build_surveys()

# Скользящее окно запросов к паспорту — для проверки E-2301.
_CALLS: deque[float] = deque()


def _rate_limited() -> bool:
    """Лимит запросов сервисного токена. По умолчанию выключен.

    Включённым он мешал бы всему остальному: обход парка зовёт /passport
    сорок раз за один вопрос. Но и выкинуть его нельзя — E-2301 стоит в
    таблице кодов, а код, который никогда не приходит, не проверен ничем.
    """
    if RATE_LIMIT <= 0:
        return False
    now = time.monotonic()
    while _CALLS and now - _CALLS[0] > RATE_WINDOW_S:
        _CALLS.popleft()
    _CALLS.append(now)
    return len(_CALLS) > RATE_LIMIT


def _error(code: str, http: int, message: str, retry_after: int | None = None):
    """Ошибка в той же форме, что у боевого сервиса.

    Код, HTTP-статус и текст — из таблицы кодов ошибок в документе. Клиент
    различает ошибки по коду, а не по тексту: текст меняют, код нет.
    """
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(
        status_code=http,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


@api.get("/api/v1/pipes")
def pipes():
    """Реестр труб. ТОЛЬКО идентичность — так написано в контракте.

    Выработки ресурса здесь нет, и добавлять её нельзя: в документе это
    отдельный пункт «чего в API нет», на который ссылается вся остальная
    документация. Стенд, отвечающий удобнее боевого сервиса, — это стенд,
    на котором не воспроизводятся боевые проблемы.
    """
    return {
        "pipes": [
            {
                "pipe_id": pipe["pipe_id"],
                "well_id": pipe["well_id"],
                "steel_grade": pipe["steel_grade"],
                "outer_diameter_mm": pipe["outer_diameter_mm"],
                "length_m": pipe["length_m"],
            }
            for pipe in FLEET.values()
        ],
        "total": len(FLEET),
    }


def _rigged(pipe_id: str):
    """Заранее назначенный отказ испытательной трубы, если он есть.

    Собран в одну функцию, чтобы одна и та же труба вела себя одинаково во
    всех эндпоинтах. Разное поведение на /passport и /sections выглядело бы
    как плавающая ошибка, и мы бы искали её в клиенте.
    """
    if pipe_id == STALE_PIPE:
        return _error(
            "E-1042", 409,
            "Расчёт для трубы устарел: изменилась версия съёмки",
            retry_after=60,
        )
    if pipe_id == BROKEN_PIPE:
        return _error("E-5001", 503, "Расчётный воркер недоступен", retry_after=5)
    if pipe_id == SLOW_PIPE and SLOW_SECONDS > 0:
        time.sleep(SLOW_SECONDS)
    return None


@api.get("/api/v1/pipes/{pipe_id}/passport")
def passport(pipe_id: str):
    if _rate_limited():
        return _error("E-2301", 429, "Превышен лимит запросов сервисного токена", retry_after=10)
    pipe = FLEET.get(pipe_id.upper())
    if pipe is None:
        return _error("E-1108", 404, "Труба не найдена в парке текущей кампании")
    failure = _rigged(pipe_id.upper())
    if failure is not None:
        return failure
    return pipe


@api.get("/api/v1/pipes/{pipe_id}/sections")
def sections(pipe_id: str):
    pipe = FLEET.get(pipe_id.upper())
    if pipe is None:
        return _error("E-1108", 404, "Труба не найдена в парке текущей кампании")

    rng = random.Random(f"{SEED}-{pipe_id}")
    steps = int(pipe["length_m"])
    peak = pipe["critical_local_position_m"]
    rows = []
    for metre in range(steps + 1):
        # Повреждение падает по мере удаления от самого нагруженного сечения.
        # Повреждения сечений НЕ суммируются — это прямая цитата из контракта,
        # и состояние трубы определяется максимумом, а не суммой.
        distance = abs(metre - peak)
        share = max(0.0, pipe["miner_damage_fraction"] * (1 - distance / max(steps, 1)))
        rows.append({
            "position_m": float(metre),
            "miner_damage_fraction": round(share, 4),
            "cycles_total": int(share * rng.uniform(1.4e6, 2.1e6)),
        })
    return {
        "pipe_id": pipe["pipe_id"],
        "survey_version_id": pipe["survey_version_id"],
        "critical_local_position_m": peak,
        "sections": rows,
    }


@api.get("/api/v1/pipes/{pipe_id}/runs")
def runs(pipe_id: str):
    pipe = FLEET.get(pipe_id.upper())
    if pipe is None:
        return _error("E-1108", 404, "Труба не найдена в парке текущей кампании")

    rng = random.Random(f"{SEED}-runs-{pipe_id}")
    started = datetime(2026, 8, 1, tzinfo=timezone.utc)
    return {
        "pipe_id": pipe["pipe_id"],
        "runs": [
            {
                "run_id": f"R-{rng.randint(4000, 4999)}",
                "started_at": (started + timedelta(days=index * 3)).isoformat(),
                "hours": round(rng.uniform(18, 120), 1),
                # Версия съёмки есть в каждой строке — так сказано в контракте.
                "survey_version_id": pipe["survey_version_id"],
            }
            for index in range(rng.randint(2, 5))
        ],
    }


@api.get("/api/v1/pipes/{pipe_id}/trajectory")
def trajectory(pipe_id: str, from_step: int = Query(0), to_step: int = Query(20)):
    pipe = FLEET.get(pipe_id.upper())
    if pipe is None:
        return _error("E-1108", 404, "Труба не найдена в парке текущей кампании")
    if to_step <= from_step or to_step - from_step > 500:
        return _error("E-2210", 422, "Некорректный диапазон шагов в запросе траектории")

    rng = random.Random(f"{SEED}-traj-{pipe_id}")
    base = rng.uniform(1200, 2600)
    # Разность глубины долота и глубины низа трубы внутри рейса ПОСТОЯННА —
    # прямое требование контракта. Здесь она и постоянна.
    offset = round(rng.uniform(80, 240), 1)
    return {
        "pipe_id": pipe["pipe_id"],
        "survey_version_id": pipe["survey_version_id"],
        "steps": [
            {
                "step": step,
                "end_bit_md_m": round(base + step * 9.3, 1),
                "pipe_bottom_md_m": round(base + step * 9.3 - offset, 1),
            }
            for step in range(from_step, to_step)
        ],
    }


@api.get("/api/v1/events")
def events():
    """Жизненный цикл труб. По контракту на текущей кампании он пуст.

    Пустой ответ здесь — не заглушка, а факт из документации: поле
    `lifecycle_event_count` равно нулю у всего парка. Придумать сюда события
    значило бы развести стенд с вики.
    """
    return {"events": [], "total": 0}


@api.get("/api/v1/calculations/{calculation_id}")
def calculation(calculation_id: str):
    known = {pipe["calculation_id"] for pipe in FLEET.values()}
    if calculation_id not in known:
        return _error("E-1108", 404, "Расчёт не найден")
    return {
        "calculation_id": calculation_id,
        "method": "Майнер",
        "version": "1.1.0",
        "parameters": {"sn_curve": "DNV-D", "safety_factor": 1.0},
    }


@api.get("/api/v1/wells")
def wells(field: str = Query("", description="фильтр по месторождению")):
    """Справочник скважин со сводкой по трубам. Появился в 1.2.0.

    Сводка здесь ЕСТЬ — в отличие от `/pipes`, где её нет намеренно. Это не
    непоследовательность: в `/pipes` агрегат отсутствует по контракту
    (известное ограничение, на него ссылается вся документация), а скважина
    без сводки по трубам вообще не имеет смысла как сущность.
    """
    rows = [well for well in WELLS.values() if not field or well["field"] == field]
    rows.sort(key=lambda well: well["max_damage_fraction"], reverse=True)
    return {"wells": rows, "total": len(rows)}


@api.get("/api/v1/wells/{well_id}")
def well(well_id: str):
    found = WELLS.get(well_id.upper())
    if found is None:
        return _error("E-1108", 404, "Скважина не найдена")
    return found


@api.get("/api/v1/inspections")
def inspections(
    category: str = Query("", description="PREMIUM | CLASS_2 | CLASS_3 | SCRAP"),
    overdue: bool | None = Query(None, description="только просроченные"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Последние инспекции по парку. Появился в 1.2.0.

    Единственный эндпоинт с фильтрами и страницами — и заведён он в том
    числе ради них: пока листать было нечего, кроме выработки, вся работа с
    постраничной выдачей проверялась на одном сценарии.

    ВАЖНО ДЛЯ ПОТРЕБИТЕЛЯ: `category = SCRAP` означает решение инспектора о
    списании. Выработка ресурса к списанию отношения не имеет — по расчёту
    труба не списывается никогда, это правило INSPECT-MOBILE и оно здесь не
    нарушено: в парке есть трубы с высокой выработкой и категорией PREMIUM.
    """
    rows = list(INSPECTIONS.values())
    if category:
        if category.upper() not in CATEGORIES:
            return _error("E-2210", 422, f"Неизвестная категория: {category}")
        rows = [row for row in rows if row["category"] == category.upper()]
    if overdue is not None:
        rows = [row for row in rows if row["overdue"] is overdue]
    rows.sort(key=lambda row: row["inspected_at"], reverse=True)
    page = rows[offset : offset + limit]
    return {"inspections": page, "total": len(rows), "offset": offset, "limit": limit}


@api.get("/api/v1/pipes/{pipe_id}/inspections")
def pipe_inspections(pipe_id: str):
    found = INSPECTIONS.get(pipe_id.upper())
    if found is None:
        return _error("E-1108", 404, "Труба не найдена в парке текущей кампании")
    return {"pipe_id": pipe_id.upper(), "inspections": [found]}


@api.get("/api/v1/surveys")
def surveys():
    """Версии инклинометрической съёмки: только метаданные.

    Сама инклинометрия наружу по-прежнему не выставлена, и это по-прежнему
    записано в документе как ограничение. Здесь — даты, причина уточнения и
    сколько труб пересчитано: ровно то, чем объясняется постмортем 2025-03.
    """
    rows = sorted(SURVEYS.values(), key=lambda row: row["recorded_at"])
    return {"surveys": rows, "total": len(rows)}


@api.get("/api/v1/surveys/{survey_id}")
def survey(survey_id: str):
    found = SURVEYS.get(survey_id.upper())
    if found is None:
        return _error("E-1108", 404, "Версия съёмки не найдена")
    return found


@api.get("/api/v1/runs/{run_id}/trajectory")
def run_trajectory(run_id: str, from_step: int = Query(0), to_step: int = Query(20)):
    """Обратный срез траектории: все трубы одного рейса. Появился в 1.2.0.

    В 1.1.0 положение отдавалось только для одной трубы, и это стояло
    отдельным пунктом «чего в API нет»: вьюеру рейса нужен тот же массив по
    `run_id`. Теперь есть.
    """
    if not run_id.upper().startswith("R-"):
        return _error("E-1108", 404, "Рейс не найден")
    if to_step <= from_step or to_step - from_step > 500:
        return _error("E-2210", 422, "Некорректный диапазон шагов в запросе траектории")

    rng = random.Random(f"{SEED}-run-{run_id}")
    members = sorted(rng.sample(sorted(FLEET), rng.randint(3, 6)))
    base = rng.uniform(1200, 2600)
    return {
        "run_id": run_id.upper(),
        "pipes": members,
        "steps": [
            {
                "step": step,
                "end_bit_md_m": round(base + step * 9.3, 1),
                # Разность постоянна внутри рейса — требование контракта.
                "pipe_bottom_md_m": round(base + step * 9.3 - 140.0, 1),
            }
            for step in range(from_step, to_step)
        ],
    }


@api.get("/health")
def health():
    return {
        "ok": True,
        "fleet": len(FLEET),
        "wells": len(WELLS),
        "surveys": len(SURVEYS),
        "seed": SEED,
        # Что стенд сейчас изображает. Без этой строчки «почему всё
        # отваливается по таймауту» выясняется чтением исходника.
        "rigged": {
            "stale": STALE_PIPE,
            "broken": BROKEN_PIPE,
            "slow": SLOW_PIPE if SLOW_SECONDS else None,
            "slow_seconds": SLOW_SECONDS,
            "rate_limit_per_10s": RATE_LIMIT or None,
        },
    }


if __name__ == "__main__":
    # Запуск прямо из файла, а не через `python -m uvicorn app:api`.
    #
    # Причина в первом же отчёте об ошибке: `-m uvicorn` разбирает строку
    # `модуль:объект` через обычный импорт, и какой именно модуль он найдёт,
    # зависит от текущего каталога. Здесь передаётся сам объект приложения —
    # искать нечего, перепутать не с чем.
    import os

    import uvicorn

    uvicorn.run(api, host="127.0.0.1", port=int(os.environ.get("PW_MOCK_PORT", "8100")))
