"""Поход агента в живой сервис: права, границы, честность отчёта.

Здесь проверяется не «доехал ли запрос» — это видно на глаз при первом
запуске. Проверяется то, что ломается тихо и дорого:

1. модель может подсунуть в путь что угодно, потому что текст в вики пишут
   люди, а модель управляется текстом;
2. токен сервиса может утечь в диалог с моделью вместе с ответом;
3. одна труба с устаревшим расчётом может отменить обзор всего парка —
   или, что хуже, промолчать и сделать ответ неполным незаметно;
4. живой фрагмент может не доехать до контекста и вылететь при отсеве
   повторов;
5. инструмент может показываться модели, когда сервис не настроен.

Сеть здесь не нужна: httpx умеет подставной транспорт, и все ответы
сервиса задаются прямо в тесте.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from app.config import Settings
from app.rag.live import LiveApi
from app.rag.tools import LIVE_SPECS, Toolbox


FLEET = {
    "PP-0001": 0.12,
    "PP-0002": 0.55,
    "PP-0003": 0.91,
    "PP-0004": 0.83,
    "PP-0007": None,  # устаревший расчёт: сервис отдаёт E-1042
}


def _handler(seen: list[str]):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        path = request.url.path

        if path == "/api/v1/pipes":
            return httpx.Response(200, json={
                "pipes": [{"pipe_id": name} for name in FLEET],
                "total": len(FLEET),
            })

        if path.endswith("/passport"):
            pipe_id = path.split("/")[-2]
            if pipe_id not in FLEET:
                return httpx.Response(404, json={
                    "error": {"code": "E-1108", "message": "Труба не найдена"}
                })
            damage = FLEET[pipe_id]
            if damage is None:
                return httpx.Response(409, json={
                    "error": {"code": "E-1042", "message": "Расчёт устарел"}
                })
            return httpx.Response(200, json={
                "pipe_id": pipe_id,
                "well_id": "W-100",
                "miner_damage_fraction": damage,
                "cycles_total": 1_500_000,
                "critical_local_position_m": 3.2,
                "steel_grade": "S-135",
                "outer_diameter_mm": 127.0,
                "length_m": 9.4,
                "survey_version_id": "SV-2026-09",
                "calculation_id": "CALC-9001",
                "updated_at": "2026-09-20T10:00:00+00:00",
            })

        return httpx.Response(404, json={"error": {"code": "E-1108", "message": "нет"}})

    return handle


def make(*, token: str = "", url: str = "http://fatigue.local") -> tuple[LiveApi, list[str]]:
    seen: list[str] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler(seen)))
    settings = Settings(fatigue_api_url=url, fatigue_api_token=token)
    return LiveApi(settings, client), seen


# ------------------------------------------------------------------- права


def test_the_model_never_supplies_an_address() -> None:
    """В описании инструментов нет ни одного поля под адрес.

    Соблазн дать `http_get(url)` велик: один инструмент вместо двух. Но
    адрес от модели — это адрес из документа вики, а документы пишут люди.
    Такой инструмент отправит запрос куда угодно, откуда дотянется наш
    сервер: во внутреннюю админку, в служебные адреса облака, в соседний
    сервис без пароля. Тест сторожит границу, чтобы её не сдвинули
    «временно».
    """
    for spec in LIVE_SPECS:
        fields = spec["function"]["parameters"]["properties"]
        for name in fields:
            assert name not in {"url", "path", "endpoint", "host", "address"}, name
        assert "url" not in json.dumps(spec, ensure_ascii=False).lower()


def test_a_path_in_the_pipe_id_goes_nowhere() -> None:
    """Идентификатор чистится до букв, цифр и дефиса.

    Документ, в котором написано «труба ../../admin», не должен превращать
    наш запрос в обращение к другому адресу. Проверяем не текст ответа, а
    то, КУДА ушёл запрос.
    """
    live, seen = make()

    for payload in ["../../admin", "PP-0003/../../internal", "PP-0003?token=leak"]:
        seen.clear()
        asyncio.run(live.pipe(payload))

        assert len(seen) == 1, payload
        sent = seen[0]
        # Ни выхода вверх по путям, ни лишнего сегмента, ни параметров
        # запроса: идентификатор остаётся ОДНИМ куском пути.
        assert ".." not in sent, sent
        assert "?" not in sent, sent
        assert sent.startswith("http://fatigue.local/api/v1/pipes/"), sent
        # Ровно один сегмент между /pipes/ и /passport — то есть в путь не
        # вклинился ни лишний каталог, ни другой эндпоинт.
        tail = sent.removeprefix("http://fatigue.local/api/v1/pipes/")
        assert tail.count("/") == 1 and tail.endswith("/passport"), sent


def test_the_service_token_never_reaches_the_model() -> None:
    """Токен уходит в заголовок и никогда — в текст для модели.

    Всё, что возвращает инструмент, попадает в диалог и может быть
    процитировано в ответе. Секрет, попавший туда, попадёт и на экран.
    """
    live, _ = make(token="s3cret-token-value")
    result = asyncio.run(live.pipe("PP-0003"))

    assert "s3cret-token-value" not in result.text
    assert all("s3cret-token-value" not in hit.chunk.body for hit in result.hits)


def test_tools_are_hidden_when_the_service_is_not_configured() -> None:
    """Ненастроенный инструмент модели не показывается вовсе.

    Показать и отвечать ошибкой — худший вариант: модель попробует, получит
    отказ, попробует ещё раз и потратит на это все разрешённые шаги.
    """
    live_off, _ = make(url="")
    live_on, _ = make()

    off = Toolbox(store=None, settings=Settings(), embedder=None, live=live_off)
    on = Toolbox(store=None, settings=Settings(), embedder=None, live=live_on)

    assert not any("live_" in spec["function"]["name"] for spec in off.specs())
    # Сверяем со СПИСКОМ, а не с числом: число пришлось бы править при
    # каждом новом инструменте, и правка «было два, стало четыре» ничего не
    # проверяет — она просто догоняет код.
    assert sum("live_" in spec["function"]["name"] for spec in on.specs()) == len(LIVE_SPECS)


# ------------------------------------------------------- поведение и отчёт


def _ids(result) -> set[str]:
    """Номера труб из ТАБЛИЦЫ результата.

    Проверять по тексту для модели больше нельзя: строк там нет по
    построению. Это не неудобство теста, а ровно то изменение, ради
    которого всё затевалось.
    """
    assert result.dataset is not None
    return {row["pipe_id"] for row in result.dataset.rows}


def test_the_threshold_comes_from_the_caller_not_from_the_code() -> None:
    """Порог 80 % не зашит здесь, и это принципиально.

    Регламент сам предупреждает, что пороги пересматривают и что значения
    из старых постмортемов недействительны. Зашитое в код число разошлось
    бы с документом молча — то есть система уверенно отвечала бы по
    отменённому порогу.
    """
    live, _ = make()
    strict = asyncio.run(live.fleet(0.9))
    loose = asyncio.run(live.fleet(0.5))

    # Смотрим на ТАБЛИЦУ, а не на текст для модели: строки теперь живут
    # там, и порог отбирает именно их.
    assert _ids(strict) == {"PP-0003"}
    assert {"PP-0002", "PP-0004"} <= _ids(loose)


def test_one_stale_pipe_does_not_cancel_the_survey_but_is_named() -> None:
    """Обзор парка переживает отказ по одной трубе — и сообщает о нём.

    Молча пропустить нельзя: «две трубы за порогом» и «две трубы за
    порогом, а по одной расчёт устарел» — разные ответы, и человек имеет
    право знать, что обзор неполный.
    """
    live, _ = make()
    result = asyncio.run(live.fleet(0.8))

    # Прошедшие порог — в ТАБЛИЦЕ: строк в выжимке для модели больше нет.
    assert _ids(result) == {"PP-0003", "PP-0004"}
    # А вот про пропущенную трубу модель знать ОБЯЗАНА: без этого она
    # напишет «две трубы за порогом», а честный ответ — «две за порогом,
    # и ещё по одной расчёт устарел».
    assert "PP-0007" in result.text
    assert "E-1042" in result.text
    assert result.dataset is not None and result.dataset.skipped == ["PP-0007 (E-1042)"]


def test_an_error_is_explained_with_its_code() -> None:
    """Код ошибки обязателен: по нему в вики есть таблица с инструкцией.

    Голое «сервис ответил ошибкой» превращает штатную ситуацию в тупик —
    ни модель, ни человек не могут найти, что делать дальше.
    """
    live, _ = make()

    missing = asyncio.run(live.pipe("PP-9999"))
    assert missing.ok is False
    assert missing.dataset is not None and missing.dataset.error is not None
    assert missing.dataset.error.code == "E-1108"
    # ПРИЗНАК, А НЕ ТОЛЬКО СЛОВА. «Повтор не поможет» в тексте модель может
    # пересказать как угодно; поле `retriable` не пересказывается — по нему
    # интерфейс решает, предлагать ли кнопку «повторить».
    assert missing.dataset.error.retriable is False
    # В тексте для модели то же самое сказано словами: признак нужен
    # интерфейсу, слова — модели, и расходиться они не должны.
    assert "повторять запрос бессмысленно" in missing.text

    stale = asyncio.run(live.pipe("PP-0007"))
    assert stale.ok is False
    assert stale.dataset is not None and stale.dataset.error is not None
    assert stale.dataset.error.code == "E-1042"
    # Этот отказ временный, и через минуту повтор осмыслен.
    assert stale.dataset.error.retriable is True
    assert stale.dataset.error.retry_after_s == 60


def test_live_data_enters_the_context_as_an_ordinary_fragment() -> None:
    """Иначе половина ответа проверяется, а половина нет.

    Живые числа приходят обычным фрагментом с номером, и проверка цитат
    работает над ними без единой правки. Номер куска отрицательный: по
    нему идёт отсев повторов, и совпадение с настоящим куском из базы тихо
    выбросило бы живой фрагмент из контекста.
    """
    live, _ = make()
    result = asyncio.run(live.pipe("PP-0003"))

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.chunk.chunk_id < 0
    assert hit.found_by == "живые данные сервиса"
    # В шапке — время снятия: число из API устаревает, строчка регламента нет.
    assert "снято" in hit.chunk.heading_path
    # Тело читаемое, не JSON: проверка цитат вырезает предложения, а по
    # фигурным скобкам она вырезала бы обрывок разметки.
    assert "{" not in hit.chunk.body
    # Паспорт одной трубы — одна строка, и модель её видит: на вопрос
    # «какая выработка у PP-0003» ответом служит само число.
    assert "91.0" in hit.chunk.body


def test_two_live_fragments_do_not_collide() -> None:
    """Два вызова подряд дают разные номера кусков.

    Одинаковые номера означали бы, что второй фрагмент считается повтором
    первого и выбрасывается при отсеве — инструмент отработал, данные
    получены, в контекст не попали.
    """
    live, _ = make()
    first = asyncio.run(live.pipe("PP-0003"))
    second = asyncio.run(live.pipe("PP-0004"))

    assert first.hits[0].chunk.chunk_id != second.hits[0].chunk.chunk_id


def test_the_fleet_survey_is_cached() -> None:
    """Обзор парка — это вызов на каждую трубу, и повторять его незачем.

    В контракте прямо сказано: обход всего парка укладывается в лимит, но
    занимает около сорока секунд, и потребителям рекомендовано кэшировать
    агрегат у себя. Делаем ровно это.
    """
    live, seen = make()
    asyncio.run(live.fleet(0.8))
    after_first = len(seen)
    asyncio.run(live.fleet(0.8))

    assert len(seen) == after_first


def test_a_nonsense_threshold_is_refused_before_any_request() -> None:
    """Порог вне диапазона не стоит ни одного вызова сервиса."""
    live, seen = make()
    result = asyncio.run(live.fleet(42))

    assert result.ok is False
    assert seen == []


def test_the_next_page_is_a_page_and_not_a_bigger_list() -> None:
    """«Следующие пять» — это страница, а не запрос списка побольше.

    Без смещения оболочка честно сообщала «показано 5 из 39» и не давала
    никакого способа получить остальные. Модели оставалось одно: взять
    первые десять и назвать словами строки с шестой по десятую — то самое
    переписывание таблицы, ради избавления от которого всё затевалось.

    Оболочка, объявляющая о скрытых данных, обязана давать способ их
    достать. Иначе она не описывает ограничение, а провоцирует обход.
    """
    live, _ = make()

    first = asyncio.run(live.fleet(None, 2))
    second = asyncio.run(live.fleet(None, 2, 2))

    assert first.dataset is not None and second.dataset is not None
    # Страницы не пересекаются и идут подряд по убыванию износа.
    assert not (_ids(first) & _ids(second))
    assert second.dataset.offset == 2
    # И заголовок соответствует содержимому: «топ-2» на третьей-четвёртой
    # строке был бы заголовком, который врёт.
    assert "топ" not in second.dataset.title.lower()


def test_the_hint_names_an_action_not_just_a_fact() -> None:
    """«Показано 5 из 39» — факт, из которого непонятно, что делать."""
    live, _ = make()

    page = asyncio.run(live.fleet(None, 2))

    assert page.dataset is not None
    assert "следующие 2" in page.dataset.hint


def test_paging_past_the_end_is_not_an_error() -> None:
    """Человек просто пролистал дальше конца — это не повод для отказа."""
    live, _ = make()

    beyond = asyncio.run(live.fleet(None, 5, 999))

    assert beyond.ok is True
    assert beyond.dataset is not None and beyond.dataset.rows == []


def test_the_model_does_no_arithmetic_to_turn_the_page() -> None:
    """Смещение модель поняла иначе, чем я его назвал, — и была вправе.

    На «следующие 5» после первой пятёрки она передала offset=1 и
    получила строки со второй по шестую. Это не ошибка в арифметике: она
    прочла «смещение» как «номер начальной строки». Спорить бесполезно —
    любое название параметра кто-нибудь поймёт иначе.

    Метку толковать не нужно: её передают обратно как есть.
    """
    live, _ = make()

    first = asyncio.run(live.fleet(None, 2))
    assert first.dataset is not None and first.dataset.next_cursor

    second = asyncio.run(live.fleet(cursor=first.dataset.next_cursor))

    assert second.dataset is not None
    assert second.dataset.offset == 2
    assert not (_ids(first) & _ids(second))


def test_a_broken_cursor_is_refused_not_silently_reset() -> None:
    """Молчаливый откат на первую страницу — худшее из возможного.

    Человек просит следующую, получает первую и решает, что дальше
    ничего нет.
    """
    live, _ = make()

    result = asyncio.run(live.fleet(cursor="что-то не то"))

    assert result.ok is False
    assert result.dataset is None


# ------------------------------------------------- инспекции и скважины


def test_inspections_and_fleet_are_different_questions() -> None:
    """Списание и выработка ресурса — разные вещи, и инструменты разные.

    Правило домена: по расчёту труба не списывается НИКОГДА, решение
    принимает инспектор по физическому осмотру. Склей мы категорию осмотра
    с выработкой в одну таблицу — предложили бы модели ровно то сравнение,
    которое регламент запрещает, и получили бы «под списание PP-0035, у неё
    97 % ресурса»: правдоподобно и неверно.
    """
    names = {spec["function"]["name"] for spec in LIVE_SPECS}
    assert {"live_fleet", "live_inspections", "live_wells"} <= names

    inspections = next(
        spec for spec in LIVE_SPECS if spec["function"]["name"] == "live_inspections"
    )
    # Описание обязано отправлять вопрос про списание СЮДА, иначе модель
    # ответит по выработке — она ближе лежит.
    assert "списание" in inspections["function"]["description"]


def test_a_live_tool_name_cannot_be_forgotten_in_the_rights_check() -> None:
    """Набор имён рядом со списком инструментов разъезжается с ним молча.

    Третий инструмент показался бы модели в списке, она бы его позвала, а
    проверка прав не узнала имени и ответила «такого инструмента нет» —
    ошибка выглядела бы как галлюцинация модели.
    """
    from app.rag.tools import LIVE_TOOL_NAMES

    assert LIVE_TOOL_NAMES == {spec["function"]["name"] for spec in LIVE_SPECS}


def test_a_typo_in_the_category_is_a_refusal_not_an_empty_list() -> None:
    """«Таких труб нет» на опечатку — неправда, и стоит она неверного ответа."""
    from app.rag.live import _as_category

    value, error = _as_category("SCRAPP")

    assert value == ""
    assert "Неизвестная категория" in error
    assert _as_category("scrap") == ("SCRAP", "")
    assert _as_category(None) == ("", "")


def test_overdue_has_three_states_not_two() -> None:
    """«Инспекция в срок» и «не фильтровать» — разные запросы.

    Булев флаг свёл бы их в одно, и половина вопросов стала бы
    непередаваемой.
    """
    from app.rag.live import _as_flag

    assert _as_flag(None) == (None, "")
    assert _as_flag(True) == (True, "")
    assert _as_flag("нет") == (False, "")
    assert _as_flag("может быть")[1]


def test_the_inspection_cursor_carries_the_filter() -> None:
    """Иначе «следующие пять» молча сменят фильтр на середине списка."""
    from app.rag.live import _make_inspection_cursor, _read_inspection_cursor

    mark = _make_inspection_cursor("SCRAP", True, 5, 5)

    assert _read_inspection_cursor(mark) == ("SCRAP", True, 5, 5)
    assert _read_inspection_cursor("inspect|SCRAP|1|5") is None
    assert _read_inspection_cursor("fleet|0.0|5|5") is None


def test_the_inspection_page_does_not_claim_a_scan_it_did_not_do() -> None:
    """Фильтрует сервис — значит «просмотрено» ставить нечего.

    «Просмотрено 5» после серверного фильтра отчитывается о работе,
    которой не было, и заодно намекает модели, что в парке пять труб.
    """
    import inspect as _inspect

    from app.rag import live

    source = _inspect.getsource(live.LiveApi.inspections)
    assert "scanned=0" in source


def test_the_fleet_tool_comes_first() -> None:
    """Маленькая модель тяготеет к первому инструменту списка.

    Замером показано: за весь прогон агент не позвал live_fleet ни разу, а
    live_pipe звал на всё подряд — и на парк, и на скважины, и на
    «следующие 5». live_pipe стоял первым.

    Самый частый вопрос — про парк, значит первым стоит парк. Это не
    подкрутка под модель: порядок в списке и есть подсказка о том, что
    вероятнее.
    """
    assert LIVE_SPECS[0]["function"]["name"] == "live_fleet"


def test_a_passport_call_without_a_pipe_id_is_refused() -> None:
    """Паспорт одной трубы без идентификатора трубы не имеет смысла.

    Описание инструмента про это написано и не сработало. Вместо похода в
    сервис с выдуманным PP-0000 возвращаем отказ, который НАЗЫВАЕТ нужный
    инструмент: модель прочитает его на том же шаге и исправится, не
    потратив вызов впустую.
    """
    import asyncio

    from app.config import Settings
    from app.rag.tools import Toolbox

    live, _ = make()
    box = Toolbox(store=None, settings=Settings(), embedder=None, live=live)

    outcome = asyncio.run(box.run("live_pipe", {"pipe_id": "парк"}))

    assert not outcome.ok
    assert "live_fleet" in outcome.text
    assert "live_inspections" in outcome.text


def test_a_real_pipe_id_still_goes_through() -> None:
    """Проверка не должна перекрыть тот случай, ради которого инструмент есть."""
    from app.rag.tools import _PIPE_ID

    assert _PIPE_ID.fullmatch("PP-0035")
    assert _PIPE_ID.fullmatch("W-122")
    assert not _PIPE_ID.fullmatch("")
    assert not _PIPE_ID.fullmatch("все трубы")
