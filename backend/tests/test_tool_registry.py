"""Описание инструмента и его выполнение не имеют права разойтись.

Учебник советует держать реестр «имя -> обработчик» в одном месте — иначе
описания и реализации расползаются по коду и расходятся между собой. У нас
они и правда лежат порознь: описания в `LIVE_SPECS`, выполнение — в цепочке
`if/elif` внутри `Toolbox`.

Заводить реестр ради реестра в конце дня — плохая сделка: риск сломать
работающее выше выгоды. А вот выгоду реестра — «разойтись молча нельзя» —
можно получить тестом, и он же переживёт рефакторинг, если тот однажды
случится.

Проверять это стоит именно на нас: сегодня мы уже поймали ровно такую
рассинхронизацию. Поля бланка решения были переписаны руками и отстали от
описаний инструментов на два фильтра — модель физически не могла попросить
инспекции по категории. Стоило прогона.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.rag.tools import LIVE_SPECS, LIVE_TOOL_NAMES, Toolbox


class RecordingLive:
    """Сервис живых данных, который только запоминает, что у него спросили."""

    configured = True

    def __init__(self) -> None:
        self.called: list[str] = []

    async def _record(self, name):
        self.called.append(name)

        class Result:
            text = "ок"
            hits: list = []
            ok = True
            dataset = None

        return Result()

    async def pipe(self, *args, **kwargs):
        return await self._record("pipe")

    async def fleet(self, *args, **kwargs):
        return await self._record("fleet")

    async def inspections(self, *args, **kwargs):
        return await self._record("inspections")

    async def wells(self, *args, **kwargs):
        return await self._record("wells")


EXPECTED = {
    "live_pipe": "pipe",
    "live_fleet": "fleet",
    "live_inspections": "inspections",
    "live_wells": "wells",
}


def test_every_described_tool_has_its_own_handler() -> None:
    """У каждого описанного инструмента своя ветка выполнения.

    Раньше парк стоял веткой `else`, то есть отдавался всему, что не
    опознано. Седьмой инструмент, добавленный без ветки, молча возвращал бы
    таблицу парка на вопрос про скважины — не ошибку, а правдоподобные НЕ ТЕ
    данные, которых никто не заметит.
    """
    live = RecordingLive()
    box = Toolbox(store=None, settings=Settings(), embedder=None, live=live)

    for name in sorted(LIVE_TOOL_NAMES):
        live.called.clear()
        outcome = asyncio.run(box.run(name, {"pipe_id": "PP-0035"}, live_allowed=True))
        assert outcome.ok, f"{name} не выполнился: {outcome.text}"
        assert live.called == [EXPECTED[name]], (
            f"{name} ушёл в обработчик {live.called}, а должен был в {EXPECTED[name]}"
        )


def test_the_map_covers_exactly_the_described_tools() -> None:
    """Список в тесте не имеет права отстать от списка в коде.

    Иначе получится сторож, который сторожит вчерашний набор: добавили
    инструмент, тест о нём не знает, и расхождение снова проходит молча.
    """
    described = {spec["function"]["name"] for spec in LIVE_SPECS}

    assert described == set(EXPECTED), (
        "набор живых инструментов изменился — добавьте ветку выполнения и строку сюда"
    )
    assert described == set(LIVE_TOOL_NAMES)


def test_an_unknown_live_name_is_an_honest_failure_not_the_fleet() -> None:
    """Неопознанное имя обязано вернуть отказ, а не чужую таблицу."""
    live = RecordingLive()
    box = Toolbox(store=None, settings=Settings(), embedder=None, live=live)

    outcome = asyncio.run(box._live_call("live_unicorn", {}))

    assert outcome.ok is False
    assert live.called == [], "неопознанное имя не должно никуда сходить"
