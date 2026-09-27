"""Правило «нужны ли живые данные» — на всём золотом наборе, без модели.

В этом и смысл переноса решения в код. Уговор проверяется только полным
прогоном на полтора часа, и мы этих прогонов на один этот выбор потратили
шесть. Правило проверяется здесь за миллисекунды, и видно ИМЕННО те вопросы,
на которых оно врёт.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.route import needs_live

GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden.jsonl"


def questions() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_every_question_needing_the_service_is_caught() -> None:
    """Ни один живой вопрос не должен проскочить мимо правила.

    Набор сам говорит, каким вопросам нужен инструмент: у них заполнено
    `expect_tools`. Список признаков подгонять под них нельзя бесконечно,
    поэтому промах здесь — это ошибка правила, а не повод поправить набор.
    """
    missed = [
        (item["id"], item["question"])
        for item in questions()
        if item.get("expect_tools")
        # «следующие 5» осмысленно только при открытой таблице — признак
        # приходит от браузера, и в замере он воспроизводится так же.
        and not needs_live(item["question"], has_open_table=item["id"] == "a004")
    ]
    assert missed == [], f"живые вопросы мимо правила: {missed}"


def test_rule_questions_are_not_sent_to_the_service() -> None:
    """Ложное срабатывание дороже пропуска, и цена измерена.

    Прогон, в котором модель осталась без права отказаться, звал сервис на
    обычных вопросах — и `answer_contains` упал с 0.963 до 0.596, а задержка
    выросла вчетверо. Поэтому перекос правила — в сторону молчания.

    ОДНО РАСХОЖДЕНИЕ ОСТАВЛЕНО НАМЕРЕННО: q018 «сколько труб в парке на
    текущей кампании». Набор считает его вопросом к документам, правило
    отправляет в сервис. Подгонять под него правило я не стал: вопрос про
    «сколько труб сейчас» честнее закрывать сервисом. Расхождение записано
    здесь, чтобы оно было осознанным, а не случайным.
    """
    known = {"q018"}
    false_alarms = {
        item["id"]: item["question"]
        for item in questions()
        if not item.get("expect_tools") and needs_live(item["question"])
    }
    unexpected = {k: v for k, v in false_alarms.items() if k not in known}
    assert unexpected == {}, f"вопросы о правилах ушли в сервис: {unexpected}"


@pytest.mark.parametrize(
    "question",
    [
        "когда труба списывается",
        "какой порог по усталости считается аварийным",
        "списывается ли труба по расчёту усталости",
        "что означает ошибка E-1042",
        "сколько сечений у трубы",
        "какие категории инспекции труб бывают",
        "как дела?",
    ],
)
def test_the_shape_of_the_question_decides_not_the_word_pipe(question: str) -> None:
    """Слово «труба» в вопросе само по себе ничего не значит.

    «Когда труба списывается» — это правило, «какие трубы под списание» — это
    список объектов, и отличаются они не темой, а формой вопроса. Поэтому
    признаки «это правило» проверяются первыми и перебивают остальное.
    """
    assert needs_live(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "дай топ 5 труб по выработке",
        "какие трубы под списание",
        "у кого просрочена инспекция",
        "на какой скважине трубы хуже всего",
        "какая выработка у PP-0035",
        "что с трубой PP-0007",
    ],
)
def test_objects_and_their_numbers_go_to_the_service(question: str) -> None:
    assert needs_live(question) is True


def test_next_page_needs_an_open_table() -> None:
    """«Следующие пять» без таблицы на экране — это просто «следующие пять».

    Листает человек, а не агент: метку следующей страницы даёт браузер. Без
    открытой таблицы просьба о продолжении ни к чему не относится, и лезть в
    сервис по ней значило бы угадывать, что человек имел в виду.
    """
    assert needs_live("следующие 5", has_open_table=True) is True
    assert needs_live("следующие 5", has_open_table=False) is False


def test_the_router_names_the_right_source() -> None:
    """Инструмент угадывается правилом, чтобы не жечь шаг на промахе.

    В замере «route» пять живых вопросов из девяти начинались с `live_pipe` —
    паспорта ОДНОЙ трубы, — притом что идентификатора трубы в вопросе не было.
    Проверка отбивала вызов и подсказывала верное имя, модель со второго шага
    шла правильно. Работало и стоило половины бюджета: шагов у агента два.

    Девять примеров на четыре класса — этого мало для уверенности, поэтому имя
    уходит подсказкой, а не запретом. Но раз мерить можно, надо мерить.
    """
    from app.rag.route import live_tool

    wrong = []
    for item in questions():
        expected = (item.get("expect_tools") or [None])[0]
        if not expected:
            continue
        got = live_tool(item["question"], has_open_table=item["id"] == "a004")
        if got != expected:
            wrong.append((item["id"], item["question"], expected, got))
    assert wrong == [], f"правило назвало не тот источник: {wrong}"


def test_a_rule_question_gets_no_source_at_all() -> None:
    """Если вопрос не про сегодняшние данные, источника нет — и подсказки нет."""
    from app.rag.route import live_tool

    assert live_tool("когда труба списывается") is None
    assert live_tool("какой порог по усталости считается аварийным") is None


def test_the_pipe_id_is_taken_from_the_question_not_asked_for_again() -> None:
    """Идентификатор трубы извлекает код, а не модель.

    Из замера: на «какая выработка у PP-0035» агент дважды позвал паспорт
    трубы и ни разу не передал саму трубу. Проверка отбивала вызов, таблица
    не приходила, и ответ выходил «данных конкретно по трубе PP-0035 не
    указано» — при том, что имя стоит и в вопросе, и в самом ответе.

    Правило и так его извлекает: по нему выбирается `live_pipe`.
    """
    from app.rag.route import pipe_id

    assert pipe_id("какая выработка у PP-0035") == "PP-0035"
    assert pipe_id("что с трубой pp-0007") == "PP-0007"
    assert pipe_id("дай топ 5 труб по выработке") == ""
    assert pipe_id("когда труба списывается") == ""


# --- ОДИН КРАЙНИЙ СЛУЧАЙ ПРОТИВ СПИСКА --------------------------------------


def test_a_superlative_wants_one_row() -> None:
    """Вопрос про один крайний случай: ответом служит верхняя строка.

    Из замера: «на какой скважине трубы хуже всего». Таблица пришла верная,
    ответ назвал выдуманную трубу, предложение выброшено — и на его место
    пошло «Данные в таблице выше». Безопасно и бесполезно: человек и сам
    видит, что таблица выше, а ответ в ней первой строкой.
    """
    from app.rag.route import wants_one_row

    assert wants_one_row("на какой скважине трубы хуже всего")
    assert wants_one_row("самая изношенная труба парка")
    assert wants_one_row("где наибольшая выработка")


def test_a_list_beats_a_superlative() -> None:
    """«Топ-5 по максимальной выработке» — это список, и подменять его нельзя.

    Слово «максимальной» здесь есть, а ответом служит вся выборка. Отдать
    вместо неё одну строку значило бы завести новый дефект на месте починки
    старого: человек решит, что остальных четырёх нет.
    """
    from app.rag.route import wants_one_row

    for question in (
        "дай топ 5 труб по выработке",
        "дай топ 10 труб парка",
        "какие трубы за порогом аварии",
        "у кого просрочена инспекция",
        "следующие 5",
        "сколько труб с максимальной выработкой",
        "перечисли самые изношенные трубы",
    ):
        assert not wants_one_row(question), question


def test_the_rule_is_measured_on_the_whole_set() -> None:
    """ЗАМЕР, а не уверенность: правило прогнано по всему набору.

    Так же, как `needs_live`: проверка по всем вопросам за миллисекунды,
    вместо прогона на полтора часа. Срабатываний ровно два, и оба разобраны:

        a009 «на какой скважине трубы хуже всего» — цель правила;
        q037 «какой максимальный размер пул-реквеста» — вопрос по документам.

    Второе безвредно по построению: откат вызывается только при пришедшей
    таблице, а у вопроса по документам таблиц ноль. Правило намеренно НЕ
    подгонялось под этот случай — подгонка под вопрос, до которого дело не
    доходит, добавила бы условие без выигрыша.
    """
    import json
    from pathlib import Path

    from app.rag.route import wants_one_row

    fired = []
    for line in Path("eval/golden.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if wants_one_row(payload["question"]):
            fired.append(payload["id"])

    assert fired == ["q037", "a009"], (
        f"правило сработало иначе, чем замерено: {fired}. "
        f"Если это намеренно — перемеряйте и перепишите этот тест"
    )


def test_no_live_question_asking_for_a_list_is_caught() -> None:
    """Ни один живой вопрос про выборку не должен попасть под подмену.

    Это та проверка, которая обязана падать при расширении правила: цена
    ошибки здесь односторонняя, и сторона известна.
    """
    import json
    from pathlib import Path

    from app.rag.route import wants_one_row

    for line in Path("eval/golden.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("type") != "live":
            continue
        if (payload.get("expect_tables") or 0) <= 1 and payload["id"] == "a009":
            continue
        assert not wants_one_row(payload["question"]), payload["id"]
