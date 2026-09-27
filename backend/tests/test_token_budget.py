"""Влезают ли части запроса в окно модели.

Бюджеты у нас заданы по частям: столько фрагментам, столько истории, столько
ответу. Удобно и имеет одну дырку: сумму частей никто не сверял с окном.
Перебор не падает с внятной ошибкой — он обрезает ввод молча, а виноватым
выглядит разбор ответа. Мы это уже проходили: `max_answer_tokens`,
перекрытый в `.env` значением 700, стоил прогона и выглядел как порча
структуры.
"""

from __future__ import annotations

import pytest

from app.config import Settings, budget_problems


def test_a_fitting_budget_is_silent() -> None:
    """Проверка, срабатывающая на исправной настройке, будет отключена."""
    assert budget_problems(Settings(context_window=32768), system_tokens=900) == []


def test_an_overflowing_budget_is_named_with_the_arithmetic() -> None:
    """Сообщение обязано содержать разбор по частям, а не только итог.

    «Не влезает» без слагаемых заставляет складывать руками, а складывать
    надо шесть чисел из трёх файлов.
    """
    problems = budget_problems(
        Settings(context_window=4096, context_token_budget=3200, max_answer_tokens=1200),
        system_tokens=900,
    )

    assert len(problems) == 1
    message = problems[0]
    assert "4096" in message
    assert "фрагменты 3200" in message
    assert "место под ответ 1200" in message
    # И что делать — тоже в сообщении: ошибка настройки без имени настройки
    # отправляет читать код.
    assert "PW_CONTEXT_TOKEN_BUDGET" in message


def test_the_answer_reserve_is_counted() -> None:
    """Место под ответ — часть бюджета, а не то, что «как-нибудь влезет».

    Классическая ошибка книжного списка: модель доходит до конца окна
    посреди фразы и отдаёт `finish_reason: length`.
    """
    window = 5000
    base = Settings(
        context_window=window, context_token_budget=3000,
        history_token_budget=500, max_answer_tokens=100,
    )
    assert budget_problems(base, system_tokens=500) == []

    greedy = base.model_copy(update={"max_answer_tokens": 2000})
    assert budget_problems(greedy, system_tokens=500), "потолок ответа обязан учитываться"


def test_zero_window_turns_the_check_off() -> None:
    """У модели с неизвестным окном честнее выключить проверку, чем врать.

    Проверка по выдуманному числу хуже отсутствующей: она отказывает в
    запуске рабочей настройке, и первое, что с ней сделают, — обойдут.
    """
    settings = Settings(context_window=0, context_token_budget=999_999)
    assert budget_problems(settings, system_tokens=999_999) == []


def test_the_pipeline_refuses_to_start_on_an_overflow() -> None:
    """Отказ, а не предупреждение, и проверка стоит в конвейере.

    Конвейер — единственная дорога, по которой идут оба пути: и сервер, и
    прогоны. Сервис, поднявшийся с заведомо обрезанным контекстом, не выдаёт
    ошибку — он выдаёт тихо ухудшённые ответы, которых никто не заметит.
    """
    from app.pipeline import Pipeline

    settings = Settings(context_window=1024, context_token_budget=3200)

    with pytest.raises(RuntimeError, match="не влезает в окно"):
        Pipeline(
            settings=settings, store=None, providers=None, traces=None,
        )


def test_a_label_with_a_path_inside_cannot_lose_the_run() -> None:
    """Прогон на полтора часа потерялся из-за метки. Больше — нет.

    Метка приехала как `verdict.\\run.ps1`: человек вставил команду дважды, и
    второе «.\\run.ps1 verdict» ушло в аргумент. Файл сохранился по пути
    `...-answer-verdict.\\run.ps1.json`, то есть в подпапку `verdict.` — и
    после этого не находился ни по метке, ни в списке прогонов. Работа
    сделана, результат недоступен.

    Уговором «набирайте метки аккуратно» это не лечится: метку пишут руками
    в конце длинной команды, и ошибаться там будут всегда.
    """
    from eval.runner import safe_label

    assert "/" not in safe_label("a/b")
    assert "\\" not in safe_label("verdict.\\run.ps1")
    assert not safe_label("verdict.").endswith("."), "точка на конце — ловушка Windows"
    # Пустая метка обязана превратиться в имя, а не в пустоту.
    assert safe_label("   ") == "run"
    assert safe_label("..") == "run"
    # Нормальную метку трогать нельзя: иначе сравнение прогонов по имени
    # начнёт промахиваться.
    assert safe_label("enum-order") == "enum-order"
    assert safe_label("ok-1_2") == "ok-1_2"
