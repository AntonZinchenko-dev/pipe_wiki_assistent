"""Починка должна быть ПОДКЛЮЧЕНА, а не только написана.

Зачем отдельная проверка на подключение. За два дня это ломалось трижды, и
каждый раз одинаково: функция написана, покрыта тестами, проходит — и не
вызывается из работающего пути. `agent_version` трижды не видел того, что
обязан был видеть. Подсказка с идентификатором трубы была написана, стояла в
промпте и не превращалась в аргумент. Тесты самой функции этого не ловят по
определению: они зовут её напрямую.

Проверка читает исходник конвейера. Это грубо, и грубость здесь осознанная:
поднять весь конвейер с поддельными провайдерами дороже, чем убедиться, что
вызов стоит на месте и стоит в нужном порядке.
"""

from __future__ import annotations

from pathlib import Path


def _pipeline() -> str:
    return Path("app/pipeline.py").read_text(encoding="utf-8")


def test_the_repair_is_called_from_the_pipeline() -> None:
    """Функция вызвана из рабочего пути, а не просто существует.

    Импорт проверяем по МОДУЛЮ, а не по строке исходника: строку я уже один
    раз сломал, переписав импорт в многострочный, — и тест упал на форме
    записи вместо сути. Проверка, которая ловит переформатирование, шумит и
    её начинают отключать.
    """
    from app import pipeline

    assert hasattr(pipeline, "drop_invented"), "починка не импортирована в конвейер"
    assert "drop_invented(" in _pipeline(), "починка не вызывается из конвейера"


def test_denial_is_repaired_before_inventions() -> None:
    """Порядок важен, и вот почему.

    Предложение вроде «данных по PP-9999 нет» — это И отрицание, И выдуманное
    обозначение. Убрать его должно правило отрицания: оно про оформление и
    ничего не скрывает. Поставь проверку выдумки первой — и то же самое
    предложение уедет в `invented_dropped`, то есть уронит `live_clean` там,
    где дефекта нет. Счётчик начнёт показывать выдумку на честных отказах.
    """
    source = _pipeline()
    assert source.index("repair_denial(") < source.index("drop_invented("), (
        "проверка выдумки стоит раньше починки отрицаний — счётчик соврёт"
    )


def test_the_repair_runs_before_the_leak_guard() -> None:
    """Утечка промпта подменяет ответ ЦЕЛИКОМ — после неё чинить нечего."""
    source = _pipeline()
    assert source.index("drop_invented(") < source.index("leaked_lines("), (
        "починка стоит после подмены ответа на отказ — она ничего не изменит"
    )


def test_the_dropped_names_leave_the_user_warning() -> None:
    """Предупреждение человеку не должно называть убранное.

    Интерфейс печатает `unknown_refs` словами: «в ответе названы обозначения,
    которых нет в источниках». Оставить там убранное обозначение — отправить
    человека искать в тексте то, чего в тексте больше нет.
    """
    source = _pipeline()
    assert "envelope.invented_dropped = dropped" in source
    assert "if name not in dropped" in source, (
        "убранные обозначения остались в предупреждении для человека"
    )


def test_the_payload_carries_the_field() -> None:
    """Поле уходит наружу — иначе прогон о починке не узнает."""
    from app.rag.answer import AnswerEnvelope, AnswerStatus

    envelope = AnswerEnvelope(status=AnswerStatus.ANSWERED, answer="ответ")
    envelope.invented_dropped = ["PP-0035"]
    assert envelope.to_dict()["invented_dropped"] == ["PP-0035"]


def test_the_run_counts_the_field() -> None:
    """Прогон считает убранное нечистым ответом."""
    source = Path("eval/runner.py").read_text(encoding="utf-8")
    where = source.index("live_clean=")
    window = source[where:where + 400]
    assert "invented_dropped" in window, (
        "live_clean не учитывает убранное — метрика станет 1.000 от самой починки"
    )
