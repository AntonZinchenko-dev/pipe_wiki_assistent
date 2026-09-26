"""Команды, которые запускают другие команды изнутри.

`noise` делает три прогона подряд, `gate` — один поисковый. Обе зовут не
подпроцесс, а прямо функцию другой команды, и значит должны передать ей её
аргументы. Раньше набор аргументов собирался руками:

    argparse.Namespace(label=..., note=..., limit=..., with_judge=False)

Потом у команды `answer` появился флаг `--nudge`, `_answer` начал его читать, а
рукописный набор про него не знал. `noise --mode answer` стал падать с
`AttributeError` на первом же прогоне.

Сломалось при этом не абы что. `noise` — единственный источник порога
значимости, и в режиме ответа он не работал вовсе. То есть порог для метрик
ответа был не «решили пока не мерить», а «мерить было нечем»: `compare` всё это
время подставлял в ответную половину таблицы ноль, измеренный на поисковом
прогоне, где генерации нет и разброс честно нулевой.

Дефект того же класса, что и константа в двух местах: **рукописная копия чужого
интерфейса однажды разойдётся с оригиналом, и каждая половина по отдельности
будет выглядеть правильной.** Падения не было ни в одном тесте, потому что
тесты проверяли команды по отдельности.

Здесь проверяется стык. Тест интроспективный — читает исходник и смотрит, какие
поля функция берёт из `args`. Это выглядит непривычно, но альтернатива —
перечислить поля списком, то есть завести третью рукописную копию рядом с той,
из-за которой всё и сломалось.
"""

from __future__ import annotations

import inspect
import re

import pytest

import scripts.eval as cli


def attributes_read_from_args(function) -> set[str]:
    """Какие поля функция читает из `args`. По исходнику, а не по догадке."""
    source = inspect.getsource(function)
    return set(re.findall(r"\bargs\.(\w+)", source))


@pytest.mark.parametrize(
    "command, runner",
    [("answer", cli._answer), ("search", cli._search)],
)
def test_synthetic_namespace_has_everything_the_command_reads(command, runner) -> None:
    """Главный тест файла, и он бы поймал падение `noise`.

    Набор аргументов, который `noise` и `gate` передают внутрь, обязан содержать
    ВСЁ, что вызываемая функция читает из `args`. Не то, что читала когда-то.
    """
    parser = cli.run_parser(command)
    namespace = cli.run_namespace(parser)
    missing = {
        attribute
        for attribute in attributes_read_from_args(runner)
        if not hasattr(namespace, attribute)
    }
    assert not missing, (
        f"команда {command} читает из args поля {sorted(missing)}, которых нет в "
        "наборе по умолчанию — значит вызов изнутри другой команды упадёт"
    )


def test_nudge_is_off_by_default_in_a_generated_run() -> None:
    """Конкретно тот флаг, на котором всё сломалось.

    Мало, чтобы поле существовало: замер разброса обязан идти на НЕИЗМЕНЁННОМ
    промпте. Если бы значение по умолчанию однажды стало `True`, три прогона
    «без изменений» молча измеряли бы разброс испорченной системы.
    """
    namespace = cli.run_namespace(cli.run_parser("answer"))
    assert namespace.nudge is False


def test_overrides_are_applied() -> None:
    namespace = cli.run_namespace(
        cli.run_parser("answer"), label="noise1", note="замер разброса", limit=7
    )
    assert namespace.label == "noise1"
    assert namespace.note == "замер разброса"
    assert namespace.limit == 7


def test_unknown_override_is_loud() -> None:
    """Опечатка в подмене обязана ронять вызов, а не уходить в прогон молча.

    Молчаливо проигнорированная подмена — это прогон, который измерил не то,
    что просили, и не сказал об этом. Ровно та ошибка, которую мы уже ловили с
    константой слияния: значение выглядело изменённым и не было.
    """
    with pytest.raises(SystemExit):
        cli.run_namespace(cli.run_parser("answer"), labell="опечатка")


def test_noise_asks_for_the_parser_of_the_mode_it_runs() -> None:
    """`noise --mode search` не должен требовать аргументы команды `answer`.

    Режимы разные, и набор аргументов у них разный: у поискового прогона нет ни
    судьи, ни подмены промпта. Проверяем, что оба режима собираются.
    """
    for mode in ("search", "answer"):
        namespace = cli.run_namespace(cli.run_parser(mode), label="noise1", limit=0)
        assert namespace.label == "noise1"
