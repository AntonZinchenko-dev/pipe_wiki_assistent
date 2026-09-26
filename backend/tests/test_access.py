"""Права пользователя: где именно они проверяются и почему там.

Эта штука ломается не так, как обычный код. Неверно работающий поиск видно
по метрикам; неверно работающие права не видно вообще — система отвечает,
выглядит исправной и тихо показывает лишнее.

Пять мест, в которых это происходит, и все пять проверяются здесь:

1. фильтр стоит при показе, а не в поиске: модель всё равно прочитала текст
   и пересказала его в ответе;
2. фильтр стоит на одной половине поиска: закрытый фрагмент поднимается
   второй;
3. право проверяется только при выдаче списка инструментов, но не при
   вызове;
4. право спрашивают у сервиса, а не у человека: у сервиса токен есть
   всегда, и ответ «можно» получается для всех;
5. заголовку с именем доверяют без прокси впереди — тогда имя ставит сам
   клиент.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.access import (
    ANONYMOUS, RIGHT_AGENT, RIGHT_LIVE_READ, User,
    denied_projects, identify, parse_roles,
)
from app.config import Settings
from app.rag.chunk import Chunk
from app.rag.search import hybrid_search
from app.rag.store import Store


def user(roles: str) -> User:
    return User(name="кто-то", roles=parse_roles(roles))


# ------------------------------------------------------------------ роли


def test_roles_are_the_ones_from_the_regulation() -> None:
    """Своя система ролей рядом с регламентной — два ответа на один вопрос.

    В вики есть таблица ролей (РЛ-4.2.3, 4.2.3.2). Одна из них описывает
    наши инструменты дословно: `data-reader` — чтение боевых данных через
    API. Именно это делают live_pipe и live_fleet.
    """
    assert user("data-reader").may(RIGHT_LIVE_READ) is True
    assert user("data-admin").may(RIGHT_LIVE_READ) is True
    # А эти роли про репозитории и релизы, не про боевые данные.
    assert user("dev").may(RIGHT_LIVE_READ) is False
    assert user("ops").may(RIGHT_LIVE_READ) is False
    assert user("dev").may(RIGHT_AGENT) is True


def test_an_unknown_role_gives_nothing() -> None:
    """Опечатка в роли — это роль без прав, а не роль со всеми.

    Заголовок приходит из прокси, и опечатка в его настройке не должна
    оборачиваться ни повышением прав, ни аварией.
    """
    assert parse_roles("data-readr, датаридер") == frozenset()
    assert user("data-readr").may(RIGHT_LIVE_READ) is False


# --------------------------------------------------------------- личность


def test_header_identity_is_off_by_default() -> None:
    """Доверие к заголовку без прокси впереди — это дыра, которая выглядит защитой.

    Клиент поставит заголовок сам и объявит себя кем угодно. Та же ловушка,
    что с X-Forwarded-For в лимитере, и безопасное значение то же: по
    умолчанию не доверять.
    """
    settings = Settings()
    assert settings.auth_mode == "off"

    # В режиме off заголовок игнорируется целиком: подменить личность
    # через него нельзя, даже зная его имя.
    подделка = {"X-Remote-User": "директор", "X-Remote-Roles": "data-admin"}
    assert identify(подделка, settings).name != "директор"


def test_header_identity_works_when_switched_on() -> None:
    settings = Settings(auth_mode="header")
    who = identify({"X-Remote-User": "anton", "X-Remote-Roles": "dev,data-reader"}, settings)

    assert who.name == "anton"
    assert who.may(RIGHT_LIVE_READ) is True


def test_a_broken_proxy_gives_no_rights_rather_than_all() -> None:
    """Прокси включён, имени нет — это неработающий вход, а не аноним-админ.

    Пусть лучше не хватит прав, чем хватит чужих: первое заметят за минуту,
    второе может не заметить никто.
    """
    who = identify({}, Settings(auth_mode="header"))
    assert who == ANONYMOUS
    assert who.may(RIGHT_AGENT) is False
    assert who.may(RIGHT_LIVE_READ) is False


# ------------------------------------------------------- закрытые проекты


def test_restrictions_are_a_denylist_not_an_allowlist() -> None:
    """Новый проект в вики обязан быть виден, пока не решили иначе.

    Со списком РАЗРЕШЁННОГО его пришлось бы держать полным, и добавление
    документации молча ломало бы поиск по ней для всех. Такую поломку
    ищут в поиске, а она в правах.
    """
    settings = Settings(restricted_projects="SURVEYD:ops")
    assert denied_projects(user("dev"), settings) == {"SURVEYD"}
    # Проект, про который в правилах ничего нет, виден всем.
    assert "TELEMETRY" not in denied_projects(user("dev"), settings)
    # А без правил не закрыто ничего.
    assert denied_projects(user("dev"), Settings()) == set()


def test_seniority_is_taken_from_the_regulation_order() -> None:
    settings = Settings(restricted_projects="SURVEYD:ops")
    assert denied_projects(user("ops"), settings) == set()
    assert denied_projects(user("data-admin"), settings) == set()
    assert denied_projects(user("dev-extended"), settings) == {"SURVEYD"}


# ----------------------------------------------- отсев в САМОМ поиске


@pytest.fixture()
def store(tmp_path) -> Store:
    store = Store(tmp_path / "access.sqlite3")
    for doc_id, project in [("OPEN-1", "TELEMETRY"), ("SEC-1", "SURVEYD")]:
        store.upsert_document({
            "doc_id": doc_id, "title": doc_id, "project": project, "owner": "x",
            "updated": "2026-01-01", "version": "1", "status": "действующий",
            "source_path": "", "pdf_path": "", "page_count": 1,
        })
    rng = np.random.default_rng(7)
    vector = rng.normal(size=8).astype(np.float32)
    vector /= np.linalg.norm(vector)
    for doc_id, body in [
        ("OPEN-1", "кэш телеметрии отключается параметром"),
        ("SEC-1", "кэш телеметрии в закрытом контуре отключается иначе"),
    ]:
        store.add_chunk(
            Chunk(doc_id=doc_id, ordinal=0, heading_path="раздел",
                  text=body, body=body, page_from=1, page_to=1),
            vector,
        )
    yield store
    store.close()


def test_a_closed_fragment_never_reaches_the_context(store: Store) -> None:
    """Самое главное свойство всей затеи.

    Спрятать закрытый фрагмент в интерфейсе НЕДОСТАТОЧНО: модель прочитает
    текст и перескажет его своими словами. Утечка произойдёт через ответ,
    при пустом списке источников на экране, и выглядеть будет как обычная
    работа. Поэтому фильтр стоит в поиске, а не в показе.
    """
    rng = np.random.default_rng(7)
    query = rng.normal(size=8).astype(np.float32)
    query /= np.linalg.norm(query)

    everything = asyncio.run(hybrid_search(
        store=store, query="кэш", query_vector=query,
        top_k=10, floor=0.0, rrf_k=10, keyword_weight=0.45,
    ))
    limited = asyncio.run(hybrid_search(
        store=store, query="кэш", query_vector=query,
        top_k=10, floor=0.0, rrf_k=10, keyword_weight=0.45,
        denied_projects={"SURVEYD"},
    ))

    assert {hit.chunk.doc_id for hit in everything.hits} == {"OPEN-1", "SEC-1"}
    assert {hit.chunk.doc_id for hit in limited.hits} == {"OPEN-1"}


def test_both_halves_of_the_search_are_filtered(store: Store) -> None:
    """Отсечь одну половину мало: закрытый кусок поднимется второй.

    Гибридный поиск на то и гибридный — у него два независимых входа, и
    забыть один из них проще всего именно потому, что второй работает.
    """
    rng = np.random.default_rng(7)
    query = rng.normal(size=8).astype(np.float32)
    query /= np.linalg.norm(query)

    by_vector = store.vector_search(query, 10, denied_projects={"SURVEYD"})
    by_keyword = store.keyword_search("кэш", 10, denied_projects={"SURVEYD"})
    closed = {
        chunk_id
        for chunk_id, project in store._chunk_projects().items()
        if project == "SURVEYD"
    }

    assert closed, "в фикстуре обязан быть закрытый кусок, иначе тест ничего не проверяет"
    assert not ({chunk_id for chunk_id, _ in by_vector} & closed)
    assert not ({chunk_id for chunk_id, _ in by_keyword} & closed)


# -------------------------------------------- право проверяется при ВЫЗОВЕ


def test_the_tool_checks_the_right_itself_not_only_the_listing() -> None:
    """Скрытый инструмент — не то же самое, что запрещённый.

    Модель может назвать его по памяти о прошлых диалогах или потому, что
    имя мелькнуло в документе. Проверка при показе списка — удобство,
    проверка при вызове — собственно ограничение. Оставить только первую
    значит защищаться от вежливой модели.
    """
    from app.rag.tools import Toolbox

    class FakeLive:
        configured = True

    box = Toolbox(store=None, settings=Settings(), embedder=None, live=FakeLive())

    listed = {spec["function"]["name"] for spec in box.specs(live_allowed=False)}
    assert "live_fleet" not in listed

    outcome = asyncio.run(box.run("live_fleet", {"min_damage": 0.8}, live_allowed=False))
    assert outcome.ok is False
    # В отказе назван регламент: человеку надо знать, какую роль просить.
    assert "data-reader" in outcome.text


def test_the_right_is_asked_of_the_person_not_of_the_service() -> None:
    """У сервиса токен есть всегда — спрашивать его бессмысленно.

    Если проверять «настроен ли доступ к API», ответ будет «да» для всех
    пользователей сразу, и проверка выродится в украшение. Смысл имеет
    только вопрос «можно ли ЭТОМУ человеку».
    """
    import inspect

    from app import pipeline as pipeline_module

    source = inspect.getsource(pipeline_module.Pipeline._run)
    assert "user.may(RIGHT_LIVE_READ)" in source
    assert "user.may(RIGHT_AGENT)" in source
