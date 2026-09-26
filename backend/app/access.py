"""Права ПОЛЬЗОВАТЕЛЯ, а не права сервиса.

Раньше система работала от своего имени: один сервисный токен, один набор
прав, одинаковый ответ всем. Пока в вики лежит только общедоступное, это
незаметно. Как только появляется документ, который видно не всем, такая
схема превращается в канал утечки — причём молчаливый, потому что никакой
ошибки не происходит: система честно нашла документ и честно его пересказала.

РОЛИ ВЗЯТЫ ИЗ РЕГЛАМЕНТА, А НЕ ПРИДУМАНЫ

В вики есть таблица ролей (РЛ-4.2.3, пункт 4.2.3.2): `dev`, `dev-extended`,
`ops`, `data-reader`, `data-admin`. Заводить рядом свою систему ролей значило
бы, что на вопрос «кому что можно» в компании два ответа. Одна из ролей
описывает наши инструменты дословно: `data-reader` — «чтение боевых данных
через API, без выгрузок». Ровно это делают `live_pipe` и `live_fleet`.

ГДЕ ПРОВЕРЯЕТСЯ ДОСТУП — ГЛАВНОЕ РЕШЕНИЕ

В ПОИСКЕ, а не при показе. Фрагмент, который человеку не положен, не должен
попасть в контекст вообще. Спрятать его в интерфейсе недостаточно: модель
всё равно прочитает текст и перескажет его в ответе своими словами. Утечка
произойдёт через ответ, при пустом списке источников на экране, — и выглядеть
это будет как нормальная работа.

Аналогия: закрыть посетителю доступ в архив — не то же самое, что выдать ему
пересказ секретной папки без указания источника. Второе хуже: следов нет.

ЛИЧНОСТЬ ПРИХОДИТ ИЗ ЗАГОЛОВКА, И ЭТО ОПАСНО РОВНО В ОДНОМ СЛУЧАЕ

Паролей мы не проверяем и хранить их не собираемся: внутренняя система стоит
за обратным прокси, который уже выполнил вход, и передаёт нам имя и роли
заголовком. Это обычная схема, и у неё одна ловушка: заголовок, которому
доверяют без прокси впереди, ставит сам клиент. Тогда любой желающий
объявляет себя кем угодно — и защита выглядит работающей.

Поэтому режим `header` включается ЯВНО и по умолчанию выключен. Ровно та же
логика, что у доверия к `X-Forwarded-For` в лимитере: обе ошибки тихие,
поэтому безопасное значение — то, которое не создаёт дыру.
"""

from __future__ import annotations

from dataclasses import dataclass

# Роли из РЛ-4.2.3, пункт 4.2.3.2. Имена — как в регламенте.
ROLE_DEV = "dev"
ROLE_DEV_EXTENDED = "dev-extended"
ROLE_OPS = "ops"
ROLE_DATA_READER = "data-reader"
ROLE_DATA_ADMIN = "data-admin"

KNOWN_ROLES = frozenset({
    ROLE_DEV, ROLE_DEV_EXTENDED, ROLE_OPS, ROLE_DATA_READER, ROLE_DATA_ADMIN,
})

# Права — то, что система умеет делать. Их немного нарочно: список прав
# должен расти медленнее, чем список возможностей, иначе он перестаёт быть
# инструментом ограничения.
#
# `live:read` — чтение боевых данных через API. Это дословно роль
# `data-reader` из регламента, и именно это делают инструменты агента.
RIGHT_LIVE_READ = "live:read"
# `agent:run` — право на агентский режим. Отдельно от чтения живых данных:
# агент тратит вызовы модели и деньги, и разрешать его стоит осознанно.
RIGHT_AGENT = "agent:run"

RIGHTS_BY_ROLE: dict[str, frozenset[str]] = {
    ROLE_DEV: frozenset({RIGHT_AGENT}),
    ROLE_DEV_EXTENDED: frozenset({RIGHT_AGENT}),
    ROLE_OPS: frozenset({RIGHT_AGENT}),
    # Чтение боевых данных — ровно то, что даёт эта роль по регламенту.
    ROLE_DATA_READER: frozenset({RIGHT_AGENT, RIGHT_LIVE_READ}),
    ROLE_DATA_ADMIN: frozenset({RIGHT_AGENT, RIGHT_LIVE_READ}),
}

# Старшинство ролей: кто кого включает. Нужно для ограничений на проекты,
# где указывается МИНИМАЛЬНАЯ роль.
ROLE_ORDER = [ROLE_DEV, ROLE_DEV_EXTENDED, ROLE_OPS, ROLE_DATA_READER, ROLE_DATA_ADMIN]


@dataclass(frozen=True, slots=True)
class User:
    """Кто спрашивает. Не «сервис», а человек с ролями."""

    name: str
    roles: frozenset[str]

    def may(self, right: str) -> bool:
        return any(right in RIGHTS_BY_ROLE.get(role, frozenset()) for role in self.roles)

    def rank(self) -> int:
        """Старшинство: место самой старшей роли в порядке регламента."""
        places = [ROLE_ORDER.index(role) for role in self.roles if role in ROLE_ORDER]
        return max(places) if places else -1


ANONYMOUS = User(name="аноним", roles=frozenset())


def parse_roles(raw: str) -> frozenset[str]:
    """Роли из строки. Неизвестные ОТБРАСЫВАЮТСЯ, а не пропускаются.

    Опечатка в роли не должна давать прав: неизвестная роль — это роль без
    прав, а не роль со всеми. Отбрасываем молча по той же причине, по
    которой не падаем: заголовок приходит из прокси, и уронить запрос
    из-за чужой опечатки в настройке прокси значило бы устроить аварию
    вместо отказа в доступе.
    """
    names = {part.strip().lower() for part in raw.replace(";", ",").split(",")}
    return frozenset(name for name in names if name in KNOWN_ROLES)


def identify(headers, settings) -> User:
    """Кто прислал этот запрос.

    Режим `off` — один и тот же пользователь с ролями из настроек. Это
    рабочий режим для разработки и для установки, где вход ещё не
    настроен: система при этом ведёт себя предсказуемо, а не «как будто
    прав нет».

    Режим `header` — имя и роли из заголовков обратного прокси. Включать
    его без прокси впереди нельзя: тогда заголовок ставит сам клиент.
    """
    if settings.auth_mode != "header":
        return User(
            name=settings.auth_default_user or "локальный пользователь",
            roles=parse_roles(settings.auth_default_roles),
        )

    name = (headers.get(settings.auth_user_header) or "").strip()
    if not name:
        # Прокси включён, а имени нет — это не «аноним со всеми правами»,
        # это неработающий вход. Отдаём пользователя вообще без ролей:
        # пусть лучше не хватит прав, чем хватит чужих.
        return ANONYMOUS
    return User(name=name[:120], roles=parse_roles(headers.get(settings.auth_roles_header) or ""))


def parse_restricted(raw: str) -> dict[str, str]:
    """Настройка вида `SURVEYD:data-reader,FATIGUE-API:ops`.

    Проект — минимальная роль. Пусто означает «ограничений нет», и это
    сознательное значение по умолчанию: включение ограничений — решение
    владельца установки, а не побочный эффект обновления.
    """
    rules: dict[str, str] = {}
    for part in raw.split(","):
        if ":" not in part:
            continue
        project, role = part.split(":", 1)
        project, role = project.strip(), role.strip().lower()
        if project and role in KNOWN_ROLES:
            rules[project] = role
    return rules


def denied_projects(user: User, settings) -> set[str]:
    """Проекты, которые этому человеку видеть нельзя.

    Возвращаем именно ЗАПРЕЩЁННЫЕ, а не разрешённые. Разница важна: список
    разрешённых пришлось бы держать полным, и новый проект по умолчанию
    оказался бы невидимым для всех — то есть обновление вики молча ломало
    бы поиск. Список запрещённых по умолчанию пуст, и новый проект виден,
    пока кто-то явно не решит иначе.
    """
    rules = parse_restricted(settings.restricted_projects)
    if not rules:
        return set()
    place = user.rank()
    return {
        project
        for project, role in rules.items()
        if place < ROLE_ORDER.index(role)
    }
