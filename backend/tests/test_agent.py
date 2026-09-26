"""Агентский шаг: границы цикла, права инструментов, честность отчёта.

Проверять здесь надо не «вызвался ли инструмент» — это видно и на глаз. В
цикле, которым управляет модель, ломается другое, и ломается тихо:

1. цикл не кончается, потому что граница описана словами в промпте;
2. модель просит одно и то же по кругу и тратит шаги впустую;
3. неудачный необязательный шаг роняет весь ответ;
4. фрагмент, который модель попросила сама, не доезжает до контекста —
   инструмент «работает», а толку ноль, и понять это неоткуда;
5. в отчёте не отличить «агент не участвовал» от «участвовал и решил, что
   хватает».

Все пять — про поведение на границе, и все пять дешевле поймать здесь, чем
на живой модели.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.agent import Agent, _ordered
from app.config import Settings
from app.providers.base import ChatChunk, FinishReason, ProviderError, ToolCall, Usage
from app.rag.search import Hit, SearchResult
from app.rag.tools import ToolOutcome, tool_specs


# ------------------------------------------------------------------ подделки


@dataclass
class FakeChunk:
    chunk_id: int
    doc_id: str = "DOC-1"
    heading_path: str = "Раздел"
    body: str = "текст фрагмента"
    doc_title: str = "Документ"
    doc_version: str = "1.0"
    doc_status: str = "действующий"
    doc_updated: str = "2026-01-01"
    page_from: int = 1
    page_to: int = 1
    text: str = "текст"
    doc_project: str = "PRJ"


def hit(chunk_id: int, *, score: float = 0.5, origin: str = "") -> Hit:
    return Hit(
        chunk=FakeChunk(chunk_id),
        vector_score=None if origin else 0.6,
        keyword_score=None,
        vector_rank=None if origin else 1,
        keyword_rank=None,
        fused_score=score,
        origin=origin,
    )


class FakeProviders:
    """Отдаёт заранее заданную последовательность решений модели."""

    # Цепочка нужна агенту, чтобы запомнить, какая пара «провайдер +
    # модель» не умеет вызывать функции. Подделка обязана иметь тот же
    # интерфейс, иначе тест проверяет совместимость с выдумкой.
    chain = ["ollama"]

    def __init__(self, calls: list[ToolCall | None], *, fails: bool = False) -> None:
        self._calls = list(calls)
        self._fails = fails
        self.asked = 0

    async def stream_chat(self, request, *, prefer="", on_provider=None):
        self.asked += 1
        if self._fails:
            raise ProviderError("модель занята", retryable=True, provider="ollama")
        call = self._calls.pop(0) if self._calls else None
        if call is not None:
            yield ChatChunk(tool_calls=[call])
        else:
            # Явное решение словом, а не молчание: модель, которая умеет
            # инструменты и решила, что хватает, отвечает именно так.
            yield ChatChunk(text="ХВАТИТ")
        yield ChatChunk(done=True, finish_reason=FinishReason.STOP, usage=Usage())


class FakeToolbox:
    def __init__(self, outcomes: dict[str, ToolOutcome]) -> None:
        self._outcomes = outcomes
        self.ran: list[tuple[str, dict]] = []

    def specs(self, *, live_allowed: bool = True) -> list[dict]:
        # Набор инструментов спрашивают у ящика, а не у модуля: он зависит
        # и от того, что настроено на машине, и от прав пользователя.
        return tool_specs()

    async def run(self, name: str, arguments: dict, *, live_allowed: bool = True) -> ToolOutcome:
        self.ran.append((name, arguments))
        return self._outcomes.get(name, ToolOutcome(text="нет такого", ok=False))


class FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict = {}


class FakeTrace:
    def __init__(self) -> None:
        self.spans: list[str] = []

    def span(self, name: str, **attributes):
        self.spans.append(name)

        class Ctx:
            def __enter__(self_inner):
                return FakeSpan()

            def __exit__(self_inner, *exc):
                return False

        return Ctx()


def build(calls, outcomes=None, *, max_steps=2, fails=False):
    settings = Settings(agent_enabled=True, agent_max_steps=max_steps)
    toolbox = FakeToolbox(outcomes or {})
    providers = FakeProviders(calls, fails=fails)
    return Agent(settings=settings, toolbox=toolbox, providers=providers), toolbox, providers


def empty_result(hits=()) -> SearchResult:
    return SearchResult(
        hits=list(hits), best_vector_score=0.6, passed_floor=True, floor=0.45,
        dropped_duplicates=0,
    )


def run(agent, result):
    return asyncio.run(agent.gather("вопрос", result, trace=FakeTrace()))


# -------------------------------------------------------------------- тесты


def test_model_saying_nothing_means_the_chain() -> None:
    """Модель не попросила инструмент — работаем как раньше.

    Это же и есть поведение старой модели, которая инструменты не умеет:
    вызовов просто не будет. Агент — надстройка над работающей системой, и
    отсутствие надстройки обязано давать в точности прежний результат.
    """
    agent, toolbox, _ = build([None])
    outcome = run(agent, empty_result([hit(1)]))

    assert toolbox.ran == []
    assert outcome.used_tools is False
    assert [h.chunk.chunk_id for h in outcome.hits] == [1]


def test_the_loop_stops_at_the_step_limit() -> None:
    """Граница держится счётчиком, а не просьбой в промпте.

    Модель здесь просит инструмент бесконечно. Если бы границу держал
    текст системного промпта, цикл крутился бы, пока не кончатся деньги
    или терпение — и заметили бы это ночью, на редком вопросе.
    """
    forever = [ToolCall("search_wiki", {"query": f"попытка {n}"}) for n in range(50)]
    agent, toolbox, providers = build(
        forever, {"search_wiki": ToolOutcome(text="нашлось", hits=[hit(9)])}, max_steps=3
    )
    outcome = run(agent, empty_result([hit(1)]))

    assert len(toolbox.ran) == 3
    assert providers.asked == 3
    assert outcome.hit_limit is True


def test_the_same_call_twice_ends_the_loop() -> None:
    """Повтор того же вызова — тупик, а не повод потратить ещё шаг.

    Модели застревают на этом регулярно: результат не понравился, и они
    просят ровно то же самое. Ответ будет тот же, а шаг и токены уйдут.
    """
    same = ToolCall("search_wiki", {"query": "одно и то же"})
    agent, toolbox, _ = build(
        [same, same, same], {"search_wiki": ToolOutcome(text="нашлось", hits=[hit(9)])}
    )
    outcome = run(agent, empty_result([hit(1)]))

    assert len(toolbox.ran) == 1
    assert outcome.steps[-1].ok is False
    assert "повтор" in outcome.steps[-1].result


def test_a_broken_agent_step_does_not_break_the_answer() -> None:
    """Необязательный шаг не имеет права уронить обязательный.

    Провайдер отказал на решении «искать ли ещё». Это неприятно, но
    фрагменты с первого поиска уже есть, и ответ по ним лучше, чем ошибка
    вместо ответа.
    """
    agent, toolbox, _ = build([ToolCall("search_wiki", {"query": "x"})], fails=True)
    outcome = run(agent, empty_result([hit(1), hit(2)]))

    assert toolbox.ran == []
    assert [h.chunk.chunk_id for h in outcome.hits] == [1, 2]


def test_already_known_fragments_are_not_added_twice() -> None:
    """Повторная находка не должна занимать место в контексте дважды.

    Второй поиск другими словами почти всегда возвращает часть того же
    самого. Без отсева бюджет контекста уходил бы на копии, вытесняя
    настоящие новые фрагменты.
    """
    agent, _, _ = build(
        [ToolCall("search_wiki", {"query": "иначе"})],
        {"search_wiki": ToolOutcome(text="нашлось", hits=[hit(1), hit(7)])},
    )
    outcome = run(agent, empty_result([hit(1), hit(2)]))

    assert sorted(h.chunk.chunk_id for h in outcome.hits) == [1, 2, 7]
    assert outcome.steps[0].added == 1


def test_requested_fragments_survive_the_context_budget() -> None:
    """Самая тихая поломка этой затеи, если её не предусмотреть.

    У фрагмента, который модель попросила сама, нулевой балл слияния: он
    не участвовал в поиске. Сортировка по баллу утопила бы его в самый
    низ, где бюджет контекста просто отрежет — инструмент отработал бы,
    фрагмент бы добавился, а в ответ не попал. И объяснить это было бы
    нечем: в отчёте всё «сработало».

    Правило «лучший фрагмент поиска идёт первым» при этом сохраняется:
    модели хуже используют середину длинного контекста.
    """
    ordered = _ordered([
        hit(1, score=0.9),
        hit(2, score=0.8),
        hit(3, score=0.7),
        hit(99, score=0.0, origin="запрошен моделью"),
    ])
    assert [h.chunk.chunk_id for h in ordered] == [1, 99, 2, 3]


def test_requested_fragment_says_how_it_got_here() -> None:
    """Отчёт «как получен ответ» не должен врать про происхождение.

    До появления инструментов способ вычислялся по рангам, и фрагмент без
    рангов объявлялся найденным «ключевым». То есть отчёт называл бы
    неверный способ — а по этому полю разбирают, почему нашлось не то.
    """
    assert hit(99, origin="запрошен моделью").found_by == "запрошен моделью"
    assert hit(1).found_by == "вектор"


def test_tools_are_described_in_the_neutral_shape() -> None:
    """Форма одна на всех, перевод — в адаптере провайдера.

    Если бы перевод жил рядом с инструментами, каждый новый провайдер
    означал бы правку агента. Это ровно та же граница, по которой уже
    разведены схема ответа и история вызовов.
    """
    specs = tool_specs()
    assert {spec["function"]["name"] for spec in specs} == {"search_wiki", "read_section"}
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["parameters"]["type"] == "object"


def test_no_tool_changes_anything() -> None:
    """Список инструментов — это список прав, и он только на чтение.

    Документ в вики может положить любой сотрудник, а модель управляется
    текстом. Инструмент с побочным эффектом превращает «положил документ»
    в «выполнил команду на сервере». Тест сторожит границу: новый
    инструмент с записью не пройдёт молча.
    """
    names = {spec["function"]["name"] for spec in tool_specs()}
    forbidden = {"write", "delete", "update", "send", "fetch", "execute", "run"}
    for name in names:
        assert not any(word in name for word in forbidden), name


# ------------------------------------------- разбор вызова у локальной модели


def test_tool_call_arguments_are_normalised_to_an_object() -> None:
    """Часть моделей отдаёт аргументы строкой с JSON внутри.

    Разница видна только на конкретной модели, и наверх она уходить не
    должна: агент разбирает `arguments` как словарь. Приведение делает
    адаптер — ровно там же, где приводится форма схемы ответа.
    """
    from app.providers.ollama import _parse_tool_calls

    as_object = _parse_tool_calls(
        [{"function": {"name": "search_wiki", "arguments": {"query": "кэш"}}}]
    )
    as_string = _parse_tool_calls(
        [{"function": {"name": "search_wiki", "arguments": '{"query": "кэш"}'}}]
    )
    assert as_object == as_string
    assert as_object[0].arguments == {"query": "кэш"}


def test_a_broken_tool_call_is_skipped_not_fatal() -> None:
    """Одна аномалия не должна стоить всего ответа.

    Тот же принцип, что и при разборе строки потока: битый элемент
    пропускаем, поток живёт. Исключение здесь означало бы, что галлюцинация
    в имени функции роняет пользователю весь запрос.
    """
    from app.providers.ollama import _parse_tool_calls

    calls = _parse_tool_calls([
        "мусор",
        {"function": {"name": "", "arguments": {}}},
        {"function": {"name": "search_wiki", "arguments": "не json"}},
        {"function": {"name": "read_section", "arguments": {"doc_id": "D", "heading": "4.2"}}},
    ])
    assert [call.name for call in calls] == ["search_wiki", "read_section"]
    # У сломанных аргументов пустой объект, а не падение: инструмент сам
    # скажет «пустой запрос», и модель сможет исправиться.
    assert calls[0].arguments == {}


# ------------------------ модель, которая не умеет вызывать инструменты


class TextOnlyProviders:
    """Модель, которая на любой запрос отвечает текстом.

    Так вела себя облачная Lightning: пять запросов подряд, ни одного
    вызова функции, включая прямую просьбу пользователя сходить в сервис.
    Зато бланк ответа по схеме на ней работает — им отдаётся весь основной
    ответ системы.
    """

    chain = ["gigachat"]

    def __init__(self, schema_reply: str) -> None:
        self._schema_reply = schema_reply
        self.requests: list[object] = []

    async def stream_chat(self, request, *, prefer="", on_provider=None):
        self.requests.append(request)
        # Вызовов не отдаём никогда. На запрос со схемой отвечаем бланком,
        # на запрос с инструментами — обычным текстом, как и настоящая.
        text = self._schema_reply if request.json_schema else "Самая изношенная труба — это та…"
        yield ChatChunk(text=text)
        yield ChatChunk(done=True, finish_reason=FinishReason.STOP, usage=Usage())


def test_a_model_without_tool_calls_still_decides() -> None:
    """Запасной путь — не парашют, а рабочий механизм для таких моделей.

    Без него агент на облачной модели молчал: ноль шагов, ноль вызовов, и
    в отчёте «посмотрел и решил, что достаточно» — при том, что решения не
    было вовсе.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="auto")
    toolbox = FakeToolbox({"search_wiki": ToolOutcome(text="нашлось", hits=[hit(9)])})
    providers = TextOnlyProviders(
        '{"why": "вопрос про текущее состояние", "action": "search_wiki", "query": "износ труб"}'
    )
    agent = Agent(settings=settings, toolbox=toolbox, providers=providers)

    outcome = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert toolbox.ran == [("search_wiki", {"query": "износ труб"})]
    assert outcome.mechanism == "схема"


def test_the_report_tells_refusal_apart_from_inability() -> None:
    """Главная улика, которой раньше не было.

    «Агент ничего не сделал» выглядело одинаково и когда модель решила,
    что хватает, и когда она не умеет вызывать инструменты. Чинить
    приходилось наугад. Теперь видно, чем решали и что модель сказала
    вместо вызова.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="auto")
    providers = TextOnlyProviders('{"why": "и так ясно", "action": "ХВАТИТ"}')
    agent = Agent(settings=settings, toolbox=FakeToolbox({}), providers=providers)

    outcome = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert outcome.used_tools is False
    # Не просто «схема», а «схема (хватит)»: модель приняла решение, и это
    # ДРУГОЙ диагноз, чем оборванный бланк или пустое поле действия. В
    # отчёте все три выглядели как «ничего не звал», и один прогон я на этом
    # уже потратил, гадая, какой из них.
    assert outcome.mechanism == "схема (хватит)"
    # И видно, что именно ответила модель, а не просто «ничего не сделал».
    assert outcome.declined_with


def test_an_explicit_enough_is_not_treated_as_inability() -> None:
    """Модель, умеющая инструменты и сказавшая «ХВАТИТ», решила, а не сломалась.

    Переходить на запасной путь здесь незачем: это стоило бы лишнего
    вызова модели на каждый вопрос, где всего хватает, — то есть на
    большинстве.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="auto")
    agent, toolbox, providers = build([None], max_steps=1)
    agent._s = settings

    outcome = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert outcome.mechanism == "инструменты"
    assert providers.asked == 1, "лишнего вызова на запасной путь быть не должно"


def test_an_invented_action_is_refused() -> None:
    """По имени действия мы вызываем код, поэтому имя проверяется.

    Модели придумывают названия инструментов регулярно — и в бланке это
    делать проще, чем при родном вызове: там имя ограничено грамматикой,
    здесь это просто строка.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="schema")
    toolbox = FakeToolbox({})
    providers = TextOnlyProviders('{"why": "хочу", "action": "delete_everything"}')
    agent = Agent(settings=settings, toolbox=toolbox, providers=providers)

    outcome = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert toolbox.ran == []
    assert outcome.used_tools is False


class LiveToolbox:
    """Ящик, в котором настроен сервис живых данных.

    `tool_specs()` сам по себе живых инструментов не отдаёт: они появляются,
    только когда сервис настроен И человеку можно читать боевые данные.
    Для проверок про `live_*` это условие надо воспроизвести, иначе имя
    действия не пройдёт сверку по списку доступных — и тест покажет
    «модель не позвала» там, где её просто не пускали.
    """

    def __init__(self, inner: FakeToolbox) -> None:
        self._inner = inner

    @property
    def ran(self) -> list[tuple[str, dict]]:
        return self._inner.ran

    def specs(self, *, live_allowed: bool = True) -> list[dict]:
        from app.rag.tools import LIVE_SPECS

        return tool_specs() + (list(LIVE_SPECS) if live_allowed else [])

    async def run(self, name: str, arguments: dict, *, live_allowed: bool = True) -> ToolOutcome:
        return await self._inner.run(name, arguments, live_allowed=live_allowed)


def test_a_cut_blank_is_not_a_refusal() -> None:
    """Пять живых вопросов прогона стояли в отчёте как «ничего не звал».

    В `why` модель определяла вопрос ПРАВИЛЬНО — «это про конкретные трубы
    и их состояние» — и вызова всё равно не было. Причина оказалась не в
    промпте и не в списке инструментов: бланк обрывался посреди `why`,
    поля `action` в нём не появлялось, JSON не разбирался. Снаружи это
    выглядело как решение модели, и чинить шли не туда.

    Отсюда и проверка: оборванный бланк обязан называться обрывом.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="schema")
    toolbox = FakeToolbox({})
    # Ровно то, что приходило из прогона: начало бланка без конца.
    providers = TextOnlyProviders(
        '{\n    "why": "Вопрос о том, какие трубы под списание — это про конкретные трубы'
    )
    agent = Agent(settings=settings, toolbox=toolbox, providers=providers)

    outcome = asyncio.run(agent.gather("какие трубы под списание", empty_result([hit(1)]), trace=FakeTrace()))

    assert toolbox.ran == []
    assert outcome.mechanism == "схема (обрыв)", "обрыв нельзя показывать как отказ"
    assert outcome.declined_with


def test_a_parsed_blank_without_an_action_says_so() -> None:
    """Четвёртый случай, и найден он ценой прогона.

    Я решил, что пять живых вопросов остались без вызова из-за обрыва
    бланка, поднял лимит токенов — и обрыв не подтвердился: бланки
    разобрались, а вызова всё равно не было. Значит модель дописывает
    бланк и оставляет поле действия пустым, а отчёт называл это тем же
    словом «схема», что и осознанное «хватит».
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="schema")
    toolbox = FakeToolbox({})
    providers = TextOnlyProviders('{"why": "это про конкретные трубы", "action": ""}')
    agent = Agent(settings=settings, toolbox=toolbox, providers=providers)

    outcome = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert toolbox.ran == []
    assert outcome.mechanism == "схема (без действия)"


def test_an_invented_action_is_named_in_the_report() -> None:
    """Выдуманное имя инструмента — отдельный диагноз, и его надо видеть.

    «Без действия» чинится схемой и строгим режимом, выдуманное имя —
    списком инструментов и промптом. Одно слово на оба случая снова
    отправило бы чинить не туда.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="schema")
    providers = TextOnlyProviders('{"why": "хочу", "action": "delete_everything"}')
    agent = Agent(settings=settings, toolbox=FakeToolbox({}), providers=providers)

    outcome = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert "выдумал действие" in outcome.mechanism
    assert "delete_everything" in outcome.mechanism


def test_the_blank_is_not_truncated_before_it_is_parsed() -> None:
    """Обрезка ответа рубила то, что мы потом разбираем.

    В журнале обрезать длинный текст правильно. Но в схеме текст — это сам
    бланк решения: 400 символов хватало на многословное `why` и не хватало
    на `action` за ним. Вызов инструмента терялся внутри нашего же кода.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="schema")
    toolbox = FakeToolbox({"live_fleet": ToolOutcome(text="таблица", hits=[hit(9)])})
    verbose = "вопрос про конкретные трубы и их числа, " * 12  # ~470 символов
    providers = TextOnlyProviders(
        '{"why": "' + verbose + '", "action": "live_fleet", "top_n": 5}'
    )
    agent = Agent(settings=settings, toolbox=LiveToolbox(toolbox), providers=providers)

    outcome = asyncio.run(
        agent.gather(
            "дай топ 5 труб", empty_result([hit(1)]), trace=FakeTrace(), live_allowed=True
        )
    )

    assert toolbox.ran == [("live_fleet", {"top_n": 5})]
    assert outcome.mechanism == "схема"


def test_the_blank_asks_for_a_short_reason() -> None:
    """Границу длины держит схема, а не надежда.

    `why` идёт первым — так измерено, менять порядок нельзя. Значит длину
    обоснования надо ограничивать там же, где оно объявлено, иначе модель
    тратит на него весь бюджет ответа.
    """
    from app.agent import decision_schema

    schema = decision_schema(LiveToolbox(FakeToolbox({})).specs())

    assert schema["properties"]["why"]["maxLength"] <= 200
    assert schema["properties"]["why"].get("description")
    assert schema["properties"]["action"].get("description")
    # Порядок полей — часть измеренного решения, проверяем и его.
    assert list(schema["properties"])[:2] == ["why", "action"]


def test_enough_is_the_last_choice_not_the_first() -> None:
    """Порядок значений в перечислении — тот же дефект, что ярлык впереди.

    Провайдер строит из перечисления грамматику, и первое значение — самый
    дешёвый путь. Со `ХВАТИТ` впереди замер дал четыре живых вопроса без
    вызова, и в трёх из них собственное поле `why` говорило обратное: «это
    про конкретные трубы и их состояние» — и тут же решение не ходить за
    ними.

    Наклон списка должен смотреть в сторону дешёвой ошибки. Лишний запрос к
    сервису стоит запроса; неслучившийся вызов стоит ответа не на тот
    вопрос.
    """
    from app.agent import decision_schema

    choices = decision_schema(LiveToolbox(FakeToolbox({})).specs())["properties"]["action"]["enum"]

    assert choices[-1] == "ХВАТИТ", "отказ обязан быть последним в списке"
    assert "live_fleet" in choices[:-1]


# --------------------------- разбор двух логов: что чинилось и чем проверено


class ChattyThenNative:
    """Модель, которая один раз ответила текстом, а потом вызывает как надо.

    Так выглядит НЕ неумение, а осечка: формулировка вопроса не легла, и
    модель вместо вызова порассуждала. Отличить осечку от неумения по
    одному ответу нельзя, и раньше мы и не пытались — первый же текст
    вместо вызова навсегда переводил пару «провайдер + модель» на схему.
    """

    chain = ["ollama"]

    def __init__(self) -> None:
        self.asked = 0
        self.native_asked = 0

    async def stream_chat(self, request, *, prefer="", on_provider=None):
        self.asked += 1
        if request.tools:
            self.native_asked += 1
            if self.native_asked == 1:
                yield ChatChunk(text="Здесь стоит обратиться к сервису живых данных.")
            else:
                yield ChatChunk(
                    tool_calls=[ToolCall(name="search_wiki", arguments={"query": "износ"})]
                )
        else:
            yield ChatChunk(text='{"why": "поищем", "action": "search_wiki", "query": "износ"}')
        yield ChatChunk(done=True, finish_reason=FinishReason.STOP, usage=Usage())


def test_one_chatty_answer_does_not_disable_tools_forever() -> None:
    """Одно наблюдение — не свойство модели.

    Раньше первый же текст вместо вызова записывал пару в «не умеет» на
    весь процесс. Дальше весь сервер работал схемой, хотя родной механизм
    был исправен, и в отчёте стояло «способ: схема» — то есть улика
    указывала на несуществующую поломку.
    """
    # Один шаг на вопрос: иначе в `mechanism` останется способ ПОСЛЕДНЕГО
    # решения, и первый вопрос отчитается тем же «инструменты», ради
    # которого проверяется второй.
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="auto")
    toolbox = FakeToolbox({"search_wiki": ToolOutcome(text="нашлось", hits=[hit(9)])})
    providers = ChattyThenNative()
    agent = Agent(settings=settings, toolbox=toolbox, providers=providers)

    first = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))
    assert first.mechanism == "схема", "после осечки на этом шаге переходим на схему"

    # Второй вопрос той же моделью: родной механизм обязан быть предложен
    # снова, а не заменён схемой навсегда.
    second = asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))
    assert second.mechanism == "инструменты"


class AlwaysChatty(ChattyThenNative):
    """Модель, которая текстом отвечает всегда: настоящее неумение."""

    async def stream_chat(self, request, *, prefer="", on_provider=None):
        self.asked += 1
        if request.tools:
            self.native_asked += 1
            yield ChatChunk(text="Думаю, надо посмотреть в сервисе.")
        else:
            yield ChatChunk(text='{"why": "поищем", "action": "ХВАТИТ"}')
        yield ChatChunk(done=True, finish_reason=FinishReason.STOP, usage=Usage())


def test_a_model_that_never_calls_is_asked_natively_only_twice() -> None:
    """Терпение не бесконечное: два промаха — и пара переводится на схему.

    Иначе за возврат родного механизма мы платили бы лишним вызовом на
    КАЖДОМ шаге для модели, которая вызывать не умеет вовсе.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="auto")
    providers = AlwaysChatty()
    agent = Agent(settings=settings, toolbox=FakeToolbox({}), providers=providers)

    for _ in range(4):
        asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert providers.native_asked == Agent.NATIVE_MISS_LIMIT


def test_arguments_are_filtered_by_the_schema_of_the_chosen_tool() -> None:
    """Бланк решения плоский, инструменты — нет.

    В логе это выглядело как «дочитал раздел «IM-APP…», «Синхронизация с
    сервером», «0.8»»: порог выработки уехал в чтение раздела, потому что
    модель заполнила все поля бланка сразу. Исполнение не ломалось —
    инструмент берёт только своё, — но по журналу разбирают, что агент
    делал, и журнал врал.
    """
    settings = Settings(agent_enabled=True, agent_max_steps=1, agent_decision="schema")
    toolbox = FakeToolbox({"read_section": ToolOutcome(text="раздел", hits=[hit(9)])})
    providers = TextOnlyProviders(
        '{"why": "дочитаю", "action": "read_section", "doc_id": "IM-APP", '
        '"heading": "Синхронизация", "min_damage": 0.8, "query": "лишнее"}'
    )
    agent = Agent(settings=settings, toolbox=toolbox, providers=providers)

    asyncio.run(agent.gather("вопрос", empty_result([hit(1)]), trace=FakeTrace()))

    assert toolbox.ran == [("read_section", {"doc_id": "IM-APP", "heading": "Синхронизация"})]


def test_running_out_of_steps_after_an_empty_one_is_not_a_cut_trail() -> None:
    """Предупреждение обязано срабатывать по делу, иначе его перестают читать.

    В логе «Кончились разрешённые шаги — контекст может быть неполным»
    стояло под полным и правильным ответом: второй шаг не принёс ничего,
    и обрывать было нечего.
    """
    found = hit(9)
    agent, toolbox, providers = build(
        [ToolCall(name="search_wiki", arguments={"query": "раз"}),
         ToolCall(name="read_section", arguments={"doc_id": "X", "heading": "Y"})],
        {"search_wiki": ToolOutcome(text="нашлось", hits=[found]),
         "read_section": ToolOutcome(text="ничего", hits=[])},
        max_steps=2,
    )

    outcome = run(agent, empty_result([hit(1)]))

    assert outcome.hit_limit is True, "шаги действительно кончились"
    assert outcome.trail_cut is False, "последний шаг был пустой — терять было нечего"


def test_running_out_of_steps_mid_trail_is_worth_a_warning() -> None:
    """Обратный случай: последний шаг принёс новое, значит могло быть ещё."""
    agent, toolbox, providers = build(
        [ToolCall(name="search_wiki", arguments={"query": "раз"}),
         ToolCall(name="read_section", arguments={"doc_id": "X", "heading": "Y"})],
        {"search_wiki": ToolOutcome(text="нашлось", hits=[hit(9)]),
         "read_section": ToolOutcome(text="и ещё", hits=[hit(11)])},
        max_steps=2,
    )

    outcome = run(agent, empty_result([hit(1)]))

    assert outcome.trail_cut is True


def test_what_the_model_found_is_visible_in_the_next_state() -> None:
    """Иначе инструмент работает, а состояние об этом молчит.

    Выжимка показывает первые шесть фрагментов. Находки модели
    дописывались в конец списка из двух десятков — и на следующем шаге
    модель видела состояние без собственной находки. Ровно поэтому второй
    шаг в логе ушёл читать случайный раздел.
    """
    from app.agent import _state

    searched = [hit(number, score=0.5 - number / 100) for number in range(1, 20)]
    mine = hit(99, score=0.0, origin="живые данные сервиса")

    shown = _state("вопрос", [*searched, mine])

    assert "99" in shown or mine.chunk.heading_path in shown


def test_the_agent_does_not_turn_the_page_on_its_own() -> None:
    """Листает человек. Агент — только по метке, пришедшей из браузера.

    Из живого прогона: на «дай топ 10 труб» агент брал десять строк, видел
    в ответе метку следующей страницы и тут же шёл за одиннадцатой-
    двадцатой; на «как дела?» притащил строки с 21-й по 38-ю. Никто этого
    не просил — он просто увидел, что дальше есть ещё.

    Запретом в промпте это не лечилось: абзац «сам за следующей страницей
    не ходи» модель игнорировала через раз. Правило проверяется кодом,
    значит его место в коде.
    """
    import inspect

    from app.agent import Agent

    source = inspect.getsource(Agent.stream)

    # Разрешены ровно те метки, что вернул браузер вместе с вопросом.
    assert "allowed_pages" in source
    assert "open_tables" in source
    assert "page not in allowed_pages" in source


def test_the_tool_result_no_longer_offers_a_cursor_to_the_agent() -> None:
    """Приглашение листать убрано из выжимки — иначе запрет бессмыслен.

    Оставить метку в ответе инструмента и запретить ей пользоваться
    значило бы тратить шаг агента на вызов, который мы всё равно отклоним.
    """
    import inspect

    from app.rag import result

    assert "cursor=" not in inspect.getsource(result.Dataset.agent_note)
