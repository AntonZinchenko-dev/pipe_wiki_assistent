"""Тесты на то, что стенд ВИДИТ вторую реализацию.

Отдельный файл от `test_alt_specimen.py`, и разделение содержательное. Там
проверяется, что образец честно помечает неизмеренное. Здесь — что наши
собственные приборы не выключаются молча, когда им подсовывают прогон,
сделанный не нашим кодом.

Три дефекта, найденные на первом же сравнении. Все три — одного класса:
инструмент не измерил и не пожаловался. Ни один из них не ронял тестов, ни
один не писал в лог, и каждый выглядел в коде правильным.

ПЕРВЫЙ. Судья пропускал весь прогон образца. Условие допуска было написано
как `row.status != "answered"`, а у образца поле статуса пустое — сознательно,
своей схемы ответа у него нет. `judge alt1` печатал «к оценке 0 ответов» и
возвращал успех. То есть единственный прибор, который смотрит на СМЫСЛ
ответа, не оценил ни одной строки — и это выглядело как выполненная работа.

ВТОРОЙ. Тринадцать вопросов без ответа в корпусе засчитывались образцу как
пройденные: `status_ok` у него None, а `None is not False` — это True. На двух
из них образец выдумал ответ про то, чего в корпусе нет вообще. Вопросы-ловушки
держатся в наборе ровно ради этого — и оказались выключены.

ТРЕТИЙ, не дефект, а слепое пятно: у стенда не было прибора на язык ответа.
Каждый пятый ответ образца оказался не на русском, и увидеть это было нечем.
`answer_contains` на таких ответах падает — но падает он и когда факт не
найден, то есть показывает не ту болезнь.
"""

from __future__ import annotations

from eval import dataset
from eval.metrics import answer_language
from eval.report import _row_ok
from eval.runner import RowResult, aggregate, judgeable


def _row(**overrides) -> RowResult:
    """Строка прогона с обязательными полями. Остальное — по умолчанию."""
    base = dict(
        question_id="q001",
        question="как отключить кэш последних значений в шлюзе телеметрии",
        type="negation",
        difficulty="easy",
        answerable=True,
        critical=False,
        retrieved_docs=["TG-GW"],
        retrieval={
            "recall@5": 1.0, "recall@10": 1.0, "precision@5": 0.2, "mrr": 1.0,
            "ndcg@10": 1.0, "context_hit": True, "best_cosine": 0.0,
            "passed_floor": True, "chunk_rank": 1, "chunk_mrr": 1.0, "margin": 0.0,
        },
    )
    base.update(overrides)
    return RowResult(**base)


# --------------------------------------------------------------------------
# Дефект первый: судья не видел образец


def test_judge_accepts_a_row_without_a_status() -> None:
    """Главный тест файла.

    Пустой статус означает «статус не измерялся», а не «система не ответила».
    Строка образца — отвечаемый вопрос, текст ответа есть, ошибки нет — обязана
    попасть к судье.
    """
    assert judgeable(_row(status="", answer="Переменная TG_SNAPSHOT_CACHE=off.")) is True


def test_judge_still_skips_refusals_of_our_own_system() -> None:
    """Починка не должна открыть дверь тому, что отсекалось правильно.

    Наш отказ — это статус `not_found`, и судить его нечем: правильность
    отказа проверяется статусом, без модели и без затрат.
    """
    assert judgeable(_row(status="not_found", answer="Сведений об этом нет.")) is False
    assert judgeable(_row(status="no_context", answer="Близость ниже порога.")) is False


def test_judge_skips_unanswerable_questions() -> None:
    assert judgeable(_row(answerable=False, status="", answer="Сведений нет.")) is False


def test_judge_skips_a_row_that_failed() -> None:
    """Обломок прогона судить бессмысленно: это не ответ системы, а её падение."""
    assert judgeable(
        _row(status="", answer="частичный текст", error="TimeoutError: провайдер молчит")
    ) is False


def test_judge_skips_an_empty_answer_without_a_status() -> None:
    """Вырожденный случай НАШЕЙ системы, который прикрывал старый фильтр.

    Пайплайн не отдал ни ответа, ни ошибки: статус пустой И текст пустой.
    Такая строка судье по-прежнему не идёт — иначе починка первого дефекта
    завела бы второй.
    """
    assert judgeable(_row(status="", answer="")) is False
    assert judgeable(_row(status="", answer="   \n  ")) is False


def test_judgeable_gives_the_same_answer_for_a_dict_and_an_object() -> None:
    """Строка прогона живёт в двух видах, и допуск обязан быть один.

    В памяти это `RowResult`, на диске — словарь. Судейство работает с
    объектами, ручная разметка и замер чувствительности — со словарями. Пока
    условие было написано в каждом месте своё, эти два представления одной
    строки расходились: судья оценил у образца сто ответов, а `label` не нашёл
    ни одного и напечатал «в прогоне нет ответов для разметки», как будто
    прогон пустой.

    Каппа сравнивает человека ИМЕННО с судьёй. Если они видят разные множества
    строк, сравнивать нечего — а выглядит это как отсутствие данных, не как
    поломка.
    """
    cases = [
        dict(status="", answer="Переменная TG_SNAPSHOT_CACHE=off.", answerable=True),
        dict(status="answered", answer="ответ", answerable=True),
        dict(status="not_found", answer="Сведений нет.", answerable=True),
        dict(status="", answer="", answerable=True),
        dict(status="", answer="ответ", answerable=False),
        dict(status="", answer="частичный", answerable=True, error="Timeout"),
    ]
    for case in cases:
        as_object = judgeable(_row(**case))
        as_dict = judgeable({**case, "error": case.get("error", "")})
        assert as_object == as_dict, f"расхождение на {case}"


def test_judgeable_accepts_a_row_dict_straight_from_a_run_file() -> None:
    """Словарь из файла может не содержать поля вовсе — это не «пусто по смыслу».

    Прогоны, сделанные до появления какого-нибудь поля, его просто не имеют.
    Допуск обязан читать такой словарь без падения.
    """
    from_file = {
        "question_id": "q001",
        "answerable": True,
        "status": "",
        "answer": "Переменная TG_SNAPSHOT_CACHE=off.",
    }
    assert judgeable(from_file) is True


# --------------------------------------------------------------------------
# Дефект второй: неизмеренный отказ считался успехом


def _dict_row(**overrides) -> dict:
    row = {
        "question_id": "q069",
        "answerable": False,
        "critical": False,
        "retrieval": {"context_hit": False},
        "status": "",
        "status_ok": None,
        "answer_contains": None,
        "judge_verdict": None,
        "error": "",
    }
    row.update(overrides)
    return row


def test_unmeasured_refusal_is_not_a_pass() -> None:
    """Дословно тот случай, который прошёл незамеченным.

    Неотвечаемый вопрос, отказ мерить нечем. Это ПОТЕРЯ ИЗМЕРЕНИЯ, и строка
    обязана выпасть из обоих списков, а не пополнить успешные.
    """
    assert _row_ok(_dict_row()) is None


def test_measured_refusal_still_counts_both_ways() -> None:
    """Там, где отказ измерен, поведение прежнее."""
    assert _row_ok(_dict_row(status="not_found", status_ok=True)) is True
    assert _row_ok(_dict_row(status="answered", status_ok=False)) is False


def test_a_hallucinated_answer_to_an_unanswerable_question_is_not_a_pass() -> None:
    """Смысл всей проверки, на настоящем примере из прогона `alt1`.

    Вопрос «как настроить мониторинг Grafana для шлюза телеметрии» — ловушка:
    про Grafana в корпусе нет ни слова. Образец собрал правдоподобный ответ из
    соседних кусков документации. Засчитывать это успехом нельзя, и объявлять
    провалом тоже нельзя — мы не измеряли. Правильный ответ ровно один: None.
    """
    hallucination = _dict_row(
        question_id="q073",
        answer=(
            "Для настройки мониторинга Grafana для шлюза телеметрии TELEMETRY-GW "
            "можно использовать метрики, описанные в документации."
        ),
    )
    assert _row_ok(hallucination) is None


# --------------------------------------------------------------------------
# Слепое пятно третье: язык ответа


def test_language_detects_russian_technical_text() -> None:
    """Русский технический ответ полон латиницы, и это НЕ английский.

    Имена переменных, коды ошибок и названия сервисов дают до половины
    латинских букв при совершенно русском тексте. Порог, который этого не
    учитывает, объявил бы брак на правильных ответах — то есть метрика начала
    бы врать в первый же день.
    """
    assert answer_language(
        "Для отключения кэша установите переменную TG_SNAPSHOT_CACHE=off, "
        "затем перезапустите процесс. Код ошибки E-1042 в логе REPORTD."
    ) == "ru"


def test_language_detects_chinese() -> None:
    """Настоящий ответ образца на вопрос q069 из прогона `alt1`."""
    assert answer_language(
        "根据提供的上下文信息，无法得知PIPEPASS的订阅费用是多少。"
        "文档中没有提到关于定价或费用的信息。"
    ) == "zh"


def test_language_detects_english() -> None:
    assert answer_language(
        "The provided context does not contain any information about the salary "
        "of a shift engineer."
    ) == "en"


def test_chinese_wins_over_latin_service_names() -> None:
    """Порядок проверок, а не порог: иероглифы проверяются первыми.

    Китайский ответ про наш корпус почти всегда содержит латинские имена
    сервисов, и по доле латиницы его легко принять за английский. Разница
    существенная: английский ответ читатель вики хотя бы прочтёт.
    """
    assert answer_language("FATIGUE-API 部署在哪个云平台上无法确定。") == "zh"


def test_language_is_blank_when_there_are_no_letters() -> None:
    """Прочерк, а не «язык неизвестен и потому плохой».

    Поисковый прогон, пустой ответ и строка из одних цифр — это отсутствие
    измерения. Ровно та же разница, что между нулём и прочерком в остальных
    графах.
    """
    assert answer_language("") == ""
    assert answer_language("   ") == ""
    assert answer_language("42 — 0.866 (2026-05-14)") == ""


def test_aggregate_reports_language_share_and_breakdown() -> None:
    """Доли мало: нужна разбивка.

    «11 % не на том языке» не говорит, английский это или китайский, — а
    лечится это разными строчками промпта.
    """
    rows = [
        _row(question_id="q001", answer="русский ответ", answer_language="ru"),
        _row(question_id="q002", answer="русский ответ", answer_language="ru"),
        _row(question_id="q003", answer="中文回答", answer_language="zh"),
        _row(question_id="q004", answer="english answer", answer_language="en"),
    ]
    overall = aggregate(rows)["overall"]
    assert overall["language_ok"] == 0.5
    assert overall["languages"] == {"en": 1, "ru": 2, "zh": 1}


def test_language_share_is_blank_for_a_search_run() -> None:
    """У поискового прогона ответов нет вообще.

    Ноль здесь читался бы как «вся система отвечает не на том языке» — то есть
    как катастрофа вместо отсутствия измерения.
    """
    overall = aggregate([_row(answer="", answer_language="")])["overall"]
    assert overall["language_ok"] is None
    assert overall["languages"] == {}


def test_refusal_language_is_measured_too() -> None:
    """Язык считается и по неотвечаемым вопросам, и это главный случай.

    Отказ на китайском — формально правильное поведение, которое пользователь
    вики не прочтёт. Ограничить проверку отвечаемыми значило бы не видеть
    худшую половину случаев: из 22 иноязычных ответов образца девять пришлись
    именно на отказы.
    """
    rows = [
        _row(question_id="q068", answerable=False, answer="要迁移服务", answer_language="zh"),
        _row(question_id="q069", answerable=False, answer="Сведений нет.", answer_language="ru"),
    ]
    assert aggregate(rows)["overall"]["language_ok"] == 0.5


# --------------------------------------------------------------------------
# Судья и язык: годный ответ — верный по смыслу И читаемый


def test_judge_rejects_a_foreign_language_answer_without_calling_the_model() -> None:
    """Ответ не на языке корпуса — неверный, и это решается без вызова модели.

    Требование продуктовое, а не методическое: ответ на китайском, безупречный
    по содержанию, пользователь русской вики не прочтёт. Судья сам этого не
    ловил — на прогоне образца он признал верными шесть таких ответов из
    одиннадцати, то есть `judge_ok` был завышен на величину, которой никто не
    видел.

    Проверка стоит ДО обращения к модели. Поэтому `provider` и `store` здесь
    заведомо непригодны: если короткое замыкание однажды перестанет работать,
    тест упадёт на попытке ими воспользоваться, а не тихо разойдётся с
    ожиданием.
    """
    import asyncio

    from eval.judge import Judge

    class Unusable:
        def __getattr__(self, name):
            raise AssertionError(
                "судья полез к модели или в хранилище, хотя язык ответа "
                "определился как чужой — короткое замыкание не сработало"
            )

    judge = Judge(provider=Unusable(), model="неважно", store=Unusable())
    verdict = asyncio.run(
        judge.verdict(
            question=dataset.load()[0],
            answer="不可以删除 рейс。删除 рейс将使基于它的疲劳计算结果无法复现。",
        )
    )
    assert verdict.correct is False
    assert "не на языке корпуса" in verdict.reason


def test_judge_does_not_reject_russian_technical_text() -> None:
    """Страховка от ложной тревоги на нормальном ответе.

    Русский технический ответ полон латиницы — имена переменных, коды ошибок.
    Если бы короткое замыкание срабатывало на них, судья браковал бы
    правильные ответы, не глядя, и `judge_ok` рухнул бы по причине, не имеющей
    отношения к системе.
    """
    from eval.metrics import CORPUS_LANGUAGE, answer_language

    assert answer_language(
        "Установите TG_SNAPSHOT_CACHE=off и перезапустите процесс. "
        "См. код ошибки E-1042 в журнале REPORTD."
    ) == CORPUS_LANGUAGE


def test_empty_answer_still_goes_to_the_judge() -> None:
    """Букв нет — определять нечего, и это не нарушение языка.

    Пустой язык означает отсутствие измерения, а не измерение со значением
    «плохо». Превращать прочерк в вердикт — та же подмена, которую мы чиним
    по всему стенду.
    """
    from eval.metrics import answer_language

    assert answer_language("") == ""
    assert answer_language("42 — 0.866") == ""


# --------------------------------------------------------------------------
# Шестой дефект: сравнение не видело различий внутри index_meta


def test_compare_sees_a_setting_that_lives_inside_index_meta() -> None:
    """Два прогона с разным переписыванием запроса — это НЕ замер разброса.

    `index_meta` игнорировался целиком, потому что в нём лежит `built_at`,
    меняющийся при каждой перестройке индекса. Вместе с ним игнорировались и
    настройки: у образца там `fusion_queries` — включено ли переписывание
    вопроса.

    Итог: два прогона, отличающиеся ровно этим, разошлись на 13 пунктов
    `chunk_top1`, а `compare` напечатал «конфигурации совпадают — значит это
    замер разброса». Прочитать это можно было только как «система скачет на
    13 пунктов между одинаковыми прогонами» — вывод катастрофический и
    полностью ложный.
    """
    from eval.report import config_diff

    was = {"label": "alt1", "index_meta": {"stack": "llamaindex", "fusion_queries": "1"}}
    now = {"label": "alt4", "index_meta": {"stack": "llamaindex", "fusion_queries": "4"}}
    differences = config_diff(was, now)
    assert any("fusion_queries" in d for d in differences), differences


def test_compare_ignores_the_rebuild_timestamp() -> None:
    """А вот перестройка индекса различием конфигурации не является.

    Иначе каждый `ingest --rebuild` объявлял бы систему изменившейся, и
    предупреждение «различий больше одного» печаталось бы всегда — то есть
    перестало бы что-либо значить.
    """
    from eval.report import config_diff

    was = {"index_meta": {"built_at": "1789457686.9", "documents": "16"}}
    now = {"index_meta": {"built_at": "1789999999.1", "documents": "16"}}
    assert config_diff(was, now) == []


def test_a_new_setting_in_index_meta_shows_up_by_itself() -> None:
    """Ключ, добавленный завтра, попадёт в различия без правки списка.

    Список пропускаемого — это перечень изменчивого, а не белый список
    видимого. Разница в том, что забытая настройка тогда видна, а не спрятана:
    ошибаться этот код должен в сторону лишнего шума, а не молчания.
    """
    from eval.report import config_diff

    was = {"index_meta": {"reranker": "нет"}}
    now = {"index_meta": {"reranker": "bge-reranker-v2-m3"}}
    assert any("reranker" in d for d in config_diff(was, now))
