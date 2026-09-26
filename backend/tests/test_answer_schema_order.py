"""Тесты на порядок полей схемы и на дешёвую проверку факта в ответе.

Про первый тест отдельно, потому что он выглядит формальностью, а не тестом.

Порядок полей в схеме — это порядок рассуждения модели. Провайдер строит из
схемы грамматику и требует поля ровно в том порядке, в котором они
перечислены; генерация авторегрессивная, поэтому ярлык, выданный первым,
становится частью контекста и определяет всё, что написано после него.

В нашем первом прогоне поле `status` стояло первым, и система отказалась
отвечать на 60 вопросов из 67 отвечаемых — при том, что нужный абзац лежал в
контексте первым номером. Это не опечатка, которую видно на ревью: схема
выглядела аккуратно и проходила все остальные тесты.

Поэтому здесь стоит тест, который смотрит не на поведение, а на порядок
ключей. Он существует ровно для того, чтобы это нельзя было вернуть назад
случайно, «причесав» схему.
"""

import json

from app.rag.prompt import ANSWER_SCHEMA, SYSTEM_PROMPT
from eval.metrics import answer_contains


def test_answer_goes_before_status_in_properties():
    keys = list(ANSWER_SCHEMA["properties"])
    assert keys.index("answer") < keys.index("status"), (
        "status не должен стоять раньше answer: модель выставит ярлык до того, "
        "как прочитает фрагменты, и допишет текст под уже выданный ярлык"
    )


def test_answer_goes_before_status_in_required():
    required = ANSWER_SCHEMA["required"]
    assert required.index("answer") < required.index("status")
    assert "citations" in required


def test_status_enum_has_explicit_refusal():
    # Обратная сторона: вариант «не найдено» обязан существовать. Категоризация
    # без явного выхода заставляет модель выбрать что-то из имеющегося.
    assert "not_found" in ANSWER_SCHEMA["properties"]["status"]["enum"]


def test_prompt_forbids_both_kinds_of_error():
    # В промпте должны быть обе половины правила: и «не выдумывай», и «не
    # отказывайся, когда ответ есть». Одна без другой даёт перекос.
    assert "хуже честного" in SYSTEM_PROMPT
    assert "Обратная ошибка" in SYSTEM_PROMPT


# --------------------------------------------------------- проверка факта


def test_answer_contains_finds_fact():
    assert answer_contains("Нужно выставить TG_SNAPSHOT_CACHE=off", ["TG_SNAPSHOT_CACHE=off"])


def test_answer_contains_ignores_case_spacing_and_yo():
    assert answer_contains("...не  менее\nсуток...", ["не менее суток"])
    assert answer_contains("Раскатка идёт через канарейку", ["канарейку"])


def test_answer_contains_catches_the_real_failure():
    # Ровно тот случай из прогона: во фрагменте «не менее суток», в ответе
    # «конкретное время не уточняется». Подстрока это видит без судьи-модели.
    answer = "Длительность шага не менее одного рабочего дня, конкретное время не уточняется."
    assert answer_contains(answer, ["не менее суток"]) is False


def test_answer_contains_requires_all_needles():
    assert answer_contains("только первое", ["только первое", "и второе"]) is False


def test_answer_contains_returns_none_when_nothing_to_check():
    assert answer_contains("любой текст", []) is None


# ------------------------------------------- страховка «ответ без цитат»


def test_answered_without_citations_becomes_not_found():
    """Ярлык, противоречащий содержанию, исправляется на нашей стороне.

    Случай из прогона: модель написала «сведений во фрагментах нет» и
    пометила это `answered`. Помечать такое недостаточно — интерфейс покажет
    человеку уверенный ответ. То, что проверяется у себя, не оставляют на
    совесть модели.
    """
    from app.rag.answer import AnswerStatus, finalize

    raw = json.dumps(
        {"answer": "Сведений о миграции в Kubernetes во фрагментах нет.",
         "citations": [], "status": "answered"},
        ensure_ascii=False,
    )
    envelope = finalize(raw, [], streamed_text="")
    assert envelope.status is AnswerStatus.NOT_FOUND
    assert "без единой цитаты" in envelope.schema_error
    assert envelope.schema_valid is False


def test_answered_with_citations_stays_answered():
    from app.rag.answer import AnswerStatus, finalize
    from app.rag.context import Fragment

    fragment = Fragment(
        number=1, chunk_id=1, doc_id="TG-GW", doc_title="t", doc_version="1",
        doc_status="действующий", doc_updated="2026-01-01", heading_path="р",
        page_from=1, page_to=1, body="Переменная TG_SNAPSHOT_CACHE=off отключает срез.",
        found_by="vector", vector_score=0.7, fused_score=0.7,
    )
    raw = json.dumps(
        {"answer": "Выставить TG_SNAPSHOT_CACHE=off.",
         "citations": [{"fragment": 1, "quote": "TG_SNAPSHOT_CACHE=off отключает срез"}],
         "status": "answered"},
        ensure_ascii=False,
    )
    envelope = finalize(raw, [fragment], streamed_text="")
    assert envelope.status is AnswerStatus.ANSWERED
    assert envelope.citations_failed == 0


# ------------------------------------------------ разметка ответа отдельно


def test_answer_needles_do_not_fall_back_to_must_contain():
    """Раньше здесь проверялся ОТКАТ на разметку поиска, и это было ошибкой.

    Тест честно фиксировал поведение, которое оказалось генератором тихих
    ошибок: как только разметка поиска стала длинной цитатой из документа,
    откат начал требовать от пересказа дословности документа. Поведение
    отменено, тест заменён — а не удалён, потому что «раньше было наоборот»
    надо где-то зафиксировать.
    """
    from eval.dataset import Question

    question = Question(
        id="q", question="?", type="fact", difficulty="easy", answerable=True,
        docs=["D"], must_contain=["не заменяет"],
    )
    assert question.answer_needles == []


def test_answer_needles_override_corpus_wording():
    from eval.dataset import Question

    question = Question(
        id="q", question="?", type="fact", difficulty="easy", answerable=True,
        docs=["D"], must_contain=["не заменяет"],
        answer_must_contain=["не списывается", "осмотр"],
    )
    assert question.answer_needles == ["не списывается", "осмотр"]


def test_alternatives_inside_one_needle():
    # `|` внутри подстроки — это «или»: тот же факт, другие слова.
    needles = ["не заменяет | не списывается | не является основанием", "осмотр"]
    assert answer_contains("Труба не списывается по расчёту, решает осмотр.", needles)
    assert answer_contains("Расчёт не заменяет осмотр.", needles)
    assert answer_contains("Труба списывается по расчёту усталости.", needles) is False


# ------------------------------------------ обрыв по лимиту токенов


def test_truncated_json_is_not_called_answered():
    """Неразобранная структура не помечается «ответил».

    Случай из прогона: массив citations в схеме не имел верхнего предела,
    модель набивала его до упора в лимит токенов, поток оборвался, JSON не
    закрылся. Текст ответа при этом был полезен — и уходил наружу со статусом
    `answered`, то есть «структура получена и проверена». Она не получена.
    """
    from app.rag.answer import AnswerStatus, finalize

    truncated = '{"answer": "Кэш отключается переменной.", "citations": [{"fragment": 1, "qu'
    envelope = finalize(truncated, [], streamed_text="Кэш отключается переменной.")
    assert envelope.status is AnswerStatus.ERROR
    assert envelope.answer == "Кэш отключается переменной."  # прочитанное не стираем
    assert envelope.schema_valid is False
    assert "структура не разобрана" in envelope.schema_error


def test_citations_array_is_bounded():
    # Массив без верхнего предела — разрешение генерировать бесконечно.
    assert ANSWER_SCHEMA["properties"]["citations"]["maxItems"] >= 1


# --------------------------------------------- отпечаток кода в конфиге


def test_code_version_is_stable_and_short():
    from app.version import code_version, tracked_files

    assert len(tracked_files()) > 5
    first = code_version()
    assert first == code_version()
    assert len(first) == 12


def test_code_version_is_part_of_run_config():
    # Конфиг прогона без отпечатка кода не отвечает на вопрос «та же это
    # система или другая»: код мог измениться, а настройки — нет.
    from eval.runner import RunConfig

    assert "code_version" in RunConfig.__dataclass_fields__


# ------------------------------------------------------- схема судьи


def test_judge_reason_goes_before_verdict():
    """У судьи то же правило, что у отвечающей модели, и цена выше.

    `correct` первым — это приговор до единого шага рассуждения. У судьи это
    опаснее, чем у отвечающего: смещённый прибор портит не один ответ, а все
    выводы, которые по нему делаются.
    """
    from eval.judge import JUDGE_SCHEMA

    keys = list(JUDGE_SCHEMA["properties"])
    assert keys.index("reason") < keys.index("correct")
    assert JUDGE_SCHEMA["required"].index("reason") < JUDGE_SCHEMA["required"].index("correct")


def test_verdict_allows_unresolved():
    # «Судья сказал неверно» и «судья не смог сказать» — разные факты.
    from eval.judge import Verdict

    verdict = Verdict(correct=None, reason="вердикт судьи не разобран")
    assert verdict.correct is None


def test_unresolved_verdict_is_not_counted_as_wrong():
    from eval.runner import RowResult, aggregate

    def row(qid, verdict, reason):
        return RowResult(
            question_id=qid, question="?", type="fact", difficulty="easy",
            answerable=True, critical=False, retrieved_docs=["D"],
            retrieval={"recall@5": 1.0, "mrr": 1.0, "ndcg@10": 1.0, "context_hit": True,
                       "chunk_mrr": 1.0, "chunk_rank": 1},
            status="answered", status_ok=True, answer="текст",
            judge_verdict=verdict, judge_reason=reason,
        )

    rows = [
        row("q1", True, "ок"),
        row("q2", False, "противоречит"),
        row("q3", None, "вердикт судьи не разобран"),
    ]
    block = aggregate(rows)["overall"]
    # Один верный из ДВУХ судимых, а не из трёх: несудимая строка не
    # размывает метрику и считается отдельно.
    assert block["judge_ok"] == 0.5
    assert block["judged"] == 2
    assert block["judge_unresolved"] == 1


# ------------------------------------------- опора судьи: номер выдержки


def test_judge_schema_asks_for_reference_before_reasoning_before_label():
    """Порядок полей у судьи: опора → рассуждение → ярлык.

    Тот же принцип, что в схеме ответа. Опора здесь — НОМЕР выдержки, а не её
    текст: требование дословной цитаты отвергло 39 вердиктов из 67, потому что
    маленькая модель не переписывает длинный отрывок символ в символ. Проверка,
    выбрасывающая больше половины данных, уничтожает измерение.
    """
    from eval.judge import JUDGE_SCHEMA

    keys = list(JUDGE_SCHEMA["properties"])
    assert keys.index("fragment") < keys.index("reason") < keys.index("correct")
    assert JUDGE_SCHEMA["required"] == ["fragment", "reason", "correct"]


def test_judge_prompt_explains_numbering():
    from eval.judge import JUDGE_SYSTEM

    assert "пронумерованы" in JUDGE_SYSTEM
    # Явный выход обязателен и здесь: ноль означает «ни одна не решает».
    assert "поставь 0" in JUDGE_SYSTEM


# ------------------------------- формат прогона переживает переименования


def test_run_rows_tolerate_unknown_fields():
    """Прогон, записанный другой версией стенда, должен читаться.

    Файл прогона — это формат исторических записей. Переименование поля
    ломает не текущий прогон, а все прошлые: мы переименовали judge_quote в
    judge_fragment, и прогон двадцатиминутной давности перестал открываться.
    Измерение, которое нельзя перечитать, перестаёт быть измерением.
    """
    from eval.runner import rows_from_run

    run = {
        "rows": [
            {
                "question_id": "q1", "question": "?", "type": "fact",
                "difficulty": "easy", "answerable": True, "critical": False,
                "retrieved_docs": ["D"], "retrieval": {"recall@5": 1.0},
                "judge_quote": "поле из прошлой версии",
                "judge_verdict": True,
            }
        ]
    }
    rows = rows_from_run(run)
    assert len(rows) == 1
    assert rows[0].question_id == "q1"
    assert rows[0].judge_verdict is True


def test_run_rows_fill_missing_fields_with_defaults():
    from eval.runner import rows_from_run

    run = {
        "rows": [
            {
                "question_id": "q1", "question": "?", "type": "fact",
                "difficulty": "easy", "answerable": True, "critical": False,
                "retrieved_docs": [], "retrieval": {},
            }
        ]
    }
    rows = rows_from_run(run)
    assert rows[0].judge_fragment == 0
    assert rows[0].answer_contains is None


def test_judge_cannot_approve_without_a_fragment():
    """«Ни одна выдержка не решает» и «ответ верен» — противоречие.

    Верен по отношению к чему? Это та же проверка, что «ответил без единой
    цитаты» у отвечающей модели: уверенный вердикт без опоры не считается.
    """
    from eval.judge import JUDGE_SCHEMA, JUDGE_SYSTEM

    # Ноль как явный выход должен существовать, иначе судья начнёт указывать
    # первую попавшуюся выдержку, чтобы заполнить поле.
    assert "поставь 0" in JUDGE_SYSTEM
    assert JUDGE_SCHEMA["properties"]["fragment"]["type"] == "integer"
