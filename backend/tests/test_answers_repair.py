"""Починки статуса: когда сервер вправе переписать ярлык модели."""

from __future__ import annotations

import json

from app.rag.answer import AnswerStatus, finalize
from app.rag.context import Fragment


def fragment(number: int, body: str) -> Fragment:
    return Fragment(
        number=number, chunk_id=number, doc_id=f"D{number}", doc_title="Документ",
        doc_version="1.0", doc_status="действующий", doc_updated="2026-01-01",
        heading_path="Раздел", page_from=1, page_to=1, body=body,
        found_by="оба", vector_score=0.7, fused_score=0.03,
    )


def test_a_combined_reference_still_counts_as_a_reference() -> None:
    """«[1, 2]» одной скобкой — это ссылки, а не их отсутствие.

    Из живого прогона: на вопрос «когда мы можем списать трубу» модель
    ответила по делу и сослалась на два источника одной скобкой. Поле
    citations она при этом не заполнила — обычный её промах, ради которого
    починка и заведена. Но починка читала только «[1]» и «[2]» по
    отдельности, ссылок не увидела и переписала статус на «в документации
    нет ответа» — над текстом, который отвечал и ссылался.

    Проверка, которая врёт о собственном предмете, хуже отсутствующей: по
    её отчёту чинят не то место.
    """
    answer = (
        "Труба списывается, если инспектор проставил категорию SCRAP, "
        "либо при сквозной трещине [1, 2]."
    )
    raw = json.dumps({"status": "answered", "answer": answer, "citations": []})

    envelope = finalize(
        raw, [fragment(1, "Категория SCRAP"), fragment(2, "Сквозная трещина")],
        streamed_text=answer,
    )

    assert envelope.status is AnswerStatus.ANSWERED
    assert "только в тексте" in envelope.schema_error


def test_an_answer_with_no_reference_at_all_is_still_demoted() -> None:
    """Починка не должна обессмыслиться: без ссылок ответ остаётся отказом."""
    answer = "Трубу можно списать, когда она износилась."
    raw = json.dumps({"status": "answered", "answer": answer, "citations": []})

    envelope = finalize(raw, [fragment(1, "Категория SCRAP")], streamed_text=answer)

    assert envelope.status is AnswerStatus.NOT_FOUND


def test_a_reference_to_a_fragment_that_was_not_there_does_not_count() -> None:
    """«[7]» при семи отсутствующих фрагментах — не ссылка, а число."""
    answer = "Списывается по осмотру [7]."
    raw = json.dumps({"status": "answered", "answer": answer, "citations": []})

    envelope = finalize(raw, [fragment(1, "Категория SCRAP")], streamed_text=answer)

    assert envelope.status is AnswerStatus.NOT_FOUND


def test_the_live_header_does_not_order_the_model_to_retell_rows() -> None:
    """Два противоположных приказа в одном запросе — это не строгость.

    Шапка живого фрагмента требовала «переписывай строки как есть», а блок
    <о таблицах> запрещал пересказывать таблицу и сообщал, что видна только
    первая строка. Модель выполняла первый приказ, упиралась в отсутствие
    строк и додумывала: «другие трубы не указаны, но можно предположить…
    рекомендую обратиться в поддержку».

    Таблица уходит в браузер отдельным событием — значит шапка обязана
    говорить то же, что и правила: строк не видно, перечислять нечего.
    """
    from app.rag.context import render_fragments

    live = fragment(1, "Таблица T98D4: топ-5 труб.")
    live.live = True
    block = render_fragments([live])

    assert "переписывай их как есть" not in block
    assert "УЖЕ ПОКАЗАНА ЧЕЛОВЕКУ НА ЭКРАНЕ" in block
    assert "не досчитывай" in block


def test_the_header_does_not_offer_a_document_id_as_a_source() -> None:
    """Иначе модель сошлётся «[PM-2025-03]» вместо «[1]».

    Атрибут назывался «источник» — ровно тем словом, которым в промпте
    просят сослаться. Модель брала его значение, проверка такой ссылки не
    находила, детектор считал её выдуманным обозначением, и над правильным
    ответом висело «ссылок не подтверждено».
    """
    from app.rag.context import render_fragments

    block = render_fragments([fragment(1, "текст")])

    assert 'источник="' not in block
    assert 'документ="D1"' in block
    assert 'номер="1"' in block


def test_no_answer_over_a_full_table_is_a_lie() -> None:
    """Живые данные в контексте означают, что сервис ответил.

    Из живого прогона: «следующие 5 скважин покажи» — таблица пришла, пять
    строк на экране, а над ней ярлык «в документации нет ответа». Человек
    видит данные и сообщение, что данных нет.

    Ярлык ставит модель, и здесь она оценивает не результат, а свою
    осведомлённость: строк она не видит, их видит человек. Мы знаем
    больше — значит поправить ярлык обязаны мы.
    """
    live = fragment(1, "Таблица T246A: топ-5 скважин по выработке.")
    live.live = True
    answer = "Извините, я могу показать только первую строку таблицы."
    raw = json.dumps({"status": "not_found", "answer": answer, "citations": []})

    envelope = finalize(raw, [live], streamed_text=answer)

    assert envelope.status is AnswerStatus.ANSWERED
    assert "живых данных" in envelope.schema_error
    # Текст НЕ трогаем: он остался неудачным, и это должно быть видно.
    assert envelope.answer == answer


def test_no_answer_without_live_data_stays_a_refusal() -> None:
    """Починка не должна съесть честный отказ по документам."""
    answer = "В документации нет ответа на этот вопрос."
    raw = json.dumps({"status": "not_found", "answer": answer, "citations": []})

    envelope = finalize(raw, [fragment(1, "текст документа")], streamed_text=answer)

    assert envelope.status is AnswerStatus.NOT_FOUND


def test_a_long_invented_identifier_is_caught() -> None:
    """«WELL-00123456789» проезжало мимо детектора: хвост был коротким.

    После седьмой цифры стояла восьмая, границы слова не получалось,
    совпадения не было — детектор молчал ровно на том случае, ради
    которого заведён.
    """
    from app.rag.answer import unknown_identifiers

    found = unknown_identifiers(
        "Скважина WELL-00123456789, статус Drilled and completed.",
        [fragment(1, "Скважина W-122, выработка 97.7 %.")],
    )

    assert "WELL-00123456789" in found
