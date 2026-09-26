"""Разбор ОДНОГО вопроса: почему нужный фрагмент оказался там, где оказался.

Зачем это отдельный инструмент. Метрика `chunk_top1 = 0.821` говорит, что в
каждом пятом вопросе нужный фрагмент не первый, — и ничего не говорит о том,
что с этим делать. Под этой цифрой скрыты четыре совершенно разные болезни, и
лечатся они четырьмя разными способами:

1. нужного фрагмента нет НИ В ОДНОМ из двух списков — значит запрос написан
   не теми словами, которыми написан документ (лечится переписыванием
   запроса), либо текст разрезан так, что нужное разъехалось по двум
   фрагментам (лечится разбиением);
2. он есть в КЛЮЧЕВОМ списке, но не в векторном — слова совпадают, а смысл
   эмбеддинг не поймал; это вопрос веса ключевого поиска при слиянии;
3. он есть в ОБОИХ, но после слияния стоит ниже границы контекста — вот это и
   есть работа для реранкера, и только это;
4. он дошёл до контекста, но НЕ первым — это и есть то, что мерит chunk_top1;
5. он дошёл первым — поиск сделал всё, что мог, и разбираться надо с ответом.

Разделение четвёртого и пятого случая появилось сразу после первого прогона
разбора и оказалось важнее всего остального. Сводка показала 66 «поиск
справился» из 67 при chunk_top1 = 0.821. Обе цифры верны и говорят о разном:
нужный фрагмент доезжает до модели почти всегда, но в каждом пятом вопросе он
не первый. Пока эти два исхода были слиты в один, разбор отвечал «всё хорошо»
на ту самую разницу, ради которой ставят реранкер.

Ставить реранкер, не разделив эти четыре случая, — как менять масло в машине,
которая не едет, потому что кончился бензин: работа настоящая, к поломке
отношения не имеет.

Аналогия для двух списков. Векторный поиск — это библиотекарь, который понял,
О ЧЁМ вы спрашиваете, и несёт книги по теме, даже если вы не назвали ни одного
слова из них. Ключевой (BM25) — указатель в конце книги: он ищет буквально то
слово, которое вы произнесли, и не понимает вопроса вообще. Первый ошибается на
жаргоне и опечатках, второй — на синонимах. Мы держим оба и складываем их
мнения, поэтому при разборе поломки надо всегда смотреть, кто именно из двоих
не справился.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.search import SearchResult
from app.rag.store import Store, StoredChunk


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower().replace("ё", "е")


def chunk_has_needles(chunk: StoredChunk, needles: list[str]) -> bool:
    """Лежит ли в этом фрагменте ВЕСЬ ожидаемый текст.

    Проверяем по `text`, а не по `body`: в `text` входит и заголовок раздела,
    а разметка вопроса иногда указывает именно на него.
    """
    if not needles:
        return False
    haystack = _normalize(chunk.text)
    return all(_normalize(needle) in haystack for needle in needles)


@dataclass(slots=True)
class Verdict:
    code: str
    text: str
    remedy: str


def diagnose(
    *,
    needle_chunks: list[int],
    vector_ranks: dict[int, int],
    keyword_ranks: dict[int, int],
    fused_ranks: dict[int, int],
    context_chunks: set[int],
) -> Verdict:
    """Ставит ОДИН диагноз. Не список наблюдений, а вывод.

    Порядок проверок — от самого дешёвого исхода к самому дорогому, потому что
    первый подходящий и есть ответ: если нужный фрагмент дошёл до контекста,
    обсуждать его место в векторном списке уже незачем.
    """
    if not needle_chunks:
        return Verdict(
            "нет в индексе",
            "ни один фрагмент индекса не содержит ожидаемый текст целиком",
            "Это не поломка поиска. Либо разметка вопроса указывает на текст, "
            "которого в корпусе нет, либо нужное разъехалось по двум фрагментам "
            "при разбиении — тогда поиск не найдёт его никогда, сколько ни "
            "улучшай ранжирование.",
        )

    if context_chunks & set(needle_chunks):
        place = _rank_of(fused_ranks, needle_chunks)
        if place == 1:
            return Verdict(
                "дошёл первым",
                "нужный фрагмент дошёл до контекста и стоит первым",
                "Поиск сделал всё, что мог. Разбираться надо с ответом.",
            )
        return Verdict(
            "дошёл не первым",
            f"нужный фрагмент дошёл до контекста, но стоит на месте {place}",
            "Ровно это и мерит chunk_top1. Разделение с «дошёл первым» нужно "
            "затем, что именно здесь работал бы реранкер — и затем, чтобы "
            "видеть, СКОЛЬКО таких случаев: если их мало, дорогой реранкер "
            "лечит малую долю, а если много, надо сначала доказать, что место "
            "фрагмента вообще влияет на ответ.",
        )

    in_vector = [cid for cid in needle_chunks if cid in vector_ranks]
    in_keyword = [cid for cid in needle_chunks if cid in keyword_ranks]
    in_fused = [cid for cid in needle_chunks if cid in fused_ranks]

    if not in_vector and not in_keyword:
        return Verdict(
            "запрос другими словами",
            "нужного фрагмента нет ни в векторном, ни в ключевом списке",
            "Лечится переписыванием запроса: вопрос задан не теми словами, "
            "которыми написан документ. Реранкер здесь бесполезен — он "
            "переставляет то, что нашлось, а нашлось не то.",
        )

    if in_keyword and not in_vector:
        return Verdict(
            "смысл не поймался",
            f"нужный фрагмент есть только в КЛЮЧЕВОМ списке (место "
            f"{min(keyword_ranks[cid] for cid in in_keyword)})",
            "Слова совпали, а близость эмбеддинга — нет. Это вопрос доверия к "
            "точному совпадению при слиянии, а не ранжирования.",
        )

    if in_vector and not in_keyword:
        return Verdict(
            "слова не совпали",
            f"нужный фрагмент есть только в ВЕКТОРНОМ списке (место "
            f"{min(vector_ranks[cid] for cid in in_vector)})",
            "Смысл поймался, буквального совпадения нет — обычное дело для "
            "пересказа. Если место низкое, это работа для реранкера.",
        )

    if in_fused:
        return Verdict(
            "порядок",
            f"нужный фрагмент есть в обоих списках и после слияния стоит на "
            f"месте {min(fused_ranks[cid] for cid in in_fused)}, а в контекст "
            f"попадают не все",
            "Вот это и есть работа для реранкера: состав выдачи правильный, "
            "порядок — нет.",
        )

    return Verdict(
        "срезано слиянием",
        "нужный фрагмент есть в исходных списках, но после слияния в выдачу "
        "не попал",
        "Слияние RRF складывает места, а не баллы: фрагмент, стоящий низко в "
        "обоих списках, проигрывает тому, кто стоит средне в обоих. Смотреть "
        "надо на константу k и на глубину списков, а не на реранкер.",
    )


def explain(
    *,
    question_id: str,
    question: str,
    needles: list[str],
    store: Store,
    vector_hits: list[tuple[int, float]],
    keyword_hits: list[tuple[int, float]],
    result: SearchResult,
    context_chunks: set[int],
    top: int = 10,
) -> str:
    """Печатает разбор одного вопроса: оба списка, слияние, контекст, диагноз."""
    lines: list[str] = []
    needle_chunks = [
        chunk.chunk_id for chunk in store.all_chunks() if chunk_has_needles(chunk, needles)
    ]

    vector_ranks = {cid: rank for rank, (cid, _) in enumerate(vector_hits, start=1)}
    keyword_ranks = {cid: rank for rank, (cid, _) in enumerate(keyword_hits, start=1)}
    vector_scores = dict(vector_hits)
    keyword_scores = dict(keyword_hits)
    fused_ranks = {hit.chunk.chunk_id: rank for rank, hit in enumerate(result.hits, start=1)}

    lines.append(f"вопрос {question_id}: {question}")
    lines.append(f"ожидаемый текст: {needles or '— (разметки нет)'}")
    if needle_chunks:
        loaded = store.load_chunks(needle_chunks)
        for chunk_id in needle_chunks:
            chunk = loaded[chunk_id]
            lines.append(
                f"  лежит в чанке {chunk_id}: {chunk.doc_id} · {chunk.heading_path}"
            )
    lines.append("")

    def mark(chunk_id: int) -> str:
        return " <<< НУЖНЫЙ" if chunk_id in needle_chunks else ""

    lines.append(f"ВЕКТОРНЫЙ СПИСОК (первые {top} из {len(vector_hits)})")
    for rank, (chunk_id, score) in enumerate(vector_hits[:top], start=1):
        lines.append(f"  {rank:>2}. чанк {chunk_id:>4}  косинус {score:.4f}{mark(chunk_id)}")
    for chunk_id in needle_chunks:
        if chunk_id in vector_ranks and vector_ranks[chunk_id] > top:
            lines.append(
                f"  нужный чанк {chunk_id} стоит на месте {vector_ranks[chunk_id]}, "
                f"косинус {vector_scores[chunk_id]:.4f}"
            )
        elif chunk_id not in vector_ranks:
            lines.append(f"  нужного чанка {chunk_id} в этом списке НЕТ")
    lines.append("")

    lines.append(f"КЛЮЧЕВОЙ СПИСОК, BM25 (первые {top} из {len(keyword_hits)})")
    for rank, (chunk_id, score) in enumerate(keyword_hits[:top], start=1):
        lines.append(f"  {rank:>2}. чанк {chunk_id:>4}  bm25 {score:.3f}{mark(chunk_id)}")
    for chunk_id in needle_chunks:
        if chunk_id in keyword_ranks and keyword_ranks[chunk_id] > top:
            lines.append(
                f"  нужный чанк {chunk_id} стоит на месте {keyword_ranks[chunk_id]}, "
                f"bm25 {keyword_scores[chunk_id]:.3f}"
            )
        elif chunk_id not in keyword_ranks:
            lines.append(f"  нужного чанка {chunk_id} в этом списке НЕТ")
    lines.append("")

    lines.append(f"ПОСЛЕ СЛИЯНИЯ (первые {top}, дублей убрано {result.dropped_duplicates})")
    for rank, hit in enumerate(result.hits[:top], start=1):
        chunk_id = hit.chunk.chunk_id
        in_context = "в контексте" if chunk_id in context_chunks else "           "
        lines.append(
            f"  {rank:>2}. чанк {chunk_id:>4}  rrf {hit.fused_score:.5f}  "
            f"{hit.found_by:<8} {in_context}  "
            f"{hit.chunk.doc_id} · {hit.chunk.heading_path[:40]}{mark(chunk_id)}"
        )
    lines.append("")

    lines.append(
        f"порог {result.floor:.2f}: лучшая близость {result.best_vector_score:.4f}, "
        f"{'прошло' if result.passed_floor else 'НЕ прошло'}"
        + (" (по точному слову)" if result.passed_by_keyword else "")
    )
    lines.append(f"в контекст попало фрагментов: {len(context_chunks)}")
    lines.append("")

    verdict = diagnose(
        needle_chunks=needle_chunks,
        vector_ranks=vector_ranks,
        keyword_ranks=keyword_ranks,
        fused_ranks=fused_ranks,
        context_chunks=context_chunks,
    )
    lines.append(f"ДИАГНОЗ: {verdict.code}")
    lines.append(f"  {verdict.text}")
    lines.append(f"  {verdict.remedy}")
    return "\n".join(lines)


def _rank_of(ranks: dict[int, int], chunk_ids: list[int]) -> int | None:
    places = [ranks[cid] for cid in chunk_ids if cid in ranks]
    return min(places) if places else None


def summary_table(rows: list[tuple[str, Verdict]]) -> str:
    """Сводка диагнозов по всем неудачным вопросам: с чего начинать.

    Смысл сводки в том, чтобы выбирать инструмент по БОЛЬШИНСТВУ, а не по
    последнему разобранному случаю. Если девять вопросов из десяти — «запрос
    другими словами», ставить реранкер значит потратить вечер на девять
    процентов проблемы.
    """
    counts: dict[str, list[str]] = {}
    for question_id, verdict in rows:
        counts.setdefault(verdict.code, []).append(question_id)

    lines = ["СВОДКА ДИАГНОЗОВ"]
    for code, ids in sorted(counts.items(), key=lambda item: -len(item[1])):
        lines.append(f"  {len(ids):>2}  {code:<26} {', '.join(ids)}")
    return "\n".join(lines)
