#!/usr/bin/env python3
"""Единая точка входа для измерений. Запуск из папки backend.

    python scripts/eval.py validate
    python scripts/eval.py search --label baseline
    python scripts/eval.py calibrate
    python scripts/eval.py answer --label baseline
    python scripts/eval.py judge <файл-прогона>
    python scripts/eval.py noise --runs 3 --mode answer
    python scripts/eval.py list
    python scripts/eval.py show <файл-прогона>
    python scripts/eval.py recheck <файл-прогона>
    python scripts/eval.py compare <было> <стало>
    python scripts/eval.py label <файл-прогона>
    python scripts/eval.py transfer-labels <откуда> <куда>
    python scripts/eval.py kappa <файл-прогона>

Почему одна команда с подкомандами, а не восемь скриптов: измерения делаются
часто и в определённом порядке, и порядок должен быть виден из справки.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings, overridden_from_env  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.providers import build_registry  # noqa: E402
from app import pipeline as pipeline_module  # noqa: E402
from app.rag import prompt as prompt_module  # noqa: E402
from app.rag.prompt import PROMPT_VERSION, SYSTEM_PROMPT  # noqa: E402
from app.rag.context import build_context  # noqa: E402
from app.rag.embed import embed_query  # noqa: E402
from app.rag.search import fuse, hybrid_search, single_list_can_win  # noqa: E402
from app.rag.store import Store  # noqa: E402
from app.trace import TraceWriter  # noqa: E402
from app.version import code_version  # noqa: E402
from eval import calibrate as calibrate_mod  # noqa: E402
from eval import dataset, diagnose, explain, mutate, report  # noqa: E402
from eval.judge import Judge, agreement_table, language_verdict  # noqa: E402
from eval.metrics import (  # noqa: E402
    answer_contains, answer_language, cohen_kappa, mean, score_retrieval,
)
from eval.runner import (  # noqa: E402
    RunConfig, judge_rows, judgeable, load_run, rows_from_run, run_answer, run_search,
    save_run, write_run,
)


def _number(value, digits: int = 3) -> str:
    """Печать метрики, которой может не быть.

    Неизмеренное печатается прочерком. Раньше здесь стоял `get(key, 0)` с
    форматом `:.3f`, и на прогоне без судьи команда падала с TypeError:
    ключ существовал со значением None, и подстановка по умолчанию не
    срабатывала. Дефолт-ноль в таком месте — сразу две ошибки: и падение, и
    намерение печатать отсутствие измерения нулём.
    """
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def runs_dir() -> Path:
    return get_settings().db_path.parent / "eval" / "runs"


def labels_dir() -> Path:
    return get_settings().db_path.parent / "eval" / "labels"


def report_env_overrides(settings) -> None:
    """Печатает настройки, пришедшие не из кода. Вызывается перед каждым прогоном.

    Стоимость — три строки, польза — потерянный прогон. Подробности в
    `app.config.overridden_from_env`: настройка выглядела изменённой и не была,
    потому что её переопределял `.env`, и нигде не печаталось ни то, ни другое.
    """
    changed = overridden_from_env(settings)
    if not changed:
        return
    print("ПЕРЕОПРЕДЕЛЕНО ИЗ ОКРУЖЕНИЯ (.env или переменные среды):")
    for name, (current, default) in sorted(changed.items()):
        print(f"  {name} = {current!r}   (в коде {default!r})")
    print("  Если правка в коде не подействовала — она здесь и перекрыта.")
    print()


def make_config(
    *, label: str, mode: str, settings, store: Store, note: str = "",
    providers=None, agent: bool = False, prompt_version: str = "",
) -> RunConfig:
    stats = store.stats()
    return RunConfig(
        label=label,
        mode=mode,
        # Модель, которая РЕАЛЬНО отвечала, а не та, что прописана у
        # локального провайдера.
        #
        # Здесь жёстко стояло `settings.ollama_chat_model`. Пока провайдер был
        # один, это совпадало. С появлением облачного прогон через него
        # сохранился бы под подписью локальной модели — и через месяц два
        # прогона выглядели бы как одна система с разными цифрами.
        #
        # Прогон поиска провайдеров не знает, там подпись прежняя: отвечающая
        # модель в нём не участвует.
        chat_model=(
            providers.answering_model if providers is not None
            else settings.ollama_chat_model
        ),
        provider_chain=",".join(settings.providers),
        # Агентский шаг — другая система, а не настройка. В шапке он обязан
        # быть, иначе два прогона лягут под одинаковым отпечатком и разница
        # между ними будет прочитана как разброс.
        agent=agent,
        agent_max_steps=settings.agent_max_steps if agent else 0,
        restricted_projects=settings.restricted_projects,
        embed_model=settings.ollama_embed_model,
        # Версия ТОГО промпта, которым отвечали, а не константа модуля.
        #
        # Передаётся аргументом: прогон может идти по выбранной версии из
        # реестра или по временной переставленной (--nudge). Пока здесь
        # стояла импортированная константа, прогон с другим промптом
        # записывался под отпечатком обычного — в журнале лежали два разных
        # промпта под одним именем, и `compare` честно объявил бы их «одной
        # конфигурацией». Прибор, который врёт о себе, хуже отсутствующего:
        # отсутствующий заметен.
        prompt_version=prompt_version or pipeline_module.PROMPT_VERSION,
        code_version=code_version(),
        dataset_version=dataset.dataset_version(),
        **{f"labels_{key}": value for key, value in dataset.label_versions().items()},
        similarity_floor=settings.similarity_floor,
        search_top_k=settings.search_top_k,
        context_max_fragments=settings.context_max_fragments,
        context_token_budget=settings.context_token_budget,
        temperature=settings.temperature,
        rrf_k=settings.rrf_k,
        keyword_weight=settings.keyword_weight,
        index_meta=stats["meta"],
        index_chunks=stats["chunks"],
        # Настройки СУДЬИ — часть конфигурации измерения, а не деталь запуска.
        # Без них прогон, где судила одна модель, и прогон, где другая,
        # выглядят как одна конфигурация, и рост judge_ok читается как
        # улучшение системы. Улучшился прибор.
        judge_model=settings.judge_model,
        judge_think=settings.judge_think,
        judge_max_tokens=settings.judge_max_tokens,
        note=note,
    )


# ---------------------------------------------------------------- подкоманды


def cmd_validate(args) -> int:
    settings = get_settings()
    report_env_overrides(settings)
    store = Store(settings.db_path)
    questions = dataset.load()

    composition = dataset.composition(questions)
    print(json.dumps(composition, ensure_ascii=False, indent=2))

    share = composition["unanswerable_share"]
    if not 0.10 <= share <= 0.15:
        print(
            f"\nВНИМАНИЕ: доля вопросов без ответа {share:.0%}, ожидается 10–15 %. "
            f"Меньше — система не наказывается за привычку всегда отвечать; "
            f"больше — набор смещается в сторону отказов."
        )

    problems = dataset.validate(questions, store)
    print(f"\nпроблем разметки: {len(problems)}")
    for problem in problems:
        print(f"  {problem.question_id}  {problem.kind}: {problem.detail}")
    store.close()
    return 1 if problems else 0


async def _search(args) -> int:
    settings = get_settings()
    report_env_overrides(settings)
    store = Store(settings.db_path)
    questions = dataset.load()
    if args.limit:
        questions = questions[: args.limit]

    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0))
    providers = build_registry(settings, client)

    try:
        store.assert_compatible(
            embed_model=settings.ollama_embed_model, embed_dim=settings.ollama_embed_dim
        )
        rows = await run_search(
            questions, store=store, provider=providers.embedder, settings=settings
        )
    finally:
        await providers.aclose()
        await client.aclose()

    config = make_config(label=args.label, mode="search", settings=settings, store=store, note=args.note)
    path = save_run(config=config, rows=rows, directory=runs_dir())
    store.close()

    print(report.summarize(load_run(path)))
    print(f"прогон сохранён: {path}")
    return 0


def _nudge_prompt(system: str) -> str:
    """Меняет промпт, НЕ меняя его смысла: переставляет два независимых правила.

    Это прибор для измерения прибора, и вот зачем он нужен. Каждая правка
    промпта переписывает все 77 ответов сразу: генерация авторегрессивная, и
    один изменённый токен в начале меняет продолжение везде. За три прогона мы
    трижды приписывали формулировкам разницу в один-три вопроса — а надо
    сначала узнать, сколько вопросов меняется САМО, от правки, которая не
    несёт смысла вообще.

    Аналогия. Прежде чем взвешивать письма на кухонных весах, кладут на них
    одну и ту же гирю дважды. Если во второй раз показало на двадцать граммов
    больше, то разница в пятнадцать граммов между двумя письмами — это не
    разница между письмами, а разброс весов. Мы этого замера не делали и
    поэтому не знаем, что именно мерили.

    Переставляются правила 4 и 5 — они независимы друг от друга (одно про
    противоречия между документами, другое про советы вне документации),
    нумерация сохраняется, ни одного слова не добавлено и не убрано. Смысл
    промпта тот же, байты другие.
    """
    lines = system.split("\n")
    starts = [index for index, line in enumerate(lines) if re.match(r"^[45]\. ", line)]
    if len(starts) != 2:
        raise SystemExit("не нашёл правила 4 и 5 — промпт изменился, проверь _nudge_prompt")
    first, second = starts
    end = next(
        (index for index in range(second + 1, len(lines)) if not lines[index].strip()),
        len(lines),
    )
    block_four = lines[first:second]
    block_five = lines[second:end]
    renumbered = (
        [re.sub(r"^5\. ", "4. ", line) for line in block_five]
        + [re.sub(r"^4\. ", "5. ", line) for line in block_four]
    )
    return "\n".join(lines[:first] + renumbered + lines[end:])


def cmd_coverage(args) -> int:
    """Какие разделы корпуса не проверяет НИ ОДИН вопрос.

    Инструмент против самого коварного вида самообмана в наборе. Метрики
    говорят, насколько хорошо система отвечает на ТЕ вопросы, которые мы
    задали, — и ничего не говорят о том, что мы не спросили. Набор может
    стоять на 0.99 и при этом не проверять треть документации.

    Первый же запуск показал: 119 разделов в корпусе, 67 отвечаемых вопросов,
    21 содержательный раздел без единого вопроса. Причём не какие попало:
    формула метода минимальной кривизны, запреты в регламенте доступов, план
    изменений после аварии — то есть места, где ошибка дороже всего.

    Аналогия: ученик отвечает на пять вопросов из билета и получает пять.
    Вопрос, знает ли он остальные двадцать, оценкой не измеряется и вообще не
    задан.

    Считается по документам корпуса, а не по индексу: нам нужно «что вообще
    написано», а не «что попало в чанки».
    """
    root = Path(args.corpus) if args.corpus else get_settings().corpus_dir
    # Документы лежат в source/: в самом каталоге корпуса рядом могут быть
    # собранные PDF и служебные файлы, и считать их разделами нельзя.
    corpus = root / "source" if (root / "source").exists() else root
    if not corpus.exists():
        print(f"нет каталога корпуса: {corpus}")
        return 1

    def normalize(text: str) -> str:
        return " ".join(text.split()).lower().replace("ё", "е")

    sections: list[tuple[str, str, str]] = []
    for path in sorted(corpus.glob("*.md")):
        heading, buffer = "(преамбула)", []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                if buffer:
                    sections.append((path.name, heading, "\n".join(buffer)))
                heading, buffer = line.lstrip("#").strip(), []
            else:
                buffer.append(line)
        if buffer:
            sections.append((path.name, heading, "\n".join(buffer)))

    questions = dataset.load()
    needles = [
        (question.id, [normalize(n) for n in question.must_contain])
        for question in questions
        if question.answerable and question.must_contain
    ]

    blind: list[tuple[str, str, int]] = []
    covered = 0
    for name, heading, text in sections:
        words = len(text.split())
        if words < args.min_words:
            continue
        haystack = normalize(f"{heading} {text}")
        hit = [qid for qid, ns in needles if all(n in haystack for n in ns)]
        if hit:
            covered += 1
        else:
            blind.append((name, heading, words))

    print(f"корпус: {corpus}")
    print(
        f"разделов длиннее {args.min_words} слов: {covered + len(blind)}, "
        f"из них проверяется вопросами: {covered}, не проверяется: {len(blind)}"
    )
    print(f"вопросов в наборе: {len(questions)} "
          f"(отвечаемых {sum(1 for q in questions if q.answerable)})")
    if blind:
        print()
        print("РАЗДЕЛЫ БЕЗ НИ ОДНОГО ВОПРОСА")
        for name, heading, words in blind:
            print(f"  {name:32} {heading[:50]:52} {words} слов")
        print()
        print("Это не список обязательных дел: часть разделов не стоит вопроса")
        print("(преамбулы, оглавления). Но каждый раздел здесь — место, про")
        print("которое метрики не знают ничего.")
    return 0


async def _sweep(args) -> int:
    """Перебор константы слияния БЕЗ повторных прогонов.

    Ключевая экономия здесь в том, что дорогое отделено от дешёвого. Векторы
    вопросов и оба списка выдачи получаются ОДИН раз; слияние, сборка контекста
    и метрики пересчитываются для каждого значения k мгновенно. Поэтому шесть
    вариантов стоят столько же, сколько один прогон, и выбор константы
    перестаёт быть вопросом веры.

    Мерить здесь можно честно: поиск детерминирован, генерации нет, разброс
    измерен и равен нулю. Любая разница в таблице — настоящая.
    """
    settings = get_settings()
    report_env_overrides(settings)
    store = Store(settings.db_path)
    # ВЫБИРАЕМ НА НАСТРОЕЧНОЙ ЧАСТИ. Отложенную считаем тем же кодом, но
    # отдельно — и только чтобы посмотреть, выжил ли выигрыш там, где его не
    # выбирали.
    everything = [q for q in dataset.load() if q.answerable]
    questions = [q for q in everything if q.split == "tune"]
    held_out = [q for q in everything if q.split == "holdout"]

    ks = [int(part) for part in args.rrf_k.split(",") if part.strip()]
    weights = [float(part) for part in args.keyword_weight.split(",") if part.strip()]
    promotions = [False, True] if args.promote else [False]
    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0))
    providers = build_registry(settings, client)

    # Складываем в память: вопрос -> (векторный список, ключевой список).
    fetched: list[tuple[object, list, list]] = []
    try:
        store.assert_compatible(
            embed_model=settings.ollama_embed_model, embed_dim=settings.ollama_embed_dim
        )
        for question in questions:
            vector = await embed_query(providers.embedder, question.question)
            fetched.append(
                (
                    question,
                    store.vector_search(vector, settings.search_top_k),
                    store.keyword_search(question.question, settings.search_top_k),
                )
            )
    finally:
        await providers.aclose()
        await client.aclose()

    print(
        f"настроечная часть: {len(questions)} вопросов, "
        f"отложенная: {len(held_out)}, глубина списков {settings.search_top_k}"
    )
    print(
        f"условие «одиночка может выиграть по k»: k < {settings.search_top_k - 2}. "
        f"Но одной k мало:\n  вклады СКЛАДЫВАЮТСЯ, и пара мест (9, 2) бьёт "
        f"первое место при любом разумном k.\n  Поэтому перебираются ещё вес "
        f"второго списка и гарантия для лучшего по смыслу."
    )
    print()

    def measure(rrf_k: int, keyword_weight: float, promote: bool) -> dict:
        scores = []
        for question, vector_hits, keyword_hits in fetched:
            result = fuse(
                store=store,
                vector_hits=vector_hits,
                keyword_hits=keyword_hits,
                top_k=settings.search_top_k,
                floor=settings.similarity_floor,
                rrf_k=rrf_k,
                keyword_weight=keyword_weight,
                promote_best_vector=promote,
            )
            context = (
                build_context(
                    result.hits,
                    token_budget=settings.context_token_budget,
                    max_fragments=settings.context_max_fragments,
                )
                if result.passed_floor
                else None
            )
            retrieved_docs: list[str] = []
            for hit in result.hits:
                if hit.chunk.doc_id not in retrieved_docs:
                    retrieved_docs.append(hit.chunk.doc_id)
            scores.append(
                score_retrieval(
                    retrieved_docs=retrieved_docs,
                    relevant_docs=set(question.docs),
                    context_text="\n".join(f.body for f in context.fragments) if context else "",
                    must_contain=question.must_contain,
                    best_cosine=result.best_vector_score,
                    passed_floor=result.passed_floor,
                    chunk_texts=[hit.chunk.body for hit in result.hits],
                    cosines=[h.vector_score for h in result.hits if h.vector_score is not None],
                )
            )
        return {
            "k": rrf_k,
            "w": keyword_weight,
            "promote": promote,
            "context_hit": mean([1.0 if s.context_hit else 0.0 for s in scores]),
            "chunk_top1": mean([1.0 if s.chunk_rank == 1 else 0.0 for s in scores]),
            "chunk_mrr": mean([s.chunk_mrr for s in scores]),
            "recall@5": mean([s.recall_at_5 for s in scores]),
            "ndcg@10": mean([s.ndcg_at_10 for s in scores]),
        }

    rows = [
        measure(rrf_k, weight, promote)
        for rrf_k in ks
        for weight in weights
        for promote in promotions
    ]

    header = (
        f"  {'k':>4} {'вес кл.':>8} {'гарант':>7}  {'контекст':>8} "
        f"{'чанк-1':>7} {'чанк-mrr':>9} {'recall@5':>8} {'ndcg@10':>8}"
    )

    def line(row: dict) -> str:
        return (
            f"  {row['k']:>4} {row['w']:>8.2f} {('да' if row['promote'] else 'нет'):>7}  "
            f"{row['context_hit']:>8.3f} {row['chunk_top1']:>7.3f} {row['chunk_mrr']:>9.3f} "
            f"{row['recall@5']:>8.3f} {row['ndcg@10']:>8.3f}"
        )

    current = next(
        (
            row for row in rows
            if row["k"] == settings.rrf_k
            and row["w"] == settings.keyword_weight
            and not row["promote"]
        ),
        None,
    )
    if current is not None:
        print("СЕЙЧАС В НАСТРОЙКАХ")
        print(header)
        print(line(current))
        print()

    # Сортируем по «дошёл» и лишь потом по «дошёл высоко». Порядок целей
    # важен: сначала нужный текст обязан доехать, и только потом имеет смысл
    # спорить о его месте.
    rows.sort(key=lambda row: (row["context_hit"], row["chunk_mrr"], row["chunk_top1"]), reverse=True)
    print(f"ЛУЧШИЕ {min(args.show, len(rows))} из {len(rows)} вариантов")
    print(header)
    for row in rows[: args.show]:
        print(line(row))

    best = rows[0]

    # ПРОВЕРКА ЛУЧШЕГО ВАРИАНТА НА ОТЛОЖЕННОЙ ЧАСТИ.
    #
    # Перебор из пятидесяти вариантов на одном наборе — это подгонка по
    # построению: где-то среди пятидесяти всегда окажется тот, кому повезло на
    # этих вопросах. Единственная честная проверка — посчитать выбранное на
    # вопросах, которые в выборе не участвовали.
    if held_out:
        keep = fetched
        fetched = []
        client2 = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0)
        )
        providers2 = build_registry(settings, client2)
        try:
            for question in held_out:
                vector = await embed_query(providers2.embedder, question.question)
                fetched.append(
                    (
                        question,
                        store.vector_search(vector, settings.search_top_k),
                        store.keyword_search(question.question, settings.search_top_k),
                    )
                )
        finally:
            await providers2.aclose()
            await client2.aclose()

        checked_best = measure(best["k"], best["w"], best["promote"])
        checked_now = measure(settings.rrf_k, settings.keyword_weight, False)
        fetched = keep

        print()
        print("ОТЛОЖЕННАЯ ЧАСТЬ (в выборе не участвовала)")
        print(header)
        print(line(checked_now) + "   ← текущая настройка")
        print(line(checked_best) + "   ← лучшая по настроечной")
        if checked_best["chunk_mrr"] < checked_now["chunk_mrr"]:
            print("  ВНИМАНИЕ: на отложенной части «лучший» вариант ХУЖЕ текущего.")
            print("  Значит его преимущество состояло из особенностей настроечной")
            print("  выборки. Менять настройку на таком основании нельзя.")

    print()
    if current is not None and (
        best["context_hit"] > current["context_hit"]
        or (best["context_hit"] == current["context_hit"]
            and best["chunk_mrr"] > current["chunk_mrr"])
    ):
        print(
            f"лучше текущего: k={best['k']}, вес ключевого={best['w']}, "
            f"гарантия={'да' if best['promote'] else 'нет'}"
        )
        print("  Прежде чем менять настройки: это перебор на ТОМ ЖЕ наборе, то есть")
        print("  подгонка под 67 вопросов. Разница в один-два вопроса тут ничего не")
        print("  значит — смотреть надо на разницу, которая держится на нескольких")
        print("  соседних значениях k, а не на выигрыш одного варианта.")
    else:
        print("текущая настройка не хуже всех перебранных — менять нечего")
    store.close()
    return 0


async def _sensitivity(args) -> int:
    """Чувствительность приборов: ловят ли они ЗАВЕДОМО неверные ответы.

    Команда отвечает на вопрос, на который каппа на нашей системе ответить не
    может. Каппа измеряет согласие с человеком сверх случайного, и для этого
    ей нужны оба класса в разумной пропорции. У нас 36 «верно» против 2
    «неверно»: система права в 95 % случаев, и случайная выборка
    сбалансированной не будет никогда. Каппа на таком перекосе формально
    считается и выдаёт 0.2 — цифру, которую легко прочитать как «приборы
    плохи», хотя означает она «мерить нечем».

    Поэтому задача переворачивается. Неверные ответы делаем сами из верных —
    заменяем отказом, оставляем эхо вопроса, меняем число, несущее факт,
    переворачиваем отрицание и дописываем к верному ответу ложное
    утверждение. Про каждый такой ответ мы ТОЧНО знаем, что он неверен, и
    спрашиваем приборы, поднимут ли они тревогу.

    Аналогия: пожарный датчик нельзя проверить, сидя в комнате и отмечая, что
    он молчит. Его проверяют дымом — своим, заранее известным.

    Ключевой вид порчи — ПРИПИСКА: верный ответ плюс ложное утверждение
    рядом. Проверка подстрокой к ней слепа по построению (все нужные слова на
    месте), и именно на ней решается, окупает ли судья свои вызовы.

    Оговорка, без которой цифры соврут: чувствительность к подделкам говорит,
    что прибор не слеп. Она НЕ говорит, как часто он ошибается на настоящем
    потоке. Это два разных вопроса, и смешивать их нельзя.
    """
    settings = get_settings()
    report_env_overrides(settings)
    path = resolve_run(args.run, mode="answer")
    if path is None:
        return run_not_found(args.run, mode="answer")

    run = load_run(path)
    questions = {question.id: question for question in dataset.load()}
    # И здесь тот же допуск. Замер чувствительности берёт ВЕРНЫЕ ответы и портит
    # их, проверяя, заметят ли приборы порчу. На прогоне без нашей схемы ответа
    # старое условие не находило ни одного верного ответа — то есть проверить
    # приборы на чужом стеке было нельзя, и команда сообщала об этом так, будто
    # система не ответила верно ни разу.
    rows = [
        row
        for row in run["rows"]
        if judgeable(row)
        and row.get("answer_contains") is True
        and questions.get(row["question_id"]) is not None
    ][: args.limit]

    if not rows:
        print("нет верных ответов, из которых можно сделать порчу")
        return 1

    store = Store(settings.db_path)
    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0))
    judge = Judge(
        provider=_judge_provider(settings, client),
        model=settings.judge_model,
        store=store,
        think=settings.judge_think,
        max_tokens=settings.judge_max_tokens,
    )

    # Итог: по каждому виду порчи — сколько поймала подстрока и сколько судья.
    stats: dict[str, dict[str, int]] = {}
    misses: list[tuple[str, str, str]] = []

    try:
        for row in rows:
            question = questions[row["question_id"]]
            for mutant in mutate.mutants_for(
                question=question.question,
                answer=row["answer"],
                needles=question.answer_needles,
            ):
                bucket = stats.setdefault(
                    mutant.kind, {"всего": 0, "подстрока": 0, "судья": 0}
                )
                bucket["всего"] += 1

                if answer_contains(mutant.answer, question.answer_needles) is False:
                    bucket["подстрока"] += 1

                verdict = await judge.verdict(question=question, answer=mutant.answer)
                if verdict.correct is False:
                    bucket["судья"] += 1
                elif verdict.correct is True:
                    misses.append((question.id, mutant.kind, verdict.reason[:120]))
    finally:
        await client.aclose()
        store.close()

    print(f"прогон: {path.name}")
    print(f"верных ответов взято: {len(rows)}, порч сделано: "
          f"{sum(b['всего'] for b in stats.values())}")
    print()
    print(f"  {'вид порчи':<12} {'всего':>6} {'подстрока':>10} {'судья':>7}")
    for kind, bucket in stats.items():
        print(
            f"  {kind:<12} {bucket['всего']:>6} "
            f"{bucket['подстрока']:>10} {bucket['судья']:>7}"
        )

    appended = stats.get("приписка")
    if appended and appended["всего"]:
        print()
        print("ГЛАВНЫЙ ВОПРОС — ПРИПИСКА (верный ответ плюс ложь рядом):")
        print(
            f"  подстрока поймала {appended['подстрока']} из {appended['всего']} "
            f"(она слепа к этому по построению),"
        )
        print(f"  судья поймал {appended['судья']} из {appended['всего']}.")
        if appended["судья"] > appended["подстрока"]:
            print("  Судья видит то, чего не видит подстрока — вызовы окупает.")
        else:
            print("  Судья не видит и этого. Тогда платить за него нечем:")
            print("  бесплатная проверка справляется не хуже, и он уходит из гейта.")

    if misses:
        print()
        print(f"СУДЬЯ НЕ ЗАМЕТИЛ ПОРЧУ ({len(misses)}) — читать по одному:")
        for question_id, kind, reason in misses[:12]:
            print(f"  {question_id}  {kind:<11} {reason}")
    return 0


async def _why(args) -> int:
    """Разбор одного вопроса или всех неудачных: ПОЧЕМУ фрагмент оказался там.

    Команда отвечает на вопрос, на который метрика ответить не может. Метрика
    говорит «нужный фрагмент не первый в каждом пятом вопросе», а под этой
    цифрой лежат четыре разные болезни с четырьмя разными лекарствами. Ставить
    реранкер, не разделив их, — менять масло в машине, у которой кончился
    бензин.
    """
    settings = get_settings()
    report_env_overrides(settings)
    store = Store(settings.db_path)
    questions = {question.id: question for question in dataset.load()}

    if args.question:
        chosen = [questions[qid] for qid in args.question if qid in questions]
        missing = [qid for qid in args.question if qid not in questions]
        if missing:
            print(f"нет таких вопросов: {', '.join(missing)}")
        if not chosen:
            return 1
    else:
        # Без аргументов разбираем все отвечаемые вопросы: сводка диагнозов
        # важнее одного случая, потому что инструмент выбирают по большинству.
        chosen = [question for question in questions.values() if question.answerable]

    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0))
    providers = build_registry(settings, client)
    verdicts: list[tuple[str, explain.Verdict]] = []
    try:
        store.assert_compatible(
            embed_model=settings.ollama_embed_model, embed_dim=settings.ollama_embed_dim
        )
        for question in chosen:
            vector = await embed_query(providers.embedder, question.question)
            vector_hits = store.vector_search(vector, settings.search_top_k)
            keyword_hits = store.keyword_search(question.question, settings.search_top_k)
            result = await hybrid_search(
                store=store,
                query=question.question,
                query_vector=vector,
                top_k=settings.search_top_k,
                floor=settings.similarity_floor,
                rrf_k=settings.rrf_k,
            )
            context = (
                build_context(
                    result.hits,
                    token_budget=settings.context_token_budget,
                    max_fragments=settings.context_max_fragments,
                )
                if result.passed_floor
                else None
            )
            context_chunks = (
                {fragment.chunk_id for fragment in context.fragments} if context else set()
            )
            needle_chunks = [
                chunk.chunk_id
                for chunk in store.all_chunks()
                if explain.chunk_has_needles(chunk, question.must_contain)
            ]
            verdict = explain.diagnose(
                needle_chunks=needle_chunks,
                vector_ranks={cid: r for r, (cid, _) in enumerate(vector_hits, start=1)},
                keyword_ranks={cid: r for r, (cid, _) in enumerate(keyword_hits, start=1)},
                fused_ranks={
                    hit.chunk.chunk_id: r for r, hit in enumerate(result.hits, start=1)
                },
                context_chunks=context_chunks,
            )
            verdicts.append((question.id, verdict))

            # Подробный разбор печатаем только когда вопросы названы руками
            # или когда поиск не справился: иначе семьдесят семь разборов
            # заливают консоль, и сводка в них теряется.
            if args.question or verdict.code != "поиск справился":
                print(
                    explain.explain(
                        question_id=question.id,
                        question=question.question,
                        needles=question.must_contain,
                        store=store,
                        vector_hits=vector_hits,
                        keyword_hits=keyword_hits,
                        result=result,
                        context_chunks=context_chunks,
                        top=args.top,
                    )
                )
                print()
    finally:
        await providers.aclose()
        await client.aclose()
        store.close()

    if len(verdicts) > 1:
        print(explain.summary_table(verdicts))
    return 0


async def _answer(args) -> int:
    settings = get_settings()
    report_env_overrides(settings)
    store = Store(settings.db_path)
    questions = dataset.load()
    if args.limit:
        questions = questions[: args.limit]

    # Версия промпта выбирается ЯВНО и передаётся в конвейер объектом.
    #
    # Раньше перестановка правил патчила атрибут модуля конвейера. Работало,
    # но держалось на том, что никто не импортирует константу по значению, —
    # а это не свойство кода, а везение. Теперь версия передаётся как
    # аргумент, и подменять глобальное состояние не нужно вовсе.
    try:
        spec = prompt_module.resolve(getattr(args, "prompt", "") or settings.prompt_name)
    except ValueError as error:
        print(f"ОШИБКА: {error}")
        store.close()
        return

    if args.nudge:
        # Смысл тот же, байты другие. Отпечаток обязан отличаться, иначе мы
        # сами себе подложим два разных прогона под одним именем.
        spec = prompt_module.PromptSpec(
            name=spec.name + "-nudge",
            system=_nudge_prompt(spec.system),
            schema=spec.schema,
            note="перестановка двух независимых правил: замер чувствительности",
            # Отпечаток заморозки здесь не нужен: версия временная, живёт
            # один прогон и в реестр не попадает.
            frozen_at="",
        )
        print(f"ПРОМПТ ПЕРЕСТАВЛЕН: {spec.version}")
        print("  смысл тот же, байты другие — это замер чувствительности, не улучшение")
    elif spec.version != prompt_module.PROMPT_VERSION:
        print(f"ПРОМПТ: {spec.version}  ({spec.note})")

    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0))
    providers = build_registry(settings, client)
    pipeline = Pipeline(
        settings=settings,
        store=store,
        providers=providers,
        traces=TraceWriter(settings.trace_dir),
        # Клиент нужен и здесь: иначе прогон с агентом ходил бы в вики, но
        # не в сервис живых данных — то есть измерял бы не ту систему,
        # которая работает у пользователя.
        http=client,
        prompt=spec,
    )

    # Просьбу включить агента проверяем ЗДЕСЬ и говорим вслух, если нельзя.
    #
    # Молча прогнать без агента было бы худшим из вариантов: прогон
    # сохранился бы с пометкой «агент» в шапке, а работала бы цепочка — то
    # есть в журнал легло бы измерение системы, которой не существовало.
    use_agent = bool(getattr(args, "agent", False))
    if use_agent and not settings.agent_enabled:
        print("АГЕНТ ЗАПРОШЕН, НО ВЫКЛЮЧЕН НА СЕРВЕРЕ: поставьте PW_AGENT_ENABLED=true")
        print("  Прогон остановлен: иначе в журнал попадёт прогон под чужой подписью.")
        store.close()
        await client.aclose()
        return

    # ОБРАТНАЯ ПРОВЕРКА, и она важнее прямой.
    #
    # В наборе есть вопросы, ответом на которые служат живые данные сервиса.
    # Прогнать их БЕЗ агента можно — и получится отчёт, в котором
    # `tools_ok 0.000` выглядит как приговор системе, хотя инструменты никто
    # не включал. Такой отчёт хуже отсутствия отчёта: он выглядит
    # измерением.
    #
    # Мы на это уже попались: первый же прогон агентского набора ушёл без
    # флага `--agent`, и двенадцать вопросов доложили о провале, которого не
    # было.
    live = [question for question in questions if question.type == "live"]
    if live and not use_agent:
        print(
            f"В НАБОРЕ {len(live)} ВОПРОСОВ ПРО ЖИВЫЕ ДАННЫЕ, А АГЕНТ НЕ ВКЛЮЧЁН."
        )
        print("  Без него инструменты не вызываются, и tools_ok будет нулём —")
        print("  не потому, что система плоха, а потому, что её не спрашивали.")
        print("  Добавьте --agent или исключите эти вопросы фильтром.")
        store.close()
        await client.aclose()
        return

    judge = None
    if args.with_judge:
        # Судья в одном прогоне с генерацией держит в памяти ДВЕ модели. На
        # одной видеокарте это валит llama-server с ошибкой CUDA, поэтому по
        # умолчанию судейство вынесено во второй проход: scripts/eval.py judge.
        if settings.judge_model == settings.ollama_chat_model:
            print(
                f"ВНИМАНИЕ: судья и отвечающий — одна модель ({settings.judge_model}). "
                f"Это завышает метрику: модель систематически одобряет свои же ответы."
            )
        judge = Judge(
            provider=_judge_provider(settings, client),
            model=settings.judge_model,
            store=store,
            think=settings.judge_think,
            max_tokens=settings.judge_max_tokens,
        )

    config = make_config(
        label=args.label, mode="answer", settings=settings, store=store, note=args.note,
        # Реестр передаётся, чтобы в шапку попала та модель, которая
        # отвечала, а не та, что прописана у локального провайдера.
        providers=providers,
        agent=use_agent,
        prompt_version=spec.version,
    )
    path = runs_dir() / (
        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(config.started_at))}"
        f"-answer-{args.label}.json"
    )
    config_dict = asdict(config)

    def checkpoint(number: int, total: int, rows: list) -> None:
        """Промежуточное сохранение.

        Прогон на семьдесят семь вопросов идёт минуты, и падение на середине
        не должно стоить всей работы. Файл пишется после каждого вопроса — это
        дешевле, чем потом гадать, докуда дошло.
        """
        write_run(path, config=config_dict, rows=rows)
        if number % 10 == 0 or number == total:
            print(f"  {number}/{total} вопросов")

    try:
        rows = await run_answer(
            questions, pipeline=pipeline, judge=judge, on_progress=checkpoint,
            agent=use_agent,
        )
    finally:
        # Освобождаем видеопамять от отвечающей модели: следующим шагом,
        # скорее всего, пойдёт судья, и вдвоём они не поместятся.
        release = getattr(providers.get("ollama"), "release", None)
        if release:
            await release(settings.ollama_chat_model)
        # Свои клиенты провайдеров закрываем отдельно: у облачного свой
        # список доверенных сертификатов, и общий клиент про него не знает.
        # Без этого после каждого прогона остаётся открытое соединение.
        await providers.aclose()
        await client.aclose()

    write_run(path, config=config_dict, rows=rows)
    store.close()

    failures = [row for row in rows if row.error]
    print(report.summarize(load_run(path)))
    if failures:
        print(f"ОШИБОК НА ВОПРОСАХ: {len(failures)}")
        for row in failures[:10]:
            print(f"  {row.question_id}  {row.error[:110]}")
        print()
    print(f"прогон сохранён: {path}")
    if judge is None:
        print(f"\nтеперь судейство вторым проходом:\n  python scripts/eval.py judge {path}")
    return 0


async def _judge(args) -> int:
    """Второй проход: судья по сохранённым ответам.

    Разнесение этапов — требование железа, а не стиля: две модели на одной
    видеокарте не помещаются. Проход идемпотентный, прерванное судейство можно
    запустить снова.
    """
    settings = get_settings()
    report_env_overrides(settings)
    path = resolve_run(args.run, mode="answer")
    if path is None:
        return run_not_found(args.run, mode="answer")
    print(f"прогон: {path.name}")
    run = load_run(path)
    rows = rows_from_run(run)
    questions = dataset.load()

    if settings.judge_model == settings.ollama_chat_model:
        print(
            f"ВНИМАНИЕ: судья и отвечающий — одна модель ({settings.judge_model}). "
            f"Метрика будет завышена: модель одобряет свои же ответы."
        )

    store = Store(settings.db_path)
    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0))
    provider = _judge_provider(settings, client)

    # ПРЕДПРОВЕРКА до начала работы. Судейство идёт минуты; выяснять, что
    # модель не установлена, на первом же вопросе — значит потратить время
    # впустую и получить непонятную ошибку вместо внятной. Дёшево проверить
    # заранее то, без чего дальше нельзя.
    health = await provider.health()
    if health.get("ok") and not health.get("chat_model_present"):
        installed = health.get("models") or []
        print(f"модель судьи не установлена: {settings.judge_model}")
        print("установлено на этой машине:")
        for name in installed:
            print(f"  {name}")
        print()
        print("поставить другую модель судьи можно в backend\\.env:")
        print("  PW_JUDGE_MODEL=<имя из списка выше>")
        print()
        print("судить той же моделью, что отвечала, тоже допустимо — но тогда")
        print("цифра завышена, и это надо помнить: модель одобряет свои ответы.")
        await client.aclose()
        store.close()
        return 1
    if not health.get("ok"):
        print(f"Ollama недоступна: {health.get('error', '')[:200]}")
        await client.aclose()
        store.close()
        return 1

    # Сначала выгружаем отвечающую модель, потом поднимаем судью.
    from app.providers.ollama import OllamaProvider

    await OllamaProvider(settings, client).release(settings.ollama_chat_model)

    if args.retry_unresolved:
        # Сбрасываем причину — и строка снова становится «не судимой».
        # Повторять по умолчанию нельзя: неразобранный вердикт обычно
        # воспроизводится, и тогда каждый проход тратил бы время заново.
        retried = 0
        for row in rows:
            if row.judge_verdict is None and row.judge_reason:
                row.judge_reason = ""
                retried += 1
        print(f"сброшено для повтора: {retried}")

    judge = Judge(
        provider=provider,
        model=settings.judge_model,
        store=store,
        think=settings.judge_think,
        max_tokens=settings.judge_max_tokens,
    )

    if args.probe:
        # Пробный вызов на ОДНОМ вопросе с печатью сырого ответа.
        #
        # Дешёвая проверка связки перед семьюдесятью семью вызовами. Судейство
        # идёт минуты; выяснять на нём, что модель отдаёт не тот формат, —
        # значит потратить эти минуты, чтобы получить одну и ту же ошибку
        # шестьдесят семь раз.
        target = next((row for row in rows if judgeable(row)), None)
        if target is None:
            print("в прогоне нет ответов для пробы")
        else:
            question = next(q for q in questions if q.id == target.question_id)
            print(f"пробный вопрос: {target.question_id}  {target.question}")
            verdict = await judge.verdict(question=question, answer=target.answer)
            print(f"модель судьи: {settings.judge_model}  "
                  f"размышления={settings.judge_think}  "
                  f"лимит токенов={settings.judge_max_tokens}")
            print(f"вердикт: {verdict.correct!r}")
            print(f"причина: {verdict.reason}")
            print(f"диагностика: {verdict.raw or '(пусто)'}")
        await provider.release(settings.judge_model)
        await client.aclose()
        store.close()
        return 0

    # Условие допуска — ТА ЖЕ функция, которой пользуется сам проход.
    #
    # Раньше оно было переписано здесь копией, и копия разошлась бы с
    # оригиналом при первой же правке: счётчик обещал бы одно число, а проход
    # оценивал другое. Тот же случай, что и с константой в двух местах — оба
    # экземпляра по отдельности выглядят правильными.
    pending = sum(
        1
        for row in rows
        if row.judge_verdict is None and not row.judge_reason and judgeable(row)
    )
    print(f"судья {settings.judge_model}: к оценке {pending} ответов")
    if not pending:
        # Ноль к оценке — это новость, а не тишина.
        #
        # Прошлая версия печатала «к оценке 0 ответов» и молча возвращала
        # успех. Ровно так `judge alt1` не оценил ни одной строки второй
        # реализации, и выглядело это как выполненная работа.
        already = sum(1 for row in rows if row.judge_verdict is not None)
        print(
            f"  оценивать нечего: уже с вердиктом {already} из {len(rows)}."
            if already
            else "  ВНИМАНИЕ: судье не досталось НИ ОДНОЙ строки. Это не "
                 "«всё оценено»,\n  а прогон, который судья целиком не увидел: "
                 "проверьте статусы и ошибки."
        )

    def checkpoint(number: int, total: int, current: list) -> None:
        write_run(path, config=run["config"], rows=current)

    try:
        judged = await judge_rows(
            rows, questions=questions, judge=judge, on_progress=checkpoint
        )
    finally:
        await provider.release(settings.judge_model)
        await client.aclose()

    write_run(path, config=run["config"], rows=rows)
    store.close()

    print(f"оценено в этот раз: {judged}")
    print(report.summarize(load_run(path)))
    unresolved = [
        row for row in rows if row.judge_verdict is None and row.judge_reason and row.answerable
    ]
    if unresolved:
        print(f"судья не смог вынести вердикт в {len(unresolved)} случаях:")
        for row in unresolved[:5]:
            print(f"  {row.question_id}  {row.judge_reason[:80]}")
            if row.judge_raw:
                # Сырой ответ печатается сразу: без него «не разобран» —
                # это сообщение без информации.
                print(f"    что прислал судья: {row.judge_raw[:220]!r}")
        print(
            "  Эти строки НЕ засчитаны как «неверно» — они вообще не попали в "
            "judge_ok.\n"
            "  Чтобы попробовать снова, очисти их причину: "
            "python scripts/eval.py judge <прогон> --retry-unresolved"
        )
    print(f"\nдальше калибровка судьи на людях:\n"
          f"  python scripts/eval.py label {path}\n"
          f"  python scripts/eval.py kappa {path}")
    return 0


def _judge_provider(settings, client):
    """Провайдер для судьи: тот же Ollama, но с другой моделью.

    Подменяем модель копией настроек, а не флагом внутри провайдера: провайдер
    не должен знать, что кто-то его использует для судейства.
    """
    from app.providers.ollama import OllamaProvider

    judge_settings = settings.model_copy(update={"ollama_chat_model": settings.judge_model})
    return OllamaProvider(judge_settings, client)


def run_namespace(parser, **overrides):
    """Аргументы прогона — из НАСТОЯЩЕГО парсера команды, а не собранные руками.

    Зачем отдельная функция, если `argparse.Namespace(...)` короче.

    Потому что короткий вариант уже сломался. `noise` собирал набор аргументов
    вручную — метка, пометка, лимит, судья, — и вызывал им `_answer`. Потом у
    команды `answer` появился флаг `--nudge`, `_answer` начал его читать, а
    рукописный набор про него не знал. Результат: `noise --mode answer` падал с
    `AttributeError` на первом же прогоне.

    Сломалось при этом не абы что. `noise` — единственный источник порога
    значимости, и в режиме ответа он не работал вовсе. То есть порог для
    метрик ответа был не «решили пока не мерить», а «мерить было нечем», и
    `compare` всё это время подставлял в ответную половину таблицы ноль,
    измеренный на поисковом прогоне.

    Это та же болезнь, что и с константой в двух местах: **рукописная копия
    чужого интерфейса однажды разойдётся с оригиналом, и каждая половина по
    отдельности будет выглядеть правильной.** Лечится не добавлением
    недостающего поля, а устранением копии: значения по умолчанию берутся у
    самого парсера, поэтому следующий новый флаг приедет сюда сам.

    Подмены проверяются по имени: опечатка или переименованный флаг роняют
    вызов здесь, а не тихо уходят в прогон со значением по умолчанию. Молчаливо
    проигнорированная подмена — это прогон, который измерил не то, что просили,
    и об этом не сказал.
    """
    namespace = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(namespace, key):
            raise SystemExit(
                f"внутренняя ошибка: у команды нет аргумента «{key}». "
                "Похоже, флаг переименовали, а подмену в run_namespace не поправили."
            )
        setattr(namespace, key, value)
    return namespace


async def _noise(args) -> int:
    """Несколько прогонов БЕЗ изменений. Разброс между ними — порог значимости."""
    runs: list[dict] = []
    for number in range(1, args.runs + 1):
        print(f"--- прогон {number} из {args.runs} ---")
        overrides = {
            "label": f"noise{number}",
            "note": "замер разброса",
            "limit": args.limit,
        }
        if args.mode == "answer":
            # Судья в замере разброса не участвует — и это ограничение самого
            # замера, а не забывчивость: измеренный здесь порог годится для
            # `answer_contains` и `language_ok`, но не для `judge_ok`.
            overrides["with_judge"] = False
        namespace = run_namespace(args.run_parsers[args.mode], **overrides)
        code = await (_answer(namespace) if args.mode == "answer" else _search(namespace))
        if code != 0:
            return code
        latest = sorted(runs_dir().glob(f"*-{args.mode}-noise{number}.json"))[-1]
        runs.append(load_run(latest))

    # ПРИБАВЛЯЕМ УЖЕ СДЕЛАННЫЕ ПРОГОНЫ ТОЙ ЖЕ КОНФИГУРАЦИИ.
    #
    # Урок из живого замера. Три прогона подряд дали 0.967, 0.967, 0.967 —
    # разброс ноль. А прогон «опоры», сделанный десятью минутами раньше с
    # ТЕМ ЖЕ отпечатком кода и промпта, дал 0.951. Четвёртая точка той же
    # конфигурации лежала в той же папке, и её просто не посмотрели.
    #
    # Из-за этого порог вышел заниженным ровно там, где он важнее всего:
    # три совпавших значения выглядят как «метрика детерминирована», и
    # следующая разница в два вопроса будет объявлена значимой.
    #
    # Правило общее: оценка разброса по трём точкам — сама по себе
    # случайная величина, и выбрасывать имеющиеся точки нельзя. Тем более
    # что они уже оплачены.
    reference = runs[0]["config"]
    for path in sorted(runs_dir().glob(f"*-{args.mode}-*.json")):
        if any(path.name.endswith(f"-noise{number}.json") for number in range(1, args.runs + 1)):
            continue
        other = load_run(path)
        if report.config_diff(other["config"], reference):
            continue
        runs.append(other)
        print(f"  + прежний прогон той же конфигурации: {other['config']['label']}")

    print()
    print(report.noise_report(runs))
    return 0


def cmd_calibrate(args) -> int:
    path = resolve_run(args.run, mode="search")
    if path is None:
        return run_not_found(args.run, mode="search")

    run = load_run(path)
    from eval.runner import RowResult

    rows = rows_from_run(run)

    # ПОРОГ ВЫБИРАЕТСЯ ТОЛЬКО ПО НАСТРОЕЧНОЙ ЧАСТИ.
    #
    # Это последний незакрытый пункт аудита. Раньше порог перебирался по всему
    # набору и на нём же докладывался результат — то есть выбиралось значение,
    # наилучшее для этих самых вопросов. Из десятка вариантов такой всегда
    # найдётся, и его преимущество может целиком состоять из особенностей
    # выборки.
    #
    # Аналогия: составить контрольную по разобранным на уроке задачам и
    # объявить высокие оценки доказательством знаний. Оценки настоящие, вывод
    # ложный.
    splits = {question.id: question.split for question in dataset.load()}
    tune_rows = [row for row in rows if splits.get(row.question_id, "tune") == "tune"]
    holdout_rows = [row for row in rows if splits.get(row.question_id) == "holdout"]

    points = calibrate_mod.sweep(tune_rows)
    recommended = calibrate_mod.recommend(points, min_answerable=args.min_answerable)

    print(f"калибровка по прогону {path.name}")
    print(
        f"настроечная часть: {len(tune_rows)} вопросов "
        f"(отвечаемых {sum(1 for r in tune_rows if r.answerable)}), "
        f"отложенная: {len(holdout_rows)}"
    )
    print()
    print(calibrate_mod.render(points, recommended=recommended))
    print()
    print("рекомендация (выбрана на настроечной части):",
          json.dumps(recommended.as_dict(), ensure_ascii=False))

    # И сразу проверка на отложенной: если там выигрыш исчез, выигрыша не было.
    if holdout_rows:
        checked = calibrate_mod.sweep(holdout_rows)
        same = next((p for p in checked if abs(p.floor - recommended.floor) < 1e-9), None)
        print()
        if same is None:
            print("на отложенной части этот порог не проверить: нет такой точки в переборе")
        else:
            print("ПРОВЕРКА НА ОТЛОЖЕННОЙ ЧАСТИ (её не было в выборе):")
            print(f"  {json.dumps(same.as_dict(), ensure_ascii=False)}")
            print("  Сильно хуже, чем на настроечной, — значит порог подогнан,")
            print("  а не найден. Ровно это и есть смысл отложенной части.")
    print(f"\nтекущее значение в конфиге: {run['config']['similarity_floor']}")
    print(f"поставить в .env:  PW_SIMILARITY_FLOOR={recommended.floor:.2f}")
    print(
        "\nПорог — продуктовое решение, а не математическое: параметр "
        "--min-answerable задаёт, сколько пользы допустимо потерять ради "
        "защиты от выдумок."
    )
    return 0


def cmd_show(args) -> int:
    path = resolve_run(args.run, mode=args.mode)
    if path is None:
        return run_not_found(args.run, mode=args.mode)
    print(f"файл: {path.name}")
    print()
    print(report.summarize(load_run(path)))
    return 0


def cmd_recheck(args) -> int:
    """Пересчитывает дешёвые проверки по УЖЕ сохранённым ответам.

    Зачем отдельная команда. Прогон с генерацией стоит модели и времени, а
    новая проверка ответа — нет: текст ответов уже лежит в файле прогона.
    Поэтому новую метрику не надо «домеривать» повторным прогоном, иначе она
    окажется измерена на другой генерации, и сравнивать будет нельзя.

    Правило общее: всё, что можно посчитать из сохранённого прогона, считается
    из файла, а модель дёргается только за тем, чего в файле нет.
    """
    path = resolve_run(args.run, mode="answer")
    if path is None:
        return run_not_found(args.run, mode="answer")
    run = load_run(path)
    print(f"прогон: {path.name}")
    if run["config"]["mode"] != "answer":
        print("пересчитывать нечего: это поисковый прогон, в нём нет ответов")
        return 1

    questions = {question.id: question for question in dataset.load()}
    rows = rows_from_run(run)
    changed = 0
    # Отдельный счётчик: сколько вердиктов «верно» СНЯТО по языку. Это не
    # деталь пересчёта, а изменение качества прогона, и печатать его надо
    # отдельной строкой, а не растворять в числе тронутых строк.
    overturned = 0
    for row in rows:
        # Язык досчитывается по ВСЕМ строкам, до отбора по отвечаемости.
        #
        # Это и есть главная польза пересчёта: метрика, добавленная сегодня,
        # появляется в прогонах, сделанных вчера. Без этого сравнить `alt1` с
        # `wide3` по языку было бы нельзя — новая цифра была бы только у
        # будущих прогонов, то есть измерена на другой генерации.
        touched = False

        language = answer_language(row.answer)
        if language != row.answer_language:
            row.answer_language = language
            touched = True

        question = questions.get(row.question_id)
        if question is not None and question.answerable:
            value = answer_contains(row.answer, question.answer_needles)
            if value != row.answer_contains:
                row.answer_contains = value
                touched = True

        # Вердикт по языку пересматривается ТЕМ ЖЕ кодом, что и у судьи.
        #
        # Правило «ответ не на языке корпуса — неверный» появилось позже, чем
        # были вынесены вердикты в сохранённых прогонах, а `judge` строки с
        # вердиктом пропускает: переcудить их он не может по построению.
        # Значит старые прогоны так и остались бы измерены прибором прежней
        # версии — и `judge_ok` в них был бы завышен на строках, которых никто
        # не видит.
        #
        # Пересчитывать это можно здесь ровно потому, что правило не требует
        # модели: оно детерминированное и считается по буквам. Всё, что можно
        # досчитать из сохранённого файла, считается из файла.
        #
        # Своей копии правила тут нет — вызывается та же функция, которой
        # пользуется судья. Копия разошлась бы: мы это в проекте уже проходили
        # четыре раза.
        # ТОЛЬКО по строкам, которые судья вообще судит.
        #
        # Без этой оговорки правило зацепило одиннадцать НЕОТВЕЧАЕМЫХ вопросов:
        # у образца ответы на них тоже не на русском, вердикта не было, и они
        # разом получили «неверно» — то есть попали в `judge_ok`, куда им
        # дороги нет. Правильность отказа проверяется статусом, без модели; для
        # образца она не измеряется вовсе. Записать туда вердикт значило бы
        # смешать две разные проверки и уронить `judge_ok` на строках, которые
        # он не про то.
        #
        # Я сделал ровно ту ошибку, которую чинил в стенде четыре раза подряд:
        # применил правило к множеству строк, для которого оно не определено.
        by_language = language_verdict(row.answer)
        if by_language is not None and judgeable(row) and row.judge_verdict is not False:
            was = row.judge_verdict
            row.judge_verdict = by_language.correct
            row.judge_reason = by_language.reason
            row.judge_fragment = 0
            row.judge_fragment_label = ""
            row.judge_raw = ""
            if was is True:
                overturned += 1
            touched = True

        # Счётчик считает СТРОКИ, а не срабатывания. Две изменившиеся проверки
        # в одной строке — это одна пересчитанная строка, иначе «пересчитано
        # 40» на прогоне из 113 вопросов означало бы неизвестно что.
        changed += int(touched)

    versions = dataset.label_versions()
    was_retrieval = run["config"].get("labels_retrieval")
    was_answer = run["config"].get("labels_answer")

    # Штампуем ТОЛЬКО ту половину, которую действительно пересчитали.
    #
    # answer_contains пересчитан из сохранённых ответов — эту половину можно
    # объявить новой честно. А context_hit, recall и chunk_rank считаются по
    # фрагментам, которых в файле нет: их пересчитать нечем, и если разметка
    # поиска изменилась, прогон надо делать заново. Штамповать общий отпечаток
    # здесь значило бы соврать, что весь прогон мерян новой линейкой.
    run["config"]["labels_answer"] = versions["answer"]

    # Отпечаток НАБОРА обновляем только когда разметка поиска та же.
    #
    # `dataset_version` покрывает набор целиком — и вопросы, и обе разметки.
    # Пока мы его не трогали, после пересчёта получалась полуправда: половина
    # отпечатков новая, половина старая, и `compare` показывал различие
    # конфигурации там, где менялась только та половина, которую мы честно
    # пересчитали. Но если изменилась разметка ПОИСКА, штамповать общий
    # отпечаток нельзя: он объявит новой линейкой весь прогон, включая
    # метрики, которые пересчитать нечем.
    if not was_retrieval or was_retrieval == versions["retrieval"]:
        run["config"]["dataset_version"] = dataset.dataset_version()

    write_run(path, config=run["config"], rows=rows)
    print(f"пересчитано строк: {changed}")
    if overturned:
        print(
            f"вердиктов «верно» снято по языку ответа: {overturned}"
            "\n  Это не ухудшение системы, а починка прибора: прежний судья"
            "\n  язык не проверял, и judge_ok был завышен на эти строки."
        )

    if was_answer and was_answer != versions["answer"]:
        print(f"разметка ответа: {was_answer} -> {versions['answer']}")
    elif not was_answer:
        print(f"разметка ответа: проставлен отпечаток {versions['answer']}")

    if was_retrieval and was_retrieval != versions["retrieval"]:
        print()
        print("ВНИМАНИЕ: изменилась разметка ПОИСКА (docs / must_contain).")
        print("  Её пересчитать из файла нельзя — context_hit, recall и место")
        print("  чанка считаются по фрагментам, которых в прогоне не сохранено.")
        print("  Этот прогон остаётся мерян старой линейкой поиска, и сравнивать")
        print("  его метрики поиска с новыми прогонами нельзя. Нужен новый прогон.")
    print()
    print(report.summarize(load_run(path)))
    return 0


def cmd_diagnose(args) -> int:
    """Разделимость: существует ли числовой признак, отличающий «ответа нет».

    Нужна после первого же прогона, если доля правильных отказов низкая: она
    отвечает на вопрос, лечится ли это порогом вообще, или порог — слабый
    рубеж и решение должна принимать модель.
    """
    path = resolve_run(args.run, mode="search")
    if path is None:
        return run_not_found(args.run, mode="search")

    run = load_run(path)
    print(f"диагностика по прогону {path.name}")
    print(f"эмбеддинги {run['config']['embed_model']}, порог {run['config']['similarity_floor']}")
    print()
    print(diagnose.analyze(run["rows"]))
    return 0


def cmd_compare(args) -> int:
    before_path = resolve_run(args.before, mode=args.mode)
    after_path = resolve_run(args.after, mode=args.mode)
    if before_path is None:
        return run_not_found(args.before, mode=args.mode)
    if after_path is None:
        return run_not_found(args.after, mode=args.mode)
    print(f"было:  {before_path.name}")
    print(f"стало: {after_path.name}")
    print()
    print(report.compare(load_run(before_path), load_run(after_path), noise=args.noise))
    return 0


def cmd_list(args) -> int:
    directory = runs_dir()
    if not directory.exists():
        print("прогонов ещё нет")
        return 0
    for path in sorted(directory.glob("*.json")):
        run = load_run(path)
        overall = run["aggregate"]["overall"]
        print(
            f"{path.name:<44} порог {run['config']['similarity_floor']:<5} "
            f"recall@5 {overall.get('recall@5', 0):.3f}  "
            f"контекст {overall.get('context_hit', 0):.3f}  "
            f"статус {overall.get('status_ok', 0):.3f}  "
            f"судья {_number(overall.get('judge_ok'))}"
        )
    return 0


def cmd_label(args) -> int:
    """Ручная разметка ответов — то, без чего судье нельзя доверять.

    Размечать надо 30–50 ответов и именно руками: смысл упражнения в том,
    чтобы узнать, совпадает ли судья с человеком, а не в том, чтобы получить
    ещё одну машинную оценку.

    Две вещи здесь сделаны нарочно, и обе — про то, чтобы разметка осталась
    независимой.

    ПЕРВОЕ: вердикт судьи по умолчанию НЕ показывается. Человек, который видит
    «судья сказал: верно» прежде, чем решить сам, соглашается с судьёй заметно
    чаще — это эффект привязки, он работает и на внимательных людях. А каппа
    измеряет именно НЕЗАВИСИМОЕ согласие: если человек подсматривал, каппа
    покажет не качество судьи, а силу подсказки. Это тот же дефект, что порядок
    полей в схеме, только вместо модели — человек: ярлык, увиденный до решения,
    определяет решение.

    ВТОРОЕ: вопросы перемешиваются, а не идут по порядку номеров. Иначе
    размеченные тридцать — это первые тридцать по идентификатору, то есть
    первые темы корпуса, а не срез набора. Перемешивание детерминированное
    (зерно из имени прогона): один и тот же прогон даёт один и тот же порядок,
    и разметку можно продолжить с того места, где остановился.
    """
    path = resolve_run(args.run, mode="answer")
    if path is None:
        return run_not_found(args.run, mode="answer")
    run = load_run(path)
    # Условие допуска — то же, что у судьи, и по той же причине.
    #
    # Разметка человеком существует ради каппы, а каппа сравнивает человека
    # именно с судьёй. Если человек и судья видят РАЗНЫЕ множества строк,
    # сравнивать нечего: у прогона образца судья оценил сто ответов, а разметка
    # не нашла ни одного — и печатала «в прогоне нет ответов для разметки», как
    # будто прогон пустой.
    rows = [row for row in run["rows"] if judgeable(row)]
    if not rows:
        print("в прогоне нет ответов для разметки")
        return 1

    import random

    random.Random(path.stem).shuffle(rows)

    if args.suspect_first:
        # Обогащённая выборка: подозрительные вперёд.
        #
        # Зачем это нужно. Система отвечает правильно примерно в девяти случаях
        # из десяти, поэтому случайные сорок ответов — это тридцать шесть
        # верных и четыре неверных. На таком перекосе каппа неинформативна:
        # случайное согласие само по себе огромно. Чтобы калибровать детектор,
        # в выборке нужны оба класса.
        #
        # Цена честности: каппа по обогащённой выборке — это НЕ каппа по всему
        # набору, и выдавать её за неё нельзя. Она отвечает на другой вопрос:
        # «согласен ли инструмент с человеком там, где вообще есть о чём
        # спорить». Для отладки детектора нужен как раз этот вопрос.
        def suspicious(row: dict) -> int:
            score = 0
            if row.get("answer_contains") is False:
                score += 2
            if row.get("judge_verdict") is False:
                score += 2
            if row.get("citations_failed"):
                score += 1
            if (row["retrieval"].get("chunk_rank") or 0) > 1:
                score += 1
            return score

        rows.sort(key=suspicious, reverse=True)
        print("порядок: сначала подозрительные (обогащённая выборка для каппы)")

    labels_dir().mkdir(parents=True, exist_ok=True)
    target = labels_dir() / f"{path.stem}.json"
    labels: dict[str, bool] = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}

    # Источник выдержек. Судья тут не вызывается — нужен только его разбор
    # документов, а он работает от индекса и модели не требует.
    questions = {question.id: question for question in dataset.load()}
    excerpt_source = None
    try:
        from eval.judge import Judge

        excerpt_source = Judge(
            provider=None, model="", store=Store(get_settings().db_path)
        )
    except Exception as error:  # индекса нет — размечаем без выдержек
        print(f"выдержки из документов недоступны ({error.__class__.__name__}), "
              f"размечаем без них")

    print(f"прогон: {path.name}")
    print(f"размечено ранее: {len(labels)} из {len(rows)}. y — верно, n — неверно, "
          f"s — пропустить, q — выйти и сохранить.")
    print("Вердикт судьи скрыт нарочно: увидев его, человек соглашается чаще, "
          "и каппа\nизмерит силу подсказки вместо качества судьи. А выдержка из "
          "документа\nпоказывается: это источник, по которому и надо судить, а не "
          "чьё-то мнение.\n")

    only = {qid.strip() for qid in args.only.split(",") if qid.strip()} if args.only else set()
    if only:
        # Переразметка названных вопросов: старая оценка стирается, иначе цикл
        # ниже пропустит их как уже размеченные. Нужна затем, что человек
        # ошибается — и должен иметь возможность вернуться к конкретному
        # вопросу, не переразмечая все сорок.
        for qid in only:
            labels.pop(qid, None)
        print(f"переразметка по списку: {', '.join(sorted(only))}\n")

    for row in rows:
        if only and row["question_id"] not in only:
            continue
        if row["question_id"] in labels or (not only and len(labels) >= args.limit):
            continue
        print("=" * 78)
        print(f"{row['question_id']}  [{row['type']}/{row['difficulty']}]  {row['question']}")
        # ВЫДЕРЖКА ИЗ ДОКУМЕНТА — человеку показываем, вердикт судьи скрываем.
        #
        # Разница между этими двумя вещами принципиальная, и я её сначала не
        # разделил. Вердикт судьи — МНЕНИЕ, и показывать его до решения нельзя:
        # человек соглашается чаще, и каппа измерит силу подсказки. А выдержка
        # из документа — ИСТОЧНИК, по которому и надо судить; скрывать его
        # значит требовать оценки по памяти.
        #
        # Цена этой ошибки измерена. Из восьми расхождений с человеком два
        # оказались его промахами по документу: он отметил неверными ответы
        # «аварийный порог 80 %» и «пороги 40 / 70 %», а в регламенте стоит
        # ровно «50 % внимание и 80 % авария», и в постмортеме ровно «40 %» и
        # «70 %». Оба ответа были верны. Разметка — точка отсчёта для всех
        # приборов, и ошибки в ней дороже ошибок в самих приборах.
        #
        # Аналогия: проверять контрольную по памяти, без задачника под рукой.
        # Проверяющий добросовестен, а часть верных решений зачёркнута.
        question = questions.get(row["question_id"])
        if question is not None and excerpt_source is not None:
            excerpt = excerpt_source.excerpts_for(question)
            if excerpt:
                print(f"\nчто написано в документах:\n{excerpt[:1200]}")

        print(f"\nответ системы:\n{row['answer'][:900]}")
        if args.show_judge:
            print(f"\nсудья сказал: {row.get('judge_verdict')}  ({row.get('judge_reason','')})")
        # НЕИЗВЕСТНАЯ КЛАВИША — ПЕРЕСПРОСИТЬ, а не пропустить молча.
        #
        # Он нажал «t» вместо «y» в трёх вопросах, и они тихо ушли в пропуск:
        # ни сообщения, ни повторного вопроса. Человек уверен, что оценил
        # сорок три ответа, а записано тридцать пять, и какие именно потеряны
        # — неизвестно. Ручная разметка — самое дорогое, что есть в стенде:
        # полчаса внимательного чтения, и она же единственная точка отсчёта
        # для всех остальных приборов. Молча терять её нельзя.
        #
        # Аналогия: урна, которая не принимает бюллетень и не сообщает об этом.
        while True:
            answer = input(
                "\nверно? [y — верно, n — неверно, s — пропустить, q — выход] "
            ).strip().lower()
            if answer in {"y", "n", "s", "q"}:
                break
            print(f"  не понял «{answer}». Нужна одна буква: y, n, s или q.")

        if answer == "q":
            break
        if answer == "s":
            continue
        labels[row["question_id"]] = answer == "y"

    target.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено {len(labels)} оценок: {target}")
    return 0


def cmd_transfer_labels(args) -> int:
    """Переносит ручную разметку с одного прогона на другой.

    Зачем. Ручная разметка — самое дорогое, что есть в стенде: сорок ответов
    это полчаса внимательного чтения. А прогоны меняются часто: поправил код —
    новый прогон, и разметка осталась у старого.

    Условие переноса строгое: текст ответа на этот вопрос должен совпадать
    ДОСЛОВНО. Человек оценивал конкретный текст, а не идентификатор вопроса;
    перенести оценку на другой текст — значит подделать разметку. Поэтому
    несовпавшие не переносятся, и команда говорит, сколько их.

    Кстати, ровно эта проверка попутно измеряет воспроизводимость: если при
    температуре ноль и той же версии промпта все ответы совпали дословно, то
    разброс генерации на этом стенде нулевой, и его можно не мерить отдельно.
    """
    source = resolve_run(args.source, mode="answer")
    target = resolve_run(args.target, mode="answer")
    if source is None:
        return run_not_found(args.source, mode="answer")
    if target is None:
        return run_not_found(args.target, mode="answer")

    source_labels_path = labels_dir() / f"{source.stem}.json"
    if not source_labels_path.exists():
        print(f"у прогона {source.name} нет разметки")
        return 1

    labels = json.loads(source_labels_path.read_text(encoding="utf-8"))
    source_answers = {row["question_id"]: row["answer"] for row in load_run(source)["rows"]}
    target_answers = {row["question_id"]: row["answer"] for row in load_run(target)["rows"]}

    moved: dict[str, bool] = {}
    skipped: list[str] = []
    for question_id, value in labels.items():
        if source_answers.get(question_id) == target_answers.get(question_id):
            moved[question_id] = value
        else:
            skipped.append(question_id)

    target_labels_path = labels_dir() / f"{target.stem}.json"
    existing = (
        json.loads(target_labels_path.read_text(encoding="utf-8"))
        if target_labels_path.exists()
        else {}
    )
    existing.update(moved)
    labels_dir().mkdir(parents=True, exist_ok=True)
    target_labels_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"перенесено оценок: {len(moved)} из {len(labels)}")
    if skipped:
        print(f"НЕ перенесено (текст ответа изменился): {', '.join(skipped)}")
        print("  Их надо размечать заново: человек оценивал другой текст.")
    else:
        print("все ответы совпали дословно — генерация на этом стенде "
              "воспроизводима, разброс нулевой")
    print(f"файл: {target_labels_path}")
    return 0


def cmd_kappa(args) -> int:
    path = resolve_run(args.run, mode="answer")
    if path is None:
        return run_not_found(args.run, mode="answer")
    run = load_run(path)
    target = labels_dir() / f"{path.stem}.json"
    if not target.exists():
        print(f"нет ручной разметки {target.name} — сначала: "
              f"python scripts/eval.py label {path}")
        return 1

    labels: dict[str, bool] = json.loads(target.read_text(encoding="utf-8"))
    rows = {row["question_id"]: row for row in run["rows"]}
    print(f"прогон: {path.name}")
    print(f"ручных оценок: {len(labels)}  "
          f"(из них «верно» {sum(1 for value in labels.values() if value)})")
    print()

    # Баланс классов проверяется ДО всякой каппы.
    #
    # Каппа измеряет согласие сверх случайного. Когда почти все оценки одного
    # класса, случайное согласие само по себе огромно, и каппа перестаёт
    # что-либо различать: она валится в ноль и у хорошего детектора, и у
    # болвана. Это не поломка формулы, это её область применимости.
    positives = sum(1 for value in labels.values() if value)
    negatives = len(labels) - positives
    if min(positives, negatives) < 8:
        print(f"ВНИМАНИЕ: в разметке {positives} «верно» и {negatives} «неверно». "
              f"Каппа на таком\nперекосе неинформативна — ей нужно хотя бы по 8 "
              f"случаев каждого класса.")
        print("Размечай выборку, обогащённую подозрительными: "
              "label --suspect-first.")
        print()

    def report_pair(name: str, field: str) -> float | None:
        pairs = [
            (bool(rows[qid][field]), human)
            for qid, human in labels.items()
            if qid in rows and rows[qid].get(field) is not None
        ]
        if len(pairs) < 10:
            print(f"{name}: пар всего {len(pairs)} — считать нечего")
            return None
        machine = [pair[0] for pair in pairs]
        human_side = [pair[1] for pair in pairs]
        value = cohen_kappa(machine, human_side)
        print(f"{name}: каппа {value:.3f}  (пар: {len(pairs)})")
        print(json.dumps(agreement_table(machine, human_side), ensure_ascii=False, indent=2))
        soft = sum(1 for m, h in pairs if m and not h)
        strict = sum(1 for m, h in pairs if not m and h)
        if soft > strict:
            print(f"  мягче человека ({soft} против {strict}): метрика завышена, "
                  f"и это опаснее занижения")
        elif strict > soft:
            print(f"  строже человека ({strict} против {soft}): метрика занижена")
        print()
        return value

    # Дешёвая проверка идёт ПЕРВОЙ и считается всегда. Она не зависит от того,
    # получилось ли у судьи: это её главное достоинство.
    cheap = report_pair("проверка подстрокой (бесплатная)", "answer_contains")
    judged = report_pair("судья-модель", "judge_verdict")

    if judged is None:
        unresolved = [
            row for row in run["rows"]
            if row.get("judge_verdict") is None and row.get("judge_reason")
        ]
        if unresolved:
            print(f"Судья не вынес ни одного вердикта ({len(unresolved)} попыток). "
                  f"Причина первой:\n  {unresolved[0].get('judge_reason', '')[:120]}")
            if unresolved[0].get("judge_raw"):
                print(f"  прислал: {unresolved[0]['judge_raw'][:200]!r}")
            print("Проверить связку одним вызовом: "
                  "python scripts/eval.py judge <прогон> --probe")
    elif cheap is not None and cheap >= judged:
        print("Подстрока согласуется с человеком не хуже судьи. Значит судья пока не "
              "окупает\nвызовов: дорогой инструмент обязан быть лучше бесплатного.")

    # Расхождения по именам — самое полезное в этой команде.
    #
    # Каппа говорит, НАСКОЛЬКО инструмент согласен с человеком. Список
    # расхождений говорит, ГДЕ именно, — и читать надо его: там либо дефект
    # системы, который метрика пропустила, либо слишком строгая разметка,
    # которая метрику оклеветала. И то и другое чинится, но по-разному.
    print("РАСХОЖДЕНИЯ С ЧЕЛОВЕКОМ (читать по одному, это чинится руками)")
    found = False
    for qid, human in labels.items():
        row = rows.get(qid)
        if row is None:
            continue
        marks = []
        if row.get("answer_contains") is not None and bool(row["answer_contains"]) != human:
            marks.append("подстрока: " + ("верно" if row["answer_contains"] else "НЕВЕРНО"))
        if row.get("judge_verdict") is not None and bool(row["judge_verdict"]) != human:
            marks.append("судья: " + ("верно" if row["judge_verdict"] else "НЕВЕРНО"))
        if marks:
            found = True
            print(f"  {qid}  человек: {'верно' if human else 'НЕВЕРНО':<8} "
                  f"{'; '.join(marks):<32} {row['question'][:42]}")
    if not found:
        print("  расхождений нет")
    return 0


async def _gate(args) -> int:
    """Обязательная проверка перед слиянием.

    Три правила чек-листа здесь работают вместе:

    - золотой набор прогоняется как тест, а не «иногда руками». Промпт и
      настройки поиска — это код без типов и без компилятора, и единственная
      доступная им защита это тест;
    - допуск равен ИЗМЕРЕННОМУ шуму, а не удобному числу. И он не поднимается,
      чтобы починить красный прогон: так планка качества снижается незаметно;
    - критичные вопросы проверяются поимённо и без допуска. Среднее может
      расти, пока ломается что-то важное.

    Проверка сделана на поисковом прогоне: он детерминирован, занимает секунды
    и не требует генерации, поэтому годится для каждого пул-реквеста. Прогон с
    моделью и судьёй остаётся ручным — он медленный и шумный.
    """
    # База сравнения задаётся ЯВНО и не может быть прошлым прогоном гейта.
    #
    # Раньше бралось «последнее поисковое», а гейт сам пишет прогон с меткой
    # gate — значит со второго запуска он сравнивался сам с собой. Десять
    # пул-реквестов, каждый роняет метрику на 0.015 при допуске 0.02: все
    # десять зелёные, суммарно потеряно 0.15. Гейт рапортует «проверка
    # пройдена», а система на пятнадцать процентов хуже эталона. Это храповик:
    # планка опускается по одной щелчке за раз, и каждый шаг законен.
    baseline_path = resolve_run(args.baseline, mode="search") if args.baseline else None
    if baseline_path is None:
        candidates = [
            path for path in sorted(runs_dir().glob("*-search-*.json"))
            if "-search-gate" not in path.name
        ]
        baseline_path = candidates[-1] if candidates else None
        if baseline_path is not None:
            print(f"база не указана, беру последний НЕ-гейтовый прогон: {baseline_path.name}")
            print("  В CI базу надо задавать явно: --baseline <метка>")
    if baseline_path is None:
        print("нет базового прогона. Сделайте его и зафиксируйте:")
        print("  python scripts/eval.py search --label baseline")
        return 1

    # Тот же приём, что и в `noise`, по той же причине. Сегодня `_search`
    # обходится тремя аргументами, и рукописный набор работал — но работал по
    # везению, а не по устройству: первый же новый флаг у команды `search`
    # уронил бы `gate` ровно так, как `--nudge` уронил `noise`.
    namespace = run_namespace(
        args.run_parsers["search"], label="gate", note="проверка перед слиянием", limit=0
    )
    code = await _search(namespace)
    if code != 0:
        return code

    current = load_run(sorted(runs_dir().glob("*-search-gate.json"))[-1])
    baseline = load_run(baseline_path)

    print()
    print(f"база: {baseline_path.name}")
    failures: list[str] = []
    waived: list[str] = []

    # Исключения задаются парами «метрика=причина». Причина обязательна: без
    # неё через месяц невозможно понять, была это взвешенная уступка или
    # способ сделать красное зелёным.
    accepted: dict[str, str] = {}
    for item in args.accept:
        metric, _, why = item.partition("=")
        if not why.strip():
            print(f"исключение без причины: {metric}. Формат: --accept 'recall@5=почему'")
            return 1
        accepted[metric.strip()] = why.strip()

    if args.tolerance > 0:
        # Допуск на детерминированной метрике не защищает ни от чего.
        #
        # Поисковый прогон повторяется байт в байт: замер разброса дал ровно
        # ноль. Значит любое отличие здесь — настоящее изменение системы, и
        # ненулевой допуск может только ПРОПУСТИТЬ его, а не отфильтровать шум,
        # которого нет.
        print(
            f"ВНИМАНИЕ: допуск {args.tolerance} на поисковых метриках. Поиск "
            f"детерминирован\n  (разброс измерен и равен нулю), поэтому допуск "
            f"тут не фильтрует шум, а прячет регрессию."
        )

    # Метрики, за которыми следим. Насыщенные проверяем всё равно — но честно
    # говорим, что они ничего не сторожат.
    for key in ("context_hit", "recall@5", "mrr", "chunk_mrr", "chunk_top1"):
        was = baseline["aggregate"]["overall"].get(key)
        now = current["aggregate"]["overall"].get(key)
        if was is None or now is None:
            print(f"  {key:<14} нет в одном из прогонов — не проверяется")
            continue
        delta = now - was
        verdict = "ок"
        if delta < -args.tolerance:
            if key in accepted:
                # ОСОЗНАННОЕ ИСКЛЮЧЕНИЕ, а не поднятый допуск.
                #
                # Разница между ними принципиальная. Поднятый допуск прощает
                # ЛЮБУЮ просадку ЛЮБОЙ метрики во всех будущих проверках — это
                # храповик, планка опускается по щелчку за раз. Исключение
                # названо по имени, требует письменной причины и действует
                # только на этот запуск: следующий снова покраснеет, если
                # причину не повторить.
                verdict = f"принято: {accepted[key]}"
                waived.append(f"{key}: {was:.3f} -> {now:.3f} ({delta:+.3f}) — {accepted[key]}")
            else:
                verdict = "ПРОВАЛ"
                failures.append(
                    f"{key}: {was:.3f} -> {now:.3f} ({delta:+.3f}), допуск {args.tolerance}"
                )
        note = ""
        if was >= 0.999:
            # Метрика на потолке не сторожит НИЧЕГО, пока всё не сломается
            # целиком. У нас так с recall@5: в выдачу попадает четверть
            # индекса, и «нужный документ где-то в топе» достижением быть
            # перестало. Молча держать такую метрику в гейте — значит считать,
            # что охрана есть, когда её нет.
            note = "  (на потолке: сторожит только полную поломку)"
        print(f"  {key:<14} {was:.3f} -> {now:.3f}  ({delta:+.3f})  {verdict}{note}")

    # ПОИМЁННОЕ сравнение, а не только средние.
    #
    # Это главный урок вечера, переложенный в код. Непрошедших цитат было 7 и
    # стало 7 — при том, что состав сменился наполовину: три вопроса
    # починились, три сломались. Среднее не шевельнулось, и по нему всё было
    # «стабильно». Так же может пройти и здесь: один вопрос потерял нужный
    # фрагмент, другой приобрёл, дельта ноль, гейт зелёный.
    #
    # Поэтому вопрос, который В БАЗЕ доводил нужный текст до контекста, а
    # теперь не доводит, — это провал независимо от любых средних.
    was_hit = {
        row["question_id"]: row["retrieval"].get("context_hit")
        for row in baseline["rows"]
    }
    lost = [
        row["question_id"]
        for row in current["rows"]
        if was_hit.get(row["question_id"]) and not row["retrieval"].get("context_hit")
    ]
    gained = [
        row["question_id"]
        for row in current["rows"]
        if was_hit.get(row["question_id"]) is False and row["retrieval"].get("context_hit")
    ]
    print(f"\nпоимённо: потеряли контекст {len(lost)}, приобрели {len(gained)}")
    for question_id in lost:
        print(f"  ПРОВАЛ {question_id}: нужный текст доходил до контекста, теперь нет")
        failures.append(f"{question_id} потерял контекст")
    for question_id in gained:
        print(f"  плюс {question_id}: нужный текст стал доходить до контекста")

    # Ухудшение МЕСТА нужного фрагмента — не провал, но и не пустяк: именно
    # его мерит chunk_top1, и именно оно теряется в среднем при обмене.
    worse_rank = [
        (row["question_id"], was_rank, row["retrieval"].get("chunk_rank"))
        for row in current["rows"]
        for was_rank in [
            next(
                (
                    other["retrieval"].get("chunk_rank")
                    for other in baseline["rows"]
                    if other["question_id"] == row["question_id"]
                ),
                None,
            )
        ]
        if was_rank and row["retrieval"].get("chunk_rank")
        and row["retrieval"]["chunk_rank"] > was_rank
    ]
    if worse_rank:
        print(f"\nнужный фрагмент опустился в выдаче: {len(worse_rank)}")
        for question_id, was_rank, now_rank in worse_rank[:10]:
            print(f"  {question_id}: место {was_rank} -> {now_rank}")

    broken_critical = [
        row
        for row in current["rows"]
        if row["critical"] and row["answerable"] and not row["retrieval"]["context_hit"]
    ]
    print(f"\nкритичные вопросы: {len(broken_critical)} провалов")
    for row in broken_critical:
        print(f"  ПРОВАЛ {row['question_id']}  {row['question'][:60]}")
        failures.append(f"критичный {row['question_id']} не дошёл до контекста")

    if waived:
        print("\nПРИНЯТО КАК ОСОЗНАННАЯ УСТУПКА:")
        for item in waived:
            print(f"  {item}")
        print("  Действует только на этот запуск. Новый эталон надо зафиксировать")
        print("  явно: search --label <новая метка> — иначе следующая проверка")
        print("  сравнит с прежним и снова покраснеет.")

    if failures:
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА:")
        for failure in failures:
            print(f"  {failure}")
        print(
            "\nПоднимать допуск, чтобы это позеленело, нельзя: допуск равен "
            "измеренному шуму, а не тому, что удобно сегодня."
        )
        return 1

    print("\nпроверка пройдена")
    return 0


def _latest(mode: str) -> Path | None:
    directory = runs_dir()
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(f"*-{mode}-*.json"))
    return candidates[-1] if candidates else None


def resolve_run(value: str | None, *, mode: str = "answer") -> Path | None:
    """Находит прогон по чему угодно: по пути, по имени файла, по метке, никак.

    Зачем это нужно. Папка прогонов задаётся конфигом (она рядом с базой), и
    её расположение — не то, что человек обязан помнить: в справке оно
    устареет при первом же переносе. Команда, требующая полного пути к файлу,
    который она сама только что создала, — это лишний шаг, на котором
    спотыкаешься каждый раз.

    Порядок разрешения: готовый путь → файл в папке прогонов → совпадение по
    метке (берём самый свежий) → последний прогон нужного режима.
    """
    directory = runs_dir()

    if not value:
        return _latest(mode)

    direct = Path(value)
    if direct.exists():
        return direct
    if (directory / value).exists():
        return directory / value
    if not directory.exists():
        return None
    # Путь мимо папки, но с правильным именем файла: частый случай, когда
    # путь скопирован из чужой инструкции. Имя файла уникально — берём его.
    if (directory / direct.name).exists():
        return directory / direct.name

    # Метка: `recheck baseline` вместо полного имени со штампом времени.
    matches = sorted(directory.glob(f"*-{mode}-{value}.json"))
    if matches:
        return matches[-1]
    matches = sorted(path for path in directory.glob("*.json") if value in path.stem)
    return matches[-1] if matches else None


def run_not_found(value: str | None, *, mode: str) -> int:
    """Внятная ошибка вместо FileNotFoundError с сырым путём."""
    directory = runs_dir()
    print(f"прогон не найден: {value or f'(последний в режиме {mode})'}")
    print(f"папка прогонов: {directory}")
    available = sorted(directory.glob("*.json")) if directory.exists() else []
    if available:
        print("есть такие:")
        for path in available[-10:]:
            print(f"  {path.name}")
        print()
        print("достаточно метки, например:  python scripts/eval.py show baseline")
    else:
        print(f"прогонов ещё нет — сначала: python scripts/eval.py {mode}")
    return 1


# ------------------------------------------------------------------- разбор


def build_parser() -> argparse.ArgumentParser:
    """Всё дерево команд. Отделено от `main`, чтобы его можно было проверить.

    Раньше парсер собирался прямо в `main`, и добраться до него из теста было
    нельзя — а значит нельзя было проверить стык между командами, которые
    запускают друг друга изнутри. Именно на этом стыке и сломался `noise`:
    падения не было ни в одном тесте, потому что тесты проверяли команды по
    отдельности, а сломался переход между ними.
    """
    parser = argparse.ArgumentParser(description="измерения качества")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="состав набора и достижимость разметки")
    validate.set_defaults(func=lambda args: cmd_validate(args))

    search = sub.add_parser("search", help="прогон только поиска, без модели")
    search.add_argument("--label", default="run")
    search.add_argument("--note", default="")
    search.add_argument("--limit", type=int, default=0)
    search.set_defaults(func=lambda args: asyncio.run(_search(args)))

    answer = sub.add_parser("answer", help="полный прогон с генерацией и судьёй")
    answer.add_argument("--label", default="run")
    answer.add_argument("--note", default="")
    answer.add_argument("--limit", type=int, default=0)
    answer.add_argument(
        "--nudge", action="store_true",
        help="переставить два независимых правила промпта, не меняя смысла: "
             "замер того, сколько ответов меняется от бессмысленной правки",
    )
    answer.add_argument(
        "--with-judge", action="store_true",
        help="судить в том же прогоне — только если хватает видеопамяти на две модели",
    )
    answer.add_argument(
        "--prompt", default="",
        help="версия промпта из реестра (по умолчанию — активная, PW_PROMPT_NAME). "
             "Два прогона с разными версиями сравниваются командой compare",
    )
    answer.add_argument(
        "--agent", action="store_true",
        help="агентский шаг: модель сама решает, дособрать ли контекст инструментами. "
             "Это ДРУГАЯ система — прогоняйте её отдельной меткой и сравнивайте "
             "с обычным прогоном командой compare",
    )
    answer.set_defaults(func=lambda args: asyncio.run(_answer(args)))

    coverage = sub.add_parser(
        "coverage", help="какие разделы корпуса не проверяет ни один вопрос"
    )
    coverage.add_argument("--corpus", default="", help="каталог корпуса (по умолчанию из настроек)")
    coverage.add_argument("--min-words", type=int, default=25,
                          help="не считать разделы короче этого числа слов")
    coverage.set_defaults(func=cmd_coverage)

    sweep = sub.add_parser("sweep", help="перебор константы слияния без повторных прогонов")
    sweep.add_argument(
        "--rrf-k", default="4,6,8,10,12,14,16,20,30,60",
        help="значения k через запятую",
    )
    sweep.add_argument(
        "--keyword-weight", default="1.0,0.6,0.45,0.3,0.2",
        help="вес ключевого списка при слиянии, через запятую",
    )
    sweep.add_argument(
        "--promote", action="store_true",
        help="перебрать ещё и гарантию: лучший по близости фрагмент — в начало",
    )
    sweep.add_argument("--show", type=int, default=12, help="сколько лучших вариантов печатать")
    sweep.set_defaults(func=lambda args: asyncio.run(_sweep(args)))

    sensitivity = sub.add_parser(
        "sensitivity", help="ловят ли приборы заведомо неверные ответы (порча)"
    )
    sensitivity.add_argument("run", nargs="?", default="", help="метка или файл прогона")
    sensitivity.add_argument(
        "--limit", type=int, default=20,
        help="сколько верных ответов портить (каждый даёт 4-5 порч и столько же вызовов судьи)",
    )
    sensitivity.set_defaults(func=lambda args: asyncio.run(_sensitivity(args)))

    why = sub.add_parser("why", help="разбор поиска по вопросу: где нужный фрагмент и почему")
    why.add_argument("question", nargs="*", help="номера вопросов; без них — все отвечаемые")
    why.add_argument("--top", type=int, default=10, help="сколько мест печатать в каждом списке")
    why.set_defaults(func=lambda args: asyncio.run(_why(args)))

    noise = sub.add_parser("noise", help="несколько прогонов без изменений: порог значимости")
    noise.add_argument("--runs", type=int, default=3)
    noise.add_argument("--mode", choices=["search", "answer"], default="answer")
    noise.add_argument("--limit", type=int, default=0)
    # Парсеры прогонов едут вместе с командой.
    #
    # `noise` и `gate` запускают чужие команды изнутри, и им нужны их
    # аргументы по умолчанию. Передать сам парсер — единственный способ
    # получить их из одного места: список полей, переписанный руками, отстанет
    # от оригинала при первом же новом флаге. Именно так и вышло с `--nudge`.
    noise.set_defaults(
        func=lambda args: asyncio.run(_noise(args)),
        run_parsers={"search": search, "answer": answer},
    )

    calibrate = sub.add_parser("calibrate", help="калибровка порога отсечения")
    calibrate.add_argument("run", nargs="?", default="")
    calibrate.add_argument("--min-answerable", type=float, default=0.95)
    calibrate.set_defaults(func=cmd_calibrate)

    recheck = sub.add_parser(
        "recheck", help="пересчитать дешёвые проверки по сохранённым ответам, без модели"
    )
    recheck.add_argument("run", nargs="?", default="", help="путь, имя файла или метка; по умолчанию последний прогон")
    recheck.set_defaults(func=cmd_recheck)

    show = sub.add_parser("show", help="сводка по прогону")
    show.add_argument(
        "run", nargs="?", default="",
        help="путь, имя файла или метка; по умолчанию последний прогон",
    )
    show.add_argument("--mode", default="answer", choices=["answer", "search"])
    show.set_defaults(func=cmd_show)

    diagnosis = sub.add_parser(
        "diagnose", help="разделимость: отличимы ли неотвечаемые вопросы по числам"
    )
    diagnosis.add_argument("run", nargs="?", default="")
    diagnosis.set_defaults(func=cmd_diagnose)

    compare = sub.add_parser("compare", help="сравнить два прогона")
    compare.add_argument("before", help="путь, имя файла или метка")
    compare.add_argument("after", help="путь, имя файла или метка")
    compare.add_argument("--mode", default="answer", choices=["answer", "search"])
    compare.add_argument("--noise", type=float, default=0.0)
    compare.set_defaults(func=cmd_compare)

    listing = sub.add_parser("list", help="список прогонов")
    listing.set_defaults(func=cmd_list)

    label = sub.add_parser("label", help="ручная разметка ответов для калибровки судьи")
    label.add_argument(
        "--only", default="",
        help="переразметить названные вопросы через запятую (q029,q106): "
             "старая оценка по ним стирается",
    )
    label.add_argument(
        "--show-judge", action="store_true",
        help="показывать вердикт судьи (по умолчанию скрыт: подсказка портит каппу)",
    )
    label.add_argument(
        "--suspect-first", action="store_true",
        help="сначала подозрительные ответы: нужно, чтобы в разметке были оба класса",
    )
    label.add_argument(
        "run", nargs="?", default="",
        help="путь, имя файла или метка; по умолчанию последний прогон",
    )
    label.add_argument("--limit", type=int, default=50)
    label.set_defaults(func=cmd_label)

    transfer = sub.add_parser(
        "transfer-labels",
        help="перенести ручную разметку на другой прогон (только где ответ совпал дословно)",
    )
    transfer.add_argument("source", help="прогон, где разметка есть")
    transfer.add_argument("target", help="прогон, куда переносить")
    transfer.set_defaults(func=cmd_transfer_labels)

    kappa = sub.add_parser("kappa", help="каппа Коэна: судья против человека")
    kappa.add_argument(
        "run", nargs="?", default="",
        help="путь, имя файла или метка; по умолчанию последний прогон",
    )
    kappa.set_defaults(func=cmd_kappa)

    judging = sub.add_parser(
        "judge", help="второй проход: судья по сохранённым ответам (одна модель в памяти)"
    )
    judging.add_argument("run", nargs="?", default="", help="путь, имя файла или метка; по умолчанию последний прогон")
    judging.add_argument(
        "--retry-unresolved", action="store_true",
        help="повторить те строки, где судья не смог вынести вердикт",
    )
    judging.add_argument(
        "--probe", action="store_true",
        help="один пробный вопрос с печатью сырого ответа судьи, без прогона",
    )
    judging.set_defaults(func=lambda args: asyncio.run(_judge(args)))

    gate = sub.add_parser("gate", help="проверка перед слиянием: набор как тест")
    gate.add_argument("--baseline", default="")
    gate.add_argument(
        "--accept", action="append", default=[], metavar="МЕТРИКА=ПРИЧИНА",
        help="принять просадку одной метрики с письменной причиной "
             "(например: --accept 'recall@5=просел один документ из топ-5, "
             "нужный текст всё равно дошёл до контекста')",
    )
    gate.add_argument(
        "--tolerance", type=float, default=0.0,
        help="допуск ОБЯЗАН равняться измеренному шуму (python scripts/eval.py noise)",
    )
    gate.set_defaults(
        func=lambda args: asyncio.run(_gate(args)),
        run_parsers={"search": search},
    )

    return parser


def run_parser(name: str):
    """Подпарсер команды прогона: «search» или «answer».

    Берётся из настоящего дерева команд и через тот же механизм, которым
    пользуются `noise` и `gate`. Это важно: тест, который построил бы парсер
    по-своему, проверял бы собственный макет, а не то, что запускается.
    """
    parsers = build_parser().parse_args(["noise"]).run_parsers
    if name not in parsers:
        raise KeyError(f"нет парсера прогона «{name}»: есть {sorted(parsers)}")
    return parsers[name]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
