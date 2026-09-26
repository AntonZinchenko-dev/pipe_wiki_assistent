"""Отчёты: сводка по прогону, сравнение двух прогонов, разброс между прогонами.

Три правила раздела 4, зашитые в этот файл.

**Одно изменение — один замер — одна строка.** Сравнение всегда идёт между
двумя конкретными прогонами, и в отчёте печатается, чем их конфигурации
различались. Если различий больше одного, об этом сказано прямо: такой
результат нельзя приписать ни одному из изменений.

**Разброс — это порог значимости.** Три прогона без изменений дают величину,
ниже которой «улучшение» неотличимо от случайности. Сравнение печатает
дельту рядом с разбросом и честно пишет «в пределах шума», когда это так.

**Среднее скрывает структуру.** Поэтому сравнение показывает не только
дельты метрик, но и построчно: что починилось и что сломалось. «Метрика не
изменилась» при двух починенных и трёх сломанных вопросах — это не отсутствие
изменений.
"""

from __future__ import annotations

import textwrap

from .metrics import CORPUS_LANGUAGE, mean, spread

SUMMARY_KEYS = (
    "recall@5", "mrr", "ndcg@10", "chunk_mrr", "chunk_top1",
    "context_hit", "status_ok", "answer_contains", "language_ok",
    # Доля подтвердившихся ссылок. Добавлена после прогона, в котором правка
    # чинила ровно цитаты — и в сравнении показала «починилось 0», потому что
    # сравнивать было нечего: счётчик неудачных ссылок в сводку не входил.
    "citations_ok",
    # Агентский режим. Добавлены после дня, в который всё, что чинилось в
    # агенте — лишняя страница, выдуманный сервис, ссылка именем документа, —
    # проверялось живыми логами и ни одной метрикой.
    #
    # `tables_ok`  — столько ли таблиц, сколько просили. Лишняя таблица такой
    #                же дефект, как её отсутствие.
    # `tools_ok`   — в тот ли эндпоинт сходил.
    # `live_clean` — не выдумал ли обозначений и не разошлись ли числа.
    "tables_ok", "tools_ok", "live_clean",
    "judge_ok",
)

# Метрика, не измеренная в этом прогоне, печатается прочерком, а не нулём.
# Ноль читается как «всё плохо», прочерк — как «мы этого не мерили», и это
# ровно та разница, из-за которой можно полдня чинить несуществующую беду.
NOT_MEASURED = "—"


def _fmt(value) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _measured(value) -> float | None:
    """Величина, если она измерена, и None если нет.

    Существует затем, чтобы сравнение с порогом нельзя было написать через
    `get(ключ, 0)`. Ноль по умолчанию выглядит безопасным и не работает:
    ключ-то в словаре есть, в нём лежит None.
    """
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _cell(value, width: int = 8) -> str:
    if value is None:
        return f"{NOT_MEASURED:>{width}}"
    return f"{value:>{width}.3f}"


def summarize(run: dict) -> str:
    config = run["config"]
    aggregate = run["aggregate"]
    rows = run["rows"]

    lines: list[str] = []
    lines.append(f"прогон: {config['label']}  режим: {config['mode']}")
    lines.append(
        f"модель {config['chat_model']} · эмбеддинги {config['embed_model']} · "
        f"промпт {config['prompt_version']} · код {config.get('code_version', '—')} · "
        + (f"роль {config['run_as']} · " if config.get("run_as") else "")
        + (f"агент {config['agent_version']} · " if config.get("agent_version") else "")
        + f"разметка поиска {config.get('labels_retrieval', '—')} / "
        f"ответа {config.get('labels_answer', '—')}"
    )
    lines.append(
        f"порог {config['similarity_floor']} · top_k {config['search_top_k']} · "
        f"фрагментов {config['context_max_fragments']} · бюджет {config['context_token_budget']} "
        f"· температура {config['temperature']}"
    )
    # Настройки слияния — в шапку.
    #
    # Их отсутствие уже стоило нам прогона. Я поменял константу k в коде с 60
    # на 12, прогон отработал, в шапке было написано «порог 0.45 · top_k 24»,
    # и всё выглядело нормально. А k осталась шестьюдесятью: значение было
    # переопределено в .env, и код его не применял. Заметил я это только по
    # строчке «СЕЙЧАС В НАСТРОЙКАХ» в переборе — то есть случайно.
    #
    # Правило: то, что влияет на результат, печатается в шапке результата.
    lines.append(
        f"слияние: rrf_k {config.get('rrf_k', '—')} · "
        f"вес ключевого {config.get('keyword_weight', '—')}"
    )
    lines.append(f"индекс: {config['index_chunks']} чанков, {config['index_meta']}")
    # Пометка прогона — В ШАПКЕ, до всех чисел.
    #
    # Она появилась ради второй реализации: у образца на LlamaIndex статус
    # ответа и цитаты НЕ ИЗМЕРЯЮТСЯ, и в соответствующих графах стоят нули.
    # Ноль непрошедших цитат читается как безупречная работа, а означает «мы
    # не проверяли». Если оговорка живёт в исходнике скрипта, а таблицу
    # смотрят через `show`, то через неделю сравнение будет прочитано в нашу
    # пользу — и прочитает его тот, кто ничего не подтасовывал.
    #
    # Правило то же, что и со настройками слияния: то, что меняет ЧТЕНИЕ
    # результата, печатается рядом с результатом.
    note = (config.get("note") or "").strip()
    if note:
        lines.append("")
        for piece in textwrap.wrap(note, width=76):
            lines.append(f"  ! {piece}")
    lines.append("")

    # Разрез по частям набора — сразу под шапкой, до всех остальных чисел.
    # Это не деталь: цифра на настроечной части может быть следствием того,
    # что настройки под неё и подбирали.
    by_split = aggregate.get("by_split") or {}
    if by_split.get("holdout"):
        lines.append("НАСТРОЕЧНАЯ И ОТЛОЖЕННАЯ ЧАСТИ НАБОРА")
        lines.append(
            f"  {'часть':<12} {'n':>3}  {'контекст':>8} {'чанк-1':>7} "
            f"{'статус':>7} {'факт':>7}"
        )
        for key, title in (("tune", "настроечная"), ("holdout", "отложенная")):
            piece = by_split.get(key) or {}
            if not piece:
                continue
            lines.append(
                f"  {title:<12} {piece.get('n', 0):>3}  "
                f"{_cell(_measured(piece.get('context_hit'))):>8} "
                f"{_cell(_measured(piece.get('chunk_top1')), 7):>7} "
                f"{_cell(_measured(piece.get('status_ok')), 7):>7} "
                f"{_cell(_measured(piece.get('answer_contains')), 7):>7}"
            )
        lines.append(
            "  Отложенная часть в выборе настроек не участвовала. Разрыв между"
        )
        lines.append(
            "  строками — это мера подгонки, а не шум: числа считаны одинаково."
        )
        lines.append("")

    overall = aggregate["overall"]
    lines.append("ОБЩЕЕ")
    for key in SUMMARY_KEYS:
        if key not in overall:
            continue
        lines.append(f"  {key:<16} {_fmt(overall[key])}")
    if overall.get("judged"):
        lines.append(f"  {'судьёй оценено':<16} {overall['judged']}")
    if overall.get("judge_bad_reference"):
        lines.append(
            f"  {'ссылка судьи в пустоту':<16} {overall['judge_bad_reference']}"
        )
    if overall.get("judge_unresolved"):
        # Отдельная строка, а не сноска. Судья, который не смог вынести
        # вердикт, — это не «неверно»: это сломанный прибор, и его поломка не
        # должна выглядеть как падение качества системы.
        lines.append(
            f"  {'судить не удалось':<16} {overall['judge_unresolved']}"
            f"   ← в judge_ok НЕ входят"
        )
    # Разбивка по языкам печатается всегда, когда язык определялся, — даже
    # когда всё по-русски.
    #
    # Соблазн — показывать её только при браке, чтобы не занимать строку. Но
    # тогда отсутствие строки означает сразу две вещи: «всё на своём языке» и
    # «эту проверку в прогоне не считали», а различать их надо: прогон,
    # сделанный до появления метрики, выглядел бы безупречным.
    languages = overall.get("languages") or {}
    if languages:
        breakdown = " · ".join(f"{name} {count}" for name, count in languages.items())
        lines.append(f"  {'язык ответа':<14} {breakdown}")
        foreign = sum(count for name, count in languages.items() if name != CORPUS_LANGUAGE)
        if foreign:
            lines.append(
                f"  {'':<14} ← не на языке корпуса: {foreign}; "
                f"answer_contains на них падает НЕ из-за поиска"
            )
    lines.append(f"  {'цитат не прошло':<14} {overall.get('citations_failed', 0)}")
    if overall.get("citations_repaired"):
        lines.append(
            f"  {'ссылок починено':<14} {overall['citations_repaired']}"
            f"   ← указание было неверным, текст нашёлся"
        )
    if overall.get("citations_on_refusal"):
        lines.append(
            f"  {'ссылок при отказе':<14} {overall['citations_on_refusal']}"
            f"   ← модель отказалась и всё равно сослалась"
        )
    lines.append(f"  {'задержка, мс':<14} {_fmt(overall.get('latency_ms_avg', 0))}")
    lines.append("")

    unanswerable = aggregate.get("unanswerable") or {}
    answerable = aggregate.get("answerable") or {}
    if unanswerable:
        lines.append("ОТКАЗЫ — ДВЕ СТОРОНЫ ОДНОЙ МОНЕТЫ")
        lines.append(
            f"  правильные отказы (вопросов без ответа в корпусе)  "
            f"{_fmt(unanswerable.get('status_ok', 0))}  из {unanswerable.get('n', 0)}"
        )
        # Вторая строка обязательна. Без неё высокая доля правильных отказов
        # выглядит достижением, хотя её легче всего получить, отказывая
        # ВСЕГДА — и именно это однажды и произошло.
        lines.append(
            f"  ответы по существу (вопросов с ответом в корпусе)  "
            f"{_fmt(answerable.get('status_ok', 0))}  из {answerable.get('n', 0)}"
        )
        # Предупреждение печатается только когда ОБЕ половины измерены.
        #
        # `get(key, 0)` здесь не спасал: ключ существует со значением None, и
        # подстановка по умолчанию не срабатывает — сравнение None с числом
        # падает. Это тот же самый идиом, который я час назад починил в
        # `eval.py list`, оставив его копию через две функции. Ошибка не в
        # числе, повторённом в двух местах, а в ПРИЁМЕ, повторённом в двух
        # местах: `get(ключ, 0)` для величины, которая законно бывает None,
        # неверен всегда и везде.
        refusals = _measured(unanswerable.get("status_ok"))
        answers = _measured(answerable.get("status_ok"))
        if refusals is not None and answers is not None and refusals > 0.8 and answers < 0.5:
            lines.append(
                "  ВНИМАНИЕ: система отказывает почти всегда. Высокая доля "
                "правильных отказов здесь не качество, а её побочный эффект."
            )
        lines.append("")

    lines.append("ПО ТИПАМ ВОПРОСА")
    header = (
        f"  {'тип':<14}{'n':>4}{'recall@5':>10}{'чанк-mrr':>10}{'чанк-1':>8}"
        f"{'контекст':>10}{'статус':>8}{'факт':>8}{'судья':>8}"
    )
    lines.append(header)
    for name, block in aggregate["by_type"].items():
        lines.append(
            f"  {name:<14}{block['n']:>4}{_cell(block.get('recall@5'), 10)}"
            f"{_cell(block.get('chunk_mrr'), 10)}{_cell(block.get('chunk_top1'))}"
            f"{_cell(block.get('context_hit'), 10)}{_cell(block.get('status_ok'))}"
            f"{_cell(block.get('answer_contains'))}{_cell(block.get('judge_ok'))}"
        )
    lines.append("")

    critical = aggregate.get("critical") or {}
    if critical:
        lines.append("КРИТИЧНЫЕ ВОПРОСЫ (проверяются поимённо, без допуска)")
        for row in rows:
            if not row["critical"]:
                continue
            marks = []
            if not row["retrieval"]["context_hit"]:
                marks.append("контекст не дошёл")
            if row.get("status_ok") is False:
                marks.append(f"статус {row['status']}")
            if row.get("answer_contains") is False:
                marks.append("в ответе нет ключевого факта")
            if row.get("judge_verdict") is False:
                marks.append("судья: неверно")
            verdict = "ок" if not marks else "; ".join(marks)
            lines.append(f"  {row['question_id']}  {verdict:<40} {row['question'][:48]}")
        lines.append("")

    # Самый ценный список отчёта: поиск сделал свою работу, а ответ — нет.
    # Именно здесь живут ошибки промпта и схемы, и их нельзя починить
    # реранкером или порогом, сколько бы их ни крутить.
    lost = [
        row
        for row in rows
        if row["answerable"]
        and row["retrieval"]["context_hit"]
        and (row.get("answer_contains") is False or row.get("status_ok") is False)
    ]
    if lost:
        lines.append(f"НУЖНЫЙ ТЕКСТ ДОШЁЛ, А ОТВЕТ НЕ СЛОЖИЛСЯ ({len(lost)})")
        for row in lost[:20]:
            why = []
            if row.get("status_ok") is False:
                why.append(f"статус {row['status']}")
            if row.get("answer_contains") is False:
                why.append("нет факта")
            lines.append(
                f"  {row['question_id']}  место чанка "
                f"{row['retrieval'].get('chunk_rank', 0)}  "
                f"{'; '.join(why):<28} {row['question'][:44]}"
            )
            # Причина у статуса `error` известна системе — печатаем её здесь,
            # а не заставляем открывать файл прогона.
            if row.get("schema_error"):
                lines.append(f"      причина: {row['schema_error'][:96]}")
        lines.append("")

    # Непрошедшие ссылки — с причиной и с тем, что указала модель.
    # Число без диагноза заставляет гадать; гадание про цитаты уже стоило нам
    # одной неверной правки.
    with_bad = [row for row in rows if row.get("citations_bad")]
    if with_bad:
        lines.append(f"ССЫЛКИ, КОТОРЫЕ НЕ НАШЛИСЬ ({sum(len(r['citations_bad']) for r in with_bad)})")
        for row in with_bad[:12]:
            for citation in row["citations_bad"]:
                lines.append(
                    f"  {row['question_id']}  [{citation.get('fragment')}]  "
                    f"{citation.get('reason', '')[:44]:<46} {citation.get('handle', '')[:60]!r}"
                )
        lines.append("")

    # АГЕНТ: ЧТО ЖДАЛИ, ЧТО ПОЗВАЛ, ЧЕМ ОБЪЯСНИЛ.
    #
    # `tools_ok 0.000` — число без диагноза. По нему не отличить «решил, что
    # инструменты не нужны» от «позвал не тот» и от «не умеет вызывать
    # вовсе»; это три болезни и три ремонта. Разбирать их по файлу прогона
    # руками мы уже пробовали — дорого и каждый раз заново.
    wrong_tools = [row for row in rows if row.get("tools_ok") is False]
    if wrong_tools:
        lines.append(f"АГЕНТ ПОЗВАЛ НЕ ТО ({len(wrong_tools)})")
        for row in wrong_tools[:12]:
            expected = ", ".join(row.get("tools_expected") or []) or "—"
            actual = ", ".join(row.get("tools_used") or []) or "ничего не звал"
            lines.append(
                f"  {row['question_id']}  ждали {expected:<18} позвал {actual:<28} "
                f"способ: {row.get('agent_mechanism') or '—'}"
            )
            said = row.get("agent_declined") or ""
            if said and "обрыв" in (row.get("agent_mechanism") or ""):
                # У оборванного бланка интересен ХВОСТ: там видно, на чём
                # модель срезало и сколько ей не хватило. Голова бланка при
                # обрыве всегда одна и та же — начало поля `why`.
                lines.append(f"      бланк не дописан, оборвано на: «…{said[-90:]}»")
            elif said:
                lines.append(f"      вместо вызова ответил: «{said[:90]}»")
        lines.append("")

    worst = [
        row
        for row in rows
        if row["answerable"] and not row["retrieval"]["context_hit"]
    ]
    if worst:
        lines.append(f"НЕ ДОШЛИ ДО КОНТЕКСТА ({len(worst)})")
        for row in worst[:15]:
            lines.append(
                f"  {row['question_id']}  косинус {row['retrieval']['best_cosine']:.3f}  "
                f"нашлось {','.join(row['retrieved_docs'][:3]) or '—'}  {row['question'][:46]}"
            )
        lines.append("")

    return "\n".join(lines)


# Что внутри index_meta меняется само по себе и различием не является.
# Всё остальное там — настройки, и их различия обязаны быть видны.
VOLATILE_META = {"built_at", "corpus_dir"}


def config_diff(before: dict, after: dict) -> list[str]:
    """Чем различаются конфигурации двух прогонов.

    `index_meta` раньше игнорировался ЦЕЛИКОМ, и это стоило нам вывода.

    Замысел был правильный: внутри лежит `built_at`, он меняется при каждой
    перестройке индекса, и объявлять это различием конфигурации бессмысленно.
    Реализация выплеснула вместе с ним всё остальное — а там живут настройки.
    У образца на LlamaIndex там `fusion_queries`, то есть включено ли
    переписывание запроса.

    Что из этого вышло. Два прогона образца, отличающиеся ровно этим
    параметром, дали разницу в 13 пунктов `chunk_top1` — и `compare` напечатал
    «конфигурации совпадают, включая отпечаток кода — значит это замер
    разброса». То есть инструмент объявил измерение функции измерением шума.

    Прочитать это можно было только одним способом, и он хуже правды: «система
    скачет на тринадцать пунктов между одинаковыми прогонами». Вывод
    катастрофический и полностью ложный.

    Поэтому теперь `index_meta` разбирается по ключам: изменчивое пропускаем
    поимённо, остальное сравниваем. Список изменчивого короткий и явный —
    добавить туда ключ можно только осознанно, а новый ключ с настройкой
    попадёт в различия сам.
    """
    ignored = {"started_at", "label", "note", "index_meta"}
    differences: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key in ignored:
            continue
        if before.get(key) != after.get(key):
            differences.append(f"{key}: {before.get(key)} -> {after.get(key)}")

    was_meta = before.get("index_meta") or {}
    now_meta = after.get("index_meta") or {}
    for key in sorted(set(was_meta) | set(now_meta)):
        if key in VOLATILE_META:
            continue
        if was_meta.get(key) != now_meta.get(key):
            differences.append(
                f"index_meta.{key}: {was_meta.get(key)} -> {now_meta.get(key)}"
            )
    return differences


def compare(before: dict, after: dict, *, noise: float = 0.0) -> str:
    lines: list[str] = []
    differences = config_diff(before["config"], after["config"])

    lines.append(f"было: {before['config']['label']}   стало: {after['config']['label']}")
    if not differences:
        lines.append(
            "конфигурации совпадают, включая отпечаток кода — значит это замер "
            "разброса, а не сравнение изменений"
        )
        if not before["config"].get("code_version"):
            # Старый прогон отпечатка кода не содержит, и тогда утверждение
            # «конфигурации совпадают» ничем не подкреплено: код мог
            # измениться, а мы этого не видим.
            lines.append(
                "  ОГОВОРКА: в одном из прогонов нет отпечатка кода — "
                "совпадение настроек НЕ доказывает, что система та же."
            )
    else:
        lines.append("различия конфигурации:")
        for difference in differences:
            lines.append(f"  {difference}")
        if len(differences) > 1:
            lines.append(
                "  ВНИМАНИЕ: различий больше одного. Результат нельзя приписать "
                "ни одному из них — нужен отдельный прогон на каждое изменение."
            )
    # Смена разметки — это не «ещё одно различие конфигурации». Это смена
    # линейки: метрики двух прогонов посчитаны по-разному, и дельта между ними
    # не значит ничего, пока оба не пересчитаны одной разметкой.
    for field, name, remedy in (
        (
            "labels_retrieval",
            "разметка ПОИСКА (docs / must_contain)",
            "Пересчётом это не лечится: метрики поиска считаются по фрагментам,\n"
            "  которых в файле прогона нет. Нужны новые прогоны на одной разметке.",
        ),
        (
            "labels_answer",
            "разметка ОТВЕТА (answer_must_contain)",
            "Лечится пересчётом обоих прогонов:\n"
            f"    python scripts/eval.py recheck {before['config'].get('label', '')}\n"
            f"    python scripts/eval.py recheck {after['config'].get('label', '')}",
        ),
    ):
        was = before["config"].get(field)
        now = after["config"].get(field)
        if was and now and was != now:
            lines.append("")
            lines.append(f"СТОП: {name} РАЗНАЯ ({was} против {now}).")
            lines.append("  Это не изменение системы, а смена линейки.")
            lines.append(f"  {remedy}")
    lines.append("")

    before_overall = before["aggregate"]["overall"]
    after_overall = after["aggregate"]["overall"]

    lines.append(f"метрики (порог значимости по измеренному шуму: {noise:.3f})")
    if (before_overall.get("judge_ok") is not None
            and after_overall.get("judge_ok") is not None):
        # Шум судьи ничем не измерен: команда noise гоняет прогоны без
        # судейства. Применять к judge_ok порог, посчитанный по
        # детерминированному поиску, значит объявлять значимым собственный
        # разброс прибора.
        lines.append(
            "  ОГОВОРКА: шум судьи не измерен (noise гоняет прогоны без него). "
            "Дельту\n  judge_ok порогом ниже не проверить — читать как наблюдение, "
            "не как вывод."
        )
    for key in SUMMARY_KEYS:
        if before_overall.get(key) is None or after_overall.get(key) is None:
            continue
        delta = after_overall[key] - before_overall[key]
        if abs(delta) <= noise:
            verdict = "в пределах шума"
        else:
            verdict = "значимо лучше" if delta > 0 else "значимо ХУЖЕ"
        lines.append(
            f"  {key:<14} {before_overall[key]:.3f} -> {after_overall[key]:.3f}  "
            f"({delta:+.3f}, {verdict})"
        )
    lines.append("")

    before_rows = {row["question_id"]: row for row in before["rows"]}
    after_rows = {row["question_id"]: row for row in after["rows"]}

    fixed: list[str] = []
    broken: list[str] = []
    unmeasured: list[str] = []
    vanished: list[str] = []

    for question_id in before_rows:
        if question_id not in after_rows:
            # Вопрос исчез из набора. Это не «ничего не изменилось»: удаление
            # провальных вопросов поднимает средние, а прежний цикл шёл по
            # новому прогону и таких строк вообще не видел.
            vanished.append(f"  {question_id}  {before_rows[question_id]['question'][:60]}")

    for question_id, row in after_rows.items():
        previous = before_rows.get(question_id)
        if previous is None:
            continue
        was = _row_ok(previous)
        now = _row_ok(row)
        if now is None and was is not None:
            unmeasured.append(f"  {question_id}  {row['question'][:60]}")
        elif now is True and was is False:
            fixed.append(f"  {question_id}  {row['question'][:60]}")
        elif was is True and now is False:
            broken.append(f"  {question_id}  {row['question'][:60]}")

    lines.append(f"ПОЧИНИЛОСЬ: {len(fixed)}")
    lines.extend(fixed)
    lines.append(f"СЛОМАЛОСЬ: {len(broken)}")
    lines.extend(broken)
    if unmeasured:
        lines.append(f"СТАЛО НЕЧЕМ МЕРИТЬ: {len(unmeasured)}")
        lines.extend(unmeasured)
        lines.append(
            "  Это не починка и не регресс: измерение пропало. Обычно значит, "
            "что\n  сломался прибор или убрали разметку."
        )
    if vanished:
        lines.append(f"ИСЧЕЗЛИ ИЗ НАБОРА: {len(vanished)}")
        lines.extend(vanished)
        lines.append(
            "  Вопросы были в базовом прогоне и отсутствуют в новом. Средние "
            "после\n  такого сравнивать нельзя."
        )
    if not fixed and not broken and not unmeasured and not vanished:
        lines.append("  построчных изменений нет")
    elif len(fixed) == len(broken) and fixed:
        lines.append("")
        lines.append(
            "  Обратите внимание: починилось и сломалось одинаковое число вопросов. "
            "Среднее при этом не изменится, хотя поведение системы изменилось дважды."
        )

    return "\n".join(lines)


def _row_ok(row: dict) -> bool | None:
    """Успешна ли строка. None — судить нельзя, измерения не хватает.

    Три состояния, а не два, и это важно именно в сравнении прогонов.
    Раньше отсутствие измерения считалось успехом, и поломка прибора выглядела
    как починка вопроса: в прогоне A судья сказал «неверно», в прогоне B не
    смог вынести вердикт — и compare печатал «ПОЧИНИЛОСЬ: q0xx». Тем же
    способом «починить» вопрос можно было, убрав у него разметку.

    Теперь строка без измерения выпадает из обоих списков и считается
    отдельно: «стало нечем мерить» — это не улучшение и не регресс, это
    потеря измерения, и говорить о ней надо своими словами.
    """
    if row.get("error"):
        return None
    if row["answerable"]:
        if not row["retrieval"]["context_hit"]:
            return False
        if row.get("status") and row.get("status_ok") is False:
            return False
        if row.get("answer_contains") is False:
            return False
        if row.get("judge_verdict") is False:
            return False
        # Ни одна проверка ответа не сработала — значит ответа в прогоне нет
        # (поисковый прогон) или мерить его нечем.
        if row.get("status_ok") is None and row.get("answer_contains") is None:
            return None
        return True

    # Неотвечаемый вопрос. Здесь правильное поведение — отказ, и проверяет его
    # только статус.
    #
    # Раньше стояло `return row.get("status_ok") is not False`, и это была та
    # же самая ошибка, что и в ветке выше, — просто в соседней строке, куда
    # при починке не посмотрели. У второй реализации `status_ok` равен None
    # (схемы ответа нет, отказ мерить нечем), а `None is not False` — это
    # True. То есть все тринадцать вопросов-ловушек засчитывались образцу как
    # пройденные — включая два, на которых он выдумал ответ про то, чего в
    # корпусе нет вообще.
    #
    # Вопросы без ответа в корпусе держатся в наборе ровно для ловли такого
    # вранья. Засчитывать их непроверенными — значит выключить ту часть
    # экзамена, ради которой он и составлен.
    if row.get("status_ok") is None:
        return None
    return row.get("status_ok") is not False


def noise_report(runs: list[dict]) -> str:
    """Разброс одной и той же конфигурации между прогонами.

    Это и есть порог значимости. Для поискового прогона он обычно нулевой:
    эмбеддинги и BM25 детерминированы. Для прогона с генерацией — нет, и
    именно поэтому мерить его обязательно: без этой цифры любое «стало на два
    процента лучше» не имеет смысла.
    """
    # ТРИ ТОЧКИ — ЭТО МАЛО, и об этом надо сказать прямо в отчёте.
    #
    # Оценка разброса по трём прогонам сама по себе случайная величина. В
    # живом замере три подряд дали одинаковые 0.967 — и выглядело это как
    # «метрика детерминирована», хотя четвёртый прогон той же конфигурации
    # отличался на 0.016. Ноль в графе «разброс» — самый опасный результат
    # этого отчёта: по нему любая следующая разница объявляется значимой.
    lines = [f"прогонов: {len(runs)}"]
    if len(runs) < 5:
        lines.append(
            "  ВНИМАНИЕ: прогонов меньше пяти. Ноль в графе «разброс» скорее "
            "означает «не поймали», чем «разброса нет»."
        )
    for key in SUMMARY_KEYS:
        values = [
            run["aggregate"]["overall"][key]
            for run in runs
            if run["aggregate"]["overall"].get(key) is not None
        ]
        if not values or not any(values):
            continue
        mark = "  ← ноль на малой выборке доверия не заслуживает" if (
            spread(values) == 0 and len(values) < 5
        ) else ""
        lines.append(
            f"  {key:<14} среднее {mean(values):.3f}  разброс {spread(values):.3f}  "
            f"значения {[round(value, 3) for value in values]}{mark}"
        )

    flapping: dict[str, int] = {}
    for run in runs:
        for row in run["rows"]:
            if _row_ok(row) is False:
                flapping[row["question_id"]] = flapping.get(row["question_id"], 0) + 1
    unstable = [
        question_id
        for question_id, count in flapping.items()
        if 0 < count < len(runs)
    ]
    if unstable:
        lines.append("")
        lines.append(
            f"нестабильные вопросы ({len(unstable)}): {', '.join(sorted(unstable))}"
        )
        lines.append(
            "  Они проходят не всегда. Это не повод их удалять — это и есть шум, "
            "который надо знать в лицо."
        )
    return "\n".join(lines)
