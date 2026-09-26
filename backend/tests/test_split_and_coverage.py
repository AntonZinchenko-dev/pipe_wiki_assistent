"""Тесты честности набора: разделение на части и покрытие корпуса.

Оба инструмента здесь — про один и тот же вид самообмана. Метрика говорит,
насколько хорошо система отвечает на ТЕ вопросы, которые мы задали, и при
настройках, подобранных по ТЕМ ЖЕ вопросам. Ни то, ни другое само по себе не
ложь — ложью становится вывод «система хороша», сделанный из этих цифр.
"""

from eval.dataset import SPLITS, holdout, load, tune


def test_every_question_belongs_to_exactly_one_part():
    questions = load()
    assert all(question.split in SPLITS for question in questions)
    assert len(tune(questions)) + len(holdout(questions)) == len(questions)
    assert len(holdout(questions)) >= 20, (
        "отложенная часть слишком мала, чтобы что-то на ней проверять"
    )


def test_critical_questions_are_never_held_out():
    """Критичный вопрос отложенным быть не может.

    Мы смотрим критичные вопросы поимённо в каждом прогоне — значит «не
    подглядывать» в них невозможно. Помечать их отложенными означало бы
    сообщать себе неправду о том, на чём мы настраивались.
    """
    for question in load():
        if question.critical:
            assert question.split == "tune", question.id


def test_both_parts_cover_the_same_kinds_of_questions():
    """Отложенная часть обязана быть стратифицированной, а не случайной.

    Если в ней не окажется, скажем, ни одного вопроса на отрицание, она не
    проверит именно тот класс, где система ошибается чаще всего. Разделение
    делается по типам, поэтому каждый достаточно большой тип обязан быть в
    обеих частях.
    """
    import collections

    counts = collections.Counter((question.type, question.split) for question in load())
    types = {qtype for qtype, _ in counts}
    for qtype in types:
        total = counts[(qtype, "tune")] + counts[(qtype, "holdout")]
        if total >= 6:
            assert counts[(qtype, "holdout")] >= 1, f"тип {qtype} не попал в отложенную часть"


def test_settings_are_chosen_on_the_tuning_part_only():
    """И перебор слияния, и калибровка порога обязаны выбирать по tune.

    Это последний пункт аудита и самый незаметный: код работает одинаково в
    обоих случаях, разница только в том, на каких вопросах выбрано значение.
    Поэтому проверяем сам вызов.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "scripts" / "eval.py").read_text(
        encoding="utf-8"
    )
    sweep = source[source.index("async def _sweep("):source.index("async def _why(")]
    assert 'q.split == "tune"' in sweep, "перебор выбирает не по настроечной части"
    assert "ОТЛОЖЕННАЯ ЧАСТЬ" in sweep, "перебор не докладывает результат на отложенной"

    calibrate = source[source.index("def cmd_calibrate("):]
    calibrate = calibrate[: calibrate.index("\ndef ")]
    assert "tune_rows" in calibrate, "порог калибруется по всему набору"
    assert "ПРОВЕРКА НА ОТЛОЖЕННОЙ ЧАСТИ" in calibrate


def test_run_reports_both_parts_separately():
    """Разрез по частям набора обязан считаться в каждом прогоне.

    Если считать его «когда понадобится», он не будет посчитан никогда:
    смотреть на него неприятно ровно в тот момент, когда он важен.
    """
    from eval.runner import RowResult, aggregate

    def row(question_id: str, split: str, hit: bool) -> RowResult:
        return RowResult(
            question_id=question_id, question="q", type="fact", difficulty="easy",
            answerable=True, critical=False, split=split, retrieved_docs=["D1"],
            retrieval={"context_hit": hit, "chunk_rank": 1, "recall@5": 1.0,
                       "chunk_mrr": 1.0, "mrr": 1.0, "ndcg@10": 1.0, "precision@5": 0.2,
                       "best_cosine": 0.7, "passed_floor": True, "margin": 0.1},
        )

    summary = aggregate([row("q1", "tune", True), row("q2", "holdout", False)])
    assert summary["by_split"]["tune"]["n"] == 1
    assert summary["by_split"]["holdout"]["n"] == 1
    assert summary["by_split"]["tune"]["context_hit"] == 1.0
    assert summary["by_split"]["holdout"]["context_hit"] == 0.0


def test_corpus_has_almost_no_sections_without_questions():
    """Новый документ без вопросов не должен проходить незамеченным.

    Первый запуск покрытия дал 21 содержательный раздел без единого вопроса —
    и не какие попало: формула метода минимальной кривизны, безусловные
    запреты регламента доступов, план изменений после аварии. То есть места,
    где ошибка стоит дороже всего, метрики не проверяли вообще.

    Аналогия: ученик ответил на пять вопросов билета и получил пять. Знает ли
    он остальные двадцать — оценкой не измеряется и даже не спрошено.

    Порог намеренно не ноль: преамбулы и оглавления вопроса не стоят.
    """
    import collections
    from pathlib import Path

    corpus = Path(__file__).resolve().parent.parent.parent / "corpus" / "source"
    if not corpus.exists():
        return  # корпус рядом не лежит — проверять нечего

    def normalize(text: str) -> str:
        return " ".join(text.split()).lower().replace("ё", "е")

    sections: list[tuple[str, str]] = []
    for path in sorted(corpus.glob("*.md")):
        heading, buffer = "(преамбула)", []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                if buffer:
                    sections.append((heading, "\n".join(buffer)))
                heading, buffer = line.lstrip("#").strip(), []
            else:
                buffer.append(line)
        if buffer:
            sections.append((heading, "\n".join(buffer)))

    needles = [
        [normalize(n) for n in question.must_contain]
        for question in load()
        if question.answerable and question.must_contain
    ]
    blind = []
    for heading, text in sections:
        if len(text.split()) < 25:
            continue
        haystack = normalize(f"{heading} {text}")
        if not any(all(n in haystack for n in ns) for ns in needles):
            blind.append(heading)

    assert len(blind) <= 3, f"разделов без вопросов стало больше: {blind}"
    assert isinstance(collections.Counter(blind), collections.Counter)


def test_labels_contain_no_markdown_markup():
    """В разметке набора не должно быть вёрстки markdown.

    В индекс текст попадает после извлечения, где обратные кавычки вокруг кода
    и звёздочки выделения сняты. Игла с кавычкой не найдётся никогда, а в
    отчёте это выглядит как «поиск не доносит нужный текст» — то есть
    отправляет чинить исправный поиск.

    Поймало меня на четырёх вопросах сразу. Причём моя собственная проверка
    их пропустила: я сравнивал разметку с файлами корпуса, а система читает
    индекс. Проверять надо то, что видит система, а не то, что написано в
    источнике, — и это уже третий раз за проект, когда разница между «похоже»
    и «то же самое» стоит прогона.
    """
    for question in load():
        for needle in question.must_contain + question.answer_must_contain:
            assert "`" not in needle, f"{question.id}: обратная кавычка в {needle!r}"
            assert "**" not in needle, f"{question.id}: звёздочки в {needle!r}"


def test_answer_labels_tell_an_answer_from_a_refusal():
    """Разметка ответа обязана отличать ответ от «сведений нет».

    Дефект был массовым и найден автоматически: семь вопросов из ста, включая
    два критичных. Причина в одном слове. На вопрос «дают ли новичку токен»
    правильный ответ начинается со «нет» — и я записал «нет» в разметку как
    допустимый вариант. Но «нет» есть и в «сведений об этом нет»: проверка
    одинаково принимала верный ответ и отказ.

    Злее всего то, что это ровно тот класс вопросов, где модель и путается:
    она принимает отрицательный ответ за отказ (ловили на q020 и q066).
    Проверка, которая должна была стеречь именно этот класс, была к нему
    слепа.

    Аналогия: тест на дальтонизм, напечатанный серым по серому. Проходят все,
    различает он ноль.
    """
    from eval.dataset import REFUSAL_SAMPLES
    from eval.metrics import answer_contains

    for question in load():
        if not question.answerable or not question.answer_needles:
            continue
        for sample in REFUSAL_SAMPLES:
            assert not answer_contains(sample, question.answer_needles), (
                f"{question.id}: {question.answer_needles} срабатывает на отказе"
            )


def test_answer_labels_require_a_fact_not_an_echo():
    """Разметку нельзя пройти повтором термина из вопроса.

    Дыра того же рода, что слепота к отказу, только шире. Вопрос «что такое
    critical_local_position_m» с разметкой `critical_local_position_m`
    проходит на ответе из одного этого слова. Так и случилось: модель
    ответила ровно «critical_local_position_m», не сказав, что это такое, — и
    ОБА прибора, подстрока и судья, зачли ответ верным. Поймал человек,
    глазами.

    Таких разметок нашлось девять, включая два КРИТИЧНЫХ вопроса: «можно ли
    выкладывать в пятницу» с разметкой «пятниц» и «что тяжелее» с разметкой
    «недостоверн».

    Аналогия: экзаменатор спрашивает «что такое интеграл?» и ставит зачёт за
    ответ «интеграл». Формально слово названо.
    """

    def flat(text: str) -> str:
        return " ".join(text.split()).lower().replace("ё", "е")

    for question in load():
        if not question.answerable or not question.answer_needles:
            continue
        asked = flat(question.question)
        echo = all(
            all(flat(v) in asked for v in needle.split("|") if v.strip())
            for needle in question.answer_needles
        )
        assert not echo, (
            f"{question.id}: разметка {question.answer_needles} проходит "
            f"повтором вопроса"
        )


def test_manual_labelling_never_swallows_an_unknown_key():
    """Ручная разметка не имеет права молча терять оценку.

    Он нажал «t» вместо «y» в трёх вопросах — они ушли в пропуск без единого
    сообщения. Человек уверен, что оценил сорок три ответа, записано
    тридцать пять, и какие потеряны — неизвестно. А ручная разметка это
    единственная точка отсчёта для всех остальных приборов.

    Аналогия: урна, которая не принимает бюллетень и не сообщает об этом.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "scripts" / "eval.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("def cmd_label("):]
    block = block[: block.index("\ndef ")]
    assert "не понял" in block, "неизвестная клавиша обязана вызывать переспрос"
    assert "while True:" in block
    assert "--only" in source, "должна быть возможность переразметить один вопрос"
