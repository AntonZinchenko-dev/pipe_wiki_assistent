"""Проверки второй реализации — образца на LlamaIndex.

Здесь проверяется не качество образца (его меряет прогон), а ЧЕСТНОСТЬ
сравнения. Три вещи, каждая из которых однажды уже ломалась у нас или
ломается у всех, кто сравнивает свою систему с чужой.

ПЕРВОЕ: неизмеренное должно остаться неизмеренным. У образца нет нашей схемы
ответа: он отдаёт свободный текст, без ярлыка статуса и без ссылок на
фрагменты. Значит статус и цитаты у него мерить нечем. Соблазн — поставить в
эти графы нули: код тогда не падает, таблица заполнена. Но ноль непрошедших
цитат читается как «брака нет», то есть как ИДЕАЛЬНАЯ работа, а означает «мы
не проверяли». Сравнение, подтасованное в свою пользу, — и подтасованное
незаметно, без единой ложной цифры.

Аналогия. Два автомобиля на техосмотре. У первого проверили тормоза и нашли
одну неисправность. У второго тормоза не проверяли вообще. Написать во второй
графе «неисправностей тормозов: 0» — формально не ложь, но читатель отчёта
поймёт её ровно наоборот.

ВТОРОЕ: оговорка должна ехать вместе с цифрами. Пометку «НЕ ИЗМЕРЯЛИСЬ» мало
написать в комментарии скрипта — таблицу через неделю будут смотреть командой
`show`, а не читая исходник. Поэтому пометка лежит в конфигурации прогона и
печатается в шапке отчёта.

ТРЕТЬЕ: чужой фреймворк не должен становиться обязательным. Модуль образца
обязан импортироваться на машине, где LlamaIndex не установлен, иначе один
вспомогательный сценарий утащит за собой всё дерево зависимостей и уронит
тесты стенда.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict

import pytest

from alt.llamaindex_run import FILE_TO_DOC, NOT_MEASURED_NOTE, corpus_files, row_for
from app.config import get_settings
from eval import dataset, report
from eval.runner import RunConfig, aggregate


def _question():
    """Настоящий вопрос из золотого набора, а не выдуманный.

    Разметка — часть прибора: строка образца должна строиться на том же
    объекте, что и строка своей реализации, иначе тест проверяет свой макет.
    """
    return dataset.load()[0]


def test_module_imports_without_the_framework_installed():
    """Импорт модуля образца не требует LlamaIndex.

    Тяжёлые импорты спрятаны внутрь build_engine именно за этим. Тест
    сторожит то, что легко потерять при правке: достаточно вынести один
    импорт наверх «чтобы было аккуратнее», и весь стенд перестанет
    запускаться там, где фреймворка нет.
    """
    module = importlib.import_module("alt.llamaindex_run")
    assert module.DEFAULT_FUSION_QUERIES == 1


def test_row_marks_status_and_citations_as_not_measured():
    question = _question()
    row = row_for(
        question=question,
        answer_text="какой-то свободный текст без ссылок",
        retrieved_docs=list(question.docs),
        chunk_texts=["текст узла"],
        context_text="текст узла",
        latency_ms=123.4,
    )

    # Статус и схема — НЕ ноль и не False, а None: «мерить нечем».
    assert row.status == ""
    assert row.status_ok is None
    assert row.schema_valid is None


def test_aggregate_reports_status_as_not_measured_rather_than_zero():
    """Сводка по строкам образца печатает прочерк, а не 0.000.

    Это главный тест файла. Строка образца устроена правильно, но сводку
    считает НАШ код, и вопрос в том, что он сделает с пустым статусом.
    Если бы `status_ok` посчитался как ноль, в сравнении двух стеков
    появилась бы строка «статус 0.000» — то есть «образец не отвечает
    вообще», чего мы не измеряли.
    """
    question = _question()
    rows = [
        row_for(
            question=question,
            answer_text="свободный текст",
            retrieved_docs=list(question.docs),
            chunk_texts=["текст узла"],
            context_text="текст узла",
            latency_ms=100.0,
        )
    ]
    summary = aggregate(rows)

    assert summary["overall"]["status_ok"] is None
    # А поиск и факт в тексте — измеримы у обоих стеков, и они посчитаны.
    assert summary["overall"]["recall@5"] is not None


def test_note_is_printed_in_the_report_header():
    """Оговорка видна тому, кто смотрит таблицу, а не исходник."""
    question = _question()
    rows = [
        row_for(
            question=question,
            answer_text="свободный текст",
            retrieved_docs=list(question.docs),
            chunk_texts=["текст узла"],
            context_text="текст узла",
            latency_ms=100.0,
        )
    ]
    config = RunConfig(
        label="alt-test",
        mode="answer",
        chat_model="qwen2.5:14b",
        embed_model="bge-m3",
        prompt_version="llamaindex-встроенный",
        code_version="test",
        dataset_version="test",
        similarity_floor=0.0,
        search_top_k=24,
        context_max_fragments=8,
        context_token_budget=0,
        temperature=0.0,
        rrf_k=60,
        index_meta={"stack": "llamaindex"},
        index_chunks=1,
        note=NOT_MEASURED_NOTE,
    )
    # asdict, а не `config.__dict__`: RunConfig — dataclass со slots, и
    # атрибута `__dict__` у него нет вообще. Мелочь, но показательная: именно
    # на ней этот тест и упал в первый запуск, поймав мою же опечатку в
    # скрипте прогона.
    text = report.summarize(
        {
            "config": asdict(config),
            "aggregate": aggregate(rows),
            "rows": [],
        }
    )

    assert "НЕ ИЗМЕРЯЛИСЬ" in text
    # И пометка стоит ДО чисел: читатель видит её раньше, чем таблицу.
    assert text.index("НЕ ИЗМЕРЯЛИСЬ") < text.index("ОБЩЕЕ")
    # Графа цитат печатается — с нулём, — и ровно поэтому пометка обязательна.
    assert "цитат не прошло" in text


def test_document_map_covers_the_whole_corpus():
    """Карта «файл → doc_id» покрывает все документы золотого набора.

    Зачем отдельный тест на словарь. Чтец LlamaIndex не разбирает наши шапки
    markdown и кладёт в метаданные имя файла. Если для какого-то документа
    имени нет в карте, метрики поиска по нему считаются по чужому ключу и
    дают НОЛЬ — то есть образец выглядит катастрофически плохим по причине,
    к нему не относящейся.

    Такую поломку легко не заметить: «recall 0.000» выглядит как результат
    измерения, а является поломкой сравнения. Поэтому карта сверяется с
    разметкой автоматически.
    """
    expected = {doc for question in dataset.load() for doc in question.docs}
    missing = expected - set(FILE_TO_DOC.values())
    assert not missing, f"в карте нет документов: {sorted(missing)}"


def test_every_corpus_file_has_a_doc_id():
    """Каждый файл корпуса на диске есть в карте — и наоборот.

    Тест ловит самую дешёвую и самую незаметную поломку сравнения: документ
    добавили в корпус, а в карту не добавили. Тогда его фрагменты попадают в
    индекс образца под ключом «имя файла», метрики считаются по чужому ключу,
    и все вопросы по этому документу дают ноль — то есть выглядят как провал
    поиска, а не как наша забывчивость.

    Проверять это здесь, а не первым прогоном, важно по времени: прогон — это
    двадцать минут работы модели, а тест — полсекунды.
    """
    settings = get_settings()
    root = settings.corpus_dir
    corpus_dir = root / "source" if (root / "source").exists() else root
    if not corpus_dir.exists():
        pytest.skip(f"корпус не найден: {corpus_dir}")

    names = {path.name for path in corpus_files(corpus_dir)}
    assert names, "корпус пуст"
    assert names - set(FILE_TO_DOC) == set(), (
        f"файлы корпуса без doc_id: {sorted(names - set(FILE_TO_DOC))}"
    )
    assert set(FILE_TO_DOC) - names == set(), (
        f"в карте есть файлы, которых нет в корпусе: {sorted(set(FILE_TO_DOC) - names)}"
    )


def test_corpus_files_are_sorted_and_missing_dir_is_loud():
    """Порядок файлов устойчив, а пустой каталог — это ошибка, а не ноль узлов.

    Молча собранный индекс из нуля документов — худший из возможных исходов:
    прогон отработает, все цифры будут нулевыми, и выглядеть это будет как
    «образец ничего не умеет». Поэтому пусто = исключение с адресом каталога.
    """
    settings = get_settings()
    root = settings.corpus_dir
    corpus_dir = root / "source" if (root / "source").exists() else root
    if corpus_dir.exists():
        paths = corpus_files(corpus_dir)
        assert paths == sorted(paths)

    with pytest.raises(RuntimeError):
        corpus_files(corpus_dir / "нет-такого-каталога")
