"""Прогон золотого набора через образец на LlamaIndex.

Отдельный скрипт, а не подкоманда стенда, и это сознательно: зависимости
фреймворка не должны становиться обязательными для `scripts/eval.py`. Стенд
обязан запускаться на машине, где LlamaIndex не установлен.

Результат пишется в ТОТ ЖЕ формат прогона, что и у своей реализации. Отсюда
вся польза: дальше работают все наши команды без единой правки —

    python scripts/eval.py show alt1
    python scripts/eval.py compare wide3 alt1
    python scripts/eval.py sensitivity alt1

Сравнивать своё с чужим имеет смысл только одной линейкой. Написать для
образца отдельный замер значило бы получить две несравнимые таблицы и спор о
том, чья линейка честнее.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alt.llamaindex_run import (  # noqa: E402
    DEFAULT_FUSION_QUERIES,
    NOT_MEASURED_NOTE,
    build_engine,
    doc_id_of,
    row_for,
)
from app.config import get_settings  # noqa: E402
from app.version import code_version  # noqa: E402
from eval import dataset, report  # noqa: E402
from eval.runner import RunConfig, load_run, save_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="alt", help="метка прогона")
    parser.add_argument("--limit", type=int, default=0, help="ограничить число вопросов")
    parser.add_argument(
        "--fusion-queries", type=int, default=DEFAULT_FUSION_QUERIES,
        help="сколько запросов делает слияние: 1 — без переписывания вопроса, "
             "4 — поведение LlamaIndex по умолчанию (переписывание включено)",
    )
    parser.add_argument(
        "--corpus", default="", help="каталог с markdown корпуса (по умолчанию из настроек)"
    )
    args = parser.parse_args()

    settings = get_settings()
    root = Path(args.corpus) if args.corpus else settings.corpus_dir
    corpus_dir = root / "source" if (root / "source").exists() else root

    questions = dataset.load()
    if args.limit:
        questions = questions[: args.limit]

    print(f"корпус: {corpus_dir}")
    print(f"модели: {settings.ollama_chat_model} · эмбеддинги {settings.ollama_embed_model}")
    print(f"слияние: запросов {args.fusion_queries}"
          + ("  (переписывание вопроса ВКЛЮЧЕНО)" if args.fusion_queries > 1 else ""))
    print("сборка индекса образца...")

    engine, node_count = build_engine(
        settings=settings, corpus_dir=corpus_dir, fusion_queries=args.fusion_queries
    )
    print(f"узлов в индексе образца: {node_count}")
    print()

    rows = []
    for number, question in enumerate(questions, start=1):
        started = time.monotonic()
        response = engine.query(question.question)
        latency_ms = (time.monotonic() - started) * 1000

        nodes = list(getattr(response, "source_nodes", []) or [])
        retrieved_docs: list[str] = []
        for node in nodes:
            doc_id = doc_id_of(node.node if hasattr(node, "node") else node)
            if doc_id and doc_id not in retrieved_docs:
                retrieved_docs.append(doc_id)

        chunk_texts = [
            (node.node if hasattr(node, "node") else node).get_content() for node in nodes
        ]

        rows.append(
            row_for(
                question=question,
                answer_text=str(response),
                retrieved_docs=retrieved_docs,
                chunk_texts=chunk_texts,
                # Контекст образца — это в точности те узлы, которые он взял:
                # у него нет ни порога, ни бюджета токенов, поэтому «что
                # нашлось» и «что дошло до модели» совпадают.
                context_text="\n".join(chunk_texts),
                latency_ms=latency_ms,
            )
        )
        if number % 10 == 0 or number == len(questions):
            print(f"  {number}/{len(questions)} вопросов")

    versions = dataset.label_versions()
    config = RunConfig(
        label=args.label,
        mode="answer",
        chat_model=settings.ollama_chat_model,
        embed_model=settings.ollama_embed_model,
        # Промпт образца — его собственный, встроенный. Отпечатка у него нет, и
        # выдумывать его нельзя: пусть в шапке стоит честная пометка.
        prompt_version="llamaindex-встроенный",
        code_version=code_version(),
        dataset_version=dataset.dataset_version(),
        similarity_floor=0.0,          # порога у образца нет
        search_top_k=settings.search_top_k,
        context_max_fragments=settings.context_max_fragments,
        context_token_budget=0,        # бюджета токенов у образца нет
        temperature=settings.temperature,
        rrf_k=60,                      # значение по умолчанию внутри библиотеки
        index_meta={
            "stack": "llamaindex",
            "fusion_queries": str(args.fusion_queries),
            "node_parser": "SentenceSplitter(700/80)",
        },
        index_chunks=node_count,
        labels_retrieval=versions["retrieval"],
        labels_answer=versions["answer"],
        keyword_weight=1.0,            # у библиотеки веса списков равны
        note=NOT_MEASURED_NOTE,
    )

    path = save_run(config=config, rows=rows, directory=Path("var/eval/runs"))
    print()
    # Печатаем сводку из ЗАПИСАННОГО файла, а не из объектов в памяти.
    #
    # Это не лишний шаг. Записанный файл — то, что завтра прочтёт `show`, и
    # если он записался не так, как мы думали, увидеть это надо сейчас, а не
    # через неделю. Сводка из памяти подтвердила бы только то, что мы
    # правильно посчитали в памяти.
    print(report.summarize(load_run(path)))
    print(f"прогон сохранён: {path}")
    print()
    print("дальше — одной линейкой с нашей реализацией:")
    print(f"  python scripts/eval.py show {args.label}")
    print(f"  python scripts/eval.py compare wide3 {args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
