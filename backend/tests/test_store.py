"""Тесты индекса: идемпотентность, проверка модели, гибридные половины поиска."""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.chunk import Chunk
from app.rag.store import IndexMismatch, Store


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    store.upsert_document({
        "doc_id": "TG-GW", "title": "TELEMETRY-GW", "project": "TELEMETRY-GW",
        "owner": "Группа интеграций", "updated": "2026-08-19", "version": "2.1",
        "status": "действующий", "source_path": "", "pdf_path": "", "page_count": 3,
    })
    yield store
    store.close()


def unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def make_chunk(ordinal: int, body: str) -> Chunk:
    return Chunk(
        doc_id="TG-GW", ordinal=ordinal, heading_path="TELEMETRY-GW / Кэш",
        text=f"TELEMETRY-GW / Кэш\n\n{body}", body=body, page_from=1, page_to=1,
    )


def test_reindexing_same_content_does_not_duplicate(store: Store) -> None:
    """Идемпотентность по хешу содержимого.

    Без неё повторный запуск индексации кладёт вторую копию абзаца, и дубли
    вытесняют из топа всё остальное — при том, что ошибок нигде нет.
    """
    chunk = make_chunk(0, "Отключение кэша в продакшене запрещено.")
    first = store.add_chunk(chunk, unit_vector(1))
    second = store.add_chunk(make_chunk(0, "Отключение кэша в продакшене запрещено."), unit_vector(1))

    assert first == second
    assert store.stats()["chunks"] == 1


def test_different_content_gets_its_own_chunk(store: Store) -> None:
    store.add_chunk(make_chunk(0, "Кэш включается переменной TG_SNAPSHOT_CACHE=on."), unit_vector(1))
    store.add_chunk(make_chunk(1, "Кэш отключается переменной TG_SNAPSHOT_CACHE=off."), unit_vector(2))
    assert store.stats()["chunks"] == 2


def test_search_refuses_when_index_built_by_another_model(store: Store) -> None:
    """Самая незаметная поломка раздела 3: смена модели эмбеддингов без
    переиндексации не даёт ошибок, она даёт случайные результаты."""
    store.set_meta(embed_model="bge-m3:latest", embed_dim="1024")
    store.assert_compatible(embed_model="bge-m3:latest", embed_dim=1024)

    with pytest.raises(IndexMismatch, match="несравнимы"):
        store.assert_compatible(embed_model="nomic-embed-text:latest", embed_dim=1024)

    with pytest.raises(IndexMismatch, match="размерность"):
        store.assert_compatible(embed_model="bge-m3:latest", embed_dim=768)


def test_empty_index_is_reported_as_mismatch(store: Store) -> None:
    with pytest.raises(IndexMismatch, match="индекс пуст"):
        store.assert_compatible(embed_model="bge-m3:latest", embed_dim=1024)


def test_keyword_search_finds_exact_error_code(store: Store) -> None:
    """Ровно тот класс запросов, на котором векторный поиск проваливается:
    точный код. Дефис в `E-1042` — оператор синтаксиса FTS5, и без
    закавычивания запрос либо падает, либо ищет не то."""
    store.add_chunk(
        make_chunk(0, "Код E-1042 означает, что расчёт для трубы устарел."), unit_vector(1)
    )
    store.add_chunk(
        make_chunk(1, "Задержка доставки телеметрии не более 800 мс."), unit_vector(2)
    )

    hits = store.keyword_search("что значит ошибка E-1042", limit=5)
    assert hits, "ключевой поиск не нашёл точный код"
    top_chunk = store.load_chunks([hits[0][0]])[hits[0][0]]
    assert "E-1042" in top_chunk.body


def test_keyword_search_survives_fts_operators_in_user_input(store: Store) -> None:
    store.add_chunk(make_chunk(0, "Регламент РЛ-4.2.3 описывает порядок выдачи доступов."), unit_vector(1))
    for query in ['РЛ-4.2.3', 'что такое "РЛ-4.2.3"', "доступ* OR NOT", "AND OR NEAR"]:
        store.keyword_search(query, limit=5)  # не должно бросать


def test_vector_search_orders_by_cosine(store: Store) -> None:
    target = unit_vector(7)
    store.add_chunk(make_chunk(0, "первый"), target)
    store.add_chunk(make_chunk(1, "второй"), unit_vector(99))

    hits = store.vector_search(target, limit=2)
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[0][1] > hits[1][1]


def test_drop_document_chunks_clears_fts_too(store: Store) -> None:
    store.add_chunk(make_chunk(0, "Код E-2210 про некорректный диапазон шагов."), unit_vector(1))
    store.drop_document_chunks("TG-GW")
    assert store.stats()["chunks"] == 0
    assert store.keyword_search("E-2210", limit=5) == []
