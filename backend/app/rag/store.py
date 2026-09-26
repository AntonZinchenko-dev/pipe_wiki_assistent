"""Хранилище индекса: SQLite + FTS5 + векторы.

Почему SQLite, а не Postgres с pgvector или Qdrant. На корпусе в несколько
сотен чанков полнотекстовый поиск SQLite (FTS5 с настоящим BM25) и перебор
векторов в numpy отвечают за единицы миллисекунд, а инфраструктуры требуют
ноль: ни докера, ни миграций, ни отдельного процесса. Всё, чему учит раздел 3,
здесь проверяется в полном объёме — гибридный поиск, версия модели в
метаданных индекса, идемпотентность по хешу. Переезд на pgvector или Qdrant
меняет только этот файл, потому что выше него никто не знает, где лежат
векторы.

Главная вещь в этом файле — таблица `index_meta`. Она помнит, какой моделью
построен индекс, и поиск это проверяет. Смена модели эмбеддингов без
переиндексации не даёт ошибок — она даёт СЛУЧАЙНЫЕ по смыслу результаты,
потому что векторы разных моделей несравнимы. Это самая незаметная поломка из
всего раздела 3, и единственная защита от неё — явная проверка.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    project     TEXT NOT NULL DEFAULT '',
    owner       TEXT NOT NULL DEFAULT '',
    updated     TEXT NOT NULL DEFAULT '',
    version     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'действующий',
    source_path TEXT NOT NULL DEFAULT '',
    pdf_path    TEXT NOT NULL DEFAULT '',
    page_count  INTEGER NOT NULL DEFAULT 0,
    ingested_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    text         TEXT NOT NULL,
    body         TEXT NOT NULL,
    page_from    INTEGER NOT NULL,
    page_to      INTEGER NOT NULL,
    content_hash TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks(doc_id, ordinal);

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);

-- Ключевой поиск. unicode61 с remove_diacritics 2 нормально работает с
-- русским; стемминга здесь нет, и это осознанно: словоформы добирает
-- векторная половина гибридного поиска.
--
-- content='chunks' — индекс внешнего содержимого: текст не дублируется, FTS
-- читает его из основной таблицы. Синхронизацию держат триггеры ниже, а не
-- код на Python: тогда её нельзя забыть, а contentless-таблица (content='')
-- вообще не поддерживает удаление строк на большинстве сборок SQLite.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    heading_path,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading_path)
    VALUES (new.chunk_id, new.text, new.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
    VALUES ('delete', old.chunk_id, old.text, old.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
    VALUES ('delete', old.chunk_id, old.text, old.heading_path);
    INSERT INTO chunks_fts(rowid, text, heading_path)
    VALUES (new.chunk_id, new.text, new.heading_path);
END;

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

FTS_TOKEN = re.compile(r"[\w\-/\.]{2,}", re.UNICODE)


class IndexMismatch(RuntimeError):
    """Индекс построен другой моделью эмбеддингов, чем настроена сейчас."""


@dataclass(slots=True)
class StoredChunk:
    chunk_id: int
    doc_id: str
    heading_path: str
    body: str
    text: str
    page_from: int
    page_to: int
    doc_title: str
    doc_project: str
    doc_version: str
    doc_status: str
    doc_updated: str


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._matrix: np.ndarray | None = None
        self._chunk_docs: dict[int, str] | None = None
        self._matrix_ids: list[int] = []

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------- метаданные

    def set_meta(self, **values: str) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO index_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [(key, str(value)) for key, value in values.items()],
            )

    def meta(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM index_meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def assert_compatible(self, *, embed_model: str, embed_dim: int) -> None:
        """Вызывается ПЕРЕД каждым поиском.

        Дешевле некуда, а спасает от единственной поломки этого слоя, которая
        не даёт ни одной ошибки в логах.
        """
        meta = self.meta()
        if not meta:
            raise IndexMismatch("индекс пуст — выполните scripts/ingest.py")
        if meta.get("embed_model") != embed_model:
            raise IndexMismatch(
                f"индекс построен моделью {meta.get('embed_model')!r}, "
                f"а сейчас настроена {embed_model!r}. Векторы разных моделей "
                f"несравнимы — нужна переиндексация."
            )
        if int(meta.get("embed_dim", 0)) != embed_dim:
            raise IndexMismatch(
                f"размерность индекса {meta.get('embed_dim')} против {embed_dim} в конфиге"
            )

    # ----------------------------------------------------------------- запись

    def upsert_document(self, doc: dict) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO documents(doc_id, title, project, owner, updated, version,
                                      status, source_path, pdf_path, page_count, ingested_at)
                VALUES(:doc_id, :title, :project, :owner, :updated, :version,
                       :status, :source_path, :pdf_path, :page_count, :ingested_at)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title, project=excluded.project, owner=excluded.owner,
                    updated=excluded.updated, version=excluded.version, status=excluded.status,
                    source_path=excluded.source_path, pdf_path=excluded.pdf_path,
                    page_count=excluded.page_count, ingested_at=excluded.ingested_at
                """,
                {**doc, "ingested_at": time.time()},
            )

    def existing_hashes(self, doc_id: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT content_hash FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        return {row["content_hash"] for row in rows}

    def drop_document_chunks(self, doc_id: str) -> None:
        # FTS и векторы подчистятся сами: первое — триггером, второе — каскадом
        # по внешнему ключу. Ровно поэтому синхронизация и живёт в схеме.
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._matrix = None
        self._chunk_docs = None

    def drop_chunks_by_hash(self, doc_id: str, hashes: list[str]) -> int:
        """Удаляет из документа именно указанные чанки.

        Нужна для повторной индексации без полной перестройки: чанки, которых
        в новой нарезке больше нет, обязаны исчезнуть. Иначе правка документа
        не заменяет текст, а добавляет к нему новую версию, и в выдаче
        оказываются два взаимоисключающих фрагмента с одинаковой пометкой
        «действующий».

        FTS и векторы подчистятся сами — триггером и каскадом по внешнему
        ключу, как и при удалении документа целиком.
        """
        if not hashes:
            return 0
        placeholders = ",".join("?" for _ in hashes)
        with self._conn:
            cursor = self._conn.execute(
                f"DELETE FROM chunks WHERE doc_id = ? AND content_hash IN ({placeholders})",
                (doc_id, *hashes),
            )
        self._matrix = None
        self._chunk_docs = None
        return cursor.rowcount

    def add_chunk(self, chunk, vector: np.ndarray) -> int:
        """Идемпотентно по хешу содержимого: повторная индексация того же
        документа не создаёт вторую копию абзаца. Без этого в выдаче окажутся
        три копии одного текста, и они вытеснят из топа всё остальное."""
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO chunks(doc_id, ordinal, heading_path, text, body,
                                   page_from, page_to, content_hash)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO NOTHING
                """,
                (chunk.doc_id, chunk.ordinal, chunk.heading_path, chunk.text,
                 chunk.body, chunk.page_from, chunk.page_to, chunk.content_hash),
            )
            if cursor.rowcount == 0:
                row = self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE content_hash = ?", (chunk.content_hash,)
                ).fetchone()
                return int(row["chunk_id"])

            chunk_id = int(cursor.lastrowid)
            # В chunks_fts не пишем: за это отвечает триггер из схемы.
            self._conn.execute(
                "INSERT INTO vectors(chunk_id, dim, vec) VALUES(?, ?, ?)",
                (chunk_id, int(vector.shape[0]), vector.astype(np.float32).tobytes()),
            )
        self._matrix = None
        self._chunk_docs = None
        return chunk_id

    # ------------------------------------------------------------------ поиск

    def _vector_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Все векторы одной матрицей.

        Векторы нормализованы при записи, поэтому косинус здесь — это обычное
        скалярное произведение, и делить на длины при каждом поиске не на что
        (раздел 3).
        """
        if self._matrix is None:
            rows = self._conn.execute(
                "SELECT chunk_id, dim, vec FROM vectors ORDER BY chunk_id"
            ).fetchall()
            if not rows:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
                self._matrix_ids = []
            else:  # noqa: RET505
                dim = int(rows[0]["dim"])
                self._matrix = np.vstack(
                    [np.frombuffer(row["vec"], dtype=np.float32).reshape(1, dim) for row in rows]
                )
                self._matrix_ids = [int(row["chunk_id"]) for row in rows]
        return self._matrix, self._matrix_ids

    def _chunk_projects(self) -> dict[int, str]:
        """Карта «кусок -> проект». По проектам и раздаются права.

        Права описаны на уровне проектов, а не отдельных документов: в
        регламенте доступ выдаётся на систему, а не на файл. Карта
        кэшируется вместе с матрицей и сбрасывается там же — расходиться им
        нельзя, иначе отсев пойдёт по устаревшим данным и пропустит то, что
        уже запрещено.
        """
        if self._chunk_docs is None:
            rows = self._conn.execute(
                "SELECT c.chunk_id, d.project FROM chunks c "
                "JOIN documents d ON d.doc_id = c.doc_id"
            ).fetchall()
            self._chunk_docs = {int(row["chunk_id"]): str(row["project"]) for row in rows}
        return self._chunk_docs

    def _denied_chunk_ids(self, denied_projects) -> set[int]:
        if not denied_projects:
            return set()
        closed = set(denied_projects)
        return {
            chunk_id
            for chunk_id, project in self._chunk_projects().items()
            if project in closed
        }

    def vector_search(
        self, query_vector: np.ndarray, limit: int, denied_projects=None
    ) -> list[tuple[int, float]]:
        """Поиск по векторам с отсевом недоступного ДО отбора лучших.

        Отсев именно здесь, а не после. Отфильтровать готовый список
        значило бы: попросили двадцать четыре, получили двадцать четыре,
        выбросили десять — и человек с ограниченными правами получает не
        «свои лучшие двадцать четыре», а огрызок чужих. Поиск при этом
        выглядит исправным, просто хуже отвечает.
        """
        matrix, ids = self._vector_matrix()
        if matrix.size == 0:
            return []
        scores = matrix @ query_vector.astype(np.float32)

        denied = self._denied_chunk_ids(denied_projects)
        if denied:
            # Запрещённым ставим минус бесконечность, а не вырезаем строки:
            # так не сбиваются индексы и не надо пересобирать матрицу на
            # каждый запрос с другими правами.
            mask = np.array([chunk_id in denied for chunk_id in ids])
            scores = np.where(mask, -np.inf, scores)

        top = np.argsort(-scores)[:limit]
        return [(ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def keyword_search(
        self, query: str, limit: int, denied_projects=None
    ) -> list[tuple[int, float]]:
        """BM25 из FTS5.

        Запрос пользователя нельзя подставлять в MATCH как есть: дефис, кавычки
        и звёздочка — операторы синтаксиса FTS5, и «ошибка E-1042» превратится
        в синтаксическую ошибку или в другой запрос. Поэтому берём токены и
        закавычиваем каждый.
        """
        tokens = FTS_TOKEN.findall(query.lower())
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens[:24])
        # Отсев по правам делает БАЗА, до сортировки и до предела: иначе
        # человек с ограничениями получает огрызок чужого списка вместо
        # своих лучших результатов.
        denied = sorted(set(denied_projects or ()))
        sql = (
            "SELECT rowid, bm25(chunks_fts, 1.0, 2.0) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ?"
        )
        params: list = [expression]
        if denied:
            placeholders = ",".join("?" * len(denied))
            sql += (
                f" AND rowid NOT IN (SELECT c.chunk_id FROM chunks c "
                f"JOIN documents d ON d.doc_id = c.doc_id "
                f"WHERE d.project IN ({placeholders}))"
            )
            params.extend(denied)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() в SQLite отдаёт тем меньше, чем релевантнее. Знак переворачиваем
        # здесь, чтобы выше по стеку «больше — лучше» было верно для обоих
        # видов поиска.
        return [(int(row["rowid"]), -float(row["score"])) for row in rows]

    def load_chunks(self, chunk_ids: list[int]) -> dict[int, StoredChunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"""
            SELECT c.chunk_id, c.doc_id, c.heading_path, c.body, c.text,
                   c.page_from, c.page_to,
                   d.title AS doc_title, d.project AS doc_project, d.version AS doc_version,
                   d.status AS doc_status, d.updated AS doc_updated
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        return {int(row["chunk_id"]): StoredChunk(**dict(row)) for row in rows}

    def all_chunks(self) -> list[StoredChunk]:
        """Весь индекс целиком. Нужен только для диагностики.

        Продовый путь так никогда не делает — там поиск. А диагностика обязана
        уметь ответить на вопрос «где вообще лежит нужный текст», в том числе
        когда поиск его не вернул: иначе «не нашлось» и «нет в индексе»
        неотличимы, а лечатся они по-разному — первое поиском, второе
        разбиением на фрагменты.
        """
        rows = self._conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.heading_path, c.body, c.text,
                   c.page_from, c.page_to,
                   d.title AS doc_title, d.project AS doc_project, d.version AS doc_version,
                   d.status AS doc_status, d.updated AS doc_updated
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            ORDER BY c.chunk_id
            """
        ).fetchall()
        return [StoredChunk(**dict(row)) for row in rows]

    # -------------------------------------------------------------- документы

    def list_documents(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_id, title, project, owner, updated, version, status, page_count, "
            "(SELECT COUNT(*) FROM chunks WHERE chunks.doc_id = documents.doc_id) AS chunk_count "
            "FROM documents ORDER BY project, doc_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def document(self, doc_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        chunks = self._conn.execute(
            "SELECT chunk_id, ordinal, heading_path, body, page_from, page_to "
            "FROM chunks WHERE doc_id = ? ORDER BY ordinal",
            (doc_id,),
        ).fetchall()
        return {**dict(row), "chunks": [dict(chunk) for chunk in chunks]}

    def stats(self) -> dict:
        documents = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"documents": int(documents), "chunks": int(chunks), "meta": self.meta()}


def dump_meta(store: Store) -> str:
    return json.dumps(store.meta(), ensure_ascii=False, indent=2)
