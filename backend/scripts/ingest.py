#!/usr/bin/env python3
"""Индексация корпуса: PDF -> чанки -> эмбеддинги -> SQLite.

Запуск из папки backend:

    python scripts/ingest.py              # добавить новое, не трогая старое
    python scripts/ingest.py --rebuild    # перестроить с нуля
    python scripts/ingest.py --dump       # плюс выгрузить извлечённый текст

Про `--dump` отдельно. Он кладёт рядом с индексом то, что реально извлеклось
из PDF, и список выброшенных колонтитулов. Это не отладочная роскошь, а первый
пункт раздела 3: десяток извлечённых документов надо просмотреть ГЛАЗАМИ.
Автоматический пайплайн не покажет вам разорванную пополам таблицу как ошибку —
он молча проиндексирует мусор.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.providers.ollama import OllamaProvider  # noqa: E402
from app.rag.chunk import chunk_document  # noqa: E402
from app.rag.embed import embed_texts  # noqa: E402
from app.rag.extract import extract_pdf  # noqa: E402
from app.rag.store import Store  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="индексация корпуса вики")
    parser.add_argument("--rebuild", action="store_true", help="перестроить индекс с нуля")
    parser.add_argument("--dump", action="store_true", help="выгрузить извлечённый текст")
    parser.add_argument("--only", default="", help="только один doc_id (для отладки)")
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = settings.corpus_dir.resolve()
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"нет манифеста {manifest_path} — сначала соберите корпус: "
              f"python build_pdf.py в папке corpus")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if args.only:
        documents = [doc for doc in documents if doc["doc_id"] == args.only]
        if not documents:
            print(f"документ {args.only} не найден в манифесте")
            return 1

    store = Store(settings.db_path)

    # Если индекс уже построен ДРУГОЙ моделью, докладывать в него нельзя:
    # получится смесь несравнимых векторов, и поиск начнёт врать без ошибок.
    meta = store.meta()
    if meta and meta.get("embed_model") != settings.ollama_embed_model and not args.rebuild:
        print(
            f"индекс построен моделью {meta.get('embed_model')}, а настроена "
            f"{settings.ollama_embed_model}.\nВекторы разных моделей несравнимы — "
            f"нужен --rebuild."
        )
        return 2

    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0))
    provider = OllamaProvider(settings, client)

    health = await provider.health()
    if not health.get("ok"):
        print(f"Ollama недоступна: {health.get('error')}")
        print(f"Проверьте, что сервер запущен: {settings.ollama_url}")
        await client.aclose()
        return 3
    if not health.get("embed_model_present"):
        print(f"модели {settings.ollama_embed_model} нет в Ollama.")
        print(f"Установите: ollama pull {settings.ollama_embed_model}")
        print(f"Есть: {', '.join(health.get('models', [])) or '(ничего)'}")
        await client.aclose()
        return 4

    dump_dir = settings.db_path.parent / "extracted"
    if args.dump:
        dump_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    total_chunks = 0
    total_new = 0
    total_stale = 0

    try:
        for entry in documents:
            pdf_path = corpus_dir / Path(entry["pdf"]).name
            if not pdf_path.exists():
                pdf_path = corpus_dir / entry["pdf"]
            if not pdf_path.exists():
                print(f"  ! нет файла {entry['pdf']}, пропускаю")
                continue

            extracted = extract_pdf(pdf_path)
            chunks = chunk_document(extracted, doc_id=entry["doc_id"], title=entry["title"])

            store.upsert_document({
                "doc_id": entry["doc_id"],
                "title": entry["title"],
                "project": entry.get("project", ""),
                "owner": entry.get("owner", ""),
                "updated": entry.get("updated", ""),
                "version": entry.get("version", ""),
                "status": entry.get("status", "действующий"),
                "source_path": entry.get("source", ""),
                "pdf_path": entry.get("pdf", ""),
                "page_count": len(extracted.pages),
            })

            if args.rebuild:
                store.drop_document_chunks(entry["doc_id"])

            # Идемпотентность по хешу: считаем эмбеддинги только для того, чего
            # в индексе ещё нет. Повторный запуск на неизменённом корпусе не
            # тратит ни одного вызова модели.
            known = store.existing_hashes(entry["doc_id"])
            fresh = [chunk for chunk in chunks if chunk.content_hash not in known]

            # Чанки, которых в новой нарезке больше НЕТ, обязаны исчезнуть из
            # индекса. Без этого правка документа не заменяла текст, а
            # добавляла к нему новую версию: в контекст модели уезжали два
            # взаимоисключающих фрагмента, оба с меткой «действующий» и той же
            # редакцией, и какой из них процитируется — вопрос ранга.
            #
            # Ни ошибки, ни предупреждения при этом не было: повторный запуск
            # честно сообщал «новых чанков: 1». Идемпотентность по хешу
            # защищает от дублей, но сама по себе не убирает устаревшее —
            # это две разные задачи, и вторую я сначала не сделал.
            current = {chunk.content_hash for chunk in chunks}
            stale = sorted(known - current)
            if stale and not args.rebuild:
                removed = store.drop_chunks_by_hash(entry["doc_id"], stale)
                total_stale += removed

            if fresh:
                vectors = await embed_texts(provider, [chunk.text for chunk in fresh])
                for chunk, vector in zip(fresh, vectors, strict=True):
                    store.add_chunk(chunk, vector)

            total_chunks += len(chunks)
            total_new += len(fresh)
            print(
                f"  {entry['doc_id']:<12} стр. {len(extracted.pages):>2}  "
                f"чанков {len(chunks):>3}  новых {len(fresh):>3}  "
                f"колонтитулов выброшено {len(extracted.dropped_lines)}"
            )

            if args.dump:
                (dump_dir / f"{entry['doc_id']}.txt").write_text(
                    "\n\n".join(
                        f"=== страница {page.number} ===\n{page.text}"
                        for page in extracted.pages
                    ),
                    encoding="utf-8",
                )
                (dump_dir / f"{entry['doc_id']}.chunks.txt").write_text(
                    "\n\n".join(
                        f"--- [{chunk.ordinal}] {chunk.heading_path} "
                        f"(с. {chunk.page_from}–{chunk.page_to}, {len(chunk.body)} симв.)\n"
                        f"{chunk.body}"
                        for chunk in chunks
                    ),
                    encoding="utf-8",
                )
                (dump_dir / f"{entry['doc_id']}.dropped.txt").write_text(
                    "\n".join(extracted.dropped_lines), encoding="utf-8"
                )

        # Метаданные индекса пишем ПОСЛЕ успешной индексации: иначе при падении
        # на середине останется индекс, который считает себя корректным.
        store.set_meta(
            embed_model=settings.ollama_embed_model,
            embed_dim=str(settings.ollama_embed_dim),
            built_at=str(time.time()),
            corpus_dir=str(corpus_dir),
            documents=str(len(documents)),
        )
    finally:
        await client.aclose()

    stats = store.stats()
    store.close()
    print(
        f"\nготово за {time.monotonic() - started:.1f} с: "
        f"документов {stats['documents']}, чанков {stats['chunks']} "
        f"(новых в этот раз {total_new}, устаревших удалено {total_stale})"
    )
    if args.dump:
        print(f"извлечённый текст: {dump_dir} — посмотрите его глазами, это важно")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
