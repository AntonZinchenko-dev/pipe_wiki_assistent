#!/usr/bin/env python3
"""Запрос к ассистенту из терминала — проверка без фронтенда.

    python scripts/ask.py "почему нет фильтра по статусу в реестре труб"
    python scripts/ask.py --search "E-1042"

Разбор потока здесь тоже через буфер, а не по куску за раз: этот скрипт
специально написан так же, как фронтенд, чтобы на нём было видно ту же
механику (раздел 5).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

BASE = "http://127.0.0.1:8000/api"


async def do_search(question: str) -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{BASE}/search", json={"question": question})
        response.raise_for_status()
        data = response.json()

    print(f"лучшая близость {data['best_cosine']} при пороге {data['floor']} — "
          f"{'прошло' if data['passed_floor'] else 'НЕ прошло'}")
    for hit in data["hits"][:10]:
        cosine = f"{hit['cosine']:.3f}" if hit["cosine"] is not None else "  —  "
        bm25 = f"{hit['bm25']:.2f}" if hit["bm25"] is not None else "  —  "
        print(f"  cos {cosine}  bm25 {bm25:>7}  rrf {hit['rrf']:.4f}  "
              f"[{hit['found_by']:^8}] {hit['doc_id']:<12} {hit['heading_path'][:60]}")


async def do_ask(question: str, *, show_sources: bool) -> None:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{BASE}/chat", json={"question": question}) as response:
            response.raise_for_status()
            buffer = ""
            async for raw in response.aiter_text():
                buffer += raw
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    name, data = parse_event(block)
                    if name is None:
                        continue
                    handle(name, data, show_sources=show_sources)


def parse_event(block: str) -> tuple[str | None, dict]:
    name, payload = None, {}
    for line in block.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                return None, {}
    return name, payload


def handle(name: str, data: dict, *, show_sources: bool) -> None:
    if name == "meta":
        retrieval = data["retrieval"]
        print(f"[поиск] близость {retrieval['best_cosine']} / порог {retrieval['floor']}, "
              f"фрагментов в контексте {len(data['fragments'])}, "
              f"дублей убрано {retrieval['duplicates_dropped']}")
        if show_sources:
            for fragment in data["fragments"]:
                print(f"   [{fragment['number']}] {fragment['source_label']} "
                      f"({fragment['found_by']})")
        print()
    elif name == "delta":
        sys.stdout.write(data["text"])
        sys.stdout.flush()
    elif name == "answer":
        print("\n")
        print(f"[статус] {data['status']}")
        if data["citations"]:
            for citation in data["citations"]:
                mark = "ок" if citation["ok"] else f"НЕ ПРОШЛА ({citation['reason']})"
                print(f"   цитата [{citation['fragment']}] {mark}: «{citation['quote'][:80]}…»")
        if not data["schema_valid"]:
            print(f"[схема] {data['schema_error']}")
    elif name == "done":
        usage = data.get("usage", {})
        timing = data.get("timing", {})
        print(f"[итог] {data['finish_reason']} / {data['stop_reason']}, "
              f"токенов {usage.get('prompt_tokens')}+{usage.get('completion_tokens')}, "
              f"до первого токена {timing.get('ttft_ms')} мс, трейс {data.get('trace_id')}")
    elif name == "error":
        print(f"\n[ошибка {data.get('code')}] {data.get('message')}")
        if data.get("detail"):
            print(f"   {data['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="спросить ассистента")
    parser.add_argument("question")
    parser.add_argument("--search", action="store_true", help="только поиск, без генерации")
    parser.add_argument("--sources", action="store_true", help="показать источники")
    args = parser.parse_args()

    if args.search:
        asyncio.run(do_search(args.question))
    else:
        asyncio.run(do_ask(args.question, show_sources=args.sources))


if __name__ == "__main__":
    main()
