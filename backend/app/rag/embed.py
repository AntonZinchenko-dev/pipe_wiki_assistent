"""Эмбеддинги: пачками и с нормализацией.

Два требования раздела 3 в двадцати строках:

- пачками. Тысяча чанков по одному запросу на каждый — это тысяча
  round-trip'ов и полчаса ожидания; пачками по сотне — минуты;
- нормализованными при записи. Тогда косинусная близость это просто скалярное
  произведение, и на каждом поиске не приходится делить на длины векторов.

Отдельно: bge-m3 — асимметричная модель только в том смысле, что её принято
кормить одинаково и для документа, и для запроса. Никаких префиксов вроде
«query: » ей добавлять НЕ надо (в отличие от e5-моделей), и если однажды
модель поменяется на e5, префиксы придётся добавить — и переиндексировать всё.
"""

from __future__ import annotations

import numpy as np

from ..providers.base import EmbedProvider

BATCH_SIZE = 64


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Приводит длину каждого вектора к единице.

    Нулевые векторы (модель вернула пустоту) оставляем нулевыми, а не делим на
    ноль: такой чанк просто никогда не найдётся, и это лучше, чем NaN,
    расползающийся по всей матрице.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


async def embed_texts(provider: EmbedProvider, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, provider.embed_dim), dtype=np.float32)

    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors = await provider.embed(batch)
        chunks.append(np.asarray(vectors, dtype=np.float32))

    return normalize(np.vstack(chunks))


async def embed_query(provider: EmbedProvider, query: str) -> np.ndarray:
    vectors = await embed_texts(provider, [query])
    return vectors[0]
