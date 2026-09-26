"""Гибридный поиск: вектор + ключевой, слияние по рангам, порог, дедупликация.

Почему не только вектор. Векторный поиск ловит смысл и теряет точные термины:
на запрос «ошибка E-1042» он найдёт «раздел про ошибки вообще», а документ, где
буквально написано `E-1042`, может не поднять. Ровно для этого класса запросов
— коды, поля API, номера регламентов вроде РЛ-4.2.3 — и нужна ключевая
половина.

Почему слияние по рангам, а не по баллам. Косинус живёт в 0…1, BM25 — в любых
положительных числах. Сложить «0,87» и «14,3» значит отдать решение той шкале,
у которой числа крупнее, то есть тихо выключить одну из двух систем поиска.
RRF складывает не баллы, а обратные ранги, и шкалы этим уравниваются.

Почему нужен абсолютный порог, а не только топ-K. «Топ-5» всегда вернёт пять
документов, даже если релевантного в базе нет вообще: это будут лучшие пять из
плохих. Модель, получив их в контекст, сочинит правдоподобный ответ, потому что
её попросили ответить, имея что-то на входе. Пустой контекст и честное
«не найдено» — лучше.

Раздел 3 чек-листа.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .store import StoredChunk, Store


@dataclass(slots=True)
class Hit:
    chunk: StoredChunk
    vector_score: float | None      # косинус, если чанк нашёлся вектором
    keyword_score: float | None     # bm25 (уже со знаком «больше — лучше»)
    vector_rank: int | None
    keyword_rank: int | None
    fused_score: float              # RRF
    # Как фрагмент попал в контекст, если НЕ через слияние двух списков.
    #
    # Появилось вместе с инструментами: модель может попросить дочитать
    # соседний раздел, и такой фрагмент не находился ни вектором, ни
    # ключевым словом. Без этого поля свойство ниже отвечало «ключевой» —
    # то есть отчёт «как получен ответ» называл бы неверный способ. Врать
    # про происхождение фрагмента нельзя: по этому полю разбирают, почему
    # нашлось не то.
    origin: str = ""

    @property
    def found_by(self) -> str:
        if self.origin:
            return self.origin
        if self.vector_rank is not None and self.keyword_rank is not None:
            return "оба"
        return "вектор" if self.vector_rank is not None else "ключевой"


@dataclass(slots=True)
class SearchResult:
    hits: list[Hit]
    best_vector_score: float
    passed_floor: bool              # прошёл ли лучший результат абсолютный порог
    floor: float
    dropped_duplicates: int
    # Порог пройден не по близости, а по точному совпадению слова. Отдельное
    # поле, а не общий флаг: это другой путь к тому же решению, и в трейсе
    # надо видеть, какой именно сработал.
    passed_by_keyword: bool = False

    @property
    def empty(self) -> bool:
        return not self.hits or not self.passed_floor


# До какого места ключевое совпадение считается сильным. Первые три: дальше
# BM25 начинает выдавать документы, где слово встретилось случайно.
KEYWORD_TRUST_RANK = 3


def single_list_can_win(k: int, depth: int) -> bool:
    """Может ли фрагмент, найденный ОДНИМ способом, обойти найденный двумя.

    Это не вопрос вкуса, а неравенство. Лучшее, что даёт один список, — это
    первое место: 1/(k+1). Худшее, что даёт присутствие в обоих, — последнее
    место в каждом: 2/(k+depth). Одиночка способен выиграть, только если

        1/(k+1) > 2/(k+depth)   ⇔   k < depth − 2

    При depth = 24 и k = 60 неравенство не выполняется: 1/61 = 0.0164 против
    2/84 = 0.0238. Значит ЛЮБОЙ фрагмент, оказавшийся в обоих списках хотя бы
    на последних местах, гарантированно обходит ЛЮБОГО, найденного одним
    способом, — даже если тот стоит первым с близостью 0.63.

    Так и случилось на вопросе про офлайн-режим: нужный фрагмент стоял в
    векторном списке ПЕРВЫМ (0.6308), в ключевой список не попал вовсе, и
    после слияния оказался пятым — позади фрагментов, стоявших в векторном
    девятым и ниже. Мы называли это гибридным поиском, а работало оно как
    «сначала пересечение, потом всё остальное».
    """
    return k < depth - 2


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], *, k: int = 60, weights: list[float] | None = None
) -> dict[int, float]:
    """RRF: вклад документа = 1 / (k + ранг), суммируется по всем спискам.

    Константа k сглаживает разницу между первым и вторым местом: без неё
    документ, оказавшийся первым в одном списке, забивал бы всё остальное.

    ОТКУДА ВЗЯЛОСЬ 60 И ПОЧЕМУ ЭТО НЕ НАШЕ ЧИСЛО. Шестьдесят — из статьи, где
    RRF применяли к спискам длиной в тысячи: там разница между местом 1 и
    местом 1000 огромна, и k = 60 её умеряет. У нас списки по 24. При k = 60
    все места ужимаются в диапазон от 1/61 до 1/84 — разброс в 1.4 раза, то
    есть МЕСТО почти перестаёт значить, а значит решает один факт: попал
    фрагмент в оба списка или нет.

    Аналогия. Двое судей выставляют оценки. Мы договорились сглаживать их
    строгость, чтобы один судья не перевешивал другого, — и сгладили так
    сильно, что все оценки от 9.84 до 9.88. Теперь побеждает не тот, кому
    поставили высший балл, а тот, кого оценили оба: два почти одинаковых
    балла складываются и бьют один высокий. Судейство формально честное, а
    результат определяется явкой судей, а не качеством выступления.

    Правильное k зависит от глубины списков — см. `single_list_can_win`.
    """
    scores: dict[int, float] = {}
    for index, ranked in enumerate(ranked_lists):
        # Вес списка. По умолчанию единица у всех — это классический RRF.
        #
        # Зачем вообще вес. Константа k правит только КРУТИЗНУ убывания внутри
        # списка и не может отменить того, что вклады СКЛАДЫВАЮТСЯ. Считанная
        # на живом примере арифметика: нужный фрагмент стоит первым в векторном
        # списке и отсутствует в ключевом, конкурент стоит девятым и вторым.
        # Одиночка выигрывает только при k = 1, то есть при полном отказе от
        # сглаживания. Значит одной константой эта болезнь не лечится: лечится
        # она либо весом второго мнения, либо гарантией (см. `fuse`).
        weight = weights[index] if weights and index < len(weights) else 1.0
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return scores


def _dedupe(hits: list[Hit]) -> tuple[list[Hit], int]:
    """Схлопывает фрагменты с одинаковым телом.

    Дубли занимают место в бюджете контекста и создают у модели ложное
    ощущение, что несколько независимых источников подтверждают одно и то же —
    хотя это один и тот же текст, найденный двумя путями.
    """
    def authority(hit: Hit) -> tuple[int, str]:
        """Чем документ авторитетнее, тем он предпочтительнее при схлопывании.

        Раньше выживал тот, кто выше по RRF, — то есть дубль решался
        случайностью ранжирования. Для корпуса регламентов это плохо: одна и
        та же формулировка честно живёт в двух документах, и выбросить можно
        было именно ДЕЙСТВУЮЩИЙ, оставив архивный. Человек получал ссылку на
        отменённый документ, и цитата при этом была совершенно верной.
        """
        return (1 if hit.chunk.doc_status == "действующий" else 0, hit.chunk.doc_updated)

    # Ключ — весь текст, а не первые 400 символов: на общей преамбуле раздела
    # два РАЗНЫХ чанка совпадали в начале и второй выбрасывался как дубль.
    best: dict[str, Hit] = {}
    order: list[str] = []
    dropped = 0

    for hit in hits:
        key = " ".join(hit.chunk.body.split())
        if key not in best:
            best[key] = hit
            order.append(key)
            continue
        dropped += 1
        if authority(hit) > authority(best[key]):
            best[key] = hit

    return [best[key] for key in order], dropped


@dataclass(frozen=True)
class Query:
    """Один запрос поиска: текст, его вектор и вес в слиянии.

    Вес нужен, потому что запросы неравноправны. Вопрос человека — главный;
    склейка «прошлый вопрос + обрывок» — подпорка, которая должна помогать
    там, где сам обрывок не находит ничего, и не должна перетягивать на
    себя, когда вопрос самостоятельный.
    """

    text: str
    vector: np.ndarray
    weight: float = 1.0


async def hybrid_search_many(
    *,
    store: Store,
    queries: list[Query],
    top_k: int,
    floor: float,
    rrf_k: int,
    keyword_weight: float = 1.0,
    promote_best_vector: bool = False,
    denied_projects=None,
) -> SearchResult:
    """Поиск НЕСКОЛЬКИМИ запросами с тем же слиянием, что и одним.

    ЗАЧЕМ. Обрывок вроде «а 10?» сам по себе не находит ничего, и раньше мы
    лечили это отдельным вызовом модели: она переписывала вопрос, подставляя
    предмет из переписки. Вызов стоил секунд и регулярно подменял тему
    целиком — «дай топ 10 труб парка» уходило в поиск как вопрос про кэш.

    Здесь нет ни генерации, ни решения модели. Есть два списка результатов
    и та же самая формула слияния, которой мы уже сливаем векторный поиск с
    ключевым: она написана, измерена и работает. Обрывок подпирается
    склейкой, самостоятельный вопрос ищется сам по себе — и если склейка
    притащила чужую тему, согласия между списками не будет, и чужое утонет.

    Это общее правило: если задачу решает арифметика над тем, что уже есть,
    её не отдают модели. Модель ошибается молча, арифметика — одинаково.
    """
    ranked: list[list[int]] = []
    weights: list[float] = []
    vector_scores: dict[int, float] = {}
    keyword_scores: dict[int, float] = {}
    vector_ranks: dict[int, int] = {}
    keyword_ranks: dict[int, int] = {}

    for query in queries:
        vector_hits = store.vector_search(
            query.vector, top_k, denied_projects=denied_projects
        )
        keyword_hits = store.keyword_search(
            query.text, top_k, denied_projects=denied_projects
        )
        ranked.append([chunk_id for chunk_id, _ in vector_hits])
        ranked.append([chunk_id for chunk_id, _ in keyword_hits])
        weights.append(query.weight)
        weights.append(query.weight * keyword_weight)

        # Отчётные числа берём ЛУЧШИЕ по всем запросам.
        #
        # «Лучшая близость» в отчёте отвечает на вопрос «насколько близко
        # вообще что-то подошло». Взять её по первому запросу значило бы
        # сказать «0.21, ниже порога» там, где склейка нашла фрагмент с
        # 0.71, — и порог отсёк бы найденное.
        for rank, (chunk_id, score) in enumerate(vector_hits, start=1):
            if score > vector_scores.get(chunk_id, -1.0):
                vector_scores[chunk_id] = score
                vector_ranks[chunk_id] = rank
        for rank, (chunk_id, score) in enumerate(keyword_hits, start=1):
            if score > keyword_scores.get(chunk_id, -1.0):
                keyword_scores[chunk_id] = score
                keyword_ranks[chunk_id] = rank

    return fuse_ranked(
        store=store,
        ranked=ranked,
        weights=weights,
        vector_scores=vector_scores,
        keyword_scores=keyword_scores,
        vector_ranks=vector_ranks,
        keyword_ranks=keyword_ranks,
        top_k=top_k,
        floor=floor,
        rrf_k=rrf_k,
        promote_best_vector=promote_best_vector,
    )


async def hybrid_search(
    *,
    store: Store,
    query: str,
    query_vector: np.ndarray,
    top_k: int,
    floor: float,
    rrf_k: int,
    keyword_weight: float = 1.0,
    promote_best_vector: bool = False,
    denied_projects=None,
) -> SearchResult:
    """Гибридный поиск: два списка и слияние. Достаёт и сливает.

    Все настройки слияния обязаны быть здесь и прокидываться дальше НАСКВОЗЬ.
    Когда я добавил вес ключевого списка только в `fuse`, прод и стенд упали с
    `unexpected keyword argument`: они ходят не в `fuse`, а сюда. Тесты этого
    не поймали, потому что ни один из них не вызывал эту функцию — они шли в
    `fuse` напрямую, мимо настоящего входа.
    """
    # Права отсекают ОБЕ половины поиска, и обязательно до слияния.
    #
    # Отсечь одну означало бы, что закрытый фрагмент всё равно поднимается
    # вторым списком. Отсечь после слияния означало бы отдать человеку не
    # его лучшие двадцать четыре, а огрызок чужих: поиск при этом выглядит
    # исправным, просто отвечает хуже.
    vector_hits = store.vector_search(query_vector, top_k, denied_projects=denied_projects)
    keyword_hits = store.keyword_search(query, top_k, denied_projects=denied_projects)
    return fuse(
        store=store,
        vector_hits=vector_hits,
        keyword_hits=keyword_hits,
        top_k=top_k,
        floor=floor,
        rrf_k=rrf_k,
        keyword_weight=keyword_weight,
        promote_best_vector=promote_best_vector,
    )


def fuse(
    *,
    store: Store,
    vector_hits: list[tuple[int, float]],
    keyword_hits: list[tuple[int, float]],
    top_k: int,
    floor: float,
    rrf_k: int,
    keyword_weight: float = 1.0,
    promote_best_vector: bool = False,
) -> SearchResult:
    """Слияние двух готовых списков. Отделено от их получения НАРОЧНО.

    Достать списки стоит вызова модели эмбеддингов, а слить — ничего. Пока
    одно было сцеплено с другим, проверить другую константу слияния можно было
    только полным прогоном: 67 обращений к модели ради арифметики, которая
    считается мгновенно. Теперь перебор констант — это перебор, а не серия
    прогонов: вопросы векторизуются один раз, а слияние пересчитывается
    сколько нужно.

    Это общее правило: дорогое отделяют от дешёвого, иначе дешёвое становится
    дорогим и его перестают проверять.
    """
    return fuse_ranked(
        store=store,
        ranked=[
            [chunk_id for chunk_id, _ in vector_hits],
            [chunk_id for chunk_id, _ in keyword_hits],
        ],
        weights=[1.0, keyword_weight],
        vector_scores=dict(vector_hits),
        keyword_scores=dict(keyword_hits),
        vector_ranks={
            chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_hits, start=1)
        },
        keyword_ranks={
            chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_hits, start=1)
        },
        top_k=top_k,
        floor=floor,
        rrf_k=rrf_k,
        promote_best_vector=promote_best_vector,
    )


def fuse_ranked(
    *,
    store: Store,
    ranked: list[list[int]],
    weights: list[float],
    vector_scores: dict[int, float],
    keyword_scores: dict[int, float],
    vector_ranks: dict[int, int],
    keyword_ranks: dict[int, int],
    top_k: int,
    floor: float,
    rrf_k: int,
    promote_best_vector: bool = False,
) -> SearchResult:
    """Слияние ПРОИЗВОЛЬНОГО числа списков. Ядро, общее для всех поисков.

    Выделено из `fuse`, когда появился поиск двумя запросами. Списков стало
    четыре вместо двух, а формула та же — и дублировать её ради этого было
    бы худшим решением: две копии RRF разойдутся на первой же правке
    константы, и расхождение будет видно не в коде, а в замерах через месяц.
    """
    fused = reciprocal_rank_fusion(ranked, k=rrf_k, weights=weights)
    if not fused:
        return SearchResult([], 0.0, False, floor, 0)

    ordered_ids = sorted(fused, key=lambda cid: -fused[cid])[:top_k]
    chunks = store.load_chunks(ordered_ids)

    hits = [
        Hit(
            chunk=chunks[chunk_id],
            vector_score=vector_scores.get(chunk_id),
            keyword_score=keyword_scores.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id),
            keyword_rank=keyword_ranks.get(chunk_id),
            fused_score=fused[chunk_id],
        )
        for chunk_id in ordered_ids
        if chunk_id in chunks
    ]
    hits, dropped = _dedupe(hits)

    if promote_best_vector and vector_scores:
        # ГАРАНТИЯ: лучший по смыслу фрагмент обязан дойти до модели.
        #
        # Слияние по природе своей награждает СОГЛАСИЕ списков, а не силу
        # одного мнения: фрагмент, стоящий в векторном списке первым и не
        # попавший в ключевой, проигрывает фрагменту с девятого и второго
        # мест при любом разумном k. Константа тут бессильна — складываются
        # два числа против одного.
        #
        # ИЗМЕРЕНИЕ ЭТО ОПРОВЕРГЛО, и потому по умолчанию гарантия ВЫКЛЮЧЕНА:
        # перебор 42 вариантов дал chunk_top1 = 0.851 с ней против 0.866 без.
        # «Лучший по близости» и «нужный» — часто разные фрагменты, и двигая
        # вперёд первого, мы сталкивали с первого места второго. Ход оставлен
        # включаемым: он остаётся лекарством там, где ключевой поиск слеп
        # (жаргон, названия), — но включать его надо по замеру на таком
        # классе запросов, а не по красоте формулировки.
        best_id = max(vector_scores, key=lambda cid: vector_scores[cid])
        for index, hit in enumerate(hits):
            if hit.chunk.chunk_id == best_id:
                if index:
                    hits.insert(0, hits.pop(index))
                break

    # Порог считаем по КОСИНУСУ, а не по RRF: у RRF нет физического смысла,
    # его значение зависит только от позиций в списках и всегда «нормальное».
    best_vector = max(vector_scores.values(), default=0.0)

    # НО косинуса одного недостаточно, и это была серьёзная ошибка.
    #
    # Порог по косинусу выключал ключевую половину поиска целиком — ровно на
    # том классе запросов, ради которого она в системе и есть. Запрос «E-1042»
    # короткий, чанк длинный, косинус выходит около 0.38 и порог не проходит,
    # хотя BM25 нашёл точное текстовое совпадение и поставил его первым.
    # Система отвечала «в вики такого нет» при точном попадании в индексе.
    #
    # Поэтому порог проходит, если сработала ЛЮБАЯ из двух половин: близость
    # по смыслу выше порога ИЛИ есть сильное ключевое совпадение в первых
    # местах. Это не ослабление порога: у него была одна задача — отсекать
    # выдачу, в которой нет ничего похожего, а точное совпадение слова как раз
    # и означает «похожее есть».
    strong_keyword = any(
        hit.keyword_rank is not None and hit.keyword_rank <= KEYWORD_TRUST_RANK
        for hit in hits
    )

    return SearchResult(
        hits=hits,
        best_vector_score=best_vector,
        passed_floor=best_vector >= floor or strong_keyword,
        passed_by_keyword=best_vector < floor and strong_keyword,
        floor=floor,
        dropped_duplicates=dropped,
    )
