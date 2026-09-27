"""Трейс на запрос, спан на шаг.

Без этого разбор жалобы «вчера ответило ерунду» превращается в археологию по
логам. Нужны: что искали, что нашли, с какими баллами, какой промпт какой
версии ушёл в какую модель, сколько токенов, сколько миллисекунд, сколько
денег (раздел 7).

Трейсы пишутся построчным JSON в файл на диск. Это сознательно примитивно:
Langfuse или OpenTelemetry подключаются в шаге про наблюдаемость, и подключатся
они к этому же интерфейсу — а на старте важнее, чтобы след существовал вообще.

Персональные данные. В корпусе их нет, но привычка вырабатывается сразу:
`mask()` вызывается на всём, что пришло от пользователя, до записи. Иначе
система наблюдаемости незаметно становится ещё одним неучтённым хранилищем
персональных данных.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?<!\d)(?:\+7|8)[\s(-]?\d{3}[\s)-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")
LONG_DIGITS = re.compile(r"(?<!\d)\d{12,}(?!\d)")


def mask(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = PHONE.sub("[phone]", text)
    return LONG_DIGITS.sub("[digits]", text)


@dataclass(slots=True)
class Span:
    name: str
    started_at: float
    duration_ms: float = 0.0
    attributes: dict = field(default_factory=dict)


@dataclass(slots=True)
class Trace:
    trace_id: str
    question: str
    prompt_version: str = ""
    model: str = ""
    provider: str = ""
    spans: list[Span] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0
    finish_reason: str = ""
    stop_reason: str = ""          # почему остановились МЫ, а не модель
    status: str = ""
    # Текст неожиданного исключения. Пустой в норме; заполняется только когда
    # запрос упал не по предусмотренной причине — иначе такой запрос в журнале
    # неотличим от успешного с невыставленным статусом.
    error: str = ""
    citations_failed: int = 0
    # Ответ содержал кусок системного промпта и был подменён. Событие
    # безопасности: одиночное — осечка, серия — атака, и увидеть серию можно
    # только если каждый случай попал в журнал.
    prompt_leak: bool = False
    # Сколько предложений, отрицавших таблицу, убрано починкой ответа.
    table_denial_fixed: int = 0
    # Сколько предложений убрано за придуманное обозначение поверх таблицы.
    invented_dropped: int = 0
    # Ссылки, снятые с отказа. Отдельное число: это другой дефект, и в
    # журнале он обязан быть отличим от непрошедшей цитаты.
    citations_dropped: int = 0
    started_at: float = field(default_factory=time.monotonic)
    wall_started_at: float = field(default_factory=time.time)
    ttft_ms: float | None = None   # время до первого токена — отдельная величина
    total_ms: float = 0.0

    @contextmanager
    def span(self, name: str, **attributes):
        span = Span(name=name, started_at=time.monotonic(), attributes=dict(attributes))
        self.spans.append(span)
        try:
            yield span
        finally:
            span.duration_ms = (time.monotonic() - span.started_at) * 1000

    def mark_first_token(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = (time.monotonic() - self.started_at) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "ts": self.wall_started_at,
            "question": mask(self.question),
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            # Пишется только при неожиданном исключении. В норме ключа нет —
            # чтобы упавшие запросы можно было найти в журнале поиском по нему.
            **({"error": mask(self.error)} if self.error else {}),
            "citations_failed": self.citations_failed,
            "prompt_leak": self.prompt_leak,
            "table_denial_fixed": self.table_denial_fixed,
            "invented_dropped": self.invented_dropped,
            "citations_dropped": self.citations_dropped,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_rub": self.cost_rub,
            },
            "timing": {
                # Две принципиально разные величины, и мерить их надо
                # раздельно: время до первого токена зависит от длины промпта
                # и загрузки провайдера, скорость генерации — от модели
                # (раздел 1).
                "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
                "total_ms": round(self.total_ms, 1),
                "tokens_per_s": (
                    round(self.completion_tokens / (self.total_ms / 1000), 1)
                    if self.total_ms > 0 and self.completion_tokens
                    else None
                ),
            },
            "spans": [
                {
                    "name": span.name,
                    "duration_ms": round(span.duration_ms, 1),
                    **{k: v for k, v in span.attributes.items()},
                }
                for span in self.spans
            ],
        }


class TraceWriter:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        directory.mkdir(parents=True, exist_ok=True)

    def start(self, question: str) -> Trace:
        return Trace(trace_id=uuid.uuid4().hex[:16], question=question)

    def finish(self, trace: Trace) -> None:
        trace.total_ms = (time.monotonic() - trace.started_at) * 1000
        path = self._dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
