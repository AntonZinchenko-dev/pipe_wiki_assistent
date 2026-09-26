"""Золотой набор: загрузка и проверка на пригодность.

Разметка сделана по СТАБИЛЬНЫМ идентификаторам: `docs` — список документов,
в которых лежит ответ, `must_contain` — подстроки, которые обязаны оказаться
в найденном контексте.

Про два разных поля отдельно, потому что это неочевидно и мы на этом
попались. `must_contain` размечен ФОРМУЛИРОВКОЙ КОРПУСА: он проверяет, доехал
ли нужный текст до модели, и формулировка там ровно такая, как в документе
(«не заменяет», «не предусмотрено»). Требовать той же формулировки от ОТВЕТА
нельзя: ответ — это пересказ, и «расчёт не заменяет осмотр» он законно передаёт
словами «труба не списывается по расчёту». Поэтому для проверки ответа есть
отдельное поле `answer_must_contain`, размеченное так, как ответ звучал бы
по-человечески; если его нет, берётся `must_contain`.

Смешивать их в одном поле — значит получать ложные тревоги на правильных
ответах и потом объяснять себе, что метрика «немного шумит». Метрика не шумит,
метрика мерит не то. Ни одного chunk_id в разметке нет сознательно: чанки
перенумеровываются при каждой переиндексации, и набор, размеченный по ним,
сломается на первом же изменении чанкинга. Разметка обязана переживать
переиндексацию — иначе это не эталон, а снимок одного прогона.

Отдельная и обязательная процедура — `validate`. Золотой набор, в котором
ожидаемая подстрока не встречается в корпусе вообще, измеряет не систему, а
собственные опечатки. Проверять его надо при каждом изменении корпуса, и это
дешевле, чем один раз поверить в неверные метрики.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")


def dataset_version(path: Path = GOLDEN_PATH) -> str:
    """Отпечаток разметки. Попадает в конфигурацию каждого прогона.

    Зачем это нужно — и это важнее, чем кажется.

    Разметка золотого набора — это ЛИНЕЙКА. Пока она не меняется, два прогона
    сравнимы. Стоит поправить одну подстроку — и метрики двух прогонов измерены
    разными линейками, хотя в конфигурации это никак не видно.

    А соблазн править линейку огромен, потому что каждая правка выглядит
    обоснованной: «ну тут действительно разметка была слишком строгой».
    Несколько таких правок подряд — и цифра растёт, не потому что система стала
    лучше, а потому что мерить стали мягче. Это самый тихий способ обмануть
    себя в измерениях, и от него не спасает добросовестность: правки-то
    честные, беда в их накоплении.

    Единственная защита — сделать изменение линейки ВИДИМЫМ. Отпечаток в
    конфигурации превращает правку разметки из незаметной в такую, о которой
    инструмент сравнения скажет вслух.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def label_versions(path: Path = GOLDEN_PATH) -> dict[str, str]:
    """Два отпечатка разметки вместо одного, и это не педантизм.

    Разметка влияет на метрики по-разному, и пересчитать задним числом можно
    только половину.

    - Разметка ПОИСКА (`docs`, `must_contain`) определяет context_hit, recall,
      chunk_rank. Считается она по фрагментам, которых в файле прогона нет, —
      значит после её правки прогон надо делать заново, пересчёт невозможен.
    - Разметка ОТВЕТА (`answer_must_contain`) определяет answer_contains.
      Считается по тексту ответа, а он в файле есть, — значит пересчитывается
      без модели.

    Пока отпечаток был один, `recheck` штамповал его целиком, пересчитав
    только вторую половину. После правки `must_contain` и двух recheck-ов
    compare перестаёт говорить «разметка разная», хотя метрики поиска в файлах
    по-прежнему посчитаны по разным линейкам. То есть защита от подмены
    линейки сама начинала врать — ровно в том случае, для которого она и
    заведена.
    """
    retrieval: list[str] = []
    answer: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("//"):
            continue
        payload = json.loads(raw)
        retrieval.append(
            json.dumps(
                [payload["id"], payload.get("docs", []), payload.get("must_contain", []),
                 payload.get("answerable"), payload.get("critical", False)],
                ensure_ascii=False, sort_keys=True,
            )
        )
        answer.append(
            json.dumps(
                [payload["id"], payload.get("answer_must_contain", [])],
                ensure_ascii=False, sort_keys=True,
            )
        )

    def digest(parts: list[str]) -> str:
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]

    return {"retrieval": digest(retrieval), "answer": digest(answer)}

QUESTION_TYPES = {
    "fact",          # простой факт из одного раздела
    "negation",      # различается отрицанием с парным вопросом
    "exact_code",    # точный артикул: код ошибки, имя поля, номер регламента
    "procedure",     # порядок действий
    "multi_doc",     # ответ требует двух и более документов
    "version",       # в корпусе есть устаревшая и действующая версия факта
    "unanswerable",  # ответа в корпусе нет
    "followup",      # обрывок, понятный только с переписью: «а 10?»
    "topic_switch",  # новая тема после другой: переписка обязана НЕ мешать
    "live",          # ответом служат живые данные сервиса, а не текст
}

DIFFICULTIES = {"easy", "medium", "hard"}

# Требование к различающей силе подстроки разметки поиска.
#
# Подстрока обязана указывать на ОДНО место в корпусе, а не совпадать где
# угодно. Иначе метрики, построенные на ней, превращаются в константу, и это
# не видно: цифра есть, она правдоподобная, и меняться она не будет никогда.
#
# Мерой служит ЧИСЛО ВХОЖДЕНИЙ, а не длина. Длина — плохой признак: код
# `E-1042` из шести символов встречается в корпусе дважды и указывает
# идеально, а цифра «3» из одного символа встречается сотни раз. Запрет по
# длине забраковал бы лучшие подстроки набора и пропустил худшие.
MAX_NEEDLE_OCCURRENCES = 8


@dataclass(slots=True)
class Question:
    id: str
    question: str
    type: str
    difficulty: str
    answerable: bool
    docs: list[str] = field(default_factory=list)
    # Формулировка КОРПУСА: проверяет, что нужный текст доехал до модели.
    must_contain: list[str] = field(default_factory=list)
    # Формулировка ОТВЕТА: проверяет, что факт оказался в тексте ответа.
    # Пусто — значит подходит та же формулировка, что и в корпусе.
    answer_must_contain: list[str] = field(default_factory=list)
    critical: bool = False
    # НАСТРОЕЧНАЯ или ОТЛОЖЕННАЯ часть набора.
    #
    # Зачем делить. Порог отсечения, константу слияния и вес ключевого списка
    # мы выбирали перебором ПО ТОМУ ЖЕ набору, на котором потом докладывали
    # результат. Это подгонка по определению: из 50 вариантов всегда найдётся
    # тот, что на 67 вопросах чуть лучше, и его преимущество может целиком
    # состоять из особенностей этих 67.
    #
    # Аналогия: составить контрольную по тем задачам, которые разобрал на
    # уроке, а потом объявить высокие оценки доказательством знаний. Оценки
    # настоящие, вывод — нет.
    #
    # Правило: настраиваем ТОЛЬКО на tune, докладываем на обеих частях. Если
    # на отложенной выигрыш исчез — выигрыша не было, была подгонка.
    #
    # Критичные вопросы всегда в настроечной: их мы смотрим поимённо в каждом
    # прогоне, значит «не подглядывать» в них невозможно, и делать вид, что
    # они отложены, — самообман.
    split: str = "tune"
    note: str = ""
    # Переписка ПЕРЕД этим вопросом: [{"role": "user", "content": "..."}, ...].
    #
    # Без неё половина системы не измеряется вовсе. Обрывок «а 10?» и
    # смена темы после трёх реплик — самые частые жалобы в живых прогонах,
    # и ровно они не попадали ни в один замер: набор состоял из одиночных
    # вопросов, заданных на пустом месте.
    #
    # Мы почти год чинили работу с переписью по логам, то есть по жалобам,
    # и не могли отличить «стало лучше» от «жалобы сменили форму».
    history: list[dict] = field(default_factory=list)
    # Сколько таблиц должно прийти. None — вопрос не про живые данные.
    #
    # Ноль — полноценное ожидание, а не «не проверяем»: на вопрос по
    # документам агент в сервис ходить не должен, и лишняя таблица здесь
    # такой же дефект, как её отсутствие там, где она нужна.
    expect_tables: int | None = None
    # Какие инструменты обязаны быть вызваны. Проверяется вхождением, а не
    # равенством: лишний поиск по вики ошибкой не считается, а вот поход не
    # в тот эндпоинт — считается.
    expect_tools: list[str] = field(default_factory=list)

    @property
    def answer_needles(self) -> list[str]:
        """Подстроки для проверки ОТВЕТА. Подмены разметкой поиска больше нет.

        Раньше здесь стоял откат на `must_contain`, и это оказалось генератором
        тихих ошибок. Два требования тянут поле в противоположные стороны:

        - подстрока ПОИСКА обязана быть длинной, дословной и различающей —
          иначе context_hit превращается в константу;
        - подстрока ОТВЕТА обязана быть короткой и терпимой к словоформе —
          иначе правильный пересказ объявляется врущим.

        Пока `must_contain` был коротким, откат работал. Как только я сделал
        его длинной цитатой из документа (а это было обязательно), откат начал
        проверять пересказ на дословность документа — и `answer_contains` упал
        с 0.925 до 0.746 на 12 совершенно верных ответах.

        Вывод шире случая: две проверки с противоположными требованиями не
        могут делить одно поле, а откат между ними — не удобство, а
        отложенная ошибка. Поэтому отката нет, а `validate` требует разметку
        ответа у каждого отвечаемого вопроса.
        """
        return self.answer_must_contain

    @property
    def expected_status(self) -> set[str]:
        """Какие статусы ответа считаются правильными.

        Для неотвечаемого вопроса правильными считаются оба отказа: `not_found`
        (модель посмотрела контекст и сказала «нет ответа») и `no_context`
        (поиск не дал ничего выше порога, и модель вообще не вызывалась).
        Это разные пути к одному верному поведению, и требовать конкретный —
        значит наказывать систему за то, что она сэкономила вызов модели.
        """
        return {"answered"} if self.answerable else {"not_found", "no_context"}


SPLITS = frozenset({"tune", "holdout"})

# Тексты типичных ОТКАЗОВ. Разметка ответа, которая срабатывает на любом из
# них, не отличает ответ от отказа — то есть не проверяет ничего.
#
# Дефект был найден автоматически и оказался массовым: семь вопросов из ста, в
# том числе ДВА КРИТИЧНЫХ. Причина в одном символе: на вопрос «дают ли новичку
# токен» правильный ответ начинается со слова «нет», и я записал «нет» в
# разметку как допустимый вариант. Но слово «нет» есть и в «сведений об этом
# нет» — то есть проверка одинаково принимала верный ответ и отказ.
#
# Особенно зло то, что это ровно те вопросы, где модель и путается: она
# принимает отрицательный ответ за отказ, мы это уже ловили на q020 и q066.
# Проверка, которая должна была стеречь именно этот класс, была к нему слепа.
#
# Аналогия: тест на дальтонизм, напечатанный серым по серому. Проходят все,
# различает он ноль.
REFUSAL_SAMPLES = (
    "Конкретных сведений об этом в документах нет.",
    "В предоставленных фрагментах ответа на этот вопрос не найдено.",
    "Явно это в документации не указано.",
)


def load(path: Path = GOLDEN_PATH) -> list[Question]:
    questions: list[Question] = []
    seen: set[str] = set()

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("//"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}, строка {number}: {error}") from error

        question = Question(**payload)
        if question.id in seen:
            raise ValueError(f"{path.name}: повторяющийся id {question.id}")
        seen.add(question.id)

        if question.split not in SPLITS:
            raise ValueError(
                f"{question.id}: неизвестная часть набора {question.split!r}, "
                f"ожидалось одно из {sorted(SPLITS)}"
            )
        if question.critical and question.split != "tune":
            raise ValueError(
                f"{question.id}: критичный вопрос не может быть отложенным — "
                f"его смотрят поимённо в каждом прогоне"
            )
        if question.type not in QUESTION_TYPES:
            raise ValueError(f"{question.id}: неизвестный тип {question.type!r}")
        if question.difficulty not in DIFFICULTIES:
            raise ValueError(f"{question.id}: неизвестная сложность {question.difficulty!r}")
        for turn in question.history:
            if turn.get("role") not in {"user", "assistant"} or not turn.get("content"):
                raise ValueError(
                    f"{question.id}: реплика переписки должна быть "
                    f'{{"role": "user"|"assistant", "content": "..."}}'
                )
        if question.type in {"followup", "topic_switch"} and not question.history:
            raise ValueError(
                f"{question.id}: вопрос типа {question.type} без переписки "
                f"измеряет не то, ради чего заведён"
            )
        if question.type == "live" and question.expect_tables is None:
            raise ValueError(
                f"{question.id}: вопрос про живые данные без ожидания по таблицам — "
                f"проверять в нём нечего (ноль тоже ожидание)"
            )
        if question.expect_tools and question.expect_tables == 0:
            raise ValueError(
                f"{question.id}: ждём инструменты и ноль таблиц одновременно — "
                f"это противоречивая разметка"
            )
        # Вопрос про живые данные вправе не иметь ожидаемых документов.
        #
        # Это не поблажка разметке, а описание того, как система устроена:
        # «дай топ 5 труб» отвечается сервисом, и в корпусе такого текста
        # нет ни в одном документе. Требовать документ здесь значило бы
        # заставить разметчика выдумать его — и получить ложную метрику
        # поиска на вопросе, где поиск ни при чём.
        #
        # Но только когда таблицы ожидаются. «Когда труба списывается» —
        # тоже live-вопрос, ноль таблиц, и документ у него обязан быть: это
        # ловушка на то, чтобы агент НЕ ходил в сервис за правилом.
        answered_by_service = question.type == "live" and (question.expect_tables or 0) > 0
        if question.answerable and not question.docs and not answered_by_service:
            raise ValueError(f"{question.id}: отвечаемый вопрос без ожидаемых документов")
        if not question.answerable and (question.docs or question.must_contain):
            raise ValueError(f"{question.id}: неотвечаемый вопрос с разметкой ответа")
        if question.answerable and question.answer_needles:
            # Разметка обязана ОТЛИЧАТЬ ответ от отказа.
            from .metrics import answer_contains

            for sample in REFUSAL_SAMPLES:
                if answer_contains(sample, question.answer_needles):
                    raise ValueError(
                        f"{question.id}: разметка ответа {question.answer_needles} "
                        f"срабатывает на отказе ({sample!r}) — она не различает "
                        f"ответ и «сведений нет». Уберите одинокие «нет», «не», "
                        f"«без» и требуйте слово из документа"
                    )

        if question.answerable and question.answer_needles:
            # Разметка обязана требовать ФАКТ, а не повтор термина из вопроса.
            #
            # Дыра того же рода, что и слепота к отказу, только шире. Вопрос
            # «что такое critical_local_position_m» с разметкой
            # `critical_local_position_m` проходит на ответе, состоящем из
            # одного этого слова. Так и вышло: модель ответила ровно
            # «critical_local_position_m» — без единого слова о том, что это
            # такое, — и ОБА прибора, подстрока и судья, зачли ответ верным.
            # Поймал человек, глазами.
            #
            # Таких разметок нашлось девять, включая два КРИТИЧНЫХ вопроса:
            # «можно ли выкладывать в пятницу» с разметкой «пятниц» и «что
            # тяжелее» с разметкой «недостоверн». Оба проходили на ответе,
            # который просто повторяет слово из вопроса.
            #
            # Аналогия: экзаменатор спрашивает «что такое интеграл?» и ставит
            # зачёт за ответ «интеграл». Формально слово названо.
            #
            # Проверяем целиком: разметка сломана, если КАЖДАЯ её подстрока
            # содержится в тексте вопроса. Одна тривиальная подстрока рядом с
            # содержательной безобидна — требуются обе.
            def _flat(text: str) -> str:
                return " ".join(text.split()).lower().replace("ё", "е")

            asked = _flat(question.question)
            if all(
                all(
                    _flat(variant) in asked
                    for variant in needle.split("|")
                    if variant.strip()
                )
                for needle in question.answer_needles
            ):
                raise ValueError(
                    f"{question.id}: разметка ответа {question.answer_needles} "
                    f"целиком содержится в самом вопросе — её пройдёт ответ, "
                    f"просто повторивший термин. Требуйте факт из документа"
                )

        for needle in question.must_contain + question.answer_must_contain:
            if "`" in needle or "**" in needle:
                # РАЗМЕТКА MARKDOWN В РАЗМЕТКЕ НАБОРА — тихий брак.
                #
                # В индекс текст попадает ПОСЛЕ извлечения, где вёрстка снята:
                # обратные кавычки вокруг кода и звёздочки выделения в чанке
                # отсутствуют. Игла с кавычкой не найдётся никогда, а выглядит
                # это как «поиск не доносит нужный текст» — то есть отправляет
                # чинить исправный поиск.
                #
                # Меня это поймало на четырёх вопросах сразу, причём моя
                # собственная проверка их пропустила: я сравнивал разметку с
                # файлами корпуса, а система видит индекс. Проверять надо то,
                # что читает система, а не то, что написано в источнике.
                raise ValueError(
                    f"{question.id}: в разметке есть вёрстка markdown ({needle!r}). "
                    f"В индексе её нет — цитату надо брать без кавычек и звёздочек"
                )
        if question.answer_must_contain and not question.must_contain:
            raise ValueError(
                f"{question.id}: есть answer_must_contain без must_contain — "
                f"проверять ответ, не проверив контекст, бессмысленно"
            )
        if question.answerable and not question.answer_must_contain and not answered_by_service:
            raise ValueError(
                f"{question.id}: отвечаемый вопрос без answer_must_contain. "
                f"Разметка ответа обязательна и НЕ берётся из must_contain: "
                f"там формулировка документа, а ответ — это пересказ"
            )
        if answered_by_service and question.answer_must_contain:
            # Подстрока в ответе на такой вопрос — ловушка для нас самих.
            #
            # Ответом служит таблица, а текст под ней — вывод из документов
            # или честное «данные в таблице выше». Потребовав в нём «PP-0035»,
            # мы бы наградили ровно то поведение, от которого уходили:
            # пересказ строк таблицы словами. Проверять надо таблицу.
            raise ValueError(
                f"{question.id}: у вопроса про живые данные размечен ответ "
                f"подстрокой. Ответом служит таблица — проверяйте её "
                f"(expect_tables, live_clean), а не пересказ"
            )

        questions.append(question)

    return questions


def composition(questions: list[Question]) -> dict:
    """Состав набора — то, что проверяют до первого прогона.

    Чек-лист требует 10–15 % вопросов без ответа в корпусе. Если этой доли
    нет, система не наказывается за привычку всегда что-то отвечать, и
    измерить эту привычку нечем.
    """
    total = len(questions)
    unanswerable = sum(1 for question in questions if not question.answerable)
    by_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for question in questions:
        by_type[question.type] = by_type.get(question.type, 0) + 1
        by_difficulty[question.difficulty] = by_difficulty.get(question.difficulty, 0) + 1

    return {
        "total": total,
        "unanswerable": unanswerable,
        "unanswerable_share": round(unanswerable / total, 3) if total else 0.0,
        "critical": sum(1 for question in questions if question.critical),
        "by_type": dict(sorted(by_type.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
    }


@dataclass(slots=True)
class LabelProblem:
    question_id: str
    kind: str
    detail: str


def validate(questions: list[Question], store) -> list[LabelProblem]:
    """Проверяет, что разметка вообще достижима на текущем индексе.

    Три вида брака, каждый из которых делает метрику ложью:

    - ожидаемый документ отсутствует в индексе;
    - ожидаемая подстрока не встречается ни в одном чанке ожидаемых
      документов (опечатка в разметке, или текст поехал при извлечении);
    - вопрос помечен как неотвечаемый, но его формулировка почти дословно
      лежит в корпусе.
    """
    problems: list[LabelProblem] = []
    documents = {document["doc_id"] for document in store.list_documents()}


    bodies: dict[str, str] = {}
    for doc_id in documents:
        detail = store.document(doc_id) or {}
        bodies[doc_id] = " ".join(
            " ".join(chunk["body"].split()) for chunk in detail.get("chunks", [])
        ).lower()

    for question in questions:
        for doc_id in question.docs:
            if doc_id not in documents:
                problems.append(
                    LabelProblem(question.id, "нет документа", f"{doc_id} отсутствует в индексе")
                )

        for needle in question.must_contain:
            normalized = " ".join(needle.split()).lower()
            found_in = [
                doc_id for doc_id in question.docs if normalized in bodies.get(doc_id, "")
            ]
            if not found_in:
                elsewhere = [
                    doc_id for doc_id, body in bodies.items() if normalized in body
                ]
                problems.append(
                    LabelProblem(
                        question.id,
                        "подстрока не найдена",
                        f"{needle!r} нет в {question.docs}"
                        + (f", зато есть в {elsewhere}" if elsewhere else " и вообще в корпусе"),
                    )
                )

        # РАЗЛИЧАЮЩАЯ СИЛА подстроки — обязательная проверка, и её не было.
        #
        # `context_hit` ищет подстроку по всему собранному контексту, а
        # `chunk_rank` — по чанкам подряд. Значит подстрока «3» превращает обе
        # метрики в константу: в корпусе из шестнадцати документов цифра 3
        # встречается во всех шестнадцати, context_hit тождественно равен
        # единице, а место нужного чанка — всегда первое. Метрика при этом
        # выглядит измеренной и даже ненасыщаемой.
        #
        # Хуже всего то, что проверка критичных вопросов в гейте построена
        # ровно на `context_hit`: два критичных вопроса были размечены
        # подстроками «14» и «50». Нужный документ мог не попасть в контекст
        # вообще — гейт оставался зелёным.
        #
        # Достижимость подстроки (выше) и её различающая сила — разные
        # свойства. Первое проверялось с самого начала, второе нет.
        for needle in question.must_contain:
            normalized = " ".join(needle.split()).lower()
            occurrences = sum(body.count(normalized) for body in bodies.values())
            if occurrences > MAX_NEEDLE_OCCURRENCES:
                problems.append(
                    LabelProblem(
                        question.id, "подстрока не различает",
                        f"{needle!r} встречается в корпусе {occurrences} раз — "
                        f"context_hit и место чанка станут константой, "
                        f"нужна цитата поконкретнее",
                    )
                )

        # `answer_must_contain` по корпусу НЕ проверяется: это формулировка
        # ответа, и её отсутствия в документе ожидать нормально. Проверяем
        # только явный брак разметки.
        for needle in question.answer_must_contain:
            if not needle.strip() or any(
                not part.strip() for part in needle.split("|")
            ):
                problems.append(
                    LabelProblem(
                        question.id, "пустая подстрока ответа", f"{needle!r}"
                    )
                )

    return problems


def tune(questions: list[Question]) -> list[Question]:
    """Только настроечная часть. Всё, что ВЫБИРАЕТ настройки, обязано звать её.

    Функция существует затем, чтобы правило «настраиваемся не на всём наборе»
    было видно в коде вызова, а не держалось в голове. Забыть её — значит
    вернуть подгонку, и это будет незаметно: цифры останутся красивыми.
    """
    return [question for question in questions if question.split == "tune"]


def holdout(questions: list[Question]) -> list[Question]:
    """Только отложенная часть. На ней ДОКЛАДЫВАЮТ, а не выбирают."""
    return [question for question in questions if question.split == "holdout"]
