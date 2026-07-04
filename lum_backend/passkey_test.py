"""
Passkey-тест для Lum (в стиле SelfExtend / LongLM, arXiv:2401.01325).

ИДЕЯ (и почему это важно):
  Гладкий, красиво звучащий ответ ≠ правильный ответ. Модель может иметь низкую
  perplexity и при этом НЕ находить конкретный факт, зарытый в длинный текст.
  Поэтому качество RAG честно меряется так: прячем уникальный факт ("passkey")
  в середину длинного отвлекающего текста и проверяем — достаёт ли система
  ИМЕННО его, а не пересказывает общие места вокруг.

ЧТО ДЕЛАЕТ СКРИПТ:
  1. Собирает длинный текст из «наполнителя» + вставляет секретный факт в середину.
  2. Прогоняет пайплайн Lum с вопросом ровно про этот факт.
  3. Проверяет, содержит ли ответ ожидаемый ключ (passkey).
  4. Печатает вердикт PASS/FAIL + где именно факт «всплыл» (ответ / граф / схема).

ЗАПУСК:
  export GROQ_API_KEY=...        # тот же ключ, что и для сервера
  python passkey_test.py                 # один прогон с дефолтным фактом
  python passkey_test.py --position 0.1  # факт ближе к началу
  python passkey_test.py --demo          # подробный вывод для питча

Это НЕ юнит-тест на CI (он ходит в Groq и стоит вызовов) — это инструмент оценки
качества и демонстрации. Гоняй вручную.
"""

import argparse
import json
import os
import sys

import core


# ─── МАТЕРИАЛ ДЛЯ ТЕСТА ───────────────────────────────────────────────────────

# «Наполнитель» — правдоподобный, но не относящийся к факту текст.
# Нужен, чтобы факт реально утонул в контексте, а не лежал в единственном абзаце.
FILLER_PARAGRAPHS = [
    "Машинное обучение изучает алгоритмы, которые улучшаются на основе данных. "
    "Обучение с учителем использует размеченные примеры, обучение без учителя ищет "
    "структуру в данных, а обучение с подкреплением опирается на систему наград.",

    "Нейронные сети состоят из слоёв связанных нейронов. Обучение происходит через "
    "обратное распространение ошибки и градиентный спуск, которые постепенно "
    "корректируют веса соединений для уменьшения ошибки предсказания.",

    "Свёрточные сети хорошо подходят для изображений, поскольку используют локальные "
    "фильтры и разделяемые веса. Рекуррентные сети обрабатывают последовательности, "
    "сохраняя скрытое состояние между шагами.",

    "Трансформеры заменили рекуррентные архитектуры в обработке языка. Механизм "
    "самовнимания позволяет каждому токену взаимодействовать со всеми остальными, "
    "а многоголовое внимание применяет несколько таких механизмов параллельно.",

    "Векторные базы данных хранят эмбеддинги и обеспечивают поиск по косинусному "
    "сходству. Это основа современных систем поиска и рекомендаций, где близость "
    "векторов отражает семантическую близость объектов.",

    "Регуляризация помогает бороться с переобучением. Методы вроде dropout, L2-штрафа "
    "и ранней остановки ограничивают сложность модели, улучшая обобщение на новых данных.",

    "Оценка моделей требует разделения данных на обучающую, валидационную и тестовую "
    "выборки. Метрики точности, полноты и F1 позволяют сравнивать модели между собой.",
]

# Секретный факт — уникальный, его нет в наполнителе, вопрос будет ровно про него.
DEFAULT_PASSKEY = {
    "fact": "Проект Lumina был основан в городе Ташкент в 2026 году командой из четырёх человек.",
    "query": "В каком городе и году был основан проект Lumina?",
    "expected_keywords": ["ташкент", "2026"],
}


def build_long_text(fact: str, position: float) -> str:
    """Собирает длинный текст, вставляя факт на заданную относительную позицию
    (0.0 = начало, 0.5 = середина, 1.0 = конец)."""
    n = len(FILLER_PARAGRAPHS)
    insert_at = max(0, min(n, round(position * n)))
    paras = FILLER_PARAGRAPHS[:insert_at] + [fact] + FILLER_PARAGRAPHS[insert_at:]
    return "\n\n".join(paras)


# ─── ПРОВЕРКА ─────────────────────────────────────────────────────────────────

def keywords_hit(text: str, keywords: list[str]) -> list[str]:
    """Возвращает список ключевых слов, найденных в тексте (регистронезависимо)."""
    low = text.lower()
    return [kw for kw in keywords if kw.lower() in low]


def run_passkey_test(passkey: dict, position: float, demo: bool) -> bool:
    fact     = passkey["fact"]
    query    = passkey["query"]
    keywords = passkey["expected_keywords"]

    text = build_long_text(fact, position)

    print("=" * 64)
    print("PASSKEY TEST — проверка, достаёт ли Lum зарытый факт")
    print("=" * 64)
    print(f"Длина текста : {len(text.split())} слов")
    print(f"Позиция факта: {position:.2f} (0=начало, 1=конец)")
    print(f"Секретный факт: {fact}")
    print(f"Вопрос        : {query}")
    print(f"Ждём ключи    : {keywords}")
    print("-" * 64)
    print("Прогоняю пайплайн (несколько вызовов Groq, это займёт время)...\n")

    result = core.run_pipeline(text=text, query=query)

    answer_obj = result["answer"]
    answer_txt = " ".join([
        str(answer_obj.get("answer", "")),
        str(answer_obj.get("summary", "")),
        " ".join(answer_obj.get("key_points", []) or []),
    ])

    # Где факт всплыл: в тексте ответа / в узлах графа / в схеме
    hit_in_answer = keywords_hit(answer_txt, keywords)
    graph_txt = " ".join(n["name"] + " " + n["description"] for n in result["graph"]["nodes"])
    hit_in_graph = keywords_hit(graph_txt, keywords)
    schema_txt = json.dumps(result["schema"], ensure_ascii=False)
    hit_in_schema = keywords_hit(schema_txt, keywords)

    all_hits = set(hit_in_answer) | set(hit_in_graph) | set(hit_in_schema)
    passed = len(all_hits) == len(keywords)   # нашли ВСЕ ключи

    print("ОТВЕТ МОДЕЛИ:")
    print(" ", answer_obj.get("answer", "(пусто)"))
    if demo:
        print("\n  summary:", answer_obj.get("summary", ""))
        print("  key_points:")
        for p in answer_obj.get("key_points", []) or []:
            print("   -", p)
        print(f"\n  Определённое намерение вопроса: {result['explanation'].get('query_intent')}")
        print("  Узлы-путь к ответу:")
        for n in result["explanation"]["path_nodes"]:
            mark = " ✓type" if n.get("type_matched") else ""
            print(f"    [{n['type']}] {n['name']} (sim={n['similarity']}){mark}")

    print("-" * 64)
    print("НАЙДЕНЫ КЛЮЧИ:")
    print(f"  в ответе : {hit_in_answer or '—'}")
    print(f"  в графе  : {hit_in_graph or '—'}")
    print(f"  в схеме  : {hit_in_schema or '—'}")
    print("-" * 64)
    print(f"РЕЗУЛЬТАТ: {'✅ PASS' if passed else '❌ FAIL'} "
          f"({len(all_hits)}/{len(keywords)} ключей найдено)")
    print("=" * 64)

    return passed


def main():
    parser = argparse.ArgumentParser(description="Passkey-тест извлечения факта для Lum")
    parser.add_argument("--position", type=float, default=0.5,
                        help="Относительная позиция факта в тексте: 0=начало, 0.5=середина, 1=конец")
    parser.add_argument("--demo", action="store_true",
                        help="Подробный вывод (для питча/презентации)")
    parser.add_argument("--sweep", action="store_true",
                        help="Прогнать факт по позициям 0.1/0.5/0.9 и показать сводку")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Ошибка: не задан GROQ_API_KEY. Сделай: export GROQ_API_KEY=...")
        sys.exit(1)

    if args.sweep:
        # Демонстрация «эффекта середины»: факт в начале/конце достаётся легче,
        # чем зарытый в середину — классический lost-in-the-middle сюжет.
        results = {}
        for pos in (0.1, 0.5, 0.9):
            results[pos] = run_passkey_test(DEFAULT_PASSKEY, pos, demo=False)
            print()
        print("СВОДКА ПО ПОЗИЦИЯМ:")
        for pos, ok in results.items():
            print(f"  позиция {pos:.1f}: {'PASS' if ok else 'FAIL'}")
    else:
        ok = run_passkey_test(DEFAULT_PASSKEY, args.position, args.demo)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
