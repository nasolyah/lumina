"""
Lum / Lumina — ядро GraphRAG пайплайна (веб-версия).

Провайдер LLM — Google Gemini (AI Studio), endpoint generateContent.

Отличия от консольного graphrag_vectors.py:
  • Ключ читается из переменной окружения GEMINI_API_KEY (не хардкод).
  • Убраны print-спам, запись в файлы и matplotlib-визуализация — это не нужно API.
  • run_pipeline() возвращает чистый dict, готовый к отдаче как JSON.
  • Добавлен "explain path" — какие узлы/рёбра привели к ответу (для подсветки в графе).

Датафлоу:
  Текст → чанки → сущности+связи (LIGHT) → граф+векторы
       → глобальная память (LIGHT) → векторный поиск топ-K
       → ответ + схема (POWER)
"""

from __future__ import annotations

import re
import json
import time
import math
import os
import requests
from collections import defaultdict

# ─── КОНФИГ ───────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Модели Gemini (Google AI Studio). Всё через env — при смене каталога код не трогаем.
#   LIGHT — извлечение сущностей и память (много вызовов, должно быть дёшево).
#   POWER — связывание и финальный ответ (нужно качество).
# Дефолты выбраны под минимальную стоимость демо:
#   Flash-Lite ($0.10/$0.40 за 1M ток.) на извлечение,
#   Flash      ($0.30/$2.50 за 1M ток.) на ответ.
# Для максимального качества на самом питче можно поставить POWER_MODEL=gemini-2.5-pro.
LIGHT_MODEL  = os.environ.get("LIGHT_MODEL", "gemini-2.5-flash-lite")
POWER_MODEL  = os.environ.get("POWER_MODEL", "gemini-2.5-flash")

# Потолок выходных токенов одного ответа модели.
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1200"))

# "Мышление" Gemini 2.5 тратит выходные токены. Для JSON-задач оно не нужно —
# по умолчанию выключаем (0): дешевле и предсказуемее (не съедает лимит на текст).
# ВНИМАНИЕ: у gemini-2.5-pro мышление полностью выключить нельзя — там минимум 128.
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "0"))

# Сколько раз повторять вызов при транзиентной ошибке (429/5xx, «high demand»).
# Если Gemini штормит прямо перед демо — подними, напр. LLM_RETRIES=5.
RETRIES = int(os.environ.get("LLM_RETRIES", "3"))

CHUNK_SIZE   = int(os.environ.get("CHUNK_SIZE", "300"))
TOP_K        = int(os.environ.get("TOP_K", "3"))

# Пауза между вызовами модели при извлечении. На бесплатном тарифе Gemini лимит
# по запросам в минуту (RPM) ниже — подними до 2-4. На платном хватает небольшой.
CHUNK_DELAY  = float(os.environ.get("CHUNK_DELAY", "0.5"))


class PipelineError(Exception):
    """Ошибка выполнения пайплайна, которую отдаём клиенту как 4xx/5xx."""


# ─── ВЫЗОВ LLM (Gemini generateContent) ──────────────────────────────────────

def call_llm(system: str, user: str, model: str = POWER_MODEL, retries: int = RETRIES) -> str:
    """Один вызов Gemini с ретраями на rate-limit (429) и таймаут.
    Без fallback-моделей: если модель недоступна — бросаем понятную PipelineError."""
    if not GEMINI_API_KEY:
        raise PipelineError("GEMINI_API_KEY не задан в переменных окружения")

    url = GEMINI_URL_TMPL.format(model=model)
    gen_config = {
        "maxOutputTokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
    }
    # gemini-2.5-pro не умеет thinkingBudget=0 (минимум 128): если для pro задан 0,
    # не шлём thinkingConfig вовсе — модель выберет бюджет сама (иначе был бы 400).
    # Для flash/flash-lite budget=0 корректно выключает «мышление».
    if not ("pro" in model.lower() and THINKING_BUDGET <= 0):
        gen_config["thinkingConfig"] = {"thinkingBudget": THINKING_BUDGET}
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_config,
    }

    for attempt in range(retries + 1):
        try:
            r = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
            data = r.json()
            if not r.ok:
                msg = data.get("error", {}).get("message", "API error")
                # Транзиентные ошибки ретраим: 429 (лимит/квота) и 5xx (перегрузка
                # модели «high demand», временная недоступность). Раз fallback-моделей
                # нет — это главная страховка демо от кратких перебоев Gemini.
                if (r.status_code == 429 or r.status_code >= 500) and attempt < retries:
                    wait = 6.0 + attempt * 4.0          # бэкофф для 5xx/перегрузки
                    if r.status_code == 429:
                        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*s", msg)
                        if m:
                            wait = float(m.group(1)) + 2.0
                    time.sleep(wait)
                    continue
                raise PipelineError(f"Gemini API ({model}): {msg}")

            # запрос мог быть отклонён фильтрами ещё до генерации
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise PipelineError(f"Gemini заблокировал запрос ({block})")

            candidates = data.get("candidates") or []
            if not candidates:
                raise PipelineError(f"Gemini ({model}) вернул пустой ответ")
            cand = candidates[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                reason = cand.get("finishReason", "UNKNOWN")
                # MAX_TOKENS здесь чаще всего значит, что весь лимит съело "мышление":
                # увеличь MAX_OUTPUT_TOKENS или поставь THINKING_BUDGET=0.
                raise PipelineError(f"Gemini ({model}) не вернул текст (finishReason={reason})")
            return text
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(4 + attempt * 3)
                continue
            raise PipelineError(f"Таймаут запроса к Gemini API (модель {model})")
        except requests.exceptions.RequestException as e:
            raise PipelineError(f"Сетевая ошибка при обращении к Gemini: {e}")
    raise PipelineError(f"Все попытки обращения к модели {model} исчерпаны")


# ─── ВЕКТОРЫ (hash-based эмбеддинги без внешних библиотек) ────────────────────

def text_to_vector(text: str, dim: int = 64) -> list[float]:
    vector = [0.0] * dim
    for word in text.lower().split():
        for i, char in enumerate(word):
            idx = (ord(char) * 31 + i * 17 + hash(word)) % dim
            vector[idx] += 1.0 / (len(word) + 1)
    magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / magnitude for v in vector]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1)) or 1.0
    mag2 = math.sqrt(sum(b * b for b in v2)) or 1.0
    return dot / (mag1 * mag2)


def merge_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    merged = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    magnitude = math.sqrt(sum(v * v for v in merged)) or 1.0
    return [v / magnitude for v in merged]


# ─── ШАГ 1: ЧАНКИ ─────────────────────────────────────────────────────────────

def step1_chunk(text: str) -> list[str]:
    words = text.strip().split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]


# ─── ШАГ 2: ИЗВЛЕЧЕНИЕ СУЩНОСТЕЙ ─────────────────────────────────────────────

_EXTRACT_SYSTEM = """Извлеки сущности и связи из научного текста.
Ответь ТОЛЬКО валидным JSON без markdown:
{
  "entities": [
    {"id": "short_id", "type": "concept|method|person|term", "name": "...", "description": "кратко"}
  ],
  "relations": [
    {"from": "id", "to": "id", "label": "тип связи"}
  ]
}
Максимум 8 сущностей, 6 связей. Только важные."""


def step2_extract_entities(chunks: list[str]) -> list[dict]:
    all_entities = []
    for i, chunk in enumerate(chunks):
        # Сбойный чанк (сеть/лимит/битый JSON) не должен ронять весь прогон.
        # Без fallback-моделей это единственная страховка на этапе извлечения:
        # пропускаем чанк, остальные обрабатываем как обычно.
        try:
            raw = call_llm(system=_EXTRACT_SYSTEM, user=f"Текст:\n{chunk}", model=LIGHT_MODEL)
            clean = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(clean)
            for e in parsed.get("entities", []):
                e["id"] = f"c{i}_{e['id']}"
                e["chunk"] = i
            for r in parsed.get("relations", []):
                r["from"] = f"c{i}_{r['from']}"
                r["to"]   = f"c{i}_{r['to']}"
            all_entities.append(parsed)
        except (PipelineError, json.JSONDecodeError):
            pass  # пропускаем сбойный чанк
        if CHUNK_DELAY:
            time.sleep(CHUNK_DELAY)
    return all_entities


# ─── ШАГ 3: ГРАФ + ВЕКТОРЫ ────────────────────────────────────────────────────

def step3_build_graph_with_vectors(extracted: list[dict]) -> dict:
    nodes, edges, name_to_id = {}, [], {}

    for chunk_data in extracted:
        for entity in chunk_data.get("entities", []):
            name_norm = entity["name"].strip().lower()
            if name_norm in name_to_id:
                nodes[name_to_id[name_norm]]["mentions"] += 1
            else:
                name_to_id[name_norm] = entity["id"]
                vector = text_to_vector(entity["name"] + " " + entity.get("description", ""))
                nodes[entity["id"]] = {
                    "id":          entity["id"],
                    "name":        entity["name"],
                    "type":        entity.get("type", "concept"),
                    "description": entity.get("description", ""),
                    "mentions":    1,
                    "vector":      vector,
                }
        for rel in chunk_data.get("relations", []):
            if rel["from"] in nodes and rel["to"] in nodes:
                edges.append({
                    "from":  rel["from"],
                    "to":    rel["to"],
                    "label": rel.get("label", "связан с"),
                })

    for node_id, node in nodes.items():
        neighbor_ids = (
            [e["to"] for e in edges if e["from"] == node_id]
            + [e["from"] for e in edges if e["to"] == node_id]
        )
        neighbor_vectors = [nodes[nid]["vector"] for nid in neighbor_ids if nid in nodes]
        node["merged_vector"] = (
            merge_vectors([node["vector"]] + neighbor_vectors)
            if neighbor_vectors else node["vector"]
        )

    return {"nodes": nodes, "edges": edges}


# ─── ШАГ 4: ГЛОБАЛЬНАЯ ПАМЯТЬ ─────────────────────────────────────────────────

def step4_memory(graph: dict) -> str:
    top_nodes = sorted(graph["nodes"].values(), key=lambda x: x["mentions"], reverse=True)[:12]
    nodes_text = "\n".join(
        f"- [{n['type']}] {n['name']}: {n['description']} (упомянут {n['mentions']} раз)"
        for n in top_nodes
    )
    edges_text = "\n".join(f"- {e['from']} → {e['to']}: {e['label']}" for e in graph["edges"][:15])
    return call_llm(
        system="""Ты — модель глобальной памяти. Запомни граф знаний научного текста.
Выдели ключевые концепции, методы, связи между ними.
Кратко, по-русски, структурированно.""",
        user=f"УЗЛЫ:\n{nodes_text}\n\nСВЯЗИ:\n{edges_text}",
        model=LIGHT_MODEL,
    )


# ─── ШАГ 5: ВЕКТОРНЫЙ ПОИСК (type-aware) ──────────────────────────────────────
#
# Идея из «Memory Matters» (AAAI): чистый similarity-поиск деградирует, если
# не учитывать ТИП того, что ищем. Метаданные (здесь — тип узла графа)
# работают как дополнительный фильтр/буст поверх косинуса.
#
# Мы определяем «намерение» вопроса по ключевым словам и слегка повышаем скор
# узлов подходящего типа. Это не заменяет векторный поиск, а корректирует его.

# Насколько бустить узел правильного типа (0.25 = +25% к скору). В env: TYPE_BOOST.
TYPE_BOOST = float(os.environ.get("TYPE_BOOST", "0.25"))

# Маркеры намерения вопроса → какой тип узла релевантен.
# Порядок важен: первый сработавший маркер выигрывает.
_INTENT_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("person", ("кто ", "кем ", "автор", "предложил", "изобрел", "изобрёл",
                "создал", "разработал", "who ")),
    ("method", ("как работает", "каким образом", "как устроен", "механизм",
                "метод", "алгоритм", "процесс", "how ")),
    ("term",   ("что такое", "что означает", "определение", "чем является",
                "what is", "define")),
    ("concept",("почему", "зачем", "в чём смысл", "в чем смысл", "идея", "why ")),
]


def detect_query_intent(query: str) -> str | None:
    """Грубо определяет, узел какого типа вероятнее всего отвечает на вопрос.
    Возвращает тип ('person'|'method'|'term'|'concept') или None, если непонятно."""
    q = query.lower()
    for node_type, markers in _INTENT_MARKERS:
        if any(m in q for m in markers):
            return node_type
    return None


def step5_vector_retrieval(graph: dict, query: str) -> list[dict]:
    query_vector = text_to_vector(query)
    intent_type = detect_query_intent(query)

    scored = []
    for node in graph["nodes"].values():
        sim = cosine_similarity(query_vector, node["merged_vector"])
        # базовый скор: сходство × буст за частоту упоминаний
        final_score = sim * (1 + 0.1 * node["mentions"])
        # type-aware буст: если тип узла совпал с намерением вопроса
        type_matched = intent_type is not None and node.get("type") == intent_type
        if type_matched:
            final_score *= (1 + TYPE_BOOST)
        scored.append({
            **node,
            "similarity":   round(sim, 4),
            "final_score":  final_score,
            "type_matched": type_matched,   # ← для explainability / демо
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    top = scored[:TOP_K]
    # прикрепим определённое намерение к каждому узлу (пригодится выше по стеку)
    for n in top:
        n["query_intent"] = intent_type
    return top


# ─── ШАГ 6: ОТВЕТ + СХЕМА ─────────────────────────────────────────────────────

def step6_reason_and_generate(memory: str, top_nodes: list[dict], graph: dict, query: str) -> list[dict]:
    top_ids = {n["id"] for n in top_nodes}
    relevant_edges = [e for e in graph["edges"] if e["from"] in top_ids or e["to"] in top_ids]
    nodes_text = "\n".join(
        f"[{n['type']}] {n['name']} (сходство с запросом: {n['similarity']}): {n['description']}"
        for n in top_nodes
    )
    edges_text = "\n".join(f"- {e['from']} → {e['to']}: {e['label']}" for e in relevant_edges[:15])
    raw = call_llm(
        system="""Ты — мощная модель в GraphRAG. На основе памяти и найденных узлов:
1. Рассуди о связях между понятиями
2. Построй структурную схему научного текста
Ответь ТОЛЬКО JSON-массивом без markdown:
[{"type":"concept|method|term|argument|conclusion","title":"...","description":"...","connections":["..."]}]
6-10 узлов.""",
        user=f"ГЛОБАЛЬНАЯ ПАМЯТЬ:\n{memory}\n\nНАЙДЕННЫЕ УЗЛЫ (по векторному поиску):\n{nodes_text}\n\nСВЯЗИ:\n{edges_text}\n\nЗАПРОС: {query}",
        model=POWER_MODEL,
    )
    try:
        return json.loads(re.sub(r"```json|```", "", raw).strip())
    except json.JSONDecodeError:
        return [{"type": "error", "title": "Ошибка парсинга", "description": raw, "connections": []}]


def step6_generate_answer(memory: str, top_nodes: list[dict], query: str) -> dict:
    nodes_text = "\n".join(f"- {n['name']} ({n['type']}): {n['description']}" for n in top_nodes)
    raw = call_llm(
        system="""Ты — преподаватель, который объясняет студенту научный текст просто и понятно.
Ответь строго валидным JSON без markdown.
{
  "answer": "краткий ответ на вопрос",
  "summary": "одно-два предложения, в которых поясняется, как связаны ключевые понятия",
  "key_points": ["важный факт 1", "важный факт 2", "важный факт 3"]
}
""",
        user=f"ГЛОБАЛЬНАЯ ПАМЯТЬ:\n{memory}\n\nТОП-УЗЛЫ:\n{nodes_text}\n\nЗАПРОС: {query}",
        model=POWER_MODEL,
    )
    try:
        return json.loads(re.sub(r"```json|```", "", raw).strip())
    except json.JSONDecodeError:
        return {"answer": raw.strip(), "summary": "", "key_points": []}


# ─── СЕРИАЛИЗАЦИЯ ГРАФА ДЛЯ ФРОНТА ────────────────────────────────────────────

def serialize_graph(graph: dict, top_ids: set[str]) -> dict:
    """Готовит граф к отдаче: убирает тяжёлые векторы, помечает узлы/рёбра,
    попавшие в ответ (для подсветки 'пути рассуждения' на фронте)."""
    nodes = []
    for n in graph["nodes"].values():
        nodes.append({
            "id":          n["id"],
            "name":        n["name"],
            "type":        n["type"],
            "description": n["description"],
            "mentions":    n["mentions"],
            "in_answer":   n["id"] in top_ids,   # ← фронт подсветит эти узлы
        })
    edges = []
    for e in graph["edges"]:
        edges.append({
            "from":      e["from"],
            "to":        e["to"],
            "label":     e["label"],
            "in_answer": e["from"] in top_ids or e["to"] in top_ids,
        })
    return {"nodes": nodes, "edges": edges}


# ─── ГЛАВНАЯ ФУНКЦИЯ ──────────────────────────────────────────────────────────

def run_pipeline(text: str, query: str) -> dict:
    """
    Полный прогон. Возвращает dict, готовый к json-ответу API.

    Ключи ответа:
      answer      — {answer, summary, key_points}
      schema      — структурная схема (6-10 узлов)
      graph       — {nodes, edges} с флагом in_answer для explainable-подсветки
      explanation — какие именно узлы/рёбра стали "путём" к ответу
      stats       — метаданные прогона (для отладки/питча)
    """
    if not text or not text.strip():
        raise PipelineError("Пустой текст для анализа")
    if not query or not query.strip():
        raise PipelineError("Пустой запрос")
    # Проверяем ключ до извлечения: иначе ошибка "нет ключа" утонет в пер-чанковом
    # skip (см. step2) и превратится в невнятное "не удалось извлечь сущности".
    if not GEMINI_API_KEY:
        raise PipelineError("GEMINI_API_KEY не задан в переменных окружения")

    chunks      = step1_chunk(text)
    extracted   = step2_extract_entities(chunks)
    graph       = step3_build_graph_with_vectors(extracted)

    if not graph["nodes"]:
        raise PipelineError("Не удалось извлечь ни одной сущности из текста")

    memory      = step4_memory(graph)
    top_nodes   = step5_vector_retrieval(graph, query)
    answer_data = step6_generate_answer(memory, top_nodes, query)
    schema      = step6_reason_and_generate(memory, top_nodes, graph, query)

    top_ids = {n["id"] for n in top_nodes}
    # какое намерение вопроса определила система (одинаково для всех top-узлов)
    query_intent = top_nodes[0].get("query_intent") if top_nodes else None
    explanation = {
        "query_intent": query_intent,   # напр. "person" → система бустила такие узлы
        "path_nodes": [
            {"id": n["id"], "name": n["name"], "type": n["type"],
             "similarity": n["similarity"],
             "type_matched": n.get("type_matched", False)}
            for n in top_nodes
        ],
        "path_edges": [
            e for e in graph["edges"]
            if e["from"] in top_ids or e["to"] in top_ids
        ],
    }

    return {
        "query":       query,
        "answer":      answer_data,
        "schema":      schema,
        "graph":       serialize_graph(graph, top_ids),
        "explanation": explanation,
        "stats": {
            "words":  len(text.split()),
            "chunks": len(chunks),
            "nodes":  len(graph["nodes"]),
            "edges":  len(graph["edges"]),
            "top_k":  len(top_nodes),
        },
    }
