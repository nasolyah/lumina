# Lum / Lumina — Backend (GraphRAG API)

Бэкенд превращает **текст + вопрос** в **граф знаний + объяснимый ответ**.
Обёртка на FastAPI поверх GraphRAG-пайплайна (Groq / Llama).

## Структура

```
core.py            # логика пайплайна (чанки → сущности → граф+векторы → память → поиск → ответ)
main.py            # FastAPI: эндпоинты, CORS, обработка ошибок
requirements.txt   # зависимости
.env.example       # шаблон переменных окружения
render.yaml        # конфиг деплоя на Render
```

## ⚠️ Первым делом — отзови старый ключ

В исходном `graphrag_vectors.py` ключ Groq лежал прямо в коде.
Зайди в консоль Groq, **отзови его** и создай новый. В коде ключ больше нигде не хардкодится — только через переменную окружения.

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # и вставь новый GROQ_API_KEY
uvicorn main:app --reload --port 8000
```

Проверка:
```bash
curl http://localhost:8000/api/health
```

## Эндпоинты

### `POST /api/analyze`
Тело запроса:
```json
{ "text": "длинный научный текст...", "query": "как работает внимание?" }
```
Ответ:
```json
{
  "answer":  { "answer": "...", "summary": "...", "key_points": ["...", "..."] },
  "schema":  [ { "type": "concept", "title": "...", "description": "...", "connections": ["..."] } ],
  "graph":   { "nodes": [ { "id","name","type","description","mentions","in_answer" } ],
               "edges": [ { "from","to","label","in_answer" } ] },
  "explanation": { "path_nodes": [...], "path_edges": [...] },
  "stats":   { "words","chunks","nodes","edges","top_k" }
}
```

**`in_answer`** на узлах/рёбрах — это фича «объяснимого ответа»: фронт подсвечивает
именно те узлы и связи графа, через которые модель пришла к ответу.

### `GET /api/health`
Показывает, задан ли ключ и какие модели используются.

## Деплой на Render

1. Залей папку в GitHub-репозиторий.
2. Render → **New → Blueprint** → выбери репу (подхватит `render.yaml`).
3. В настройках сервиса задай секреты:
   - `GROQ_API_KEY` — новый ключ
   - `ALLOWED_ORIGINS` — домен(ы) фронта через запятую, напр. `https://lumina.uz`
4. Deploy. URL будет вида `https://lum-api.onrender.com`.

> На free-плане Render сервис «засыпает» после простоя — первый запрос после сна
> идёт ~30-50 сек. Перед демо на конкурсе сделай один «прогревочный» запрос заранее.

## Как фронт зовёт бэк

```js
const res = await fetch('https://lum-api.onrender.com/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text, query })
});
const data = await res.json();
// data.answer, data.graph (рисуй через vis-network), data.explanation (подсветка пути)
```

## Модели и надёжность демо

По умолчанию используются мощные модели Groq:
- **LIGHT_MODEL** (`openai/gpt-oss-20b`) — извлечение сущностей и память (быстро, дёшево).
- **POWER_MODEL** (`openai/gpt-oss-120b`) — связывание и финальный ответ (максимум качества).

> Старые `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` объявлены Groq устаревшими
> (июнь 2026). Сверяй актуальные строки моделей на **console.groq.com/docs/models** —
> каталог меняется часто.

**Fallback.** `call_llm` при недоступности основной модели автоматически пробует
`LIGHT_FALLBACKS` / `POWER_FALLBACKS` по порядку. Одна икнувшая модель не роняет
демо. Отсутствие ключа fallback'ом не «лечится» — сразу понятная ошибка.

**Два тарифа, которые надо поднять к демо (это делается вручную, не в коде):**
1. **Groq → developer-tier.** Free-tier это 30 запросов/мин — на живом питче пайплайн
   (много вызовов на документ) может упереться в лимит и упасть. Привяжи карту в
   консоли Groq. На ваших объёмах это центы.
2. **Render → платный instance** (в `render.yaml` сейчас `plan: free`). Free-сервис
   засыпает и первый запрос висит 30-50 сек. Либо подними план, либо сделай
   прогревочный запрос за пару минут до выхода на сцену.

## Type-aware ретрив (idea: «Memory Matters», AAAI)

Чистый косинусный поиск деградирует, если игнорировать *тип* искомого. Поэтому
`step5_vector_retrieval` теперь определяет намерение вопроса по ключевым словам
(«кто…» → person, «как работает…» → method, «что такое…» → term, «почему…» → concept)
и слегка повышает скор узлов подходящего типа (`TYPE_BOOST`, по умолчанию +25%).

Это видно в ответе API: `explanation.query_intent` показывает распознанный тип,
а у каждого узла в `explanation.path_nodes` есть флаг `type_matched`. Ставь
`TYPE_BOOST=0` в env, чтобы вернуться к чистому косинусу и сравнить.

## Passkey-тест (idea: SelfExtend / LongLM, arXiv:2401.01325)

`passkey_test.py` — инструмент оценки качества и демо-номер для питча. Он прячет
уникальный факт в середину длинного отвлекающего текста и проверяет, достаёт ли
Lum именно его (а не пересказ общих мест). Смысл: гладкий ответ ≠ правильный.

```bash
export GROQ_API_KEY=...
python passkey_test.py --demo          # подробный вывод с путём по графу
python passkey_test.py --sweep         # факт в начале / середине / конце
python passkey_test.py --position 0.5  # факт ровно в середине
```

Выдаёт PASS/FAIL и показывает, где факт всплыл — в ответе, в узлах графа или в схеме.
На питче `--sweep` наглядно демонстрирует, что граф достаёт факт даже из середины.

> Скрипт ходит в Groq (стоит вызовов) — это не CI-тест, а ручной инструмент оценки.

## Заметки по производительности

- Пайплайн делает несколько последовательных вызовов LLM → ответ 10-40 сек. На фронте нужен спиннер.
- `CHUNK_DELAY=2` — пауза между чанками против rate-limit на free-tier Groq. На платном тарифе можно поставить `0`.
- Векторы (`vector`, `merged_vector`) намеренно **не** попадают в JSON-ответ — они тяжёлые и фронту не нужны.
