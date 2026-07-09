# Lum / Lumina — Backend (GraphRAG API)

Бэкенд превращает **текст + вопрос** в **граф знаний + объяснимый ответ**.
Обёртка на FastAPI поверх GraphRAG-пайплайна (Google Gemini).

## Структура

```
core.py            # логика пайплайна (чанки → сущности → граф+векторы → память → поиск → ответ)
main.py            # FastAPI: эндпоинты, CORS, обработка ошибок
requirements.txt   # зависимости
.env.example       # шаблон переменных окружения
render.yaml        # конфиг деплоя на Render
```

## Ключ Gemini

Получи ключ в **Google AI Studio** (aistudio.google.com → *Get API key*) и положи его
в `GEMINI_API_KEY` (только через переменную окружения, в код не хардкодим).

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # и вставь GEMINI_API_KEY
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
   - `GEMINI_API_KEY` — ключ из Google AI Studio
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

## Модели (Google Gemini)

По умолчанию (выбраны под минимальную стоимость демо):
- **LIGHT_MODEL** (`gemini-2.5-flash-lite`) — извлечение сущностей и память
  (много вызовов, поэтому самая дешёвая модель: $0.10 / $0.40 за 1M вход/выход).
- **POWER_MODEL** (`gemini-2.5-flash`) — связывание и финальный ответ
  ($0.30 / $2.50 за 1M). Для максимума качества на самом питче можно временно
  поставить `POWER_MODEL=gemini-2.5-pro` ($1.25 / $10.00).

**Без fallback-моделей.** `call_llm` делает один вызов Gemini с ретраями на
rate-limit (429) и таймаут. Если модель недоступна — сразу понятная ошибка,
запасные модели не подставляются.

**«Мышление».** Модели Gemini 2.5 по умолчанию тратят выходные токены на reasoning.
Для JSON-задач это лишние деньги и риск, что лимит съест не текст, а размышления,
поэтому `THINKING_BUDGET=0` (выключено). У `gemini-2.5-pro` полностью выключить
нельзя — там минимум 128.

**Тарифы к демо (вручную, не в коде):**
1. **Gemini → платный тариф (billing).** Бесплатный тариф AI Studio ограничен по
   RPM и использует данные для обучения — для тестов ок, для конкурса лучше billing.
   На объёмах Lumina это центы (см. расчёт бюджета ниже / в чате).
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
export GEMINI_API_KEY=...
python passkey_test.py --demo          # подробный вывод с путём по графу
python passkey_test.py --sweep         # факт в начале / середине / конце
python passkey_test.py --position 0.5  # факт ровно в середине
```

Выдаёт PASS/FAIL и показывает, где факт всплыл — в ответе, в узлах графа или в схеме.
На питче `--sweep` наглядно демонстрирует, что граф достаёт факт даже из середины.

> Скрипт ходит в Gemini (стоит вызовов) — это не CI-тест, а ручной инструмент оценки.

## Заметки по производительности

- Пайплайн делает несколько последовательных вызовов LLM → ответ 10-40 сек. На фронте нужен спиннер.
- `CHUNK_DELAY` — пауза между чанками против rate-limit на бесплатном тарифе Gemini (низкий RPM). На платном можно снизить до `0.5` или `0`.
- Векторы (`vector`, `merged_vector`) намеренно **не** попадают в JSON-ответ — они тяжёлые и фронту не нужны.
