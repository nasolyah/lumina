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
import base64
import logging
import requests

# basicConfig идемпотентен (no-op, если хендлер уже есть — напр. настроен в main.py);
# это позволяет видеть логи и при отдельном запуске core.py (напр. passkey_test.py).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lumina.core")
from collections import defaultdict
import contextvars

# ─── ЯЗЫК ИНТЕРФЕЙСА ──────────────────────────────────────────────────────────
# Модель отвечает на языке ВЕРСИИ САЙТА (ru/en), а не документа. Язык приходит
# заголовком X-Lang; эндпоинт вызывает set_lang() в начале запроса. ContextVar —
# чтобы не протаскивать параметр через весь пайплайн (FastAPI гоняет sync-эндпоинты
# в threadpool со скопированным контекстом — между запросами язык не протекает).
# Правило языка добавляется ТОЛЬКО в промпты, чей вывод видит пользователь;
# извлечение сущностей/память остаются на языке документа (иначе ломается
# подсветка пути ответа: имена узлов ищутся подстрокой в тексте разделов).
SUPPORTED_LANGS = ("ru", "en")
_LANG = contextvars.ContextVar("lumina_lang", default="ru")


def set_lang(lang: str | None) -> str:
    code = (lang or "").strip().lower()[:2]
    code = code if code in SUPPORTED_LANGS else "ru"
    _LANG.set(code)
    return code


def get_lang() -> str:
    return _LANG.get()


def tr(ru: str, en: str) -> str:
    """Короткие строки (фолбэки заголовков и т.п.) на языке интерфейса."""
    return en if get_lang() == "en" else ru


def _lang_rule() -> str:
    """Хвост системного промпта: на каком языке писать всё, что увидит пользователь."""
    if get_lang() == "en":
        return ("\n\nRESPONSE LANGUAGE: write ALL user-facing text in your output (answers, "
                "titles, summaries, lists, definitions) in ENGLISH, even if the document is in "
                "another language — translate headings and facts as needed. Keep JSON keys, "
                "block ids and numbers unchanged.")
    return ("\n\nЯЗЫК ОТВЕТА: весь текст для пользователя (ответы, заголовки, резюме, списки, "
            "определения) пиши на РУССКОМ языке, даже если документ на другом языке — "
            "заголовки и факты при необходимости переводи. Ключи JSON, id блоков и числа не меняй.")

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
# Алиасы -latest всегда указывают на актуальную flash-модель и не «протухают»
# (в отличие от версий вроде gemini-2.5-flash, которые Google выводит из эксплуатации).
LIGHT_MODEL  = os.environ.get("LIGHT_MODEL", "gemini-flash-lite-latest")
POWER_MODEL  = os.environ.get("POWER_MODEL", "gemini-flash-latest")

# Модель эмбеддингов (настоящие семантические векторы вместо hash). Один batch-вызов
# на все узлы + один на запрос. При сбое — откат на hash-вектор (см. embed_texts).
EMBED_MODEL     = os.environ.get("EMBED_MODEL", "text-embedding-004")
EMBED_URL_TMPL  = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

# Модель генерации картинок (инфографика по разделу). Через тот же ключ Gemini,
# endpoint generateContent с responseModalities:[TEXT,IMAGE].
#   IMAGE_MODEL (env) — если задан, используем СТРОГО его.
#   Иначе перебираем кандидатов НОВЕЙШИЕ→старые и запоминаем первый рабочий (не 404),
#   а если все не подошли — спрашиваем у ключа список моделей (ListModels) и берём
#   любую image-способную. Так «not found» из-за смены имён у Google само лечится.
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "")
IMAGE_MODEL_CANDIDATES = [
    "gemini-3-pro-image-preview",     # Nano Banana Pro — новейшая, лучшее качество
    "gemini-2.5-flash-image",         # Nano Banana (GA)
    "gemini-2.5-flash-image-preview",
]
_resolved_image_model: str | None = None   # закэшированное рабочее имя (в рамках процесса)

# Потолок выходных токенов одного ответа модели.
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "2048"))

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

def call_llm(system: str, user: str, model: str = POWER_MODEL, retries: int = RETRIES,
             max_tokens: int | None = None) -> str:
    """Один вызов Gemini с ретраями на rate-limit (429) и таймаут.
    Без fallback-моделей: если модель недоступна — бросаем понятную PipelineError."""
    if not GEMINI_API_KEY:
        raise PipelineError("GEMINI_API_KEY не задан в переменных окружения")

    url = GEMINI_URL_TMPL.format(model=model)
    gen_config = {
        "maxOutputTokens": max_tokens or MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
    }
    # thinkingConfig поддерживают только модели с «мышлением» (2.5+, 3.x, -latest).
    # На моделях без него (напр. gemini-2.0-flash) это поле даёт ошибку — не шлём.
    name = model.lower()
    supports_thinking = ("2.5" in name) or ("latest" in name) or name.startswith("gemini-3") or ("gemini-3" in name)
    if supports_thinking:
        budget = THINKING_BUDGET
        # ВАЖНО: буквальный 0 отклоняют и pro (нужен минимум 128), и — как выяснилось
        # 22.07.2026 — flash-lite/flash под алиасом "-latest" (там, похоже, докатили
        # версию модели, которая тоже больше не берёт 0 — раньше работало). Раз "-latest"
        # может подменить модель без предупреждения, никогда не шлём буквальный 0: если
        # THINKING_BUDGET<=0, берём минимальный положительный бюджет вместо полного
        # отключения — дёшево (мышление всё равно почти не тратится), но не 400.
        if budget <= 0:
            budget = 128 if "pro" in name else 1
        gen_config["thinkingConfig"] = {"thinkingBudget": budget}
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
                # Транзиентные ошибки ретраим: 429 (лимит/квота), 5xx (перегрузка
                # «high demand»), 404 и 400 (у Gemini бывает мигающий 404/"invalid
                # argument" на generateContent при абсолютно том же запросе — проходит
                # на повторе). Раз fallback-моделей нет — это главная страховка демо
                # от кратких перебоев Gemini.
                if (r.status_code in (400, 404, 429) or r.status_code >= 500) and attempt < retries:
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


# ─── OCR СКАНОВ (Gemini Vision) ───────────────────────────────────────────────
#
# ВАЖНО — ИСКЛЮЧЕНИЕ ИЗ ПРИВАТНОСТИ: в обычном потоке в LLM уходит ТОЛЬКО текст
# (см. spatial.blocks_for_llm). OCR — единственное место, где ПИКСЕЛИ страниц
# ОСОЗНАННО отправляются в Gemini: у скана нет текстового слоя, распознать его
# иначе нельзя. Включается флагом OCR_VISION (по умолчанию ВКЛ). Для сканов
# лендинговое обещание приватности перестаёт быть полным — это осознанный выбор.

OCR_ENABLED   = os.environ.get("OCR_VISION", "1") != "0"
OCR_MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "15"))   # синхронный OCR: бережём таймаут Render
OCR_MODEL     = os.environ.get("OCR_MODEL", POWER_MODEL)

_OCR_SYSTEM = """Ты — точный OCR-движок. Перепиши ВЕСЬ видимый на странице текст ДОСЛОВНО,
сохраняя естественный порядок чтения (сверху вниз, колонки — по очереди).
- Таблицы передавай построчно, ячейки разделяй « | ».
- Заголовки/пункты — с новой строки.
- НИЧЕГО не добавляй от себя, не переводи, не комментируй, не описывай картинки.
- Если на странице нет текста — верни пустую строку.
Только распознанный текст, без markdown-обёрток."""


def _gemini_vision_ocr(webp: bytes, model: str, retries: int = RETRIES) -> str:
    """Один вызов Gemini Vision: картинка страницы → распознанный текст.
    Пиксели уходят в Gemini осознанно (см. блок выше)."""
    if not GEMINI_API_KEY:
        raise PipelineError("GEMINI_API_KEY не задан в переменных окружения")
    url = GEMINI_URL_TMPL.format(model=model)
    body = {
        "system_instruction": {"parts": [{"text": _OCR_SYSTEM}]},
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/webp", "data": base64.b64encode(webp).decode("ascii")}},
            {"text": "Перепиши весь текст с этой страницы дословно."},
        ]}],
        "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.0},
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, params={"key": GEMINI_API_KEY},
                              headers={"Content-Type": "application/json"}, json=body, timeout=90)
            data = r.json()
            if not r.ok:
                msg = data.get("error", {}).get("message", "API error")
                if (r.status_code in (400, 404, 429) or r.status_code >= 500) and attempt < retries:
                    time.sleep(6.0 + attempt * 4.0)
                    continue
                raise PipelineError(f"Gemini Vision ({model}): {msg}")
            cands = data.get("candidates") or []
            if not cands:
                return ""
            parts = (cands[0].get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts).strip()
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(4 + attempt * 3)
                continue
            raise PipelineError(f"Таймаут OCR-запроса к Gemini (модель {model})")
        except requests.exceptions.RequestException as e:
            raise PipelineError(f"Сетевая ошибка OCR Gemini: {e}")
    raise PipelineError(f"OCR: все попытки к модели {model} исчерпаны")


# ─── ПРЕЗЕНТАЦИИ (.pptx) ──────────────────────────────────────────────────────
# Слайд не рендерим (для этого нужен LibreOffice) — вытаскиваем СТРУКТУРУ: заголовок,
# пункты с уровнями вложенности, таблицы и сами картинки, которые лежат внутри файла.
# Фронт рисует из этого аккуратную «карточку слайда» вместо сплошной стены текста.
_PPTX_MIN_IMG_PX = 48          # меньше — иконки/буллеты/линии, не фото
_PPTX_MAX_IMGS_PER_SLIDE = 4
_PPTX_MAX_IMGS_TOTAL = 60      # потолок на весь файл (размер ответа/Storage)


def _pptx_iter_shapes(shapes):
    """Фигуры слайда, включая вложенные в группы."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    for sh in shapes:
        try:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from _pptx_iter_shapes(sh.shapes)
                continue
        except Exception:
            pass
        yield sh


def _pptx_image_webp(blob: bytes) -> bytes | None:
    """Картинка из .pptx → WebP (≤1200px). None для форматов, которые браузер не
    покажет (EMF/WMF), битых и мелких (иконки)."""
    import io as _io
    from PIL import Image
    try:
        im = Image.open(_io.BytesIO(blob))
        im.load()
    except Exception:
        return None
    if min(im.size) < _PPTX_MIN_IMG_PX:
        return None
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if (im.mode == "P" or "A" in im.getbands()) else "RGB")
    im.thumbnail((1200, 1200))
    buf = _io.BytesIO()
    im.save(buf, format="WEBP", quality=80, method=4)
    return buf.getvalue()


def extract_pptx(raw: bytes, image_sink=None) -> dict:
    """Презентация → {text, pages, slides, blocks}.

    text   — весь текст для графа (слайды через пустую строку, + заметки докладчика);
    slides — [{index, title, images:[{src}]}], src = data URL или путь в Storage (sink);
    blocks — строки слайдов {id, page=№слайда, kind heading|text, level, text}: по ним
             ИИ группирует разделы, а узел карты знает свои слайды и пункты.
    image_sink(name, webp) -> str; сбой sink → картинка остаётся data URL."""
    import io as _io
    import hashlib
    from pptx import Presentation

    def _dataurl(webp: bytes) -> str:
        return "data:image/webp;base64," + base64.b64encode(webp).decode("ascii")

    prs = Presentation(_io.BytesIO(raw))
    slides_out, blocks, text_parts = [], [], []
    total_imgs = 0
    for si, slide in enumerate(prs.slides):
        title_shape = None
        try:
            title_shape = slide.shapes.title
        except Exception:
            pass
        title = ""
        if title_shape is not None and getattr(title_shape, "has_text_frame", False):
            title = (title_shape.text_frame.text or "").strip()
        items = [("heading", 0, title)] if title else []   # (kind, level, text)
        # сравниваем по shape_id: python-pptx отдаёт новый объект-обёртку при каждом
        # обращении, поэтому `is` не отличает заголовок и он дублировался бы пунктом
        title_id = getattr(title_shape, "shape_id", None)
        shapes = [sh for sh in _pptx_iter_shapes(slide.shapes)
                  if title_id is None or getattr(sh, "shape_id", None) != title_id]
        shapes.sort(key=lambda sh: ((sh.top or 0), (sh.left or 0)))   # порядок чтения, не z-order
        images, seen = [], set()
        for sh in shapes:
            try:
                if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
                    for p in sh.text_frame.paragraphs:
                        line = ("".join(r.text for r in p.runs) or p.text or "").strip()
                        if line:
                            items.append(("text", int(p.level or 0), line))
                if getattr(sh, "has_table", False) and sh.has_table:
                    for row in sh.table.rows:
                        cells = [(c.text or "").strip() for c in row.cells]
                        if any(cells):
                            items.append(("text", 0, " | ".join(cells)))
                blob = None
                if hasattr(sh, "image"):          # картинка или плейсхолдер с картинкой
                    try:
                        blob = sh.image.blob
                    except Exception:
                        blob = None
                if blob and len(images) < _PPTX_MAX_IMGS_PER_SLIDE and total_imgs < _PPTX_MAX_IMGS_TOTAL:
                    digest = hashlib.sha1(blob).hexdigest()
                    if digest not in seen:
                        seen.add(digest)
                        webp = _pptx_image_webp(blob)
                        if webp:
                            src = None
                            if image_sink:
                                try:
                                    src = image_sink(f"s{si}_{len(images)}", webp)
                                except Exception as e:
                                    logger.warning("pptx: картинка не залита в Storage: %s", e)
                            images.append({"src": src or _dataurl(webp)})
                            total_imgs += 1
            except Exception:
                continue   # экзотическая фигура — пропускаем, слайд не роняем
        notes = ""
        try:
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = (slide.notes_slide.notes_text_frame.text or "").strip()
        except Exception:
            pass
        for kind, level, line in items:
            blocks.append({"id": f"b{len(blocks)}", "page": si, "kind": kind, "level": level, "text": line})
        body = "\n".join(t for _, _, t in items)
        if notes:
            body = (body + "\n" + notes).strip()
        if body:
            text_parts.append(body)
        slides_out.append({"index": si, "title": title, "images": images})
    return {"text": "\n\n".join(text_parts).strip(), "pages": len(slides_out),
            "slides": slides_out, "blocks": blocks}


def ocr_pdf(raw: bytes, max_pages: int | None = None) -> str:
    """Распознаёт скан PDF (без текстового слоя) через Gemini Vision: рендерит
    страницы и OCR-ит каждую. Возвращает склеенный текст (или '' если OCR выкл /
    ничего не распознано). Сбой отдельной страницы не роняет весь документ."""
    if not OCR_ENABLED:
        return ""
    import pypdfium2 as pdfium
    from spatial import _render_page_webp
    limit = max_pages or OCR_MAX_PAGES
    doc = pdfium.PdfDocument(raw)
    n = min(len(doc), limit)
    out = []
    for i in range(n):
        try:
            webp, _, _ = _render_page_webp(doc[i])
            txt = _gemini_vision_ocr(webp, OCR_MODEL)
            if txt:
                out.append(txt)
        except PipelineError as e:
            logger.warning("ocr_pdf: страница %d пропущена: %s", i, e)
    return "\n\n".join(out).strip()


# ─── ИНФОГРАФИКА (Gemini image) ───────────────────────────────────────────────
#
# Кнопка в модалке раздела → картинка-инфографика по тексту раздела. Два шага
# ЧЕРЕЗ ТОТ ЖЕ КЛЮЧ: (1) дешёвая текст-модель дистиллирует раздел в короткий бриф
# (заголовок + 3-6 фактов ДОСЛОВНО из текста, без выдумок); (2) image-модель рисует
# по бифу в палитре Lumina. Числа/подписи на картинке модель может исказить —
# бриф отдаём фронту рядом, чтобы фактам можно было доверять из брифа, не с картинки.

_INFOGRAPHIC_BRIEF_SYSTEM = """Ты готовишь КРАТКИЙ бриф для инфографики по разделу документа.
На основе ТОЛЬКО приведённого текста извлеки:
- "title": короткий заголовок инфографики (до 6 слов),
- "points": 3-6 самых важных тезисов/цифр/фактов, каждый ОЧЕНЬ коротко (до 8 слов);
  бери реальные числа и названия ИЗ ТЕКСТА, ничего не выдумывай.
Если содержательного текста мало — меньше пунктов (но хотя бы 2).
Отвечай ТОЛЬКО валидным JSON без markdown:
{"title":"...","points":["...","..."]}"""


def _compose_image_prompt(title: str, points: list[str], lang_hint: str | None = None) -> str:
    """Собирает текстовый промпт для image-модели из брифа.

    Палитру НЕ навязываем — просим модель подобрать под тему; упор на официальный,
    профессиональный, издательского качества результат. Подписи на картинке — на
    языке интерфейса (бриф уже сгенерирован на нём, см. _lang_rule)."""
    lang_hint = lang_hint or ("English" if get_lang() == "en" else "Russian")
    pts = "\n".join(f"- {p}" for p in points if (p or "").strip())
    return (
        "Design a PROFESSIONAL, PUBLICATION-GRADE INFOGRAPHIC — the kind used in official "
        "business reports, consulting decks and editorial pages. Clean modern flat vector style "
        "(no photorealism, no clip-art, no random decorative clutter).\n"
        "Quality bar: precise alignment to an invisible grid, balanced composition, clear visual "
        "hierarchy (one strong headline, then sections), consistent iconography (simple line/solid "
        "icons), generous white space, crisp legible typography, professional data-viz where numbers "
        "appear (neat bars/stat cards/steps). Polished and trustworthy, not playful.\n"
        "COLOR: choose the color palette that BEST FITS THIS SPECIFIC TOPIC and reads as official and "
        "professional (a restrained, cohesive scheme — 2-3 main colors plus neutrals); do NOT default "
        "to purple. Ensure strong contrast and accessibility.\n"
        f"Headline: «{title}».\n"
        "Present these key points as distinct, well-organized visual blocks (icon + short label + the "
        "number/stat where present):\n"
        f"{pts}\n"
        f"Keep all text SHORT, correctly spelled and legible, written in {lang_hint}. "
        "Use ONLY the facts and numbers above — do not invent anything."
    )


def build_infographic_brief(node_title: str, node_text: str) -> dict:
    """Текст раздела → {title, points[]} (дешёвая модель, строго по тексту)."""
    src = (node_text or "").strip() or node_title
    raw = call_llm(
        system=_INFOGRAPHIC_BRIEF_SYSTEM + _lang_rule(),
        user=f"РАЗДЕЛ «{node_title}»:\n{src}",
        model=LIGHT_MODEL,
        max_tokens=600,
    )
    try:
        data = parse_json_lenient(raw)
    except json.JSONDecodeError:
        data = {}
    title = (data.get("title") or node_title or tr("Инфографика", "Infographic")).strip()[:80]
    points = [str(p).strip()[:80] for p in (data.get("points") or []) if str(p).strip()][:6]
    if not points:
        # мягкая деградация: хоть что-то отдать image-модели
        points = [node_title.strip()[:80]] if node_title.strip() else [tr("Обзор раздела", "Section overview")]
    return {"title": title, "points": points}


class ImageModelUnavailable(PipelineError):
    """Модель картинок недоступна (404/не поддерживает generateContent) — сигнал
    перебрать следующего кандидата, а не падать."""


def _list_image_capable_models() -> list[str]:
    """Спрашивает у ключа список моделей (ListModels) и возвращает имена, которые
    умеют generateContent и похожи на image-модель. Пусто — если запрос не удался."""
    try:
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": GEMINI_API_KEY, "pageSize": 200}, timeout=30,
        )
        if not r.ok:
            return []
        out = []
        for m in (r.json().get("models") or []):
            name = (m.get("name") or "").split("/")[-1]
            methods = m.get("supportedGenerationMethods") or []
            if name and "image" in name.lower() and "generateContent" in methods:
                out.append(name)
        return out
    except requests.exceptions.RequestException:
        return []


def _gemini_generate_image(prompt: str, model: str, retries: int = RETRIES) -> tuple[str, str]:
    """Один вызов image-модели Gemini: текстовый промпт → (base64-данные, mime).
    404/«not found»/«not supported» → ImageModelUnavailable (перебрать другую)."""
    if not GEMINI_API_KEY:
        raise PipelineError("GEMINI_API_KEY не задан в переменных окружения")
    url = GEMINI_URL_TMPL.format(model=model)
    # TEXT+IMAGE — самое совместимое сочетание (принимают и preview, и GA-версии
    # image-модели); из ответа берём именно image-часть (inlineData).
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, params={"key": GEMINI_API_KEY},
                              headers={"Content-Type": "application/json"}, json=body, timeout=120)
            data = r.json()
            if not r.ok:
                msg = data.get("error", {}).get("message", "API error")
                # модель не существует/не поддерживает generateContent → сигнал перебрать другую
                low = msg.lower()
                if r.status_code == 404 or "not found" in low or "not supported" in low:
                    raise ImageModelUnavailable(f"Gemini image ({model}): {msg}")
                if (r.status_code in (429,) or r.status_code >= 500) and attempt < retries:
                    time.sleep(6.0 + attempt * 4.0)
                    continue
                raise PipelineError(f"Gemini image ({model}): {msg}")
            cands = data.get("candidates") or []
            parts = ((cands[0].get("content") or {}) if cands else {}).get("parts") or []
            for p in parts:
                blob = p.get("inlineData") or p.get("inline_data")
                if blob and blob.get("data"):
                    return blob["data"], (blob.get("mimeType") or blob.get("mime_type") or "image/png")
            raise PipelineError(f"Gemini image ({model}) не вернул картинку")
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(4 + attempt * 3)
                continue
            raise PipelineError(f"Таймаут генерации картинки Gemini (модель {model})")
        except requests.exceptions.RequestException as e:
            raise PipelineError(f"Сетевая ошибка генерации картинки Gemini: {e}")
    raise PipelineError(f"Генерация картинки: все попытки к модели {model} исчерпаны")


def _generate_image_autoresolve(prompt: str) -> tuple[str, str, str]:
    """Рисует картинку, сам подбирая рабочую image-модель. Возвращает (b64, mime, model).

    Порядок: закэшированная рабочая → IMAGE_MODEL (если задан в env, строго он) →
    кандидаты новейшие→старые → любая image-способная из ListModels. Недоступную
    (404/not supported) молча пропускаем; запоминаем первую, что реально нарисовала."""
    global _resolved_image_model
    if _resolved_image_model:
        order = [_resolved_image_model]
    elif IMAGE_MODEL:
        order = [IMAGE_MODEL]           # явно задан в env — используем только его
    else:
        order = list(IMAGE_MODEL_CANDIDATES)

    last_err: Exception | None = None
    tried = set()
    def _try(models):
        nonlocal last_err
        global _resolved_image_model
        for m in models:
            if not m or m in tried:
                continue
            tried.add(m)
            try:
                b64, mime = _gemini_generate_image(prompt, model=m)
                _resolved_image_model = m
                return b64, mime, m
            except ImageModelUnavailable as e:
                last_err = e            # модель не подошла — пробуем следующую
                logger.warning("infographic: модель %s недоступна, пробую другую", m)
            # прочие ошибки (лимит/сеть/блок) пробрасываем — не нашей моделью проблема
        return None

    got = _try(order)
    # если ни один известный кандидат не подошёл и имя не форсировано — спросим у ключа
    if got is None and not IMAGE_MODEL:
        discovered = _list_image_capable_models()
        if discovered:
            logger.info("infographic: доступные image-модели у ключа: %s", discovered)
            got = _try(discovered)
    if got is None:
        raise last_err or PipelineError(
            "Ни одна image-модель Gemini недоступна для этого ключа. "
            "Проверьте доступ к генерации картинок или задайте IMAGE_MODEL."
        )
    return got


def generate_infographic(node_title: str, node_text: str) -> dict:
    """Инфографика-картинка по разделу. Возвращает {image (data URL), title, points, model}."""
    brief = build_infographic_brief(node_title, node_text)
    prompt = _compose_image_prompt(brief["title"], brief["points"])
    b64, mime, model = _generate_image_autoresolve(prompt)
    return {
        "image": f"data:{mime};base64,{b64}",
        "title": brief["title"],
        "points": brief["points"],
        "model": model,
    }


# ─── ЭМБЕДДИНГИ ──────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """
    Настоящие семантические эмбеддинги Gemini (batchEmbedContents) — один вызов на
    весь список. Возвращает None при сбое/отсутствии ключа → вызывающий откатывается
    на hash-вектор (text_to_vector), сохраняя работоспособность пайплайна.
    """
    if not GEMINI_API_KEY or not texts:
        return None
    url = EMBED_URL_TMPL.format(model=EMBED_MODEL)
    model_path = f"models/{EMBED_MODEL}"
    body = {"requests": [
        {"model": model_path, "content": {"parts": [{"text": (t or " ")[:2000]}]}}
        for t in texts
    ]}
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(url, params={"key": GEMINI_API_KEY},
                              headers={"Content-Type": "application/json"},
                              json=body, timeout=60)
            if r.ok:
                embs = (r.json().get("embeddings") or [])
                if len(embs) == len(texts) and all(e.get("values") for e in embs):
                    return [e["values"] for e in embs]
                return None
            if (r.status_code in (429, 500, 503)) and attempt < RETRIES:
                time.sleep(4.0 + attempt * 3.0)
                continue
            logger.warning("embed_texts: %s %s", r.status_code, r.text[:200])
            return None
        except requests.exceptions.RequestException as e:
            if attempt < RETRIES:
                time.sleep(3.0 + attempt * 2.0)
                continue
            logger.warning("embed_texts: сеть — %s", e)
            return None
    return None


# ─── ВЕКТОРЫ (hash-based эмбеддинги — фолбэк без внешних вызовов) ─────────────

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

def parse_json_lenient(raw: str):
    """Разбирает JSON из ответа модели терпимо: снимает markdown-обёртку и, если
    вокруг JSON есть лишний текст (слабые модели вроде gemini-2.0-flash часто
    добавляют пояснения), вытаскивает первый {...} или [...] блок."""
    s = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", s, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


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
Максимум 8 сущностей. ВАЖНО: свяжи их между собой так, чтобы НЕ было изолированных —
каждая сущность должна участвовать хотя бы в одной связи (обычно 7-12 связей).
Только важные сущности и связи."""


def step2_extract_entities(chunks: list[str]) -> list[dict]:
    all_entities = []
    for i, chunk in enumerate(chunks):
        # Сбойный чанк (сеть/лимит/битый JSON) не должен ронять весь прогон.
        # Без fallback-моделей это единственная страховка на этапе извлечения:
        # пропускаем чанк, остальные обрабатываем как обычно.
        try:
            raw = call_llm(system=_EXTRACT_SYSTEM, user=f"Текст:\n{chunk}", model=LIGHT_MODEL)
            parsed = parse_json_lenient(raw)
            for e in parsed.get("entities", []):
                e["id"] = f"c{i}_{e['id']}"
                e["chunk"] = i
            for r in parsed.get("relations", []):
                r["from"] = f"c{i}_{r['from']}"
                r["to"]   = f"c{i}_{r['to']}"
            all_entities.append(parsed)
        except (PipelineError, json.JSONDecodeError) as e:
            # Логируем причину сбоя чанка — иначе "не удалось извлечь сущности"
            # наверху ничего не говорит о том, что именно упало (Gemini/парсинг).
            logger.warning("step2: чанк %d пропущен (%s): %s", i, type(e).__name__, e)
        if CHUNK_DELAY:
            time.sleep(CHUNK_DELAY)
    return all_entities


# ─── ШАГ 3: ГРАФ + ВЕКТОРЫ ────────────────────────────────────────────────────

def step3_build_graph_with_vectors(extracted: list[dict]) -> dict:
    nodes, edges, name_to_id = {}, [], {}
    # Карта «id сущности из чанка → id узла, оставшегося в графе после схлопывания
    # одноимённых». Нужна, чтобы связи из разных чанков сходились к общим узлам,
    # а не выбрасывались (иначе граф рассыпается на изолированные куски).
    id_to_canonical = {}

    for chunk_data in extracted:
        for entity in chunk_data.get("entities", []):
            name_norm = entity["name"].strip().lower()
            if name_norm in name_to_id:
                canonical = name_to_id[name_norm]
                nodes[canonical]["mentions"] += 1
            else:
                canonical = entity["id"]
                name_to_id[name_norm] = canonical
                nodes[canonical] = {
                    "id":          canonical,
                    "name":        entity["name"],
                    "type":        entity.get("type", "concept"),
                    "description": entity.get("description", ""),
                    "mentions":    1,
                    "vector":      None,   # заполним ниже одним batch-эмбеддингом
                }
            id_to_canonical[entity["id"]] = canonical

        for rel in chunk_data.get("relations", []):
            # переназначаем концы связи на канонические узлы
            f = id_to_canonical.get(rel["from"])
            t = id_to_canonical.get(rel["to"])
            if f and t and f != t and f in nodes and t in nodes:
                edges.append({
                    "from":  f,
                    "to":    t,
                    "label": rel.get("label", "связан с"),
                })

    # убираем дубликаты рёбер (одна и та же связь могла прийти из нескольких чанков)
    seen_edges, unique_edges = set(), []
    for e in edges:
        key = (e["from"], e["to"], e["label"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)
    edges = unique_edges

    # эмбеддинги узлов: один batch-вызов Gemini; при сбое — hash-фолбэк (тот же метод
    # применяем и к запросу в step5, чтобы размерности совпадали).
    node_ids = list(nodes.keys())
    node_texts = [f"{nodes[i]['name']} {nodes[i].get('description', '')}".strip() for i in node_ids]
    vecs = embed_texts(node_texts)
    real_embed = vecs is not None
    for i, nid in enumerate(node_ids):
        nodes[nid]["vector"] = vecs[i] if real_embed else text_to_vector(node_texts[i])

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

    return {"nodes": nodes, "edges": edges, "real_embed": real_embed}


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
                "создал", "разработал", "who ", "whom ", "author", "invented",
                "proposed", "created by", "developed by")),
    ("method", ("как работает", "каким образом", "как устроен", "механизм",
                "метод", "алгоритм", "процесс", "how ", "method", "algorithm",
                "process", "mechanism")),
    ("term",   ("что такое", "что означает", "определение", "чем является",
                "what is", "what are", "what does", "meaning of", "define", "definition")),
    ("concept",("почему", "зачем", "в чём смысл", "в чем смысл", "идея", "why ",
                "purpose", "the point of")),
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
    # эмбеддинг запроса ТЕМ ЖЕ методом, что и узлы (иначе размерности не совпадут):
    # реальный, если узлы эмбеддились реально; иначе hash-фолбэк.
    query_vector = None
    if graph.get("real_embed"):
        qv = embed_texts([query])
        if qv:
            query_vector = qv[0]
    if query_vector is None:
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
        return parse_json_lenient(raw)
    except json.JSONDecodeError:
        return [{"type": "error", "title": "Ошибка парсинга", "description": raw, "connections": []}]


def step6_generate_answer(memory: str, top_nodes: list[dict], query: str,
                          sources: list[dict] | None = None) -> dict:
    nodes_text = "\n".join(f"- {n['name']} ({n['type']}): {n['description']}" for n in top_nodes)
    # ФРАГМЕНТЫ ИСТОЧНИКА — дословный текст релевантных разделов (passage-RAG).
    # Ключевой антидот против «в тексте нет информации», когда информация ЕСТЬ:
    # сущностный граф — сжатие с потерями (таблицы/списки туда часто не попадают),
    # а разделы покрывают документ целиком. Модель отвечает В ПЕРВУЮ очередь по ним.
    sources_text = "\n\n".join(
        f"[{s.get('title') or 'раздел'}]\n{s.get('text','')}" for s in (sources or []) if s.get("text")
    )
    raw = call_llm(
        system="""Ты — преподаватель, который объясняет студенту документ просто и понятно.
ГЛАВНОЕ ПРАВИЛО: отвечай В ПЕРВУЮ ОЧЕРЕДЬ по «ФРАГМЕНТАМ ИСТОЧНИКА» — это дословный
текст документа (в т.ч. таблицы, списки требований, числа). «ТОП-УЗЛЫ» и «ПАМЯТЬ» —
лишь вспомогательный контекст, они неполны.
- Если ответ есть во ФРАГМЕНТАХ — дай его конкретно (приведи нужные числа/пункты из таблицы).
- Говори «в документе этого нет» ТОЛЬКО если реально не нашёл ни во фрагментах, ни в узлах.
  Не отказывай, если данные есть во фрагментах. Ничего не выдумывай сверх источника.
Ответь строго валидным JSON без markdown.
{
  "answer": "краткий конкретный ответ на вопрос",
  "summary": "одно-два предложения, поясняющие суть/связи",
  "key_points": ["важный факт 1", "важный факт 2", "важный факт 3"]
}
""" + _lang_rule(),
        user=f"ГЛОБАЛЬНАЯ ПАМЯТЬ:\n{memory}\n\nФРАГМЕНТЫ ИСТОЧНИКА (дословно):\n{sources_text or '(нет)'}\n\nТОП-УЗЛЫ:\n{nodes_text}\n\nЗАПРОС: {query}",
        model=POWER_MODEL,
    )
    try:
        return parse_json_lenient(raw)
    except json.JSONDecodeError:
        return {"answer": raw.strip(), "summary": "", "key_points": []}


def _embed_passages(texts: list[str], real_embed: bool) -> list[list[float]]:
    """Эмбеддит тексты ТЕМ ЖЕ методом, что и узлы графа (реальный или hash-фолбэк),
    иначе размерности векторов не совпадут при косинусе с запросом."""
    if real_embed:
        vs = embed_texts(texts)
        if vs:
            return vs
    return [text_to_vector(t) for t in texts]


def build_passages(mindmap: dict | None, real_embed: bool) -> list[dict]:
    """Дословные разделы документа + их эмбеддинги — материал для passage-RAG.

    Берём узлы mind-map (реальная структура из блоков): у каждого есть full/excerpt
    (цитата) и block_ids. В отличие от сущностного графа, разделы покрывают документ
    целиком (таблицы, списки требований), поэтому ответ можно заземлить на них.
    """
    if not mindmap or not mindmap.get("nodes"):
        return []
    items = []
    for n in mindmap["nodes"]:
        txt = (n.get("full") or n.get("excerpt") or "").strip()
        if not txt:
            continue
        items.append({
            "title": (n.get("name") or "").strip(),
            "text": txt[:1200],
            "block_ids": n.get("block_ids", []),
        })
    if not items:
        return []
    vecs = _embed_passages([f"{it['title']}\n{it['text']}" for it in items], real_embed)
    for it, v in zip(items, vecs):
        # округляем — вектор уезжает на фронт и обратно (graph_state), режем размер payload
        it["vector"] = [round(x, 5) for x in v]
    return items


def retrieve_passages(passages: list[dict], query: str, real_embed: bool, k: int = 4) -> list[dict]:
    """Топ-k дословных разделов по близости к запросу — их текст пойдёт в step6."""
    if not passages:
        return []
    qv = None
    if real_embed:
        e = embed_texts([query])
        if e:
            qv = e[0]
    if qv is None:
        qv = text_to_vector(query)
    scored = []
    for p in passages:
        v = p.get("vector")
        if not v:
            continue
        scored.append((cosine_similarity(qv, v), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


# ─── ШАГ 7: MIND-MAP (реальная структура документа) ───────────────────────────
#
# Раньше модель сама придумывала абстрактные категории поверх извлечённых сущностей
# («Определение», «Компоненты»…) — неточно отражало документ. Теперь модель
# сегментирует ИСХОДНЫЙ текст на его настоящие разделы/подразделы (реальные
# заголовки, если есть в тексте, иначе — логичные тематические блоки) и цитирует
# его ДОСЛОВНО — узел графа честно привязан к месту в источнике, а не к пересказу.

_SECTIONS_SYSTEM = """Ты анализируешь структуру документа — раздели его на РЕАЛЬНЫЕ
смысловые разделы, как оглавление книги.
ПРАВИЛА:
- Если в тексте ЕСТЬ явные заголовки/подзаголовки — используй их ДОСЛОВНО как title.
- Если явных заголовков нет — определи по смыслу естественные тематические блоки и
  дай каждому короткое название (2-5 слов), отражающее содержание.
- Может быть иерархия (раздел → под-разделы), любая глубина, если текст того
  требует. Короткий цельный раздел не дели без нужды — детей не давай.
- Для КАЖДОГО раздела и под-раздела укажи:
  * "excerpt" — 2-3 предложения ДОСЛОВНО из этой части текста (процитируй реальные
    фразы, НЕ перефразируй) — суть раздела с первого взгляда;
  * "full" — более полная дословная цитата этого раздела (5-10 предложений).
- "root" — главная тема всего документа, коротко.
Ответь ТОЛЬКО валидным JSON без markdown:
{"root":"главная тема",
 "children":[
   {"title":"...", "excerpt":"...", "full":"...",
    "children":[{того же вида, любая глубина}]}
 ]}
Раздел без под-разделов — просто объект без "children"."""


def step_document_sections(text: str, main_topic: str) -> dict | None:
    """Просит сильную модель сегментировать ИСХОДНЫЙ текст на его настоящую
    структуру (не понятия из графа, а сам документ). Возвращает
    {"root":..., "children":[...]} или None, если не удалось."""
    raw = call_llm(
        system=_SECTIONS_SYSTEM + _lang_rule(),
        user=f"ГЛАВНАЯ ТЕМА (ориентир): {main_topic}\n\nТЕКСТ:\n{text}",
        model=POWER_MODEL,
        max_tokens=4000,   # узлы несут цитаты из текста — длиннее обычного ответа
    )
    try:
        data = parse_json_lenient(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("children") else None


def flatten_sections(data: dict, in_answer_names: set[str]) -> dict:
    """Разворачивает вложенную структуру документа в {nodes, edges, root}.
    Узел с детьми — 'branch' (категория, без цитаты), лист — 'section' (несёт
    excerpt/full — дословную цитату исходника для показа на графе и в модалке).
    in_answer — эвристика для подсветки пути к ответу: пересекается ли узел
    с понятиями, которые нашёл векторный поиск (top_nodes)."""
    nodes, edges, used = [], [], set()

    def uid(name: str) -> str:
        base = (name or "?").strip() or "?"
        key, i = base, 1
        while key in used:
            i += 1
            key = f"{base}#{i}"
        used.add(key)
        return key

    def is_ans(title: str, excerpt: str) -> bool:
        blob = f"{title} {excerpt}".strip().lower()
        return any(nm and nm in blob for nm in in_answer_names)

    def add(title: str, ntype: str, excerpt: str = "", full: str = "") -> str:
        nid = uid(title)
        nodes.append({
            "id": nid, "name": (title or "").strip(), "type": ntype,
            "excerpt": (excerpt or "").strip()[:400],
            "full": (full or excerpt or "").strip()[:1500],
            "world": "",
            "mentions": 1,
            "in_answer": is_ans(title, excerpt),
        })
        return nid

    def walk(children, parent_id):
        for ch in children or []:
            if not isinstance(ch, dict) or not (ch.get("title") or "").strip():
                continue
            kids = ch.get("children") or []
            ntype = "branch" if kids else "section"
            cid = add(ch["title"], ntype, ch.get("excerpt", ""), ch.get("full", ""))
            edges.append({"from": parent_id, "to": cid, "label": "",
                          "in_answer": is_ans(ch.get("title", ""), ch.get("excerpt", ""))})
            walk(kids, cid)

    root_id = add(data.get("root") or tr("Документ", "Document"), "branch")
    walk(data.get("children"), root_id)
    return {"nodes": nodes, "edges": edges, "root": root_id}


def main_topic_fallback(graph: dict) -> str:
    """Название самого упоминаемого узла — запасной ориентир темы документа."""
    if not graph["nodes"]:
        return "Тема"
    return max(graph["nodes"].values(), key=lambda n: n["mentions"])["name"]


# ─── ПЛАН 3: РАЗДЕЛЫ ИЗ БЛОКОВ (узлы несут block_ids) ─────────────────────────
# Вместо сегментации плоского текста эвристиками — LLM группирует ГОТОВЫЕ блоки
# (у PDF — абзацы с координатами, у текста — абзацы без координат) в дерево
# разделов, и каждый узел несёт block_ids. Это убирает матчинг узел↔блок на фронте
# и делает «вырез раздела» точным по построению.

def split_text_into_blocks(text: str) -> list[dict]:
    """Не-PDF ввод → блоки-абзацы (по пустым строкам / переносам), без координат."""
    parts, cur = [], []
    for line in (text or "").split("\n"):
        if line.strip():
            cur.append(line.strip())
        elif cur:
            parts.append(" ".join(cur)); cur = []
    if cur:
        parts.append(" ".join(cur))
    # слишком длинные «абзацы» (текст без пустых строк) режем по ~600 символов
    blocks, i = [], 0
    for p in parts:
        while len(p) > 800:
            cut = p.rfind(" ", 0, 700) or 700
            blocks.append({"id": f"b{i}", "text": p[:cut].strip()}); i += 1
            p = p[cut:].strip()
        if p:
            blocks.append({"id": f"b{i}", "text": p}); i += 1
    return blocks


_SECTIONS_FROM_BLOCKS_SYSTEM = """Ты сегментируешь документ на его НАСТОЯЩУЮ структуру.
Тебе дают ДОКУМЕНТ ПОСТРОЧНО: каждая строка пронумерована как [bN]. Твоя задача —
провести ЛОГИЧЕСКИЕ ГРАНИЦЫ по смыслу: сгруппировать подряд идущие строки в цельные
абзацы/разделы (и, где нужно, подразделы), как это сделал бы внимательный читатель.

Правила:
- строки идут в порядке чтения; группируй ТОЛЬКО осмысленно связанные соседние строки
  в один раздел (один абзац/таблица/пункт = один раздел или его часть);
- начинай новый раздел там, где меняется СМЫСЛ: новый заголовок, новая тема, новая
  таблица, новый пункт списка, новая колонка;
- заголовок раздела — короткий и осмысленный (2-6 слов); если в строках есть настоящий
  заголовок — используй его;
- отнеси КАЖДУЮ строку ровно к одному разделу через block_ids; НЕ выдумывай id и НЕ теряй строки;
- строку-заголовок клади в тот раздел, который она озаглавливает;
- порядок сохраняй; 4-8 разделов верхнего уровня — не мельчи и не склеивай всё в один;
- НЕ смешивай в одном разделе РАЗНЫЕ таблицы/факультеты/подтемы, даже если они рядом
  на странице (напр. «Faculty of Architecture» и «HKU Business School» — РАЗНЫЕ разделы);
- презентации: строки помечены «@слайдN» — заголовок слайда и его пункты держи вместе;
  соседние слайды на одну тему можно объединять в один раздел;
- РАЗРЫВ СТРАНИЦЫ — НЕ граница раздела. Строка с пометкой «⤷ПРОДОЛЖЕНИЕ_АБЗАЦА»
  продолжает абзац (часто — середину предложения) с предыдущей страницы: относи её И
  идущие за ней строки этого абзаца к ТОМУ ЖЕ разделу, что и последние строки
  предыдущей страницы;
- у строк указана позиция «@стрN xM yK». Строки с СИЛЬНО разным x на одной странице —
  это РАЗНЫЕ КОЛОНКИ (соседние таблицы); НЕ клади их в один раздел. Внутри раздела
  x примерно одинаков (одна колонка).

Ответь ТОЛЬКО валидным JSON без markdown:
{"root":"тема документа (2-6 слов)",
 "children":[
   {"title":"Раздел","block_ids":["b0","b1","b2"],"children":[
      {"title":"Подраздел","block_ids":["b3"]}
   ]},
   {"title":"Другой раздел","block_ids":["b4","b5"]}
 ]}"""


# ─── РАЗРЫВ СТРАНИЦЫ ≠ КОНЕЦ АБЗАЦА ───────────────────────────────────────────
# Абзац часто переходит на следующую страницу («…используют позиционное
# кодирование,» ↵ новая страница ↵ «добавляемое к эмбеддингам…»). Модель видит смену
# @стрN и режет раздел по границе страницы — продолжение теряется. Детектим такие
# строки детерминированно: предыдущая страница кончилась НЕ концом предложения, а
# новая начинается не с заголовка. Колонтитулы (номер страницы и т.п.) пропускаем.
_PAGE_FURNITURE_RE = re.compile(
    r"^\s*(\d{1,4}|[ivxlcdm]{1,6}|(стр\.?|page|p\.)\s*\d{1,4}|\d{1,4}\s*(/|из|of)\s*\d{1,4})\s*$", re.I)
_SENT_END_RE = re.compile(r"[.!?…][»\"”’)\]]*\s*$")


def _page_continuations(blocks: list[dict]) -> dict[str, str]:
    """{id первой строки продолжения на новой странице: id последней текстовой строки
    предыдущей страницы}. Только для блоков с координатами (PDF)."""
    real = [b for b in blocks
            if b.get("page") is not None and b.get("bbox")
            and not _PAGE_FURNITURE_RE.match((b.get("text") or "").strip())]
    out = {}
    for prev, cur in zip(real, real[1:]):
        if cur["page"] == prev["page"]:
            continue
        if cur.get("kind") == "heading" or prev.get("kind") == "heading":
            continue
        if _SENT_END_RE.search((prev.get("text") or "").strip()):
            continue
        out[cur["id"]] = prev["id"]
    return out


def _continuation_paragraph(blocks: list[dict], start_idx: int) -> list[str]:
    """Строки абзаца-продолжения, начиная со start_idx: та же страница и колонка, без
    абзацного отступа по вертикали и до первого заголовка."""
    first = blocks[start_idx]
    x0 = first["bbox"][0]
    line_h = max(first["bbox"][3] - first["bbox"][1], 1.0)
    ids, prev = [first["id"]], first
    for b in blocks[start_idx + 1:]:
        if b.get("page") != first.get("page") or not b.get("bbox") or b.get("kind") == "heading":
            break
        if b["bbox"][1] - prev["bbox"][3] > max(line_h * 0.9, 4.0):   # абзацный отступ
            break
        if abs(b["bbox"][0] - x0) > 40:                                # другая колонка
            break
        ids.append(b["id"]); prev = b
    return ids


def build_sections_from_blocks(blocks: list[dict], in_answer_names: set[str]) -> dict | None:
    """
    Блоки (id+текст) → дерево разделов {nodes, edges, root}, где КАЖДЫЙ узел несёт
    block_ids. excerpt/full выводим из текста самих блоков (не выдумывает модель).
    Возвращает None при сбое — вызывающий откатывается на старый путь.
    """
    if not blocks:
        return None
    by_id = {b["id"]: b.get("text", "") for b in blocks}

    # позиция блока (страница, x, y) — даёт модели понять КОЛОНКИ и вёрстку, чтобы не
    # объединять соседние таблицы. Есть только у PDF-блоков (у текста координат нет).
    def _pos(b):
        bb = b.get("bbox")
        if not bb or len(bb) < 2:
            # презентация: координат нет, но слайд известен
            return f" @слайд{b['page'] + 1}" if b.get("page") is not None else ""
        return f" @стр{(b.get('page') or 0) + 1} x{int(bb[0])} y{int(bb[1])}"

    # строки, продолжающие абзац с предыдущей страницы, — явная метка для модели
    conts = _page_continuations(blocks)
    CONT = " ⤷ПРОДОЛЖЕНИЕ_АБЗАЦА"
    listing = "\n".join(
        f"[{b['id']}]{_pos(b)}{CONT if b['id'] in conts else ''} {by_id[b['id']][:400]}" for b in blocks
    )
    raw = call_llm(
        system=_SECTIONS_FROM_BLOCKS_SYSTEM + _lang_rule(),
        user=f"СТРОКИ ДОКУМЕНТА (по порядку):\n{listing}",
        model=POWER_MODEL,
        max_tokens=8000,   # построчный ввод → в дереве много id, нужен запас на вывод
    )
    try:
        data = parse_json_lenient(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get("children"):
        return None

    nodes, edges, used = [], [], set()

    def uid(name: str) -> str:
        base = (name or "?").strip() or "?"
        key, i = base, 1
        while key in used:
            i += 1; key = f"{base}#{i}"
        used.add(key)
        return key

    def block_text(ids: list) -> str:
        return "\n".join(by_id.get(i, "") for i in ids if i in by_id).strip()

    def is_ans(title: str, full: str) -> bool:
        blob = f"{title} {full}".lower()
        return any(nm and nm in blob for nm in in_answer_names)

    covered = set()   # какие строки уже попали в дерево (чтобы не потерять ни одной)

    def add(title: str, ntype: str, ids: list) -> str:
        nid = uid(title)
        valid_ids = [i for i in (ids or []) if i in by_id]
        covered.update(valid_ids)
        full = block_text(valid_ids)
        nodes.append({
            "id": nid, "name": (title or "").strip(), "type": ntype,
            "block_ids": valid_ids,
            "excerpt": full[:400], "full": full[:1500],
            "world": "", "mentions": 1,
            "in_answer": is_ans(title, full),
        })
        return nid

    def walk(children, parent_id):
        for ch in children or []:
            if not isinstance(ch, dict) or not (ch.get("title") or "").strip():
                continue
            kids = ch.get("children") or []
            ntype = "branch" if kids else "section"
            cid = add(ch["title"], ntype, ch.get("block_ids"))
            edges.append({"from": parent_id, "to": cid, "label": "",
                          "in_answer": nodes[-1]["in_answer"]})
            walk(kids, cid)

    root_id = add(data.get("root") or tr("Документ", "Document"), "branch", [])
    walk(data.get("children"), root_id)
    if len(nodes) <= 1:
        return None   # модель ничего осмысленного не сгруппировала

    # страховка покрытия: построчный ввод длинный, модель может пропустить часть строк —
    # ничего не теряем, собираем непокрытые (в исходном порядке) в раздел «Прочее».
    missed = [b["id"] for b in blocks if b["id"] not in covered]
    if missed:
        cid = add(tr("Прочее", "Other"), "section", missed)
        edges.append({"from": root_id, "to": cid, "label": "",
                      "in_answer": nodes[-1]["in_answer"]})

    # страховка разрыва страницы: если модель всё-таки отрезала продолжение абзаца
    # (отнесла его к другому разделу или в «Прочее») — возвращаем абзац-продолжение
    # в раздел последней строки предыдущей страницы. Детерминированно, не по модели.
    if conts:
        order = {b["id"]: i for i, b in enumerate(blocks)}
        owner = {}                                   # id строки → узел (последний = самый глубокий)
        for n in nodes:
            for i in n["block_ids"]:
                owner[i] = n
        changed = set()
        for cid_line, prev_line in conts.items():
            target = owner.get(prev_line)
            if target is None or owner.get(cid_line) is target:
                continue
            for line in _continuation_paragraph(blocks, order[cid_line]):
                src = owner.get(line)
                if src is target:
                    continue
                if src is not None:
                    src["block_ids"] = [i for i in src["block_ids"] if i != line]
                    changed.add(src["id"])
                target["block_ids"].append(line)
                owner[line] = target
            target["block_ids"].sort(key=lambda i: order.get(i, 0))
            changed.add(target["id"])
        if changed:
            for n in nodes:
                if n["id"] in changed:
                    full = block_text(n["block_ids"])
                    n["excerpt"], n["full"] = full[:400], full[:1500]
                    n["in_answer"] = is_ans(n["name"], full)
            # лист, у которого после переноса не осталось строк, — убираем вместе с ребром
            parents = {e["from"] for e in edges}
            empty = {n["id"] for n in nodes
                     if n["id"] != root_id and not n["block_ids"] and n["id"] not in parents}
            nodes = [n for n in nodes if n["id"] not in empty]
            edges = [e for e in edges if e["to"] not in empty]
            by_nid = {n["id"]: n for n in nodes}
            for e in edges:
                if e["to"] in by_nid:
                    e["in_answer"] = by_nid[e["to"]]["in_answer"]
            logger.info("sections: вернул продолжения абзацев через разрыв страницы (%d узлов)", len(changed))
    return {"nodes": nodes, "edges": edges, "root": root_id}


# ─── САБ-ЧАТ ПО ВЕТКЕ (вопрос строго по контексту одного узла) ────────────────

_NODE_ANSWER_SYSTEM = """Ты отвечаешь на вопрос пользователя ПО ДОКУМЕНТУ (раздел «{node_title}»).
ГЛАВНОЕ ПРАВИЛО: отвечай ТОЛЬКО на основе приведённого текста документа (фрагмент раздела
+ дополнительные фрагменты источника). НЕ отвечай из своих общих знаний и НЕ выдумывай —
только то, что реально есть в тексте документа.
- Если ответ есть в тексте — дай прямой конкретный ответ по сути вопроса (приведи нужные
  числа/пункты/факты из текста), не пересказывай фрагмент целиком.
- Если спрашивают про термин или слово — объясни его ТАК, КАК ОНО УПОТРЕБЛЯЕТСЯ В ДОКУМЕНТЕ.
- Если в приведённом тексте ответа НЕТ — честно скажи, что в документе это не раскрыто.
  НЕ заменяй ответ общими знаниями и не сочиняй.

Ответ станет новым узлом ментальной карты, ответвлением от этого раздела.
Ответь ТОЛЬКО валидным JSON без markdown:
{{"title":"короткий заголовок ответа (2-5 слов)",
  "excerpt":"краткий ответ, 1-2 предложения",
  "full":"более полный ответ, 3-6 предложений"}}"""


def answer_for_node(node_title: str, node_text: str, question: str,
                    sources: list[dict] | None = None) -> dict | None:
    """Отвечает на вопрос СТРОГО по тексту документа (не из общих знаний модели).

    node_text — фрагмент выбранного раздела (первичный контекст); sources — доп.
    дословные фрагменты источника (passage-RAG), чтобы ответ нашёлся, даже если он
    в другом разделе документа. Возвращает {title, excerpt, full} для нового
    узла-ответвления или None при сбое. Один вызов модели — быстро и дёшево."""
    if not (node_text or "").strip():
        node_text = node_title
    sources_text = "\n\n".join(
        f"[{s.get('title') or 'раздел'}]\n{s.get('text','')}" for s in (sources or []) if s.get("text")
    )
    extra = f"\n\nДОП. ФРАГМЕНТЫ ИСТОЧНИКА (дословно из документа):\n{sources_text}" if sources_text else ""
    raw = call_llm(
        system=_NODE_ANSWER_SYSTEM.format(node_title=node_title) + _lang_rule(),
        user=f"ФРАГМЕНТ РАЗДЕЛА «{node_title}»:\n{node_text}{extra}\n\nВОПРОС: {question}",
        model=POWER_MODEL,
        max_tokens=1200,
    )
    try:
        data = parse_json_lenient(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    title = (data.get("title") or question or tr("Ответ", "Answer")).strip()[:80]
    excerpt = (data.get("excerpt") or "").strip()[:400]
    full = (data.get("full") or excerpt).strip()[:1500]
    if not excerpt and not full:
        return None
    return {"title": title, "excerpt": excerpt, "full": full}


# ─── ФРАГМЕНТЫ ИСХОДНИКА ДЛЯ УЗЛОВ ────────────────────────────────────────────

def build_concept_info(text: str, graph: dict) -> dict:
    """Для каждого понятия собирает {description, snippet}, где snippet — 1-2
    предложения из ИСХОДНОГО текста, где это понятие упоминается. Нужно, чтобы по
    клику на узел показать фрагмент источника (grounding: откуда взялось понятие)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    low = [(s, s.lower()) for s in sentences]
    info = {}
    for n in graph["nodes"].values():
        nn = n["name"].strip().lower()
        if not nn or nn in info:
            continue
        hits = [s for (s, sl) in low if nn in sl]
        info[nn] = {
            "description": n.get("description", ""),
            "snippet": " ".join(hits[:2]).strip()[:600],
            "world": "",   # краткая справка «из общих знаний», заполняется add_world_info
        }
    return info


def _batch_world_info(names: list[str]) -> dict[str, str]:
    """Одним запросом даёт краткое общее определение каждого имени из списка
    (из общих знаний модели, не из загруженного текста). Возвращает
    {имя.lower(): определение}. Мягкая деградация: при сбое — пустой dict.
    Переиспользуется и для понятий графа, и для заголовков разделов mind-map."""
    names = [n for n in names if n and n.strip()][:35]
    if not names:
        return {}
    listing = "\n".join(f"- {nm}" for nm in names)
    try:
        raw = call_llm(
            system="""Дай КРАТКОЕ общее определение каждого понятия из списка — 1-2 предложения,
простыми словами, из общих знаний (НЕ из какого-либо текста). Ответь ТОЛЬКО валидным
JSON без markdown: {"Понятие": "краткое определение", ...}. Ключи — ДОСЛОВНО как в списке.""" + _lang_rule(),
            user=f"ПОНЯТИЯ:\n{listing}",
            model=POWER_MODEL,
            max_tokens=2500,
        )
    except PipelineError:
        return {}
    try:
        data = parse_json_lenient(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k).strip().lower(): v.strip()[:600]
            for k, v in data.items() if isinstance(v, str)}


def add_world_info(info: dict, graph: dict) -> None:
    """Заполняет info[name]['world'] краткой общей справкой по понятиям графа.
    Мутирует info на месте."""
    names = [n["name"] for n in sorted(graph["nodes"].values(),
                                       key=lambda x: x["mentions"], reverse=True)[:35]]
    for k, v in _batch_world_info(names).items():
        if k in info:
            info[k]["world"] = v


# ─── СЕРИАЛИЗАЦИЯ ГРАФА ДЛЯ ФРОНТА ────────────────────────────────────────────

def serialize_graph(graph: dict, top_ids: set[str], info: dict | None = None) -> dict:
    """Готовит граф к отдаче: убирает тяжёлые векторы, помечает узлы/рёбра,
    попавшие в ответ (для подсветки 'пути рассуждения' на фронте)."""
    info = info or {}
    nodes = []
    for n in graph["nodes"].values():
        nn = n["name"].strip().lower()
        nodes.append({
            "id":          n["id"],
            "name":        n["name"],
            "type":        n["type"],
            "description": n["description"],
            "snippet":     info.get(nn, {}).get("snippet", ""),
            "world":       info.get(nn, {}).get("world", ""),
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

def _graph_state(graph: dict, memory: str, passages: list[dict] | None = None) -> dict:
    """Компактный «слепок» концепт-графа для повторных вопросов (RAG без пересборки).

    Кладём только то, что нужно для step5 (векторный поиск) + step6 (ответ):
    имя/тип/описание/упоминания/вектор узла, рёбра, память и флаг реальных
    эмбеддингов. Позиции/блоки/мир-инфа в ответе не участвуют — их не тащим.
    Этот объект уходит на фронт и возвращается в /api/ask как есть.
    """
    return {
        "nodes": [
            {
                "id":            n["id"],
                "name":          n["name"],
                "type":          n["type"],
                "description":   n.get("description", ""),
                "mentions":      n["mentions"],
                "merged_vector": n["merged_vector"],
            }
            for n in graph["nodes"].values()
        ],
        "edges":      graph["edges"],
        "memory":     memory,
        "real_embed": graph.get("real_embed", False),
        # дословные разделы + их эмбеддинги → step6 заземляется на реальный текст,
        # а не только на сжатый сущностный граф (см. build_passages)
        "passages":   passages or [],
    }


def answer_from_state(state: dict, query: str) -> dict:
    """Лёгкий ответ поверх УЖЕ построенного графа (никакой пересборки).

    Берёт слепок графа из _graph_state, гоняет только step5 (векторный поиск)
    и step6 (генерация ответа) + собирает explanation. Дерево (mindmap) не
    трогаем — оно стабильно; подсветку in_answer фронт пересчитает сам по
    именам из in_answer_names.
    """
    if not query or not query.strip():
        raise PipelineError("Пустой запрос")
    if not GEMINI_API_KEY:
        raise PipelineError("GEMINI_API_KEY не задан в переменных окружения")

    raw_nodes = (state or {}).get("nodes") or []
    if not raw_nodes:
        raise PipelineError("Пустой граф — нужно сначала построить документ")

    graph = {
        "nodes":      {n["id"]: dict(n) for n in raw_nodes},
        "edges":      (state or {}).get("edges") or [],
        "real_embed": (state or {}).get("real_embed", False),
    }
    memory   = (state or {}).get("memory", "")
    passages = (state or {}).get("passages") or []

    top_nodes   = step5_vector_retrieval(graph, query)
    # заземляем ответ на дословные разделы (passage-RAG), а не только на сущности
    sources     = retrieve_passages(passages, query, graph["real_embed"])
    answer_data = step6_generate_answer(memory, top_nodes, query, sources)

    top_ids = {n["id"] for n in top_nodes}
    query_intent = top_nodes[0].get("query_intent") if top_nodes else None
    explanation = {
        "query_intent": query_intent,
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
        "query":           query,
        "answer":          answer_data,
        "explanation":     explanation,
        # имена узлов-ответа → фронт пересветит стабильное дерево (title+full)
        "in_answer_names": [n["name"] for n in top_nodes],
    }


def run_pipeline(text: str, query: str, blocks: list[dict] | None = None) -> dict:
    """
    Полный прогон. Возвращает dict, готовый к json-ответу API.

    blocks — готовые блоки {id, text} (у PDF из spatial-манифеста, с координатами
    на фронте). Если не переданы — режем текст на блоки-абзацы. Mind-map строится
    ИЗ блоков (узлы несут block_ids), см. build_sections_from_blocks.

    Ключи ответа:
      answer      — {answer, summary, key_points}
      schema      — структурная схема (6-10 узлов)
      graph       — {nodes, edges} с флагом in_answer для explainable-подсветки
      mindmap     — реальная структура документа (разделы + дословные цитаты)
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

    # блоки: готовые (PDF) или нарезка текста (paste/txt/md) — единый фундамент
    blocks = blocks or split_text_into_blocks(text)

    chunks      = step1_chunk(text)
    extracted   = step2_extract_entities(chunks)
    graph       = step3_build_graph_with_vectors(extracted)

    if not graph["nodes"]:
        raise PipelineError("Не удалось извлечь ни одной сущности из текста")

    memory      = step4_memory(graph)
    top_nodes   = step5_vector_retrieval(graph, query)
    schema      = step6_reason_and_generate(memory, top_nodes, graph, query)

    top_ids = {n["id"] for n in top_nodes}
    in_answer_names = {n["name"].strip().lower() for n in top_nodes}
    # фрагменты исходника по каждому понятию (для модалки по клику на узел)
    info = build_concept_info(text, graph)
    # краткая справка о понятиях «из общих знаний» (доп-инфа в модалке узла).
    # Мягкая деградация: если шаг упал — просто без справки, фрагмент источника остаётся.
    try:
        add_world_info(info, graph)
    except PipelineError:
        pass

    # Mind-map: РЕАЛЬНАЯ структура документа (заголовки/разделы, дословные цитаты),
    # а не абстрактные категории от модели — то, что рисуется на фронте.
    # Мягкая деградация: если шаг упал — mindmap=None, фронт покажет обычный граф.
    mindmap = None
    try:
        # План 3: строим разделы ИЗ блоков (узлы несут block_ids). Если не вышло —
        # откат на старую сегментацию плоского текста (совместимость).
        mindmap = build_sections_from_blocks(blocks, in_answer_names)
        if not mindmap:
            sections_raw = step_document_sections(text, main_topic_fallback(graph))
            if sections_raw:
                mindmap = flatten_sections(sections_raw, in_answer_names)
        if mindmap:
            section_titles = [n["name"] for n in mindmap["nodes"] if n["type"] == "section"]
            world_blurbs = _batch_world_info(section_titles)
            for n in mindmap["nodes"]:
                if n["type"] == "section":
                    n["world"] = world_blurbs.get(n["name"].strip().lower(), "")
    except PipelineError:
        mindmap = None

    # Passage-RAG: дословные разделы + эмбеддинги (покрывают документ целиком, в т.ч.
    # таблицы/списки, которых нет в сущностном графе). Ответ заземляем на них — иначе
    # модель отвечает «в тексте нет», когда информация есть только вне сущностей.
    passages = []
    try:
        passages = build_passages(mindmap, graph.get("real_embed", False))
    except PipelineError:
        passages = []
    sources     = retrieve_passages(passages, query, graph.get("real_embed", False))
    answer_data = step6_generate_answer(memory, top_nodes, query, sources)

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
        "graph":       serialize_graph(graph, top_ids, info),
        "mindmap":     mindmap,   # иерархическое дерево (может быть None → фронт рисует graph)
        # слепок графа + дословные разделы для повторных вопросов через /api/ask
        "graph_state": _graph_state(graph, memory, passages),
        "explanation": explanation,
        "stats": {
            "words":  len(text.split()),
            "chunks": len(chunks),
            "nodes":  len(graph["nodes"]),
            "edges":  len(graph["edges"]),
            "top_k":  len(top_nodes),
            "real_embed": graph.get("real_embed", False),   # реальные эмбеддинги или hash-фолбэк
        },
    }
