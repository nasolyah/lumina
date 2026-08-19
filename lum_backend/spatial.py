"""
Spatial PDF Mapper — layout-aware извлечение (Этап 1).

PDF (байты) → манифест документа: реальные страницы (рендер, вёрстка сохранена)
+ текст-блоки с bbox-координатами (дословно, без переписывания моделью).

  - pdfplumber (MIT)   — текст-блоки с координатами
  - pypdfium2 (Apache) — рендер страниц в WebP

Лицензионно-чистый стек (без AGPL-PyMuPDF).

MVP-ограничения (см. docs/spatial-mvp-deferred.md):
  - картинки страниц embed'ятся как data URL в манифест (позже — Supabase Storage);
  - синхронный рендер с лимитом страниц (позже — async + постраничный ленивый рендер);
  - фигуры/чарты/формулы не извлекаются (Этап 2);
  - матем-текст в блоках ненадёжен → в модель как есть не отдавать.
"""
import base64
import io
import os
import re

import pdfplumber
import pypdfium2 as pdfium

# Лимиты (env-настраиваемые — защита памяти/CPU Render)
MAX_PAGES = int(os.environ.get("SPATIAL_MAX_PAGES", "30"))
RENDER_DPI = int(os.environ.get("SPATIAL_DPI", "130"))
WEBP_QUALITY = int(os.environ.get("SPATIAL_WEBP_QUALITY", "80"))

# ── ПРАВИЛО ПРИВАТНОСТИ (контракт для Этапа 2) ────────────────────────────────
# ИНВАРИАНТ: пиксели документа (рендер страниц, вырезанные фигуры/чарты/формулы)
# НИКОГДА не уходят в LLM. В модель может идти только текст блоков и подписи фигур.
# Связывание фигур с текстом (Этап 2) делаем по подписям/ссылкам, НЕ по пикселям.
# Единственный переключатель визуального анализа — env ниже, по умолчанию ВЫКЛ.
# Под это правило можно честно писать «Приватность данных» на лендинге —
# но только пока весь LLM-bound код проходит через blocks_for_llm() (см. ниже).
LLM_VISION_ENABLED = os.environ.get("SPATIAL_LLM_VISION", "0") == "1"

_LINE_TOL = 3.0   # слова с близким `top` — одна строка (pt)
_PARA_GAP = 6.0   # разрыв больше — граница блока (pt)

# Шум ответных листов: строки из точек/подчёркиваний/тире (места для ответа).
# Их в экзаменационных PDF сотни — выкидываем, чтобы «Документ» не тонул в мусоре.
_LEADERS = ".…_·•-— \t\n"
_ONLY_LEADERS = re.compile(r"^[\s.…_·•\-—]+$")


def _is_noise(text: str) -> bool:
    """True для блоков-заполнителей (пунктир/подчёркивания линий для ответа)."""
    t = text.strip()
    if not t:
        return True
    if _ONLY_LEADERS.match(t):
        return True
    # почти сплошь заполнители (частичная линия с одиночной цифрой/буквой)
    leaders = sum(ch in _LEADERS for ch in t)
    return len(t) >= 10 and leaders / len(t) > 0.85


def _render_page_webp(pdf_page) -> tuple[bytes, int, int]:
    """Рендер страницы pypdfium2 → (WebP-байты, ширина_px, высота_px)."""
    scale = RENDER_DPI / 72.0
    pil = pdf_page.render(scale=scale).to_pil()
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=WEBP_QUALITY, method=4)
    return buf.getvalue(), pil.width, pil.height


def _dataurl_sink(page_index: int, webp: bytes) -> str:
    """Sink по умолчанию: картинка страницы как data URL (dev / без Storage)."""
    return "data:image/webp;base64," + base64.b64encode(webp).decode("ascii")


def _group_words_into_blocks(words: list[dict]) -> list[dict]:
    """
    Слова pdfplumber → строки (по близкому `top`) → блоки (по вертикальному разрыву).
    Заголовок: одиночная короткая строка с высотой глифов заметно выше медианной.
    Текст остаётся ДОСЛОВНЫМ.
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and abs(w["top"] - cur[-1]["top"]) > _LINE_TOL:
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)

    def line_box(ln):
        return (min(x["x0"] for x in ln), min(x["top"] for x in ln),
                max(x["x1"] for x in ln), max(x["bottom"] for x in ln))

    def line_text(ln):
        return " ".join(x["text"] for x in sorted(ln, key=lambda x: x["x0"]))

    # Шум режем на уровне СТРОК до сборки в блоки — иначе пунктирная линия ответа
    # слипается с соседним футером/датой в один «полу-шумный» блок и выживает.
    lines = [ln for ln in lines if not _is_noise(line_text(ln))]
    if not lines:
        return []

    heights = sorted(b[3] - b[1] for b in map(line_box, lines))
    median_h = heights[len(heights) // 2] if heights else 0

    blocks_lines: list[list] = []
    cur_lines = [lines[0]]
    for prev, ln in zip(lines, lines[1:]):
        gap = min(x["top"] for x in ln) - max(x["bottom"] for x in prev)
        if gap > _PARA_GAP:
            blocks_lines.append(cur_lines)
            cur_lines = []
        cur_lines.append(ln)
    if cur_lines:
        blocks_lines.append(cur_lines)

    out = []
    for blk in blocks_lines:
        boxes = [line_box(ln) for ln in blk]
        x0 = min(b[0] for b in boxes); top = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); bottom = max(b[3] for b in boxes)
        text = "\n".join(line_text(ln) for ln in blk).strip()
        h = (bottom - top) / max(len(blk), 1)
        is_heading = len(blk) == 1 and median_h and h >= median_h * 1.15 and len(text) < 80
        out.append({
            "bbox": [round(x0, 1), round(top, 1), round(x1, 1), round(bottom, 1)],
            "kind": "heading" if is_heading else "text",
            "text": text,
        })
    return out


def build_manifest(pdf_bytes: bytes, image_sink=None) -> dict:
    """
    PDF-байты → манифест {schema_version, pages[], blocks[]}.

    image_sink(page_index, webp_bytes) -> str: куда деть картинку страницы и что
    записать в pages[].image. По умолчанию — data URL (dev / без Storage). Для
    прода передают sink, кладущий WebP в Supabase Storage и возвращающий URL/путь
    (см. storage.py). Так извлечение остаётся чистым и тестируемым локально.
    """
    sink = image_sink or _dataurl_sink
    pages_out, blocks_out = [], []
    order = 0

    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(len(pdf.pages), MAX_PAGES)
            for pi in range(n):
                page = pdf.pages[pi]
                webp, iw, ih = _render_page_webp(doc[pi])
                img = sink(pi, webp)
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
                kept = 0
                for b in _group_words_into_blocks(words):
                    if _is_noise(b["text"]):
                        continue   # пунктир/линии для ответа — не кладём в манифест
                    b.update({"id": f"b{order}", "page": pi, "order": order})
                    blocks_out.append(b)
                    order += 1
                    kept += 1
                pages_out.append({
                    "index": pi,
                    "width_pt": round(float(page.width), 1),
                    "height_pt": round(float(page.height), 1),
                    "image_w": iw, "image_h": ih,
                    "image": img,
                    # страница без содержательных блоков (пустой ответный лист/скан) —
                    # фронт может её свернуть/приглушить (картинку всё равно показываем).
                    "sparse": kept == 0,
                })
    finally:
        doc.close()

    return {
        "schema_version": 2,
        "pages": pages_out,
        "blocks": blocks_out,
        "truncated": len(pages_out) >= MAX_PAGES,
    }


def blocks_for_llm(manifest: dict) -> list[dict]:
    """
    ЕДИНСТВЕННЫЙ санкционированный способ отдать spatial-контент в LLM.

    Возвращает только текст блоков (page, order, kind, text) — БЕЗ пикселей.
    Любой код, который шлёт контент документа в модель, обязан брать данные
    отсюда, а не тянуть `pages[].image` напрямую — так пиксели физически не
    попадут в запрос к модели (см. ПРАВИЛО ПРИВАТНОСТИ выше).

    Осознанное ограничение: матем-текст извлекается ненадёжно (степени/дроби
    коверкаются) — вызывающий должен учитывать это, а не считать текст точным.
    """
    return [
        {"page": b["page"], "order": b["order"], "kind": b["kind"], "text": b["text"]}
        for b in manifest.get("blocks", [])
    ]
