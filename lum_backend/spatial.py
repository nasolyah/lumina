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

import pdfplumber
import pypdfium2 as pdfium

# Лимиты (env-настраиваемые — защита памяти/CPU Render)
MAX_PAGES = int(os.environ.get("SPATIAL_MAX_PAGES", "30"))
RENDER_DPI = int(os.environ.get("SPATIAL_DPI", "130"))
WEBP_QUALITY = int(os.environ.get("SPATIAL_WEBP_QUALITY", "80"))

_LINE_TOL = 3.0   # слова с близким `top` — одна строка (pt)
_PARA_GAP = 6.0   # разрыв больше — граница блока (pt)


def _render_page_dataurl(pdf_page) -> tuple[str, int, int]:
    """Рендер страницы pypdfium2 → (data URL WebP, ширина_px, высота_px)."""
    scale = RENDER_DPI / 72.0
    pil = pdf_page.render(scale=scale).to_pil()
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=WEBP_QUALITY, method=4)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{b64}", pil.width, pil.height


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


def build_manifest(pdf_bytes: bytes) -> dict:
    """PDF-байты → манифест {schema_version, pages[], blocks[]}."""
    pages_out, blocks_out = [], []
    order = 0

    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(len(pdf.pages), MAX_PAGES)
            for pi in range(n):
                page = pdf.pages[pi]
                img, iw, ih = _render_page_dataurl(doc[pi])
                pages_out.append({
                    "index": pi,
                    "width_pt": round(float(page.width), 1),
                    "height_pt": round(float(page.height), 1),
                    "image_w": iw, "image_h": ih,
                    "image": img,
                })
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
                for b in _group_words_into_blocks(words):
                    b.update({"id": f"b{order}", "page": pi, "order": order})
                    blocks_out.append(b)
                    order += 1
    finally:
        doc.close()

    return {
        "schema_version": 2,
        "pages": pages_out,
        "blocks": blocks_out,
        "truncated": len(pages_out) >= MAX_PAGES,
    }
