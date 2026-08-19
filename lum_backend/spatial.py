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


_MIN_FIG_PT = 24.0   # фигуры меньше — иконки/подчёркивания/буллеты, не фигура


def _extract_figures(page) -> list[dict]:
    """
    Растровые картинки страницы (встроенные изображения) → bbox в пунктах.
    Векторные чарты/формулы (кластеры линий/кривых) — следующий срез Этапа 2:
    их надо ловить рендером региона, а не «чистым» извлечением (см. deferred).
    """
    figs = []
    for im in page.images:
        try:
            x0 = float(im["x0"]); x1 = float(im["x1"])
            top = float(im["top"]); bottom = float(im["bottom"])
        except (KeyError, TypeError, ValueError):
            continue
        if (x1 - x0) < _MIN_FIG_PT or (bottom - top) < _MIN_FIG_PT:
            continue
        figs.append({
            "bbox": [round(x0, 1), round(top, 1), round(x1, 1), round(bottom, 1)],
            "kind": "image",
        })
    return figs


_MAX_PRIMS = 800   # больше примитивов на странице — плотная таблица/шум, вектор не ищем


def _extract_vector_figures(page, words) -> list[dict]:
    """
    Векторные фигуры (чарты/чертежи): кластеры примитивов рисования (линии, кривые,
    прямоугольники). Формулы сюда обычно НЕ попадают — они текст (и уже покрыты
    вырезкой блока).

    Что решает ложные/пропуски:
      - наличие КРИВОЙ — сильный сигнал чарта (принимаем даже разреженный кластер);
      - «решётка» (много различных уровней гориз. И верт. линий) + текст в ячейках —
        это таблица, отсеиваем;
      - линейки-разделители во всю ширину/высоту, тонкие полоски, плотный текст.
    """
    pw = float(page.width); ph = float(page.height)
    # prim: (x0, top, x1, bottom, is_curve, orient)  orient: 'h'|'v'|'o'
    prims = []

    def _add(objs, is_curve):
        for obj in objs:
            try:
                x0 = float(obj["x0"]); x1 = float(obj["x1"])
                top = float(obj["top"]); bottom = float(obj["bottom"])
            except (KeyError, TypeError, ValueError):
                continue
            w = abs(x1 - x0); h = abs(bottom - top)
            if h < 3 and w > 0.6 * pw:   # горизонтальная линейка во всю ширину
                continue
            if w < 3 and h > 0.6 * ph:   # вертикальный бордюр во всю высоту
                continue
            orient = "h" if (h < 3 and w >= 3) else ("v" if (w < 3 and h >= 3) else "o")
            prims.append((min(x0, x1), min(top, bottom), max(x0, x1), max(top, bottom),
                          is_curve, orient))

    _add(page.lines, False)
    _add(page.curves, True)
    _add(page.rects, False)
    if not prims or len(prims) > _MAX_PRIMS:
        return []

    # кластеризация union-find по перекрытию расширенных bbox
    PAD = 8.0
    n = len(prims)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def overlap(a, b):
        return not (a[2] + PAD < b[0] or b[2] + PAD < a[0]
                    or a[3] + PAD < b[1] or b[3] + PAD < a[1])

    for i in range(n):
        for j in range(i + 1, n):
            if overlap(prims[i], prims[j]):
                parent[find(i)] = find(j)

    # агрегируем по кластерам: bbox, число примитивов/кривых, уровни H/V линий
    groups = {}
    for i in range(n):
        r = find(i)
        g = groups.get(r)
        if g is None:
            g = groups[r] = {"x0": 1e9, "top": 1e9, "x1": -1e9, "bottom": -1e9,
                             "cnt": 0, "curves": 0, "hlev": set(), "vlev": set()}
        p = prims[i]
        g["x0"] = min(g["x0"], p[0]); g["top"] = min(g["top"], p[1])
        g["x1"] = max(g["x1"], p[2]); g["bottom"] = max(g["bottom"], p[3])
        g["cnt"] += 1
        if p[4]:
            g["curves"] += 1
        if p[5] == "h":
            g["hlev"].add(round(p[1] / 4))    # уровень с допуском ~4pt
        elif p[5] == "v":
            g["vlev"].add(round(p[0] / 4))

    figs = []
    for g in groups.values():
        x0, top, x1, bottom = g["x0"], g["top"], g["x1"], g["bottom"]
        w = x1 - x0; h = bottom - top
        if w < 40 or h < 40:
            continue
        inside = sum(1 for wd in words
                     if x0 <= (wd["x0"] + wd["x1"]) / 2 <= x1
                     and top <= (wd["top"] + wd["bottom"]) / 2 <= bottom)
        # «решётка» гориз.+верт. линий с текстом в ячейках → таблица, пропускаем
        is_grid = len(g["hlev"]) >= 3 and len(g["vlev"]) >= 3
        if is_grid and g["curves"] == 0 and inside > 6:
            continue
        # кривая есть → чарт/чертёж даже с малым числом примитивов; иначе нужно >=6
        enough = (g["curves"] >= 1 and g["cnt"] >= 3) or (g["cnt"] >= 6)
        if not enough:
            continue
        if inside > 25:   # много текста внутри — похоже на абзац/таблицу
            continue
        figs.append({"bbox": [round(x0, 1), round(top, 1), round(x1, 1), round(bottom, 1)],
                     "kind": "drawing"})
    return figs


# подпись фигуры: блок, НАЧИНАЮЩИЙСЯ с «Рис 2» / «Рисунок 2» / «Figure 2» / «Fig. 2»
_CAP_RE = re.compile(r"^\s*(?:рис(?:унок)?|fig(?:ure)?)\.?\s*(\d{1,3})\b", re.IGNORECASE)
# ссылка на фигуру в тексте: те же формы в любом месте абзаца
_REF_RE = re.compile(r"(?:рис(?:унок|унк\w*)?|fig(?:ure)?)\.?\s*(\d{1,3})\b", re.IGNORECASE)


def _link_figures(blocks: list[dict], figures: list[dict]) -> list[dict]:
    """
    Связывает фигуры с текстом по подписям и ссылкам:
      1) блок-подпись («Рис N …») → присваивает номер ближайшей фигуре над ним;
      2) ссылки «см. рис N» в остальных блоках → links[] {block, figure}.
    Мутирует figures (number/caption/caption_block), возвращает links.
    """
    by_page = {}
    for f in figures:
        by_page.setdefault(f["page"], []).append(f)

    caption_ids = set()

    def _vgap(f, ct, cb):
        ft, fb = f["bbox"][1], f["bbox"][3]
        if fb <= ct:
            return ct - fb          # фигура выше блока
        if ft >= cb:
            return ft - cb          # фигура ниже блока
        return 0                    # пересекаются

    for b in blocks:
        m = _CAP_RE.match(b["text"] or "")
        if not m:
            continue
        figs = by_page.get(b["page"], [])
        if not figs:
            continue
        ct, cb = b["bbox"][1], b["bbox"][3]
        best = min(figs, key=lambda f: _vgap(f, ct, cb))
        # подпись ДОЛЖНА прилегать к фигуре; иначе это ссылка-предложение
        # («Figure 2 plots…»), а не подпись — пропускаем, останется ссылкой.
        if _vgap(best, ct, cb) > 60:
            continue
        best["number"] = int(m.group(1))
        best["caption"] = (b["text"] or "").strip()[:200]
        best["caption_block"] = b["id"]
        caption_ids.add(b["id"])

    num2fig = {}
    for f in figures:
        if f.get("number") is not None:
            num2fig.setdefault(f["number"], f["id"])

    links = []
    for b in blocks:
        if b["id"] in caption_ids:
            continue                                   # сама подпись — не ссылка
        nums = {int(x) for x in _REF_RE.findall(b["text"] or "")}
        for num in nums:
            fid = num2fig.get(num)
            if fid:
                links.append({"block": b["id"], "figure": fid})
    return links


def build_manifest(pdf_bytes: bytes, image_sink=None) -> dict:
    """
    PDF-байты → манифест {schema_version, pages[], blocks[], figures[], links[]}.

    image_sink(page_index, webp_bytes) -> str: куда деть картинку страницы и что
    записать в pages[].image. По умолчанию — data URL (dev / без Storage). Для
    прода передают sink, кладущий WebP в Supabase Storage и возвращающий URL/путь
    (см. storage.py). Так извлечение остаётся чистым и тестируемым локально.
    """
    sink = image_sink or _dataurl_sink
    pages_out, blocks_out, figures_out = [], [], []
    order = 0
    fig_order = 0

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
                page_figs = _extract_figures(page) + _extract_vector_figures(page, words)
                for f in page_figs:
                    f.update({"id": f"f{fig_order}", "page": pi})
                    figures_out.append(f)
                    fig_order += 1
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

    links_out = _link_figures(blocks_out, figures_out)

    return {
        "schema_version": 2,
        "pages": pages_out,
        "blocks": blocks_out,
        "figures": figures_out,
        "links": links_out,
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
