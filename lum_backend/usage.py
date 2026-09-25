"""
Учёт расхода Gemini по пользователям + месячный лимит инфографики.

Как устроено:
  • каждый вызов Gemini (текст, OCR, картинка, эмбеддинги) после успешного ответа
    зовёт record(...) — берём РЕАЛЬНЫЙ расход из usageMetadata ответа и версию модели
    (modelVersion: видно, на что на самом деле указывает алиас «-latest»);
  • записи копятся в ContextVar на время одного запроса к API (эндпоинт зовёт start());
  • в конце запроса flush(...) пишет ОДНУ строку в таблицу Supabase `usage_events`
    (пользователь, действие, токены, картинки, стоимость, детали по вызовам);
  • лимит инфографики считается по этой же таблице: картинок за текущий месяц (UTC).

Схема таблиц — supabase_usage.sql. Если Supabase не настроен — только лог в Render,
лимит не применяется (fail-open: сбой учёта не должен ломать продукт).
"""
from __future__ import annotations

import contextvars
import datetime
import logging
import os
import threading

import requests

logger = logging.getLogger("lumina.usage")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")

# ─── ЛИМИТЫ ───────────────────────────────────────────────────────────────────
# картинок в месяц по тарифу; None — без лимита. Free — инфографики нет вовсе.
INFOGRAPHIC_LIMITS = {
    "free": int(os.environ.get("INFOGRAPHIC_LIMIT_FREE", "0")),
    "pro":  int(os.environ.get("INFOGRAPHIC_LIMIT_PRO", "10")),
    "max":  None,
}
OWNER_EMAILS = {e.strip().lower() for e in
                os.environ.get("OWNER_EMAILS", "markingmark33@gmail.com").split(",") if e.strip()}

# ─── ЦЕНЫ (USD, Standard paid tier) ───────────────────────────────────────────
# Сверено с ai.google.dev/gemini-api/docs/pricing 26.09.2026. in/out — за 1M токенов
# (out включает «мышление»), image — за одну картинку 1K/2K. Ключ — префикс имени
# модели (modelVersion из ответа); порядок важен: более длинные префиксы раньше.
# Токены храним сырыми, так что стоимость всегда можно пересчитать по новым ценам.
# ВНИМАНИЕ: у gemini-3.8-flash цена удваивается с 01.01.2027 ($1.50 / $7.50).
PRICES = [
    ("gemini-3-pro-image",          {"in": 2.00, "out": 12.00, "image": 0.134}),
    # у flash-image моделей сверена только цена картинки; in/out — оценка (вклад копеечный)
    ("gemini-3.1-flash-lite-image", {"in": 0.25, "out": 1.50,  "image": 0.0336}),
    ("gemini-3.1-flash-image",      {"in": 0.50, "out": 3.00,  "image": 0.067}),
    ("gemini-2.5-flash-image",      {"in": 0.30, "out": 2.50,  "image": 0.039}),
    ("gemini-3.8-flash",            {"in": 0.75, "out": 3.75}),
    ("gemini-3.5-flash-lite",       {"in": 0.30, "out": 2.50}),
    ("gemini-3.1-flash-lite",       {"in": 0.25, "out": 1.50}),
    ("gemini-2.5-flash-lite",       {"in": 0.10, "out": 0.40}),
    ("gemini-2.5-flash",            {"in": 0.30, "out": 2.50}),
    ("gemini-embedding",            {"in": 0.15, "out": 0.0}),
    ("text-embedding",              {"in": 0.0,  "out": 0.0}),
]
# Неизвестная версия → консервативная цена своего класса (лучше переоценить расход).
_FALLBACK = {
    "image": {"in": 2.00, "out": 12.00, "image": 0.134},
    "lite":  {"in": 0.30, "out": 2.50},
    "embed": {"in": 0.15, "out": 0.0},
    "text":  {"in": 0.75, "out": 3.75},
}
IMAGE_TOKENS_1K = 1120   # столько выходных токенов «весит» картинка 1K/2K


def price_for(model: str, kind: str) -> dict:
    name = (model or "").lower().removeprefix("models/")
    for prefix, p in PRICES:
        if name.startswith(prefix):
            return p
    if kind == "image" or "image" in name:
        return _FALLBACK["image"]
    if kind == "embed" or "embed" in name:
        return _FALLBACK["embed"]
    if "lite" in name:
        return _FALLBACK["lite"]
    return _FALLBACK["text"]


# ─── СБОР ЗА ЗАПРОС ───────────────────────────────────────────────────────────
_CALLS: contextvars.ContextVar[list | None] = contextvars.ContextVar("lumina_usage", default=None)


def start() -> None:
    """Начать учёт для текущего запроса (зовёт эндпоинт в своём теле)."""
    _CALLS.set([])


def record(model: str, kind: str, data: dict | None = None, *,
           images: int = 0, est_input_tokens: int = 0) -> None:
    """Записать один успешный вызов Gemini. data — JSON-ответ generateContent
    (берём usageMetadata и modelVersion). Для эмбеддингов usageMetadata нет —
    передаём оценку est_input_tokens. Вне start() ничего не делает."""
    calls = _CALLS.get()
    if calls is None:
        return
    meta = (data or {}).get("usageMetadata") or {}
    version = (data or {}).get("modelVersion") or model
    tin = int(meta.get("promptTokenCount") or est_input_tokens or 0)
    tout = int(meta.get("candidatesTokenCount") or 0) + int(meta.get("thoughtsTokenCount") or 0)
    p = price_for(version, kind)
    if images:
        text_out = max(0, tout - IMAGE_TOKENS_1K * images)
        cost = tin * p["in"] / 1e6 + images * p.get("image", 0) + text_out * p["out"] / 1e6
    else:
        cost = tin * p["in"] / 1e6 + tout * p["out"] / 1e6
    calls.append({"model": version, "kind": kind, "in": tin, "out": tout,
                  "images": images, "cost": round(cost, 6)})


def _post(row: dict) -> None:
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/usage_events",
            headers={"apikey": SUPABASE_SECRET_KEY, "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
                     "Content-Type": "application/json", "Prefer": "return=minimal"},
            json=row, timeout=10,
        )
        if not r.ok:
            logger.warning("usage: запись в Supabase не удалась: %s %s", r.status_code, r.text[:200])
    except requests.RequestException as e:
        logger.warning("usage: запись в Supabase не удалась: %s", e)


def flush(user: dict | None, endpoint: str, ok: bool = True, wait: bool = False) -> dict | None:
    """Итог запроса → лог + строка в usage_events. Пишем и неуспешные запросы:
    Google берёт деньги за каждый ответивший вызов, даже если дальше всё упало.
    wait=True — писать синхронно (нужно инфографике: лимит считается по таблице)."""
    calls = _CALLS.get()
    _CALLS.set(None)
    if not calls:
        return None
    row = {
        "user_id": (user or {}).get("sub"),
        "endpoint": endpoint,
        "ok": ok,
        "calls": len(calls),
        "input_tokens": sum(c["in"] for c in calls),
        "output_tokens": sum(c["out"] for c in calls),
        "images": sum(c["images"] for c in calls),
        "cost_usd": round(sum(c["cost"] for c in calls), 6),
        "models": sorted({c["model"] for c in calls}),
        "detail": calls,
    }
    logger.info("usage: %s user=%s calls=%d in=%d out=%d img=%d cost=$%.4f models=%s",
                endpoint, row["user_id"], row["calls"], row["input_tokens"], row["output_tokens"],
                row["images"], row["cost_usd"], ",".join(row["models"]))
    if SUPABASE_URL and SUPABASE_SECRET_KEY and row["user_id"]:
        if wait:
            _post(row)
        else:
            threading.Thread(target=_post, args=(row,), daemon=True).start()
    return row


# ─── ТАРИФ И ЛИМИТ ИНФОГРАФИКИ ────────────────────────────────────────────────
def _rest_headers(extra: dict | None = None) -> dict:
    h = {"apikey": SUPABASE_SECRET_KEY, "Authorization": f"Bearer {SUPABASE_SECRET_KEY}"}
    h.update(extra or {})
    return h


def user_plan(user: dict) -> str:
    """free | pro | max. Владелец — max. Нет записи/ошибка → free."""
    if (user.get("email") or "").lower() in OWNER_EMAILS:
        return "max"
    if not (SUPABASE_URL and SUPABASE_SECRET_KEY):
        return "free"
    try:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/user_plans",
                         params={"user_id": f"eq.{user.get('sub')}", "select": "plan"},
                         headers=_rest_headers(), timeout=8)
        rows = r.json() if r.ok else []
        return (rows[0].get("plan") if rows else None) or "free"
    except (requests.RequestException, ValueError) as e:
        logger.warning("usage: не удалось прочитать тариф: %s", e)
        return "free"


def _month_start_utc() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


def infographics_used_this_month(user_id: str) -> int:
    """Сколько картинок пользователь получил с 1-го числа текущего месяца (UTC)."""
    if not (SUPABASE_URL and SUPABASE_SECRET_KEY):
        return 0
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/usage_events",
            params={"user_id": f"eq.{user_id}", "endpoint": "eq.infographic", "images": "gt.0",
                    "created_at": f"gte.{_month_start_utc()}", "select": "images"},
            headers=_rest_headers(), timeout=8,
        )
        if not r.ok:
            logger.warning("usage: счётчик инфографики: %s %s", r.status_code, r.text[:200])
            return 0
        return sum(int(x.get("images") or 0) for x in r.json())
    except (requests.RequestException, ValueError) as e:
        logger.warning("usage: счётчик инфографики недоступен: %s", e)
        return 0


def infographic_quota(user: dict) -> dict:
    """{plan, used, limit (None = безлимит, 0 = недоступно на тарифе), left}."""
    plan = user_plan(user)
    limit = INFOGRAPHIC_LIMITS.get(plan, INFOGRAPHIC_LIMITS["free"])
    if limit == 0:          # на тарифе инфографики нет — таблицу не спрашиваем
        return {"plan": plan, "used": 0, "limit": 0, "left": 0}
    used = infographics_used_this_month(user.get("sub"))
    if limit is None:
        return {"plan": plan, "used": used, "limit": None, "left": None}
    return {"plan": plan, "used": used, "limit": limit, "left": max(0, limit - used)}
