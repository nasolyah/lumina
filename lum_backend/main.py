"""
Lum / Lumina — FastAPI-обёртка над GraphRAG пайплайном.

Запуск локально:
    uvicorn main:app --reload --port 8000

Эндпоинты:
    GET  /                — health-check (жив ли сервис)
    GET  /api/health      — детальная проверка (задан ли ключ, какие модели)
    POST /api/analyze     — полный прогон пайплайна по тексту + запросу
    POST /api/extract     — извлечь текст из загруженного PDF (парсинг на бэке)
    POST /api/feedback    — сохранить отзыв пользователя (оценка + текст)
"""

import io
import os
import uuid
import time
import threading
import logging
import requests
from typing import Optional
import jwt
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Render показывает stdout/stderr как Application logs — а uvicorn access-логи
# ("POST /api/analyze 400") не несут причины ошибки. Логируем детали сами,
# чтобы инциденты можно было разбирать по логам Render, а не переписке/скриншотам.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lumina")
from pydantic import BaseModel, Field

import core

app = FastAPI(
    title="Lum / Lumina API",
    description="GraphRAG backend: текст + вопрос → граф знаний + объяснимый ответ",
    version="1.0.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# ALLOWED_ORIGINS в env — список доменов фронта через запятую.
# Пример: "https://lumina.uz,https://lum.pages.dev"
# Для локальной разработки по умолчанию открыто всё ("*").
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins = ["*"] if _origins_env.strip() == "*" else [
    o.strip() for o in _origins_env.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─── АВТОРИЗАЦИЯ (Supabase JWT по JWKS) ───────────────────────────────────────
# Фронт шлёт access_token сессии Supabase в заголовке Authorization: Bearer <JWT>.
# Проверяем подпись публичным ключом из JWKS проекта (алгоритм ES256) — локально,
# ключ кэшируется, запроса к Supabase на каждый вызов нет. Это закрывает бэкенд:
# без валидного токена /api/analyze и /api/extract не отдаются (защита баланса Gemini).
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
# Секретный ключ Supabase (новый формат sb_secret_… ; отзывной и ротируемый по-отдельности,
# предпочтителен легаси service_role). Нужен ТОЛЬКО серверу — обходит RLS, храним в env
# Render, не на фронте. Используется для записи фидбэка в таблицу feedback через REST.
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")
_jwks_client = (
    jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    if SUPABASE_URL else None
)
_bearer = HTTPBearer(auto_error=False)


def require_user(cred: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> dict:
    """FastAPI-зависимость: пускает только с валидным Supabase-JWT, иначе 401."""
    if _jwks_client is None:
        # SUPABASE_URL не задан — не тихо пускаем всех, а честно сообщаем о мисконфиге.
        raise HTTPException(status_code=500, detail="Авторизация не настроена: не задан SUPABASE_URL")
    if cred is None or not cred.credentials:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(cred.credentials).key
        return jwt.decode(
            cred.credentials,
            signing_key,
            algorithms=["ES256"],
            audience="authenticated",
            issuer=f"{SUPABASE_URL}/auth/v1",
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Недействительный или просроченный токен")


# ─── СХЕМЫ ЗАПРОСА/ОТВЕТА ─────────────────────────────────────────────────────

class Block(BaseModel):
    id: str
    text: str = ""
    page: Optional[int] = None
    bbox: Optional[list[float]] = None   # [x0, top, x1, bottom] в пунктах (для колонок)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для анализа")
    query: str = Field(..., min_length=1, description="Вопрос пользователя")
    # готовые блоки из spatial-манифеста (PDF): mind-map строится из них с block_ids.
    # Не переданы — бэкенд режет текст на блоки-абзацы (paste/txt/md).
    blocks: Optional[list[Block]] = None


class AskNodeRequest(BaseModel):
    node_title: str = Field(..., min_length=1, description="Заголовок ветки/узла")
    node_text: str = Field("", description="Текст (excerpt/full) этой ветки — контекст ответа")
    question: str = Field(..., min_length=1, description="Вопрос пользователя по этой ветке")


class FeedbackRequest(BaseModel):
    rating: int = 0
    text: str = ""


# Лимит на длину анализируемого текста (в символах). Защита от разорительных
# прогонов: огромный текст = десятки чанков = десятки вызовов Gemini (время + деньги).
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "100000"))  # ~16 тыс. слов


# ─── PDF ──────────────────────────────────────────────────────────────────────
# Лимит на размер загружаемого PDF (в байтах). Парсим PDF на бэке (надёжнее, чем
# в браузере) и отдаём фронту чистый текст, который дальше идёт в /api/analyze.
MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", str(15 * 1024 * 1024)))  # 15 МБ


# ─── ЭНДПОИНТЫ ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "lum-api"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_key_set": bool(core.GEMINI_API_KEY),
        "auth_enabled": _jwks_client is not None,
        "light_model": core.LIGHT_MODEL,
        "power_model": core.POWER_MODEL,
        "embed_model": core.EMBED_MODEL,
        "chunk_size": core.CHUNK_SIZE,
        "top_k": core.TOP_K,
        # лимиты — чтобы их можно было проверить на живом сервере
        "max_text_chars": MAX_TEXT_CHARS,
        "max_pdf_mb": MAX_PDF_BYTES // (1024 * 1024),
        "accepted_files": [".txt", ".md", ".pdf"],
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, user: dict = Depends(require_user)):
    """
    Прогоняет текст + вопрос через GraphRAG и возвращает:
      answer, schema, graph (с флагами in_answer), explanation, stats.
    Требует валидный Supabase-JWT (см. require_user).
    """
    if len(req.text) > MAX_TEXT_CHARS:
        logger.warning("analyze: текст превышает лимит (%d симв.)", len(req.text))
        raise HTTPException(
            status_code=400,
            detail=f"Текст слишком большой ({len(req.text)} симв., максимум {MAX_TEXT_CHARS}).",
        )
    blocks = [b.model_dump() for b in req.blocks] if req.blocks else None
    try:
        return core.run_pipeline(text=req.text, query=req.query, blocks=blocks)
    except core.PipelineError as e:
        # Ожидаемые ошибки пайплайна (пустой ввод, нет ключа, Gemini упал) → 400.
        # Логируем текст ошибки — он же уходит в HTTP detail, но здесь виден в логах Render.
        logger.error("analyze: PipelineError: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Непредвиденное → 500, но без утечки внутренних деталей наружу
        logger.exception("analyze: непредвиденная ошибка")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {type(e).__name__}")


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, user: dict = Depends(require_user)):
    """
    Сохраняет фидбэк (оценка 1-5 + текст) в таблицу Supabase `feedback`.

    Пишем именно в БД, а не в файл: у Render эфемерная ФС — feedback.jsonl исчезал
    при каждом редеплое/рестарте, отзывы терялись. Запись идёт с сервера через REST
    с секретным ключом (обходит RLS). Если ключ/URL не заданы (локальная разработка)
    — падаем на запись в файл, чтобы не терять фидбэк в dev.
    """
    import datetime, json

    entry = {
        "user_id": user.get("sub"),
        "rating": req.rating,
        "text": req.text,
    }

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        try:
            resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/feedback",
                headers={
                    "apikey": SUPABASE_SECRET_KEY,
                    "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=entry,
                timeout=10,
            )
            resp.raise_for_status()
            return {"status": "ok"}
        except requests.RequestException as e:
            # Тело ответа Supabase (если есть) поможет отладить схему/RLS в логах Render.
            body = getattr(e.response, "text", "") if getattr(e, "response", None) else ""
            logger.error("feedback: не удалось записать в Supabase: %s %s", e, body)
            raise HTTPException(status_code=502, detail="Не удалось сохранить отзыв.")

    # Фолбэк для локальной разработки (Supabase не настроен).
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    feedback_path = os.environ.get("FEEDBACK_FILE", "feedback.jsonl")
    try:
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error("feedback: не удалось записать в файл: %s", e)
        raise HTTPException(status_code=500, detail="Не удалось сохранить отзыв.")

    return {"status": "ok"}


@app.post("/api/ask-node")
def ask_node(req: AskNodeRequest, user: dict = Depends(require_user)):
    """
    Саб-чат по конкретной ветке: отвечает на вопрос СТРОГО по контексту одного узла
    и возвращает {title, excerpt, full} для нового узла-ответвления. Один вызов
    модели — не полный пайплайн (быстро/дёшево). Требует валидный Supabase-JWT.
    """
    try:
        node = core.answer_for_node(
            node_title=req.node_title, node_text=req.node_text, question=req.question
        )
        if not node:
            raise HTTPException(status_code=400, detail="Не удалось сформировать ответ по этой ветке.")
        return node
    except core.PipelineError as e:
        logger.error("ask_node: PipelineError: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("ask_node: непредвиденная ошибка")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка")


@app.post("/api/extract")
async def extract_pdf(file: UploadFile = File(...), user: dict = Depends(require_user)):
    """
    Принимает PDF, возвращает извлечённый текст: {text, chars, pages}.
    Парсинг на бэке надёжнее браузерного; фронт затем шлёт text в /api/analyze.
    """
    filename = (file.filename or "").lower()
    is_pdf = filename.endswith(".pdf") or file.content_type == "application/pdf"
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Ожидается PDF-файл (.pdf).")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл.")
    if len(raw) > MAX_PDF_BYTES:
        mb = MAX_PDF_BYTES // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"Файл слишком большой (макс. {mb} МБ).")

    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        # PDF с пустым паролем — пробуем открыть; иначе честно сообщаем.
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise HTTPException(status_code=400, detail="PDF защищён паролем — снимите защиту.")
        pages = len(reader.pages)
        parts = []
        for page in reader.pages:
            chunk = page.extract_text() or ""
            if chunk.strip():
                parts.append(chunk)
        text = "\n\n".join(parts).strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось разобрать PDF: {type(e).__name__}")

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Из PDF не удалось извлечь текст — вероятно, это скан без текстового слоя (нужен OCR).",
        )

    return {"text": text, "chars": len(text), "pages": pages}


# ─── ASYNC INGEST (job + polling) ─────────────────────────────────────────────
# Рендер страниц PDF синхронно в одном запросе на больших файлах упирался в таймаут
# Render. Теперь /api/ingest сразу отдаёт job_id, рендер идёт в фоновом потоке, а
# фронт поллит /api/ingest/{job_id}. ВАЖНО: стор в памяти процесса — рассчитан на
# ОДИН воркер uvicorn (по умолчанию так). Для нескольких воркеров нужен общий стор.
_INGEST_JOBS: dict = {}            # job_id → {status, manifest, error, user, ts}
_INGEST_TTL = 900                  # держим готовый результат до 15 мин


def _prune_ingest_jobs():
    now = time.time()
    for jid in [k for k, v in list(_INGEST_JOBS.items()) if now - v["ts"] > _INGEST_TTL]:
        _INGEST_JOBS.pop(jid, None)


def _run_ingest(job_id: str, raw: bytes, sink, doc_id: str, image_kind: str, bucket):
    """Фоновая сборка манифеста; результат/ошибка кладётся в _INGEST_JOBS[job_id]."""
    job = _INGEST_JOBS.get(job_id)
    if job is None:
        return
    try:
        import spatial
        manifest = spatial.build_manifest(raw, image_sink=sink)
        if not manifest["pages"]:
            job.update(status="error", error="PDF без страниц или нечитаемый.", ts=time.time())
            return
        manifest["doc_id"] = doc_id
        manifest["image_kind"] = image_kind
        manifest["bucket"] = bucket
        job.update(status="done", manifest=manifest, ts=time.time())
    except Exception as e:
        logger.exception("ingest job: не удалось построить манифест")
        job.update(status="error", error=f"Не удалось разобрать PDF: {type(e).__name__}", ts=time.time())


@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...), user: dict = Depends(require_user)):
    """
    Spatial-режим: PDF → фоновая сборка манифеста. Возвращает {job_id, status} сразу;
    фронт поллит /api/ingest/{job_id}. Импорт spatial ленивый — без зависимостей
    падает только этот эндпоинт, а не всё приложение.
    """
    filename = (file.filename or "").lower()
    is_pdf = filename.endswith(".pdf") or file.content_type == "application/pdf"
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Ожидается PDF-файл (.pdf).")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл.")
    if len(raw) > MAX_PDF_BYTES:
        mb = MAX_PDF_BYTES // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"Файл слишком большой (макс. {mb} МБ).")

    try:
        import spatial  # noqa: F401 — проверяем наличие зависимостей до запуска задачи
    except ImportError as e:
        logger.error("ingest: зависимости spatial не установлены: %s", e)
        raise HTTPException(status_code=501, detail="Spatial-режим недоступен на сервере.")

    # Персист: если Storage настроен (SPATIAL_STORAGE=1 + ключи), картинки страниц
    # льём в приватный бакет, а в манифест пишем пути; иначе — data URL как раньше.
    import storage
    doc_id = uuid.uuid4().hex
    sink = None
    image_kind = "dataurl"
    bucket = None
    if storage.is_configured():
        sink = storage.make_sink(f"{user.get('sub')}/{doc_id}")
        image_kind = "storage"
        bucket = storage.BUCKET

    _prune_ingest_jobs()
    job_id = uuid.uuid4().hex
    _INGEST_JOBS[job_id] = {"status": "processing", "manifest": None,
                            "error": None, "user": user.get("sub"), "ts": time.time()}
    threading.Thread(target=_run_ingest,
                     args=(job_id, raw, sink, doc_id, image_kind, bucket),
                     daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/ingest/{job_id}")
def ingest_status(job_id: str, user: dict = Depends(require_user)):
    """Статус фоновой задачи: {status:'processing'} | манифест (готово) | 400 (ошибка)."""
    job = _INGEST_JOBS.get(job_id)
    if not job or job["user"] != user.get("sub"):
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    if job["status"] == "processing":
        return {"status": "processing"}
    if job["status"] == "error":
        _INGEST_JOBS.pop(job_id, None)
        raise HTTPException(status_code=400, detail=job.get("error") or "Ошибка разбора PDF.")
    # done — отдаём манифест и освобождаем память
    manifest = job.get("manifest")
    _INGEST_JOBS.pop(job_id, None)
    return manifest
