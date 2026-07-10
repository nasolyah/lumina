"""
Lum / Lumina — FastAPI-обёртка над GraphRAG пайплайном.

Запуск локально:
    uvicorn main:app --reload --port 8000

Эндпоинты:
    GET  /                — health-check (жив ли сервис)
    GET  /api/health      — детальная проверка (задан ли ключ, какие модели)
    POST /api/analyze     — полный прогон пайплайна по тексту + запросу
    POST /api/extract     — извлечь текст из загруженного PDF (парсинг на бэке)
"""

import io
import os
from typing import Optional
import jwt
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для анализа")
    query: str = Field(..., min_length=1, description="Вопрос пользователя")


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
        raise HTTPException(
            status_code=400,
            detail=f"Текст слишком большой ({len(req.text)} симв., максимум {MAX_TEXT_CHARS}).",
        )
    try:
        return core.run_pipeline(text=req.text, query=req.query)
    except core.PipelineError as e:
        # Ожидаемые ошибки пайплайна (пустой ввод, нет ключа, Gemini упал) → 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Непредвиденное → 500, но без утечки внутренних деталей наружу
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {type(e).__name__}")


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
