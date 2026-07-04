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
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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


# ─── СХЕМЫ ЗАПРОСА/ОТВЕТА ─────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для анализа")
    query: str = Field(..., min_length=1, description="Вопрос пользователя")


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
        "groq_key_set": bool(core.GROQ_API_KEY),
        "light_model": core.LIGHT_MODEL,
        "power_model": core.POWER_MODEL,
        "chunk_size": core.CHUNK_SIZE,
        "top_k": core.TOP_K,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """
    Прогоняет текст + вопрос через GraphRAG и возвращает:
      answer, schema, graph (с флагами in_answer), explanation, stats.
    """
    try:
        return core.run_pipeline(text=req.text, query=req.query)
    except core.PipelineError as e:
        # Ожидаемые ошибки пайплайна (пустой ввод, нет ключа, Groq упал) → 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Непредвиденное → 500, но без утечки внутренних деталей наружу
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {type(e).__name__}")


@app.post("/api/extract")
async def extract_pdf(file: UploadFile = File(...)):
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
