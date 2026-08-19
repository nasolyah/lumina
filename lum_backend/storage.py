"""
Персист картинок страниц в Supabase Storage (приватный бакет).

Зачем: манифест с data URL весит мегабайты — в строку таблицы `chats` такое класть
нельзя (см. docs/spatial-mvp-deferred.md). Кладём WebP-страницы в Storage, а в манифест
пишем ПУТЬ; фронт подписывает URL через supabase-js (RLS: владелец своей папки).

Приватность: бакет приватный, путь начинается с user_id → чужой доступ закрыт RLS.
Egress: заголовок cache-control делает картинки кэшируемыми (повторные просмотры не
жгут трафик).

Включается явным флагом SPATIAL_STORAGE=1 — пока выключено, spatial работает как раньше
(data URL). Загрузка идёт секретным ключом (обходит RLS), тем же, что и фидбэк.
"""
import logging
import os

import requests

logger = logging.getLogger("lumina")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")
BUCKET = os.environ.get("SPATIAL_BUCKET", "documents")
# год иммутабельного кэша: имена файлов уникальны (doc_id), перезапись не нужна
_CACHE_CONTROL = "max-age=31536000, immutable"


def is_configured() -> bool:
    """Storage используем только при явном флаге и наличии ключей."""
    return bool(
        SUPABASE_URL and SUPABASE_SECRET_KEY
        and os.environ.get("SPATIAL_STORAGE") == "1"
    )


def _upload_webp(path: str, data: bytes) -> None:
    """Заливает WebP по пути внутри бакета (секретный ключ обходит RLS)."""
    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}",
        headers={
            "apikey": SUPABASE_SECRET_KEY,
            "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
            "Content-Type": "image/webp",
            "Cache-Control": _CACHE_CONTROL,
            "x-upsert": "true",
        },
        data=data,
        timeout=20,
    )
    resp.raise_for_status()


def make_sink(prefix: str):
    """
    Возвращает image_sink для spatial.build_manifest: заливает страницу в Storage
    и пишет в манифест ПУТЬ `{prefix}/p{index}.webp` (не URL — фронт подпишет сам).
    prefix обычно = f"{user_id}/{doc_id}".
    """
    def sink(page_index: int, webp: bytes) -> str:
        path = f"{prefix}/p{page_index}.webp"
        _upload_webp(path, webp)
        return path
    return sink
