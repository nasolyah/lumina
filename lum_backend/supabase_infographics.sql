-- Персист инфографик между сессиями. Выполнить в Supabase → SQL Editor.
--
-- Картинки инфографик лежат в приватном бакете documents (тот же, что для страниц
-- spatial-манифеста — см. supabase_storage.sql; его нужно выполнить, если ещё не).
-- Путь объекта: {user_id}/infographics/{chat_id|session}/{hash(node_id)}.png
-- (первая папка = владелец, RLS из supabase_storage.sql уже это покрывает).
--
-- В строке чата храним ТОЛЬКО лёгкие метаданные (путь + title/points/model), не сами
-- картинки. Формат: { "<node_id>": { "path": "...", "title": "...", "points": [...],
-- "model": "..." }, ... }.

alter table public.chats add column if not exists infographics jsonb;
