-- Таблица фидбэка. Выполнить один раз в Supabase → SQL Editor.
-- Бэкенд пишет сюда через REST с секретным ключом Supabase (sb_secret_…, обходит RLS),
-- см. submit_feedback в main.py.

create table if not exists public.feedback (
  id         bigint generated always as identity primary key,
  user_id    uuid,
  rating     smallint,
  text       text,
  created_at timestamptz not null default now()
);

-- Включаем RLS и НЕ добавляем политик для anon/authenticated:
-- писать может только сервер по секретному ключу (он RLS игнорирует).
-- Так фронт с publishable-ключом не сможет ни читать, ни подделывать отзывы.
alter table public.feedback enable row level security;
