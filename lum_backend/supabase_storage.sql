-- Приватный бакет для картинок страниц spatial-режима. Выполнить в Supabase → SQL Editor.
-- Путь объекта: {user_id}/{doc_id}/p{index}.webp — первая папка = владелец (для RLS).
-- Бэкенд заливает секретным ключом (обходит RLS); фронт по сессии владельца
-- создаёт signed URL для показа (нужна политика SELECT ниже).

insert into storage.buckets (id, name, public)
values ('documents', 'documents', false)
on conflict (id) do nothing;

-- RLS на storage.objects включён по умолчанию. Даём владельцу доступ ТОЛЬКО к своей
-- папке (первый сегмент пути = его auth.uid()). Чужие документы недоступны.

create policy "documents: owner read"
  on storage.objects for select to authenticated
  using (bucket_id = 'documents' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "documents: owner insert"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'documents' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "documents: owner update"
  on storage.objects for update to authenticated
  using (bucket_id = 'documents' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "documents: owner delete"
  on storage.objects for delete to authenticated
  using (bucket_id = 'documents' and (storage.foldername(name))[1] = auth.uid()::text);

-- Колонка для персиста spatial-манифеста (лёгкий: пути в бакете + блоки, не картинки).
-- Нужна, чтобы «Документ» открывался при повторном заходе в чат.
alter table public.chats add column if not exists manifest jsonb;
