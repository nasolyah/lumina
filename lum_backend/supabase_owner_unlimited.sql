-- Безлимит для владельца. Выполнить в Supabase → SQL Editor.
-- Добавляет в триггер check_chat_limit ранний выход для email из списка владельцев
-- (совпадает с OWNER_EMAILS на фронте в app.html). SECURITY DEFINER позволяет
-- функции читать auth.users. Остальная логика лимита не тронута.

CREATE OR REPLACE FUNCTION public.check_chat_limit()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
declare
  used int;
begin
  -- владелец без лимита
  if (select email from auth.users where id = auth.uid()) = 'markingmark33@gmail.com' then
    return new;
  end if;

  insert into public.usage_counters (user_id, graphs_used)
    values (auth.uid(), 0)
    on conflict (user_id) do nothing;

  select graphs_used into used from public.usage_counters where user_id = auth.uid();

  if used >= 5 then
    raise exception 'CHAT_LIMIT_REACHED';
  end if;

  update public.usage_counters
     set graphs_used = graphs_used + 1
   where user_id = auth.uid();

  return new;
end;
$function$;
