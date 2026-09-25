-- Учёт расхода Gemini по пользователям + тарифы. Выполнить в Supabase → SQL Editor.
-- Пишет только бэкенд (секретный ключ, обходит RLS); с фронта таблицы расходов не видно.

-- ── 1. Расход: одна строка на запрос к API (карта, вопрос, инфографика, OCR) ──
create table if not exists public.usage_events (
  id            bigserial primary key,
  user_id       uuid not null references auth.users(id) on delete cascade,
  endpoint      text not null,                 -- analyze | ask | ask-node | infographic | ocr
  ok            boolean not null default true,  -- false: запрос упал, но Gemini уже списал деньги
  calls         int not null default 0,         -- сколько вызовов Gemini внутри запроса
  input_tokens  int not null default 0,
  output_tokens int not null default 0,         -- включая «мышление»
  images        int not null default 0,
  cost_usd      numeric(12,6) not null default 0,
  models        text[] not null default '{}',   -- реальные версии моделей (modelVersion)
  detail        jsonb,                          -- по каждому вызову: модель, токены, цена
  created_at    timestamptz not null default now()
);
create index if not exists usage_events_user_endpoint_time
  on public.usage_events (user_id, endpoint, created_at desc);
alter table public.usage_events enable row level security;   -- политик нет: только сервер

-- ── 2. Тарифы: free (по умолчанию) | pro | max ──
-- Пока нет оплаты — тариф выдаётся вручную, например:
--   insert into public.user_plans (user_id, plan)
--   select id, 'max' from auth.users where email = 'someone@gmail.com'
--   on conflict (user_id) do update set plan = excluded.plan, updated_at = now();
create table if not exists public.user_plans (
  user_id    uuid primary key references auth.users(id) on delete cascade,
  plan       text not null default 'free' check (plan in ('free', 'pro', 'max')),
  updated_at timestamptz not null default now()
);
alter table public.user_plans enable row level security;
drop policy if exists "user_plans: read own" on public.user_plans;
create policy "user_plans: read own" on public.user_plans
  for select using (auth.uid() = user_id);

-- ── 3. Лимит документов: Pro и Max без ограничений (Free — по-прежнему 5) ──
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

  -- платные тарифы без лимита документов
  if exists (select 1 from public.user_plans
             where user_id = auth.uid() and plan in ('pro', 'max')) then
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

-- ── 4. Сводка по месяцам (смотреть в SQL Editor) ──
--   select * from public.usage_monthly order by month desc, cost_usd desc;
create or replace view public.usage_monthly
with (security_invoker = on) as
select
  date_trunc('month', e.created_at)                            as month,
  u.email,
  coalesce(p.plan, 'free')                                     as plan,
  count(*) filter (where e.endpoint = 'analyze' and e.ok)      as maps,
  count(*) filter (where e.endpoint in ('ask', 'ask-node'))    as questions,
  coalesce(sum(e.images) filter (where e.endpoint = 'infographic'), 0) as infographics,
  sum(e.input_tokens)                                          as input_tokens,
  sum(e.output_tokens)                                         as output_tokens,
  round(sum(e.cost_usd), 4)                                    as cost_usd
from public.usage_events e
join auth.users u on u.id = e.user_id
left join public.user_plans p on p.user_id = e.user_id
group by 1, 2, 3;
revoke all on public.usage_monthly from anon, authenticated;
