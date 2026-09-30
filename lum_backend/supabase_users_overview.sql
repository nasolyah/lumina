-- Все пользователи одной таблицей: аккаунт, тариф, использование, расходы.
-- Выполнить в Supabase → SQL Editor. Это представление (view): данные всегда
-- актуальные, новые пользователи появляются сами. Смотреть: Table Editor →
-- users_overview, или  select * from users_overview;
-- Доступ только из панели Supabase — с сайта (anon/authenticated) не читается.

create or replace view public.users_overview
with (security_invoker = on) as
select
  u.email,
  coalesce(u.raw_user_meta_data->>'full_name', u.raw_user_meta_data->>'name') as name,
  coalesce(p.plan, 'free')                         as plan,
  u.created_at                                     as registered_at,
  u.last_sign_in_at,
  coalesce(c.graphs_used, 0)                       as docs_used,        -- счётчик лимита Free (только растёт)
  coalesce(ch.chats, 0)                            as chats_now,        -- чатов сейчас (без удалённых)
  coalesce(ev.maps_total, 0)                       as maps_total,
  coalesce(ev.questions_total, 0)                  as questions_total,
  coalesce(ev.infographics_month, 0)               as infographics_this_month,
  coalesce(ev.infographics_total, 0)               as infographics_total,
  round(coalesce(ev.cost_month, 0), 4)             as cost_usd_this_month,
  round(coalesce(ev.cost_total, 0), 4)             as cost_usd_total,
  ev.last_activity,
  u.id                                             as user_id
from auth.users u
left join public.user_plans p     on p.user_id = u.id
left join public.usage_counters c on c.user_id = u.id
left join lateral (
  select count(*) as chats from public.chats where chats.user_id = u.id
) ch on true
left join lateral (
  select
    count(*) filter (where e.endpoint = 'analyze' and e.ok)                  as maps_total,
    count(*) filter (where e.endpoint in ('ask', 'ask-node'))                as questions_total,
    sum(e.images) filter (where e.endpoint = 'infographic'
                            and e.created_at >= date_trunc('month', now()))  as infographics_month,
    sum(e.images) filter (where e.endpoint = 'infographic')                  as infographics_total,
    sum(e.cost_usd) filter (where e.created_at >= date_trunc('month', now())) as cost_month,
    sum(e.cost_usd)                                                          as cost_total,
    max(e.created_at)                                                        as last_activity
  from public.usage_events e
  where e.user_id = u.id
) ev on true
order by u.created_at desc;

revoke all on public.users_overview from anon, authenticated;
