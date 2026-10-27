-- C integration: additive tables, independent of existing OCR-Embedding schema.
create schema if not exists extensions;
create extension if not exists vector with schema extensions;
create table if not exists public.team_ocr_documents (
  owner_id uuid not null references auth.users(id), dataset text not null,
  doc_id text not null, profile text not null, content_hash text not null,
  card_no integer not null, title text not null, text text not null,
  provenance jsonb not null, created_at timestamptz not null default now(),
  primary key(owner_id,dataset,doc_id)
);
create table if not exists public.team_ocr_chunks (
  owner_id uuid not null, dataset text not null, doc_id text not null,
  position integer not null, text text not null, embedding extensions.vector(1024) not null,
  primary key(owner_id,dataset,doc_id,position),
  foreign key(owner_id,dataset,doc_id) references public.team_ocr_documents(owner_id,dataset,doc_id)
);
alter table public.team_ocr_documents enable row level security;
alter table public.team_ocr_chunks enable row level security;
drop policy if exists team_ocr_owner_read on public.team_ocr_documents;
create policy team_ocr_owner_read on public.team_ocr_documents for select to authenticated using(owner_id=auth.uid());
drop policy if exists team_ocr_owner_read on public.team_ocr_chunks;
create policy team_ocr_owner_read on public.team_ocr_chunks for select to authenticated using(owner_id=auth.uid());
revoke all on public.team_ocr_documents, public.team_ocr_chunks from anon,authenticated;
grant select on public.team_ocr_documents,public.team_ocr_chunks to authenticated;
grant all on public.team_ocr_documents,public.team_ocr_chunks to service_role;

create or replace function public.team_ocr_ingest(payload jsonb) returns text
language plpgsql security invoker set search_path=public,extensions,pg_temp as $$
declare
  owner uuid := (payload->>'owner_id')::uuid;
  ds text := payload->>'dataset';
  ident text := payload->>'doc_id';
  prof text := payload->>'profile';
  chunk jsonb; vec extensions.vector(1024); previous text; previous_profile text; previous_hash text; previous_title text; previous_chunk text;
  existing_vec extensions.vector(1024); total integer; positions integer;
begin
  if ds is null or length(ds) not between 1 and 100 or ident !~ '^[a-f0-9]{64}$'
    or prof !~ '^[a-f0-9]{64}$' or payload->>'content_hash' !~ '^[a-f0-9]{64}$'
    or payload->>'text' is null or length(payload->>'text') not between 1 and 50000
    or jsonb_typeof(payload->'chunks') is distinct from 'array'
    or jsonb_array_length(payload->'chunks') not between 1 and 200 then
    raise exception 'invalid document payload';
  end if;
  -- Serializes idempotent retries of this immutable document.
  perform pg_advisory_xact_lock(hashtextextended(owner::text||':'||ds||':'||ident,0));
  select text,profile,content_hash,title into previous,previous_profile,previous_hash,previous_title from public.team_ocr_documents
    where owner_id=owner and dataset=ds and doc_id=ident;
  if found and (previous is distinct from payload->>'text' or previous_profile is distinct from prof
    or previous_hash is distinct from payload->>'content_hash' or previous_title is distinct from payload->>'title') then raise exception 'immutable document conflict'; end if;
  insert into public.team_ocr_documents(owner_id,dataset,doc_id,profile,content_hash,card_no,title,text,provenance)
    values(owner,ds,ident,prof,payload->>'content_hash',(payload->>'card_no')::integer,
           payload->>'title',payload->>'text',coalesce(payload->'provenance','{}'::jsonb))
    on conflict do nothing;
  select count(*),count(distinct (item->>'position')::integer) into total,positions
    from jsonb_array_elements(payload->'chunks') item;
  if total<>positions then raise exception 'duplicate chunk positions'; end if;
  for chunk in select * from jsonb_array_elements(payload->'chunks') loop
    if jsonb_typeof(chunk->'embedding') is distinct from 'array'
      or jsonb_array_length(chunk->'embedding')<>1024
      or (chunk->>'position')::integer not between 0 and total-1
      or chunk->>'text' is null or length(chunk->>'text') not between 1 and 50000 then
      raise exception 'invalid chunk';
    end if;
    vec:=(chunk->'embedding')::text::extensions.vector(1024);
    if abs((vec <#> vec)+1)>.004 then raise exception 'vector must have unit norm'; end if;
    select embedding,text into existing_vec,previous_chunk from public.team_ocr_chunks
      where owner_id=owner and dataset=ds and doc_id=ident and position=(chunk->>'position')::integer;
    if found and ((existing_vec <-> vec)>.0001 or previous_chunk is distinct from chunk->>'text') then raise exception 'immutable vector conflict'; end if;
    insert into public.team_ocr_chunks values(owner,ds,ident,(chunk->>'position')::integer,chunk->>'text',vec)
      on conflict do nothing;
  end loop;
  if (select count(*) from public.team_ocr_chunks where owner_id=owner and dataset=ds and doc_id=ident)<>total then
    raise exception 'chunk count conflict';
  end if;
  return ident;
end $$;
revoke all on function public.team_ocr_ingest(jsonb) from public,anon,authenticated;
grant execute on function public.team_ocr_ingest(jsonb) to service_role;
