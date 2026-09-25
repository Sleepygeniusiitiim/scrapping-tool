-- =====================================================================
-- Candidate Sourcing Agent — Supabase schema
-- Run this once in Supabase Dashboard → SQL Editor.
-- =====================================================================

-- Track scraped URLs to guarantee deduplication
CREATE TABLE IF NOT EXISTS scraped_urls (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    domain TEXT,
    wave_tag TEXT,
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Store structured candidate records
-- NOTE: CURRENT_ROLE is a reserved keyword in PostgreSQL, so the column name
-- must be double-quoted here. The column is still called current_role and is
-- read/written as `current_role` through the Supabase API.
CREATE TABLE IF NOT EXISTS candidates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT,
    "current_role" TEXT,
    skills TEXT[],
    current_location TEXT,
    target_countries TEXT[],
    evidence_snippet TEXT,
    source_url TEXT UNIQUE NOT NULL,
    platform TEXT,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- (The UNIQUE constraints already create indexes; these are kept as in the
--  original spec and are harmless no-ops if Postgres reuses them.)
CREATE INDEX IF NOT EXISTS idx_scraped_urls_url ON scraped_urls (url);
CREATE INDEX IF NOT EXISTS idx_candidates_source_url ON candidates (source_url);

-- ---------------------------------------------------------------------
-- OPTIONAL — only if you use the *anon* key and Row Level Security (RLS)
-- is enabled on these tables. With the service_role key, skip this block.
-- These policies let the anon key read/write both tables, so keep the anon
-- key private (do not ship it in a public front-end).
-- ---------------------------------------------------------------------
-- ALTER TABLE scraped_urls ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE candidates   ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "agent_all_scraped_urls" ON scraped_urls FOR ALL TO anon USING (true) WITH CHECK (true);
-- CREATE POLICY "agent_all_candidates"   ON candidates   FOR ALL TO anon USING (true) WITH CHECK (true);
