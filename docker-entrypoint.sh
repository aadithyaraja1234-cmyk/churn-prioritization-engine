#!/bin/sh
set -e

# SQLite won't create a missing parent directory on its own - only matters
# when DATABASE_PATH points at a fresh mounted volume (docker-compose.yml's
# `backend_data` volume) whose subpath doesn't exist yet; a no-op otherwise
# (database/'s own directory always exists), and irrelevant once
# DATABASE_URL (Postgres/Supabase) is set, since SQLite is never touched
# then at all.
mkdir -p "$(dirname "${DATABASE_PATH:-database/app.db}")"

# Unconditional, not gated on "does a local SQLite file exist yet" (the
# previous check) - that check stops meaning anything the instant
# DATABASE_URL (Postgres/Supabase) is set, since a fresh container never
# has a local file regardless of whether the REAL database already has
# data, which would have re-run this - and duplicated Telco/Banking's
# rows - on every single container restart. Safe to call every time now:
# migrate_customers()/migrate_predictions() (database/migrate_csv_to_db.py)
# check for existing rows per-tenant themselves and skip if already
# present, for either backend.
python -m database.migrate_csv_to_db

# Render (and most real hosts) assign the port dynamically via $PORT and
# route traffic to whatever the app actually binds - a hardcoded 8000
# would silently fail health checks/routing there. Falls back to 8000 for
# local `docker run`/docker-compose, where nothing sets $PORT.
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
