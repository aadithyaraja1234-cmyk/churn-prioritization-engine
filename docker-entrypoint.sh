#!/bin/sh
set -e

# Respects DATABASE_PATH (see database/db.py) so this check looks at the
# same file the app itself will actually open - defaults to the pre-
# existing in-tree path when unset, same as before.
DB_FILE="${DATABASE_PATH:-database/app.db}"

if [ ! -f "$DB_FILE" ]; then
    echo "$DB_FILE not found - running migration..."
    mkdir -p "$(dirname "$DB_FILE")"
    python -m database.migrate_csv_to_db
else
    echo "$DB_FILE already exists - skipping migration."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
