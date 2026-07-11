#!/bin/sh
set -e

if [ ! -f "database/app.db" ]; then
    echo "database/app.db not found - running migration..."
    python -m database.migrate_csv_to_db
else
    echo "database/app.db already exists - skipping migration."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
