# Single-stage build: every dependency here (pandas, scikit-learn, fastapi,
# lifelines, bcrypt, etc.) ships as a prebuilt wheel, so there's no compile
# step to separate into a builder stage - multi-stage wouldn't shrink this.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY database/ ./database/
COPY config/ ./config/
COPY models/ ./models/

# Only the two CSVs actually read at runtime (by migrate_csv_to_db.py and the
# API's Telco endpoints) - not the whole data/ directory, which also holds
# unrelated large datasets and training-only file variants.
COPY data/raw/telco.csv data/raw/bank_churn.csv ./data/raw/

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
