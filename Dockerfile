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

# Only the CSVs actually read at runtime - not the whole data/ directory,
# which also holds unrelated large datasets, training-only file variants,
# and gitignored self-registered-tenant uploads:
#   - telco.csv / bank_churn.csv: migrate_csv_to_db.py + the classifier
#     endpoints for Telco/Banking.
#   - telco_enriched.csv: business_impact.py's compute_business_impact_bulk()
#     reads Telco's real clv_data_path directly (config.yaml's
#     tenants.telco.clv_data_path) for the CLV-percentile weighting term -
#     every /api/business-impact-derived response for Telco needs this file
#     present, not just CLV training.
#   - hillstrom.csv: intervention_benchmark.py's compute_hillstrom_benchmark(),
#     called by business_impact_metadata() on every business-impact-derived
#     response (see that module's docstring) - without this file present,
#     every one of those endpoints 500s in a fresh container.
COPY data/raw/telco.csv data/raw/bank_churn.csv data/raw/telco_enriched.csv data/raw/hillstrom.csv ./data/raw/

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
