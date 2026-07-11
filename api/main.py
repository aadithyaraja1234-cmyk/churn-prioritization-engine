"""FastAPI app. Every data-returning endpoint is protected by get_current_user()."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from api.auth import CurrentUser, get_current_user
from api.auth import router as auth_router
from database.db import get_db, get_tenant_scoped_query
from database.models import Customer
from src.models.anomaly import load_flagged_anomalies
from src.models.clv import load_feature_importances as load_clv_importances
from src.models.explain import get_global_importance
from src.models.prioritize import get_priority_ranking
from src.models.backtest import run_backtest as run_backtest_module
from src.models.segment import load_cluster_profiles
from src.models.survival import median_survival_by_contract

app = FastAPI(title="Churn Engine API")
app.include_router(auth_router)

ROOT = Path(__file__).resolve().parent.parent

# Classifier assets exist per-tenant. Survival/segment/anomaly/CLV were only
# ever trained for Telco (models/v1 or models/v1_enriched) - there is no
# equivalent artifact for banking, by design (see the endpoints below).
CLASSIFIER_MODEL_DIRS = {
    "telco": ROOT / "models" / "v1",
    "banking": ROOT / "models" / "banking_v1",
}
CLASSIFIER_DATA_PATHS = {
    "telco": ROOT / "data" / "raw" / "telco.csv",
    "banking": ROOT / "data" / "raw" / "bank_churn.csv",
}
TELCO_MODEL_DIR = ROOT / "models" / "v1"
TELCO_DATA_PATH = ROOT / "data" / "raw" / "telco.csv"
CLV_MODEL_DIR = ROOT / "models" / "v1_enriched"
CLV_DATA_PATH = ROOT / "data" / "raw" / "telco_enriched.csv"

NOT_TRAINED = {"available": False, "reason": "not yet trained for this tenant"}
PRIORITY_BACKTEST_NOT_VALIDATED = {
    "available": False,
    "reason": (
        "revenue-weighted prioritization was only validated for the Telco tenant; "
        "banking's revenue-proxy column (Balance) has not been backtested"
    ),
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/model/metrics")
def model_metrics(current_user: CurrentUser = Depends(get_current_user)) -> dict:
    model_dir = CLASSIFIER_MODEL_DIRS.get(current_user.tenant_id)
    metadata_path = model_dir / "metadata.json" if model_dir else None
    if metadata_path is None or not metadata_path.exists():
        return NOT_TRAINED
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@app.get("/api/model/importance")
def model_importance(current_user: CurrentUser = Depends(get_current_user)) -> Any:
    model_dir = CLASSIFIER_MODEL_DIRS.get(current_user.tenant_id)
    if model_dir is None:
        return NOT_TRAINED
    importance_df = get_global_importance(model_dir=model_dir)
    return importance_df.to_dict(orient="records")


@app.get("/api/priority")
def priority(
    strategy: str = "revenue_weighted",
    limit: int = 50,
    current_user: CurrentUser = Depends(get_current_user),
) -> Any:
    if current_user.tenant_id != "telco":
        return PRIORITY_BACKTEST_NOT_VALIDATED
    ranking = get_priority_ranking(strategy=strategy, model_dir=TELCO_MODEL_DIR, data_path=TELCO_DATA_PATH)
    return ranking.head(limit).to_dict(orient="records")


@app.get("/api/backtest")
def backtest(
    top_pct: float = 0.2,
    current_user: CurrentUser = Depends(get_current_user),
) -> Any:
    if current_user.tenant_id != "telco":
        return PRIORITY_BACKTEST_NOT_VALIDATED
    return run_backtest_module(top_pct=top_pct, model_dir=TELCO_MODEL_DIR, data_path=TELCO_DATA_PATH)


@app.get("/api/survival/segments")
def survival_segments(current_user: CurrentUser = Depends(get_current_user)) -> Any:
    if current_user.tenant_id != "telco":
        return NOT_TRAINED
    return median_survival_by_contract(model_dir=TELCO_MODEL_DIR, data_path=TELCO_DATA_PATH)


@app.get("/api/segments")
def segments(current_user: CurrentUser = Depends(get_current_user)) -> Any:
    if current_user.tenant_id != "telco":
        return NOT_TRAINED
    profiles = load_cluster_profiles(model_dir=TELCO_MODEL_DIR, data_path=TELCO_DATA_PATH)
    return profiles.to_dict(orient="records")


@app.get("/api/anomalies")
def anomalies(limit: int = 5, current_user: CurrentUser = Depends(get_current_user)) -> Any:
    if current_user.tenant_id != "telco":
        return NOT_TRAINED
    results = load_flagged_anomalies(model_dir=TELCO_MODEL_DIR, data_path=TELCO_DATA_PATH)
    flagged = results[results["anomaly_flag"] == -1].sort_values("anomaly_score").head(limit)
    return flagged.to_dict(orient="records")


@app.get("/api/clv/importance")
def clv_importance(current_user: CurrentUser = Depends(get_current_user)) -> Any:
    if current_user.tenant_id != "telco":
        return NOT_TRAINED
    importances = load_clv_importances(model_dir=CLV_MODEL_DIR)
    return [{"feature": feature, "importance": importance} for feature, importance in importances]


@app.get("/customers")
def list_customers(
    tenant_id: str | None = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    """List customers for the caller's own tenant.

    `tenant_id` is accepted as a query parameter but is never used - it
    exists only so a client attempting to request another tenant's data
    makes a well-formed request rather than hitting a validation error. The
    tenant scoping the query is always current_user.tenant_id, taken from
    the verified JWT, never from user input.
    """
    del tenant_id  # explicitly unused - tenant comes from the token only
    rows = get_tenant_scoped_query(db, Customer, tenant_id=current_user.tenant_id).all()
    return [
        {"customer_id": row.customer_id, "tenant_id": row.tenant_id, "raw_features": row.raw_features}
        for row in rows
    ]
