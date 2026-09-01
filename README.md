# Churn Prioritization Engine

[![Backend Tests](https://github.com/aadithyaraja1234-cmyk/churn-prioritization-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/aadithyaraja1234-cmyk/churn-prioritization-engine/actions/workflows/tests.yml)
[![Frontend](https://github.com/aadithyaraja1234-cmyk/churn-prioritization-engine/actions/workflows/frontend.yml/badge.svg)](https://github.com/aadithyaraja1234-cmyk/churn-prioritization-engine/actions/workflows/frontend.yml)
[![Docker](https://github.com/aadithyaraja1234-cmyk/churn-prioritization-engine/actions/workflows/docker.yml/badge.svg)](https://github.com/aadithyaraja1234-cmyk/churn-prioritization-engine/actions/workflows/docker.yml)

A multi-tenant churn-prediction and retention-decisioning platform: it scores each customer's churn probability, translates that into a dollar figure (revenue at risk / recoverable revenue), ranks who to act on first, and lets you simulate "what if we changed X" before spending a retention budget for real. Every number the API returns is either a real, trained model's output or an explicitly labeled assumption - never fabricated.

To run it locally, see **[RUNNING.md](RUNNING.md)** (or **[RUNNING.md's Docker section](RUNNING.md#run-with-docker)** to run both services in containers with one command). To onboard a new tenant beyond self-service registration, see **[docs/ADDING_A_TENANT.md](docs/ADDING_A_TENANT.md)**.

## The two kinds of tenant

This matters more than anything else here for understanding what you're looking at:

- **Internal reference tenants (Telco, Banking)** - this project's own most rigorously validated tenants. Every `feature_flags` module (priority ranking, backtest, survival analysis, segments, anomaly detection, CLV, scenario simulator, budget optimizer, customer timeline) is trained, tested, and enabled for Telco specifically; Banking exists as a second real trained tenant validating the multi-tenant architecture itself, with every advanced-module flag deliberately left off. Neither is exposed anywhere in the UI (no demo hints on the login page) - see RUNNING.md's "Internal reference tenants" section for their credentials.
- **Self-registered companies** (register via the UI, e.g. Meridian Wireless) - a real tenant that uploads its own data, validates it, and trains the real classifier (the same `train_model()` Telco/Banking use) through the actual onboarding wizard. It starts with only the core classifier + Business Impact enabled - the other modules each need their own trained artifact and a real validation pass before they're safe to turn on (see `docs/ADDING_A_TENANT.md`), and training alone never auto-enables them. Endpoints for a not-yet-validated module return `{"available": false, "reason": "..."}` rather than a guess or a crash.

**The Copilot is a deliberate exception to this gating, not an oversight.** It carries no `feature_flags` check of its own - it's available to every tenant, including a freshly self-registered one. Its answers are automatically limited by which underlying tools have real data for that tenant, rather than being blocked at the feature-flag level: each of its 8 tools independently calls `feature_enabled()` and degrades to the same `{"available": False, "reason": ...}` shape the REST endpoints use (see `src/copilot/tools.py`'s module docstring), so a tenant with only the core classifier enabled just gets fewer real answers - correctly reflected in its own sources/refusal behavior - rather than the whole assistant being switched off. Verified per-tool against Banking in `tests/test_copilot_tools.py` (`test_*_not_trained_for_banking`).

## Structure

- `api/` - FastAPI backend: auth, company registration, self-service onboarding, async training jobs, API keys, and every scoring/business-impact/analytics endpoint.
- `frontend/` - React (Vite) UI: the dashboard, onboarding wizard, analytics/executive views, Copilot chat, admin/model-internals view. `frontend/scripts/` holds live browser-driven verification scripts (Puppeteer) - see their own headers before running one.
- `database/` - SQLAlchemy models and the tenant-isolation enforcement point (`get_tenant_scoped_query`), plus one-off operator scripts (`seed_demo_users.py`) that are deliberately not API endpoints.
- `src/data/`, `src/features/` - data loading/cleaning/splitting and feature encoding, shared by every tenant's training run.
- `src/models/` - the classifier trainer (`train.py`) and every analytical module built on top of it (business impact, recommend, alerts, backtest, survival, segments, anomaly, CLV, scenario, budget optimizer, health score, whatif). CLV (`clv.py`) has two independent paths, never both enabled for the same tenant: a real trained regression (`train_clv_model()`) against a genuine per-customer CLV column, or - for a tenant with no such column at all - a formula-based CLV *estimate* (`estimate_clv_bulk()`: revenue x expected total lifetime, survival-informed when available), clearly labeled as a proxy rather than a measured value and gated on its own structural/variance sanity check instead of an R² score.
- `src/onboarding/`, `src/tenant_registry.py` - self-service upload/validation/column-mapping logic, and the single source of truth for which tenant has which feature enabled.
- `src/copilot/` - the Gemini-backed AI assistant; requires `GEMINI_API_KEY` (see RUNNING.md's Prerequisites) and degrades to "unavailable" without one, rather than crashing.
- `config/config.yaml` - tunable training parameters plus Telco/Banking's static tenant profiles and feature flags.
- `data/raw/` - immutable input data for Telco/Banking; `data/test_onboarding_samples/` - synthetic sample CSVs (with their own README) showing the onboarding pipeline's full range (clean, thin, confusingly-named, quietly-degraded, malicious); `data/live_demo/` and `data/live_demo_v2/` - two separate libraries of clean, industry-varied synthetic companies (each with its own README) purpose-built for live, in-person manual demonstration - distinct from the pipeline-range and reference-data folders above. `v2` uses deliberately varied column-naming conventions (not Telco-shaped) to exercise the onboarding suggester's real generalization, rather than duplicating `v1`'s companies.
- `models/v1/`, `models/banking_v1/` - Telco/Banking's versioned, committed model artifacts. Self-registered tenants' artifacts land under `models/{tenant_id}/`, gitignored (regenerated by training, not source of truth).
- `tests/` - the test suite; `scripts/generate_onboarding_samples.py` - regenerates the synthetic onboarding sample CSVs deterministically; `scripts/generate_live_demo_batch.py` - regenerates the six live-demo company CSVs deterministically.
- `docs/ADDING_A_TENANT.md` - the checklist for validating and enabling an advanced module for any tenant.

## Tests

```bash
pytest -q
```

`tests/test_copilot_guardrails.py` makes real calls to the Gemini API and needs a real `GEMINI_API_KEY` in `.env` - it skips itself automatically (doesn't fail) if the key is missing.

## Regenerating the Telco reference model

```bash
python run_pipeline.py --output-dir models/v1
```

`--output-dir` has no default on purpose - `train_model()` itself defaults to `models/v1` when omitted, and this script is the one place a careless run could silently overwrite the live, validated reference model in place. This is a maintenance operation (retrains from `data/raw/telco.csv` with the current `config/config.yaml`), not part of the normal run flow - see RUNNING.md for actually running the app.

## Known issues & project history

- **The frozen-baseline staleness incident (found and fixed this session).**
  `src/data/split.py`'s train/test split was fixed at some point to be
  customerID-based and deterministic instead of depending on physical row
  order - a real, necessary correctness fix (see
  `docs/ADDING_A_TENANT.md`'s "split-determinism lesson"). What didn't
  happen at the same time was regenerating `models/v1` under the new split
  algorithm. The result: for roughly a month, every number this project
  published for Telco - ROC-AUC/PR-AUC (0.8471/0.6687), the backtest lift
  curve (57.04%/52.38%/23.09% at 20%), the priority-ranking Spearman
  correlation (~0.949) - was quietly computed from a model artifact that no
  longer matched what `split_data()` would actually produce from
  `telco.csv`. Nothing crashed and no test caught it, because nothing ever
  re-checked the deployed artifact against a fresh split computation.
  It surfaced during an unrelated hyperparameter-tuning investigation, was
  traced to this exact staleness, and was fixed by regenerating `models/v1`
  (`run_pipeline.py --output-dir models/v1`) under the current split
  algorithm. Telco's corrected, permanent baseline is **ROC-AUC 0.8422 /
  PR-AUC 0.6607**, backtest lift **56.97%/52.50%/21.24% at 20%**, and
  Spearman correlation **0.9463** - all close to, but not identical to, the
  stale figures, since the split itself changed. This is now permanently
  guarded, not just fixed once: `tests/test_multi_tenant.py`'s
  `test_v1_split_indices_match_a_fresh_split_data_call` re-derives the
  split from `telco.csv` on every test run and fails loudly if
  `models/v1/split_indices.json` ever drifts from it again, and
  `test_v1_model_metadata_matches_the_documented_baseline` pins the exact
  corrected numbers above. We're documenting this rather than quietly
  scrubbing the old numbers from history: a frozen reference artifact
  silently outliving the code change that invalidated it is a real, useful
  lesson about this class of bug, not something to hide.
