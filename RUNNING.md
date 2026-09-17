# Running the Churn Engine Locally

Two terminals: backend and frontend.

## Quick Start (fastest way)

Once dependencies are installed (step 1 below, one-time), just double-click these two files at `C:\CHURN_PREDICTION\`:

- **`start-backend.bat`** — starts the backend on port 8000
- **`start-frontend.bat`** — starts the frontend dev server

Each opens its own window and keeps running there — leave both open, same as the manual terminal steps below. No typing required. If you'd rather run things manually (or these windows close unexpectedly and you need to see the full command), the step-by-step instructions below cover the same startup process.

## Run with Docker

An alternative to the manual/quick-start steps below - no local Python or Node install needed, just Docker.

```
cd c:\CHURN_PREDICTION\churn-engine
docker compose up --build
```

Then open `http://localhost:5173`. The backend is at `http://localhost:8000` (its `/health` endpoint is used by both the container healthcheck and `docker.yml`'s CI job).

What this actually does, and why:

- **Two images, one for each service** (`Dockerfile` for the backend, `frontend/Dockerfile` for a static build served by nginx) - `docker-compose.yml` builds and wires both together. The frontend's Dockerfile bakes in `VITE_API_BASE_URL` at build time (Vite inlines `import.meta.env.*` into the bundle - it can't be changed at container start the way a backend env var can); it defaults to `http://localhost:8000`, which is correct for this compose setup because the *browser* talks to whatever's mapped on the host, not to the backend container's internal address.
- **Every permanently-committed reference tenant's model artifacts ship inside the backend image** (`Dockerfile`'s `COPY models/` + `COPY data/tenant_uploads/`) - Telco, Banking, and the three reference self-registered tenants (aurora-streaming, meridian-wireless, fernwood-retail-collective - see README.md's Structure section) - no separate training step needed to see real predictions immediately. A container restart or rebuild doesn't lose anything: `models/`, `data/tenant_uploads/`, and the SQLite DB are all on named volumes (`backend_models`, `backend_uploads`, `backend_data` in `docker-compose.yml`) - Docker seeds a fresh named volume from the image's own content on first creation, so the baked-in artifacts are still there on first run, and any OTHER self-registered tenant's own training output persists across restarts from then on.
- **Copilot's `GEMINI_API_KEY` is optional here too**, same as the manual setup - set it in your shell environment before running `docker compose up` (or create a `.env` file at the repo root; Compose reads it automatically) and it's passed through; leave it unset and Copilot just degrades to "unavailable" rather than the container failing to start.
- **Demo accounts aren't seeded automatically** (same as the manual setup requiring a separate `python -m database.seed_demo_users` step - see step 6 below) - run it inside the running container once, the first time:

  ```
  docker compose exec backend python -m database.seed_demo_users
  ```

- **CI builds and runs both images on every change** that touches a Dockerfile/compose/nginx config (`.github/workflows/docker.yml`) - not just a build check, it actually starts each container and polls the real endpoint (`/health` for the backend, `/` for the frontend) before passing, the same "verify it actually runs" standard the rest of this project's CI already holds itself to.

### Before a real (non-localhost) deployment

`docker compose up --build` as shown above is a zero-config local/demo setup - a few things need real values before this is exposed anywhere else. See `.env.example` for all of these:

- **`JWT_SECRET_KEY`** - without it, every JWT is signed with a publicly-known dev key (`api/auth.py` logs a loud warning at startup if it's missing). Generate one with `python -c "import secrets; print(secrets.token_hex(32))"`.
- **`CORS_ORIGINS`** - set to your deployed frontend's real URL(s) (comma-separated) so browsers don't block it; the `localhost:5173`/`5174` dev defaults stay allowed alongside it.
- **`DATABASE_URL`** / **`REDIS_URL`** - both optional. The default (SQLite + an in-process rate limiter) is genuinely fine for a single backend instance/worker. Uncomment `docker-compose.yml`'s `postgres`/`redis` services and the matching backend env vars only if you're running multiple backend workers or replicas that need to share state - and set a real `POSTGRES_PASSWORD` if you do (the compose default is a dev-only placeholder).

## 1. Prerequisites

- Python 3.13 installed (system Python — this project does not use a venv)
- Node.js + npm installed (`node --version`, `npm --version` should both work)
- Dependencies installed:

```
cd c:\CHURN_PREDICTION\churn-engine
pip install -r requirements.txt
```

```
cd c:\CHURN_PREDICTION\churn-engine\frontend
npm install
```

- One-time database setup (only needed if `database/app.db` doesn't exist yet — it's gitignored, so a fresh clone won't have it):

```
cd c:\CHURN_PREDICTION\churn-engine
python -m database.migrate_csv_to_db
```

- **Optional** — Copilot needs a Gemini API key to actually answer questions; without one it degrades to an "unavailable" message rather than crashing, so this step can be skipped if you don't need Copilot:

```
cd c:\CHURN_PREDICTION\churn-engine
copy .env.example .env
```

  Then open `.env` and set `GEMINI_API_KEY` to a free-tier key from [Google AI Studio](https://aistudio.google.com/apikey).

## 2. Terminal 1 — Backend

```
cd c:\CHURN_PREDICTION\churn-engine
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Successful output looks like:

```
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Leave this terminal open and running.

## 3. Terminal 2 — Frontend

```
cd c:\CHURN_PREDICTION\churn-engine\frontend
npm run dev
```

Successful output looks like:

```
  VITE vX.X.X  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**The port may not be 5173** — Vite picks whatever's free (5174, 5175, etc. if 5173 is taken). Use whatever URL it actually prints.

Leave this terminal open and running.

## 4. Get started: register your company

This is the real product flow — every company gets here the same way, including this project's own internal reference tenants originally. There is no pre-made demo account you're expected to use.

1. Open the URL Terminal 2 printed (e.g. `http://localhost:5173`).
2. Click **New company? Register your workspace** (or go directly to `/register`).
3. Fill in your company name, email, and password.
4. Walk through the onboarding wizard:
   - **Upload** a customer-level CSV (a few sample files to try live under `data/test_onboarding_samples/` — e.g. `company_meridian_wireless.csv`, `company_fernwood_retail.csv` — or bring your own; see that folder's own `README.md` for what each sample company is meant to demonstrate and what result to expect).
   - **Map columns** — confirm or correct the suggested role (customer ID / target / revenue / feature / ignore) for each column.
   - **Review the validation report** — structural checks must pass; a data-sufficiency warning (thin data) can still proceed with acknowledgment.
   - **Start training** — runs the real classifier (`src/models/train.py`'s `train_model()`, the same trainer Telco/Banking use) against your data in the background; poll status until it completes.
5. Once training succeeds with a sane result (ROC-AUC not suspiciously low or near-perfect — see `src/models/tenant_training.py`), your dashboard shows real predictions computed from your own data.

A freshly self-registered tenant only has the core classifier + Business Impact enabled at first — see step 7 below for why the rest of the advanced modules (backtest, survival, segments, CLV, scenario simulator, budget optimizer, Copilot, customer timeline) aren't turned on immediately.

A subset of data-read endpoints your dashboard calls (business impact, priority ranking, alerts, scenarios, survival, segments, anomalies, CLV, customer detail) also accepts `X-API-Key` auth for server-to-server use, instead of a JWT — create one at `/api/api-keys` once logged in (JWT-only to create; the key itself works standalone after that, rate-limited per key).

## 5. Browser

Open the URL Terminal 2 printed, then either register a new company (step 4) or log in if you've already registered.

## 6. Internal reference tenants (not part of the product flow)

Telco and Banking are this project's own two most rigorously validated tenants — every advanced module below is trained, tested, and enabled only for these two, and they exist as proof the full feature set and multi-tenant architecture genuinely work. They are intentionally **not** surfaced anywhere in the UI (no login hints, no demo buttons, nothing pointing you toward them) — the product's real flow is self-registration (step 4) — but the accounts still work if you log in with them directly. Useful for your own reference when you want to see the fully-validated feature set that a freshly self-registered tenant (e.g. Meridian Wireless) doesn't have yet.

| Tenant | Email | Password | Role |
|---|---|---|---|
| Telco | `demo-telco@churn-engine.local` | `DemoPass123` | admin |
| Banking | `demo-banking@churn-engine.local` | `DemoPass123` | admin |

Seeded via `database/seed_demo_users.py` (idempotent — safe to re-run, never duplicates or resets either password). If `database/app.db` is missing or these accounts aren't present yet (e.g. fresh clone):

```
cd c:\CHURN_PREDICTION\churn-engine
python -m database.seed_demo_users
```

Banking has a real trained model and passes its own full backend test suite (`tests/test_multi_tenant.py`, `tests/test_tenant_isolation.py`, etc.) — kept as validated evidence that the multi-tenant architecture works, with every advanced-module flag left `false` on purpose (see `docs/ADDING_A_TENANT.md` for why, and what a tenant needs before a module is safe to enable). Logging in as `demo-banking@churn-engine.local` is a useful way to directly verify tenant isolation and the `{"available": false, "reason": ...}` gating behavior.

**Optional — create an additional/custom user against Telco or Banking** (for internal testing only; a real user should register their own company via step 4 instead):

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/auth/register" -Method Post -ContentType "application/json" -Body (@{
    email    = "REPLACE_ME@example.com"
    password = "REPLACE_ME_password"
    tenant_id = "telco"
    role     = "analyst"
} | ConvertTo-Json)
```

## 7. Note on tenants and feature availability

Which advanced modules (priority ranking, backtest, survival, segments, anomalies, CLV, scenario simulator, budget optimizer, customer timeline) are enabled for a given tenant is config-driven, not hardcoded - see `config/config.yaml`'s `tenants.<name>.feature_flags` and `src/tenant_registry.py`. Telco has every module enabled. Banking exists only as backend validation evidence (a real trained classifier + a full tenant-isolation test suite) with every advanced-module flag left `false` - logging in as `demo-banking@churn-engine.local` will show `{"available": false, "reason": ...}` on all of them, by design, not as a bug. A self-registered tenant starts with only the core classifier + `business_impact_core` enabled once training succeeds - the other modules each need their own trained artifact and validation pass before being safe to enable (see `docs/ADDING_A_TENANT.md` for the checklist), and are never auto-enabled by training alone.

## 8. Troubleshooting

**Both terminals (backend + frontend) must stay open for the whole session.** Closing either one breaks the app.

- **Browser shows `ERR_CONNECTION_REFUSED`** — one of the two terminals died (crashed or was closed). Restart it using the command from step 2 or 3.
- **Browser console shows a CORS error** — the frontend's actual port isn't in the backend's allowed origins list. Open `api/main.py`, find the `CORSMiddleware` `allow_origins` list, add `http://localhost:<port>` for whatever port Terminal 2 is actually using, save, then restart Terminal 1 (Ctrl+C, re-run the uvicorn command).
- **Login fails / "Incorrect email or password"** — if you registered your own company (step 4), double-check the email/password you used at registration. If you're trying one of the internal reference tenants (step 6), double-check you're using the credentials exactly as listed there; if the account doesn't exist yet, run `python -m database.seed_demo_users`.
- **Terminal prompt shows `(.venv)` but starting the backend gives `No module named uvicorn`** (or any other missing module) — the venv is activated but nothing was installed into it. This project's dependencies are installed via system pip, not a venv (see Prerequisites). Fix: run `deactivate`, then re-run the normal startup commands using system Python.
- **Backend fails to start with `WinError 10048: only one usage of each socket address is normally permitted`** — a previous uvicorn process is still running in the background on port 8000 (commonly from a terminal that was closed directly instead of stopped with Ctrl+C first). Fix:

  ```
  netstat -ano | findstr :8000
  ```

  Find the PID in the last column of the `LISTENING` line, then:

  ```
  taskkill /PID <that_number> /F
  ```

  (replace `<that_number>` with the actual PID shown — do not leave the placeholder text in the command). Then retry the uvicorn start command.
- **General reminder:** always stop a running server with Ctrl+C in its own terminal before closing that terminal window, rather than closing the window directly — this prevents orphaned background processes from holding onto ports.
