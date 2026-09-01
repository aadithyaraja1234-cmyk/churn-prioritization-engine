# Adding a New Tenant

This is a real, followable checklist, not a template — it's written from
what actually happened when this project added its second tenant
(Banking) and, separately, what it took to get each advanced module
production-honest for Telco. Two concrete lessons from that history show
up repeatedly below:

- **The recoverable-revenue unit-mismatch lesson.** `business_impact.py`'s
  `recoverable_revenue` originally used one month of revenue per
  at-risk customer instead of their full expected remaining tenure — a
  bug that looked plausible, passed review, and was only caught during
  Budget Optimizer testing. Fixing it touched `business_impact.py`,
  `budget_optimizer.py`, and `scenario.py`, and required recomputing the
  recoverable-revenue ceiling and re-picking the default
  `cost_per_intervention` from scratch. **The lesson: a revenue formula
  validated for one tenant's revenue column does not automatically mean
  anything for another tenant's revenue column** — see step 5 below.
- **The split-determinism lesson.** `src/data/split.py`'s train/test
  split used to depend on physical row order in the CSV, not on which
  records exist. It was fixed to group and shuffle by `id_column` in
  sorted order instead, so the split is reproducible regardless of row
  order. **The lesson: don't assume a new tenant's CSV behaves like
  Telco's just because the code "already handles multiple tenants"** —
  re-verify against the new file's actual shape.
- **The stop-at-positive-ROI semantics lesson.** `optimize_budget()`'s
  `stop_at_positive_roi` mode was initially implemented as "stop at the
  first customer who fails the cost-effectiveness test," which meant a
  single low-value customer near the top of the ranking could zero out
  the entire result. The fix ("skip failures, keep selecting while
  budget remains") only shipped after a synthetic-data regression test
  was written specifically to catch that class of bug. **The lesson:
  logic that looks obviously correct on the first read still needs a
  synthetic edge-case test, not just a real-data spot check**, before you
  trust it for a new tenant.

The theme: **feature_flags is a validation gate, not a formality.**
Flipping a flag to `true` is the last step, not the first.

---

## 1. Prepare and validate the data file

Place the raw file at `data/raw/<tenant>.csv` and check, by hand, before
writing any config:

- **`id_column`** — exists, and every value is unique and non-null. If
  IDs repeat or are missing, the split logic and every per-customer
  endpoint will silently misbehave.
- **`target_column`** — exists, and is genuinely binary (or has an
  unambiguous "positive" value you can name as `target_positive_value`).
- **`revenue_column`** — exists, and you have personally confirmed **what
  kind of quantity it is**. This is the single most important check in
  this whole document:
  - A **recurring, periodic charge** (like Telco's `MonthlyCharges`) is
    what `recoverable_revenue`'s formula (`expected_remaining_tenure_months
    × revenue_column`) was built for.
  - A **lump-sum or point-in-time value** (like Banking's `Balance`) is a
    *different kind of quantity* — multiplying it by a tenure estimate
    produces a number that looks plausible but means nothing. This is
    exactly why Banking's `business_impact_core` flag is `false`: nobody
    has done the work of deciding what "recoverable revenue" should even
    mean when the revenue column isn't a recurring charge.
  - If your new tenant's revenue column is a recurring charge, you're in
    good shape. If it isn't, `business_impact_core` and everything built
    on it (budget optimizer, alerts, health score) needs a rethought
    formula before it can honestly be enabled — don't just flip the flag
    because the code will run without erroring.
- **A tenure-equivalent numeric column** (Telco has `tenure` in months).
  Needed for `survival` and for `customer_timeline`'s inferred signup
  date.
- **A contract-equivalent categorical column with a handful of distinct
  values** (Telco has `Contract`: month-to-month / one year / two year).
  Needed for `survival`'s per-segment median survival time, `segments`'
  cluster labeling, and the `contract_migration` scenario type. If your
  tenant has no such column, those three features may never be portable
  without inventing one.

## 2. Add the tenant_profiles entry

Everything lives in `config/config.yaml`'s `tenants` section — this is
the single source of truth `src/tenant_registry.py` reads from. Start
with every `feature_flags` entry `false`:

```yaml
tenants:
  insurance:
    data_path: data/raw/insurance_churn.csv
    target_column: Cancelled
    target_positive_value: "Yes"
    revenue_column: AnnualPremium   # confirm: recurring charge, not a balance/payout
    id_column: PolicyID
    model_dir: models/insurance_v1
    clv_model_dir: null            # only set once a CLV-enriched dataset + model exist
    clv_data_path: null
    unavailable_reason: "not yet trained for this tenant"
    feature_flags:
      business_impact_core: false
      priority_ranking: false
      backtest: false
      survival: false
      segments: false
      anomalies: false
      clv: false
      scenario_simulator: false
      budget_optimizer: false
      customer_timeline: false
```

`unavailable_reasons` (plural, optional) lets you override the default
reason text for one specific feature, the way Banking's does for
`priority_ranking`/`backtest` — use this when the generic reason isn't
informative enough to explain *why* (e.g. "revenue column isn't a
recurring charge" is more useful than "not yet trained").

## 3. Train and validate the base classifier

```
python -c "from pathlib import Path; from src.models.train import train_model; \
from src.config import load_config; c = load_config('config/config.yaml'); \
tc = c['tenants']['insurance']; \
train_model(Path(tc['data_path']), 'config/config.yaml', output_dir=Path(tc['model_dir']), tenant_config=tc)"
```

**Always pass `output_dir` explicitly, matching the `model_dir` you just
put in `config.yaml`.** `train_model()` defaults `output_dir` to
`models/v1` when omitted — Telco's production model directory. Forgetting
this argument silently overwrites Telco's trained model instead of
creating a new tenant's.

Add a "beats random baseline" test to `tests/test_multi_tenant.py`
mirroring `test_banking_tenant_beats_random_baseline` — don't skip this;
it's the only thing standing between "the code ran" and "the model is
actually predictive."

This step alone doesn't require any `feature_flags` to be `true` —
`/api/model/metrics` and `/api/model/importance` work for any tenant with
a `model_dir` containing `metadata.json`, regardless of feature flags,
since that's the base artifact every tenant profile declares.

## 4. Validate each advanced module before enabling its flag

Enable and re-validate **one flag at a time**, re-running the full test
suite between each. Do not batch-enable several flags on the strength of
"the base classifier worked."

| Module | What must be true before flipping the flag | Rough effort (relative to Telco's build) |
|---|---|---|
| `business_impact_core` | revenue_column confirmed as a recurring charge (step 1); a working `survival` model exists (needed for `expected_remaining_tenure_months`); the recoverable-revenue ceiling has been recomputed for this tenant and a sane default `cost_per_intervention` re-derived from it (not copied from Telco's $75); tenant-specific expected numbers written into a `test_business_impact.py`-equivalent test | Largest single item — this was itself the subject of a mid-project bug fix and a full afternoon of ceiling/default-cost analysis for Telco. Budget a full day, more if the revenue semantics need rethinking first. |
| `survival` | tenant has a contract-equivalent grouping column (step 1); Cox PH model fit; fitted curves spot-checked for plausibility per segment, not just "it fit without erroring" | Half a day, assuming a usable grouping column exists. Longer if one has to be engineered. |
| `segments` | KMeans fit; **cluster labels rewritten for this tenant's own feature meanings** — Telco's label text ("high-spend, low-tenure — highest risk") is Telco-specific prose, not a template to copy-paste | Half a day — the clustering itself is quick, the human-readable labeling is the real work. |
| `anomalies` | IsolationForest fit; a handful of flagged records manually reviewed to confirm they're genuinely unusual, not an artifact of unscaled/miscoded features | A few hours. |
| `clv` | Requires a whole **enriched dataset** (extra CLV-relevant columns) and a **separate classifier retrain** on it — this is not the same model as the base classifier. Don't underestimate this one. | The most expensive module — closer to a full day-plus, since it's a second modeling effort, not a config change. |
| `scenario_simulator` | Each scenario type (`uniform_charge_change`, `contract_migration`, `discount_offer`, `loyalty_program`) encodes assumptions about specific columns (a recurring charge, a small-cardinality contract field) — validate or explicitly exclude each type per tenant rather than assuming all four port cleanly | Half a day if the underlying columns exist for all four types; less if some types are simply not applicable and get left disabled. |
| `budget_optimizer` | Fully depends on `business_impact_core` already being solid. Additionally: write a synthetic-data regression test for `stop_at_positive_roi`-style edge cases (see the stop-at-positive-ROI lesson above) before trusting real-data output | A few hours on top of `business_impact_core`, mostly regression-test writing. |
| `customer_timeline` | Verify `clean_data`/`load_raw` work cleanly on the new raw file and that logged prediction/recommendation/scenario events read back sensibly | Low effort — around an hour, since this is mostly DB plumbing already generalized. |
| `priority_ranking`, `backtest` | A **fresh backtest run against this tenant's actual revenue semantics** — do not reuse Telco's backtest result as evidence for a different tenant's revenue column | Half a day, more if revenue semantics need rethinking first (this is exactly why Banking's `unavailable_reasons` for these two names the specific column). |

These are rough, relative estimates based on how much iteration each
module actually needed for Telco — not a guarantee, and probably an
underestimate if your tenant's schema surprises you the way Banking's
`Balance` column did.

## 5. Only then, flip the flag

Once a module's own validation step above is done, set its
`feature_flags` entry to `true` in `config.yaml`. No code change is
needed — `api/main.py` and `src/copilot/tools.py` both read
`feature_enabled(tenant_id, "<module>")` from `src/tenant_registry.py` at
request time.

## 6. Decide separately whether this tenant belongs in the live demo UI

Enabling flags makes a module *available*; it does not make a tenant
part of the live-demoed product surface. That's a separate, deliberate
decision — e.g. adding a nav entry, a login hint, or a component like
`CohortComparison.jsx`. Banking has real flags-worth of validation
evidence (its classifier, its tenant-isolation tests) but is deliberately
not part of the live UI — see `RUNNING.md` section 6. Don't conflate "the
backend can serve this" with "we're demoing this."
