"""Gemini function-calling agent - Stage 2 of the AI Copilot subsystem.

Wires src/copilot/tools.py's 8 real, already-tested tools into Gemini via
the SDK's function-calling flow: declare each tool's schema, let Gemini
decide which one(s) to call and with what arguments, execute the REAL
Python function ourselves, send the real result back, and only then ask
Gemini for its final natural-language answer. Gemini never invents a
number - every fact in its answer has to trace back to a real tool call
(enforced by SYSTEM_INSTRUCTION below, verified adversarially in
tests/test_copilot_guardrails.py).

SECURITY: tenant_id is NEVER exposed to the model as a callable parameter
- none of the schemas below have a tenant_id property. It is always
injected server-side from the authenticated request's JWT (see
api/main.py's /api/copilot/chat), so no prompt can talk its way into
querying a different tenant's data than the one the caller is actually
authenticated as. _execute_tool() also strips any "tenant_id" key the
model might still try to slip into its arguments, as defense in depth.
"""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from sqlalchemy.orm import Session

from src.copilot import tools

load_dotenv()

# Live calls during implementation ruled out the alternatives:
# gemini-2.5-flash and gemini-2.5-flash-lite are both 404 ("no longer
# available to new users"); gemini-2.0-flash has a hard 0 free-tier quota
# on this account (429); gemini-3.5-flash and gemini-flash-latest are real
# and free-tier-eligible but were under sustained 503 "high demand" at
# implementation time, surviving even a 3-attempt/5s backoff (see
# _generate_with_retry below). gemini-3.1-flash-lite is the one that
# actually answered reliably - also a good functional fit, since Google
# positions Flash-Lite specifically for cost-efficient agentic/tool-calling
# workloads. Override via GEMINI_MODEL if your key's availability differs.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MODEL_NAME = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

# Guards against a pathological call -> call -> call loop. Observed live: a
# "top opportunities with a recommended-action table" question can
# legitimately need 1 (get_top_opportunities) + one get_customer_detail per
# row, so this needs headroom beyond "one or two calls" - 8 covers a top-5
# table with per-customer detail plus one retry hop.
MAX_TOOL_HOPS = 8

SYSTEM_INSTRUCTION = (
    "You can ONLY answer using the provided tools. If no tool can answer the "
    "question, say so explicitly and name what data would be needed. NEVER "
    "generate a plausible-sounding number, percentage, or explanation that did "
    "not come from a tool call. This system has NO historical time-series data "
    "(cannot answer 'why did X change over time'), NO real intervention-outcome "
    "data (cannot claim any recommendation is 'proven' to work, though you may "
    "cite the Hillstrom benchmark if directly relevant, clearly labeled as "
    "external reference data, not this system's own measured results), and NO "
    "forecasting model exists for this tenant (only "
    "illustrative scenario simulations, which you may run and clearly label as "
    "hypothetical)."
)


class CopilotUnavailableError(RuntimeError):
    """Raised when Gemini can't be reached: missing/invalid API key, or a
    real Gemini API-level failure. api/main.py maps this to a 503 - never
    a fabricated response."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise CopilotUnavailableError("GEMINI_API_KEY is not set")

    _client = genai.Client(api_key=api_key)
    return _client


def reset_client_for_testing() -> None:
    """Test-only: clears the cached client so a test can force
    _get_client() to re-check the environment."""
    global _client
    _client = None


# --- Tool declarations: schemas only, no tenant_id (see module docstring) ---

_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_business_impact_summary",
        description=(
            "Real portfolio-level revenue-at-risk and recoverable-revenue summary for the "
            "caller's own tenant, computed from the actual trained model. No arguments."
        ),
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    types.FunctionDeclaration(
        name="get_top_opportunities",
        description=(
            "Real top-N customers ranked by opportunity_score (a heuristic combining revenue "
            "at risk, ease of saving, and CLV weight - not a probability)."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many customers to return (default 10)."}
            },
        },
    ),
    types.FunctionDeclaration(
        name="get_customer_detail",
        description=(
            "Real single-customer lookup by exact customer ID: churn_probability (from the "
            "trained classifier), top model drivers, health score components, and recommended "
            "action. Does NOT include any per-customer dollar CLV figure - none exists in this "
            "system."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Exact customer ID, e.g. '7590-VHVEG'."}
            },
            "required": ["customer_id"],
        },
    ),
    types.FunctionDeclaration(
        name="run_budget_optimization",
        description=(
            "Runs the real greedy budget-allocation optimizer over a fixed retention budget: "
            "ranks customers by opportunity_score and funds as many as the budget affords at "
            "a labeled, assumed cost-per-intervention (default $75, not a measured cost)."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "budget": {"type": "number", "description": "Total retention budget in dollars."},
                "cost_override": {
                    "type": "number",
                    "description": "Optional override for cost-per-intervention (defaults to $75 if omitted).",
                },
            },
            "required": ["budget"],
        },
    ),
    types.FunctionDeclaration(
        name="run_scenario",
        description=(
            "Runs a real, ILLUSTRATIVE what-if re-scoring of the actual test-set population "
            "under a hypothetical portfolio-wide change, using the already-trained classifier. "
            "This is a hypothetical simulation, not a forecast of what will actually happen. "
            "scenario_type must be one of: 'uniform_charge_change' (params: percent_change), "
            "'contract_migration' (params: from_contract, to_contract, migration_rate 0-1), "
            "'discount_offer' (params: discount_percent, target_segment integer id or null for "
            "all customers), 'loyalty_program' (params: adoption_rate 0-1 - an illustrative, "
            "labeled-as-assumed 15% churn-probability reduction for adopters, not a measured "
            "effect)."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "scenario_type": {
                    "type": "string",
                    "enum": ["uniform_charge_change", "contract_migration", "discount_offer", "loyalty_program"],
                },
                "params": {
                    "type": "object",
                    "description": "Scenario-type-specific parameters, see this function's description.",
                },
            },
            "required": ["scenario_type", "params"],
        },
    ),
    types.FunctionDeclaration(
        name="get_segment_summary",
        description=(
            "Real, already-fitted K-Means customer segments (clusters) with their labels and "
            "profiles. No arguments."
        ),
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    types.FunctionDeclaration(
        name="get_alerts",
        description=(
            "Real, already-computed portfolio and per-customer alerts (e.g. high-value "
            "customers at high churn risk). No arguments."
        ),
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    types.FunctionDeclaration(
        name="get_survival_likelihood",
        description=(
            "Real probability, from the fitted Cox survival model (not an assumption), that a "
            "specific customer churns within 7, 30, and 90 days. Looked up by exact customer ID."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Exact customer ID, e.g. '7590-VHVEG'."}
            },
            "required": ["customer_id"],
        },
    ),
]

_TOOLS = types.Tool(function_declarations=_TOOL_DECLARATIONS)


def _execute_tool(name: str, args: dict[str, Any], tenant_id: str, db: Session | None = None) -> Any:
    """Dispatches to the real src/copilot/tools.py function, always
    injecting the server-authenticated tenant_id - the model can never
    supply or override it (the schemas above don't expose it at all; this
    strip is defense in depth in case a model ever hallucinates the key
    anyway). Also always passes db through - see tools.py's module
    docstring for why every one of these 8 tools needs it to work for a
    self-registered tenant at all."""
    args = {key: value for key, value in args.items() if key != "tenant_id"}

    if name == "get_business_impact_summary":
        return tools.get_business_impact_summary(tenant_id, db=db)
    if name == "get_top_opportunities":
        limit = int(args["limit"]) if args.get("limit") is not None else 10
        return tools.get_top_opportunities(tenant_id, limit=limit, db=db)
    if name == "get_customer_detail":
        return tools.get_customer_detail(args["customer_id"], tenant_id, db=db)
    if name == "run_budget_optimization":
        cost_override = float(args["cost_override"]) if args.get("cost_override") is not None else None
        return tools.run_budget_optimization(
            tenant_id, budget=float(args["budget"]), cost_override=cost_override, db=db
        )
    if name == "run_scenario":
        return tools.run_scenario(tenant_id, args["scenario_type"], args.get("params") or {}, db=db)
    if name == "get_segment_summary":
        return tools.get_segment_summary(tenant_id, db=db)
    if name == "get_alerts":
        return tools.get_alerts(tenant_id, db=db)
    if name == "get_survival_likelihood":
        return tools.get_survival_likelihood(args["customer_id"], tenant_id, db=db)

    return {"error": f"Unknown tool '{name}' - not one of this system's real tools."}


# Observed live: the free tier genuinely returns transient 503s ("This
# model is currently experiencing high demand... temporary") that the
# SDK's own built-in retry doesn't absorb. ServerError (5xx) is worth a
# short backoff-retry; ClientError (4xx - bad request, quota exhausted,
# invalid key) is not transient and should fail immediately.
_SERVER_ERROR_RETRY_ATTEMPTS = 3
_SERVER_ERROR_RETRY_DELAY_SECONDS = 5


def _generate_with_retry(client: genai.Client, **kwargs: Any) -> types.GenerateContentResponse:
    for attempt in range(_SERVER_ERROR_RETRY_ATTEMPTS):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError:
            if attempt == _SERVER_ERROR_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_SERVER_ERROR_RETRY_DELAY_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover


MORNING_BRIEF_PROMPT = (
    "Generate a concise 'Morning Brief' for me, the account manager just logging in. Use "
    "your tools to report: (1) the top 3 customer opportunities right now by "
    "opportunity_score, one line each; (2) the single most severe alert currently active, "
    "if any exist; (3) run exactly one realistic retention scenario relevant to what you "
    "see (pick a sensible scenario_type and parameters yourself) and report its real "
    "projected impact, clearly labeled as a hypothetical illustration, not a prediction. "
    "Keep the whole brief under 200 words, with short section headers - do NOT include an "
    "overall 'Morning Brief' title of your own, the surrounding page already shows one. "
    "Follow all your existing rules - never invent a number, and if a tool reports "
    "something is unavailable for this tenant, say so plainly instead of guessing."
)


def generate_morning_brief(tenant_id: str, db: Session | None = None) -> dict[str, Any]:
    """A proactive, unprompted variant of chat() - identical guardrails and
    real tool-calling, just a fixed internal prompt instead of a
    user-typed question. Powers the Dashboard's Morning Brief card,
    generated fresh on every call (no caching). Raises
    CopilotUnavailableError exactly like chat() does."""
    return chat(MORNING_BRIEF_PROMPT, tenant_id=tenant_id, db=db)


def chat(message: str, tenant_id: str, db: Session | None = None) -> dict[str, Any]:
    """Sends `message` to Gemini with the real tool declarations, executes
    any tool call(s) it requests against real data, sends the real
    result(s) back, and returns Gemini's final natural-language response
    along with which tools were actually used and their real outputs.

    db is threaded through to every tool call (see tools.py's module
    docstring) - without it, every tool 100%-of-the-time reports
    "unavailable" for a self-registered tenant regardless of their real
    feature_flags_json. Defaults to None so a caller with no session (e.g.
    a quick script) still gets correct behavior for Telco/Banking;
    api/main.py's copilot endpoints (the only real callers) always pass a
    real one.

    Raises CopilotUnavailableError if the API key is missing/invalid or
    Gemini itself is unreachable - callers must not fall back to a
    fabricated response in that case (see api/main.py's 503 handling).
    """
    client = _get_client()

    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, tools=[_TOOLS])
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part.from_text(text=message)])]

    sources: list[str] = []
    tool_outputs: list[dict[str, Any]] = []

    try:
        response = _generate_with_retry(client, model=MODEL_NAME, contents=contents, config=config)

        hops = 0
        while response.function_calls and hops < MAX_TOOL_HOPS:
            hops += 1
            contents.append(response.candidates[0].content)

            function_response_parts = []
            for function_call in response.function_calls:
                args = dict(function_call.args or {})
                result = _execute_tool(function_call.name, args, tenant_id, db)
                sources.append(function_call.name)
                tool_outputs.append({"tool": function_call.name, "arguments": args, "result": result})
                function_response_parts.append(
                    types.Part.from_function_response(name=function_call.name, response={"result": result})
                )
            contents.append(types.Content(role="tool", parts=function_response_parts))

            response = _generate_with_retry(client, model=MODEL_NAME, contents=contents, config=config)

        if response.function_calls or not (response.text or "").strip():
            # Either we hit MAX_TOOL_HOPS while the model still wanted
            # another tool call, or it returned an empty completion for
            # some other reason (observed live: a 5-tool-call table request
            # hit the old hop cap mid-request and came back with an empty
            # response.text, rendering as a blank bubble). `contents` at
            # this point always ends on a resolved "tool" turn (we never
            # append an unexecuted function-call request), so asking once
            # more with tools disabled forces a real text summary grounded
            # in whatever real tool results were already gathered, instead
            # of silently showing nothing.
            final_config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
            response = _generate_with_retry(client, model=MODEL_NAME, contents=contents, config=final_config)
    except errors.APIError as exc:
        raise CopilotUnavailableError(f"Gemini API error: {exc}") from exc

    return {
        "response": response.text or "",
        "sources": sources,
        "tool_outputs": tool_outputs,
    }
