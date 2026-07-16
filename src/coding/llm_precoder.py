"""LLM-assisted coding passes for Study 1.

Two independent passes (A and B) with differently-worded prompts of the
same Appendix B protocol. The model extracts structured FEATURES from the
incident text; the layer decision itself is made deterministically by
src/coding/protocol.py, so both coders share one rule base.

Determinism: every raw model response is cached under
data/raw/coding/pass{A,B}/{candidate_id}.json; re-runs are byte-identical
from cache. (Deterministic sampling is unavailable — the API rejects
sampling parameters — so response caching is what makes the pipeline
reproducible, and Appendix B documents this configuration.)

Structured outputs (output_config.format json_schema) guarantee the
response parses against CODING_SCHEMA.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from src.coding.protocol import IncidentFeatures, assign_layers, check_inclusion

MODEL = "claude-opus-4-8"
MAX_TOKENS = 1500


class CodingRefused(RuntimeError):
    """The model declined to code a candidate (stop_reason == 'refusal').

    Raised per-candidate so the orchestrator can skip it and continue rather
    than aborting the whole batch. Refusals happen on some cybersecurity
    incident writeups (the model's cyber safety classifier), which the
    orchestrator records and reports for human coding — it never fabricates a
    coding decision for a refused candidate.
    """


FEATURE_KEYS = (
    "autonomous_multistep",
    "mechanism_detail",
    "independent_reporting",
    "self_reported_with_specifics",
    "single_agent_loop_would_prevent",
    "required_interagent_or_shared_state",
    "oversight_point_existed",
    "oversight_deficient",
    "operational_mechanism",
)

CODING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "codeable": {
            "type": "boolean",
            "description": "True if the text documents a concrete agentic "
            "incident with enough mechanism detail to code.",
        },
        "exclusion_reason": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "not_agentic",
                        "insufficient_mechanism",
                        "unverified_account",
                    ],
                },
                {"type": "null"},
            ]
        },
        "system": {"type": "string", "description": "The agent system involved."},
        "date": {"type": "string", "description": "Incident date, YYYY-MM or YYYY-MM-DD."},
        "autonomous_multistep": {"type": "boolean"},
        "mechanism_detail": {"type": "boolean"},
        "independent_reporting": {"type": "boolean"},
        "self_reported_with_specifics": {"type": "boolean"},
        "single_agent_loop_would_prevent": {"type": "boolean"},
        "required_interagent_or_shared_state": {"type": "boolean"},
        "oversight_point_existed": {"type": "boolean"},
        "oversight_deficient": {"type": "boolean"},
        "operational_mechanism": {"type": "boolean"},
        "mechanism_phrase": {
            "type": "string",
            "description": "2-5 word failure-mode phrase, e.g. 'shared-memory contamination'.",
        },
        "rationale": {"type": "string", "description": "One sentence."},
        "confidence": {"type": "number"},
    },
    "required": [
        "codeable",
        "exclusion_reason",
        "system",
        "date",
        *FEATURE_KEYS,
        "mechanism_phrase",
        "rationale",
        "confidence",
    ],
    "additionalProperties": False,
}

_PROTOCOL_CORE = """\
Feature definitions (CASE layers):
- single_agent_loop_would_prevent: a correctly designed single-agent closed
  loop (observer, budget caps, iteration limits, circuit breaker,
  least-privilege tools) would have prevented the failure. [L1]
- required_interagent_or_shared_state: the mechanism required inter-agent
  interaction or shared state (cascades, shared-memory contamination,
  feedback loops between agents). [L2]
- oversight_point_existed / oversight_deficient: a human oversight point
  existed but lacked capacity, an interpretable model, or authority to
  act. [L3]
- operational_mechanism: the mechanism was operational: no rollback path,
  silent model swap, missing telemetry, unbounded cost, deployment gap. [L4]
The primary layer is decided downstream as the EARLIEST layer whose correct
functioning would have interrupted the failure trajectory; your job is only
to report which mechanisms contributed, truthfully and conservatively.
Mark a feature true only if the text supports it."""

_PROMPT_A = """You are coder A in a dual-coding reliability study of AI \
agent failures. Apply the inclusion filter, then extract mechanism features.

Inclusion filter (ALL three must hold, else codeable=false with the reason):
1. autonomous multi-step behavior: planning plus tool execution, not
   single-completion model output (else not_agentic)
2. mechanism-level description sufficient to code (else insufficient_mechanism)
3. independent reporting, or self-reported with technical specifics
   (else unverified_account)

{protocol_core}

If the text is a paper or report describing several incidents, code the
single most concretely documented incident within it.

INCIDENT MATERIAL:
Title: {title}
Date reported: {date}
Text: {text}

Return the JSON object exactly per schema."""

_PROMPT_B = """Independent reliability check: you are the second coder and \
must judge this material afresh, strictly from what is written.

Step 1 - Screening questions:
- Did a system plan AND execute tools over multiple steps? If it was a
  single model completion with no agency, it is excluded (not_agentic).
- Can you tell HOW the failure unfolded, mechanically? If not, it is
  excluded (insufficient_mechanism).
- Is the account independently reported, or first-party with concrete
  technical detail? If neither, it is excluded (unverified_account).

Step 2 - If it passes screening, answer the mechanism questionnaire.
{protocol_core}

Where a document covers multiple incidents, select the one described in the
most concrete, mechanism-level detail and code only that one.

MATERIAL TO CODE:
Title: {title}
Date reported: {date}
Text: {text}

Answer with the JSON object exactly per schema."""


def build_prompt(pass_name: str, candidate: dict) -> str:
    template = _PROMPT_A if pass_name == "A" else _PROMPT_B
    return template.format(
        protocol_core=_PROTOCOL_CORE,
        title=candidate.get("title", ""),
        date=candidate.get("date", ""),
        text=candidate.get("text_snippet", ""),
    )


def features_from_response(resp: dict, candidate: dict) -> IncidentFeatures:
    return IncidentFeatures(
        incident_id=candidate["candidate_id"],
        system=str(resp.get("system", "")),
        date=str(resp.get("date", "") or candidate.get("date", "")),
        **{k: bool(resp[k]) for k in FEATURE_KEYS},
    )


def label_for(resp: dict, candidate: dict) -> str:
    """Pre-dedup coding label: excluded, or the primary layer."""
    feats = features_from_response(resp, candidate)
    if not resp.get("codeable", False):
        return "excluded"
    if not check_inclusion(feats, seen_keys=set()).included:
        return "excluded"
    try:
        return assign_layers(feats).primary
    except ValueError:
        return "excluded"  # codeable claimed but no contributing layer


def _default_call_model(prompt: str) -> str:
    # The anthropic SDK depends on the `jiter` native extension, which a
    # Windows Application Control policy on this box intermittently blocks.
    # The DLL loads lazily (on first client.messages access), so an ImportError
    # can surface here rather than at `import anthropic`. On any ImportError,
    # fall back to a stdlib HTTP call hitting the exact same Messages endpoint /
    # structured-output contract; cached responses stay byte-identical.
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={
                "format": {"type": "json_schema", "schema": CODING_SCHEMA}
            },
            messages=[{"role": "user", "content": prompt}],
        )
    except ImportError:
        return _http_call_model(prompt)
    if response.stop_reason == "refusal":
        raise CodingRefused("model refused the coding request")
    return next(b.text for b in response.content if b.type == "text")


def _http_call_model(prompt: str) -> str:
    """SDK-free Messages API call (structured outputs, no beta header).

    Same request shape as _default_call_model; used only when the anthropic
    SDK cannot be imported (blocked jiter DLL). Uses urllib + stdlib json.
    """
    import os
    import urllib.request

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_AUTH_TOKEN"
    )
    if not api_key:
        raise RuntimeError("no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN set")
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "output_config": {
                "format": {"type": "json_schema", "schema": CODING_SCHEMA}
            },
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("stop_reason") == "refusal":
        raise CodingRefused("model refused the coding request")
    return next(b["text"] for b in payload["content"] if b["type"] == "text")


def _prefetch(
    pass_name: str,
    candidates: list[dict],
    pass_dir: Path,
    call_model: Callable[[str], str],
    progress: Callable[[str], None],
    max_workers: int,
) -> list[str]:
    """Fill the cache for all uncached candidates with a small worker
    pool. Cache files are per-candidate, so concurrent writes are safe;
    row assembly stays sequential and deterministic.

    Returns the candidate_ids the model REFUSED to code. A refusal on one
    candidate must not abort the batch (or lose the concurrently-completed
    caches), so it is recorded and skipped; any other error propagates.
    """
    todo = [
        c for c in candidates
        if not (pass_dir / f"{c['candidate_id']}.json").exists()
    ]
    if not todo:
        return []
    done = 0
    refused: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(call_model, build_prompt(pass_name, c)): c for c in todo
        }
        for future in as_completed(futures):
            cand = futures[future]
            try:
                raw = future.result()  # propagate non-refusal API errors loudly
            except CodingRefused:
                refused.append(cand["candidate_id"])
                progress(
                    f"pass {pass_name}: REFUSED {cand['candidate_id']}", flush=True
                )
                continue
            (pass_dir / f"{cand['candidate_id']}.json").write_text(
                raw, encoding="utf-8"
            )
            done += 1
            if done % 10 == 0 or done == len(todo):
                progress(f"pass {pass_name}: {done}/{len(todo)} coded", flush=True)
    return refused


def run_pass(
    pass_name: str,
    candidates: list[dict],
    cache_dir: Path,
    call_model: Callable[[str], str] | None = None,
    progress: Callable[[str], None] = lambda s, **kw: None,
    max_workers: int = 6,
) -> tuple[list[dict], list[str]]:
    """Code every candidate, serving from cache where possible.

    Returns (rows, refused_ids): one row per successfully-coded candidate
    (raw response fields plus the protocol-derived label and secondaries),
    and the candidate_ids the model refused to code (no cache, no row).
    """
    call_model = call_model or _default_call_model
    pass_dir = Path(cache_dir) / f"pass{pass_name}"
    pass_dir.mkdir(parents=True, exist_ok=True)
    refused = _prefetch(
        pass_name, candidates, pass_dir, call_model, progress, max_workers
    )
    refused_set = set(refused)
    rows = []
    for cand in candidates:
        if cand["candidate_id"] in refused_set:
            continue
        cache_file = pass_dir / f"{cand['candidate_id']}.json"
        raw = cache_file.read_text(encoding="utf-8")
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"unparseable coding response for {cand['candidate_id']} "
                f"(pass {pass_name}): {exc}"
            ) from exc

        label = label_for(resp, cand)
        secondaries = ""
        if label not in ("excluded",):
            feats = features_from_response(resp, cand)
            secondaries = "; ".join(assign_layers(feats).secondary)
        rows.append(
            {
                "candidate_id": cand["candidate_id"],
                "source": cand.get("source", ""),
                "url": cand.get("url", ""),
                "date": str(resp.get("date", "") or cand.get("date", "")),
                "system": str(resp.get("system", "")),
                "label": label,
                "exclusion_reason": str(resp.get("exclusion_reason") or "")
                if label == "excluded"
                else "",
                "secondary_layers": secondaries,
                "mechanism_phrase": str(resp.get("mechanism_phrase", "")),
                "rationale": str(resp.get("rationale", "")),
                "confidence": float(resp.get("confidence", 0.0)),
            }
        )
    return rows, refused
