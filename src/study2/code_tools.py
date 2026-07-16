"""Study 2: code each agent-ops tool's documentation against the CASE
mechanism classes, one Full/Partial/None verdict per (tool, layer).

Coding frame (Appendix, tab:tooling protocol): for each tool and each layer,
does the documentation evidence at least one mechanism class of that layer?
  Full    — mechanism described with operational detail
  Partial — mechanism claimed with limited detail
  None    — no described mechanism (marketing claims without mechanism = None)
One line of documentation evidence (+ its URL) is recorded per non-None cell.

Single coder (no dual-coding / kappa; that is Study 1's design). Every raw
model response is cached under data/raw/tool_coding/{tool_id}.json so the
matrix is exactly reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.llm.structured import call_structured

VERDICTS = ("Full", "Partial", "None")
LAYERS = ("L1", "L2", "L3", "L4")
# Bound tokens while capturing enough of each tool's (typically large) doc
# pages that operational detail is present, not just the overview/nav header.
# 24k chars ~ 6k tokens of extracted text per tool.
DOC_CHAR_BUDGET = 24000

_LAYER_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "evidence": {
            "type": "string",
            "description": "One line of documentation evidence for a non-None "
            "verdict (the mechanism the docs describe); empty string if None.",
        },
        "evidence_url": {
            "type": "string",
            "description": "The doc URL the evidence came from; empty if None.",
        },
    },
    "required": ["verdict", "evidence", "evidence_url"],
    "additionalProperties": False,
}

TOOL_SCHEMA = {
    "type": "object",
    "properties": {layer: _LAYER_VERDICT_SCHEMA for layer in LAYERS},
    "required": list(LAYERS),
    "additionalProperties": False,
}


def _layer_frame(mechanism_classes: dict) -> str:
    """One line per layer listing its mechanism-class names (the coding frame)."""
    lines = []
    for layer in LAYERS:
        spec = mechanism_classes["layers"][layer]
        names = [c["mechanism"] for c in spec["classes"]]
        lines.append(f"{layer} ({spec['name']}): " + "; ".join(names))
    return "\n".join(lines)


def build_prompt(tool: dict, doc_text: str, mechanism_classes: dict) -> str:
    frame = _layer_frame(mechanism_classes)
    text = doc_text[:DOC_CHAR_BUDGET]
    urls = ", ".join(tool.get("doc_urls", []))
    return f"""You are coding an agent-operations tool's DOCUMENTATION against the \
CASE capability framework for a systematic tooling-gap study. Judge only from \
the documentation text provided; marketing claims without a described \
mechanism code as None.

For EACH of the four CASE layers, decide whether the documentation evidences \
at least one of that layer's mechanism classes:
{frame}

Layer-boundary clarifications (apply strictly):
- L3 is oversight of the agent's live ACTIONS and decisions: approval gates \
before consequential actions, override / kill switches, escalation or routing \
to a human, tiered autonomy, and audit trails of decision provenance. Human \
labeling or annotation of EVALUATION data (scoring runs, building eval \
datasets) is NOT L3 -- that is an L4 evaluation activity. Do not code L3 from \
annotation/labeling queues.
- L2 requires monitoring or governing interactions BETWEEN multiple agents \
(interaction-graph, cascade/contagion detection, shared-state monitoring). \
Merely running or tracing a multi-agent system, or single-agent multi-step \
traces, is NOT L2 unless the docs describe monitoring the inter-agent \
coupling itself.

Verdict scale for each layer (apply uniformly across tools):
- Full: at least one mechanism class is documented concretely enough that an \
engineer could configure or rely on it -- a named feature with described \
behavior, settings, or an API (e.g. "set a per-key token budget", "interrupt \
the graph for human approve/edit/reject"). It need not include code-level \
internals; a clear operational description of the capability is Full.
- Partial: a relevant mechanism is mentioned or claimed but only in passing, \
without enough detail to configure or rely on it, or only a weak/adjacent \
form is described.
- None: no relevant mechanism described, or a marketing claim with no \
mechanism behind it.

For every Full or Partial layer, record ONE line of evidence quoting or \
closely paraphrasing the documentation, and the URL it came from (choose from: \
{urls}). For None, leave evidence and evidence_url as empty strings.

TOOL: {tool['name']} ({tool.get('vendor', '')})
DOCUMENTATION:
{text}

Return the JSON object exactly per schema (keys L1, L2, L3, L4)."""


def code_tool(
    tool: dict,
    doc_text: str,
    mechanism_classes: dict,
    cache_dir: Path,
    call_model: Callable[[str], str] | None = None,
) -> dict:
    """Return {layer: {verdict, evidence, evidence_url}} for one tool.

    Cache-first: the raw model response is cached per tool and reused.
    """
    call_model = call_model or (lambda p: call_structured(p, TOOL_SCHEMA))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{tool['id']}.json"
    if cache_file.exists():
        raw = cache_file.read_text(encoding="utf-8")
    else:
        raw = call_model(build_prompt(tool, doc_text, mechanism_classes))
        cache_file.write_text(raw, encoding="utf-8")
    resp = json.loads(raw)
    return {layer: _normalize(resp[layer]) for layer in LAYERS}


def _normalize(cell: dict) -> dict:
    verdict = cell.get("verdict", "None")
    if verdict not in VERDICTS:
        verdict = "None"
    if verdict == "None":
        return {"verdict": "None", "evidence": "", "evidence_url": ""}
    return {
        "verdict": verdict,
        "evidence": str(cell.get("evidence", "")).strip(),
        "evidence_url": str(cell.get("evidence_url", "")).strip(),
    }
