from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pico_sr.llm_client import complete_chat
from pico_sr.db.models import Paper, get_session, init_db

logger = logging.getLogger(__name__)

SCREEN_PROMPT = """You are a systematic review screener.
Given a study title and abstract, decide if the study matches the PICO criteria.

PICO criteria:
{pico_criteria}

Respond with ONLY valid JSON (no markdown):
{{"decision": "include" or "exclude", "reason": "short string", "confidence": 0.0}}

Rules:
- confidence must be 0.0 to 1.0
- Only return "include" if the study clearly matches ALL PICO criteria
- Return "exclude" if any PICO criterion is not met
- Return higher confidence when evidence is clear

Title: {title}

Abstract:
{abstract}
"""


def _parse_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    return json.loads(text)


def screen_one_paper(
    title: str,
    abstract: str | None,
    pico_criteria: str,
) -> dict[str, Any]:
    prompt = SCREEN_PROMPT.format(
        pico_criteria=pico_criteria,
        title=title,
        abstract=(abstract or "No abstract available")[:12000],
    )
    content = complete_chat(prompt, temperature=0.1)
    try:
        data = _parse_json_blob(content)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Screen parse failed: %s — raw: %s", e, content[:300])
        return {"decision": "exclude", "reason": "parse_error", "confidence": 0.0}

    decision = str(data.get("decision", "exclude")).lower()
    if decision not in ("include", "exclude"):
        decision = "exclude"
    conf = float(data.get("confidence", 0.0))
    conf = max(0.0, min(1.0, conf))
    return {
        "decision": decision,
        "reason": str(data.get("reason", ""))[:2000],
        "confidence": conf,
    }


def apply_threshold(decision: str, confidence: float) -> tuple[str, bool, bool]:
    """
    Returns (stored_decision, uncertain_flag, include_for_extraction).

    FIX: Uncertain papers are NOT included for extraction.
    Only papers with decision=include AND confidence>=0.6 are extracted.
    This prevents low-confidence papers from polluting the meta-analysis.
    """
    # High confidence include → extract
    if decision == "include" and confidence >= 0.75:
        return ("include", False, True)

    # Medium confidence include → flag as uncertain, still extract
    # but mark for human review
    if decision == "include" and confidence >= 0.6:
        return ("uncertain", True, True)

    # Low confidence → uncertain, do NOT extract
    if confidence < 0.6:
        return ("uncertain", True, False)

    # Exclude
    return ("exclude", False, False)


def run_screening(
    paper_ids: list[int] | None = None,
    pico_criteria: str = (
        "Include RCTs and quasi-experimental studies on the intervention "
        "vs comparator reporting quantitative outcomes relevant to the review question."
    ),
) -> dict[str, Any]:
    init_db()
    session = get_session()
    try:
        q = session.query(Paper)
        if paper_ids:
            q = q.filter(Paper.id.in_(paper_ids))
        papers = q.all()

        screened = included = excluded = uncertain = 0

        for p in papers:
            if not p.title:
                continue

            result = screen_one_paper(p.title, p.abstract, pico_criteria)
            time.sleep(1)  # Prevent Groq rate limiting

            _dec, is_uncertain, extract = apply_threshold(
                result["decision"], result["confidence"]
            )

            p.screen_decision   = _dec
            p.screen_reason     = result["reason"]
            p.screen_confidence = result["confidence"]
            p.uncertain_review  = is_uncertain
            p.include_for_extraction = extract

            screened += 1
            if _dec == "include":
                included += 1
            elif _dec == "exclude":
                excluded += 1
            else:
                uncertain += 1

            logger.info(
                "Screened %d/%d — %s → %s (conf=%.2f, extract=%s)",
                screened, len(papers),
                p.title[:60], _dec,
                result["confidence"], extract,
            )

        session.commit()
        logger.info(
            "Screening done: %d included, %d excluded, %d uncertain",
            included, excluded, uncertain
        )
        return {
            "screened": screened,
            "total": len(papers),
            "included": included,
            "excluded": excluded,
            "uncertain": uncertain,
        }
    finally:
        session.close()