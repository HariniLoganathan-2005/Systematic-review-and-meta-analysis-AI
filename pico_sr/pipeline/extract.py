from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

import pdfplumber

from pico_sr.llm_client import complete_chat
from pico_sr.db.models import Extraction, Paper, get_session, init_db
from pico_sr.pipeline.validate import validate_extraction_payload

logger = logging.getLogger(__name__)

SCHEMA_FIELDS = [
    "population_desc", "population_n", "intervention", "comparator",
    "outcome_measure", "effect_size", "effect_type", "ci_lower",
    "ci_upper", "p_value", "study_design", "followup_duration",
]

EXTRACT_PROMPT = """You are a precise research data extractor for meta-analysis.
Read this paper and extract ONLY the PRIMARY outcome effect size.

═══════════════════════════════════════════════
CRITICAL RULES — FOLLOW EXACTLY:
═══════════════════════════════════════════════

1. effect_size = ONE single decimal number (e.g. 0.45 or -0.32)
   - This is Cohen's d, SMD, Hedges' g, OR, RR, or MD
   - NEVER return an object like {{"value": 0.45}}
   - NEVER return a range like "0.2-0.5"

2. ci_lower = the SMALLER number of the 95% CI
   ci_upper = the LARGER number of the 95% CI
   Example: "95% CI [0.26, 0.78]" → ci_lower=0.26, ci_upper=0.78
   Example: "95% CI [-0.12, 0.45]" → ci_lower=-0.12, ci_upper=0.45
   - ci_lower MUST always be less than ci_upper
   - NEVER return an object like {{"lower": 0.26}}

3. p_value = single decimal (e.g. 0.032)
   If paper says "p<0.001" → return 0.001
   If paper says "p<0.05" → return 0.049

4. population_n = total participants as integer (e.g. 120)

5. effect_type = exactly one of:
   "SMD" "Cohen_d" "MD" "RR" "OR" "HR"

═══════════════════════════════════════════════
WHERE TO SEARCH (in order):
═══════════════════════════════════════════════
1. Results section → find SMD/Cohen's d/Hedges g/effect size
2. Abstract → numerical summary of results
3. Tables → between-group comparisons
4. Methods → sample size

═══════════════════════════════════════════════
HOW TO CALCULATE IF NOT GIVEN:
═══════════════════════════════════════════════
If only means and SDs given:
  pooled_SD = sqrt((SD1² + SD2²) / 2)
  d = (mean1 - mean2) / pooled_SD

If only t-statistic given:
  d = 2×t / sqrt(df)

If only OR given:
  d = ln(OR) × sqrt(3) / pi = ln(OR) × 0.5513

═══════════════════════════════════════════════
RETURN ONLY THIS JSON — NO MARKDOWN, NO EXPLANATION:
═══════════════════════════════════════════════
{{
  "population_desc": "string or null",
  "population_n": integer or null,
  "intervention": "string or null",
  "comparator": "string or null",
  "outcome_measure": "string or null",
  "effect_size": number or null,
  "effect_type": "SMD" or "Cohen_d" or "MD" or "RR" or "OR" or null,
  "ci_lower": number or null,
  "ci_upper": number or null,
  "p_value": number or null,
  "study_design": "RCT" or "cohort" or "case-control" or null,
  "followup_duration": "string or null",
  "confidence_population_desc": 0.9,
  "confidence_population_n": 0.9,
  "confidence_intervention": 0.9,
  "confidence_comparator": 0.9,
  "confidence_outcome_measure": 0.9,
  "confidence_effect_size": 0.9,
  "confidence_effect_type": 0.9,
  "confidence_ci_lower": 0.9,
  "confidence_ci_upper": 0.9,
  "confidence_p_value": 0.9,
  "confidence_study_design": 0.9,
  "confidence_followup_duration": 0.9
}}

Replace each 0.9 with your actual confidence (0.0–1.0) for that field.
Use low confidence (< 0.5) if you had to calculate or estimate a value.

Paper text:
---
{text}
---
"""


# ── Value cleaning helpers ────────────────────────────────────────────────────

def _unwrap(val: Any) -> Any:
    """
    Unwrap dict-wrapped values and convert strings to floats.
    Uses explicit None check so 0.0 is never lost.
    """
    if isinstance(val, dict):
        for key in ("value", "val", "estimate", "number"):
            if key in val and val[key] is not None:
                try:
                    return float(val[key])
                except (TypeError, ValueError):
                    continue
        return None

    if isinstance(val, str):
        cleaned = val.replace(",", "").strip()
        if cleaned.lower() in ("null", "none", "n/a", "na", ""):
            return None
        # Handle "p<0.001" style
        m = re.search(r"[<>]?\s*([-]?[\d.]+(?:e[+-]?\d+)?)", cleaned, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    if isinstance(val, (int, float)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f

    return None


def _force_numeric(payload: dict[str, Any]) -> dict[str, Any]:
    """Force all numeric fields to proper Python float/int or None."""
    for field in ["effect_size", "ci_lower", "ci_upper", "p_value", "population_n"]:
        val = _unwrap(payload.get(field))
        if val is not None:
            try:
                payload[field] = float(val)
            except (TypeError, ValueError):
                payload[field] = None
        else:
            payload[field] = None

    # population_n → int
    if payload.get("population_n") is not None:
        try:
            payload["population_n"] = int(payload["population_n"])
        except (TypeError, ValueError):
            payload["population_n"] = None

    return payload


def _validate_and_fix_ci(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Validate CI values:
    - Fix swapped CI (lower > upper)
    - Check effect size is inside CI
    - Flag impossible values
    - Reject CIs that are unreasonably wide (> 10 units on Cohen's d scale)
    """
    lo = payload.get("ci_lower")
    hi = payload.get("ci_upper")
    es = payload.get("effect_size")

    if lo is None or hi is None:
        return payload

    # Fix swapped CI
    if lo > hi:
        logger.warning("CI swapped [%s, %s] — fixing", lo, hi)
        payload["ci_lower"], payload["ci_upper"] = hi, lo
        lo, hi = hi, lo

    # Reject unreasonably wide CI (likely extraction error)
    ci_width = hi - lo
    if ci_width > 10:
        logger.warning(
            "CI width = %.2f is unreasonably large [%s, %s] — clearing CI",
            ci_width, lo, hi
        )
        # Don't clear effect_size, just clear the bad CI
        # This paper will be skipped in analysis (needs valid CI)
        payload["ci_lower"] = None
        payload["ci_upper"] = None
        return payload

    # Warn if effect outside CI
    if es is not None and not (lo - 0.01 <= es <= hi + 0.01):
        logger.warning(
            "Effect size %s not inside CI [%s, %s] — likely extraction error",
            es, lo, hi
        )

    return payload


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(path: str, max_chars: int = 100_000) -> str:
    parts: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                parts.append(t)
    except Exception as e:
        logger.warning("pdfplumber failed %s: %s", path, e)
        return ""
    full = "\n".join(parts)
    full = re.sub(r"\s+\n", "\n", full)
    full = re.sub(r"[ \t]+", " ", full)
    return full.strip()[:max_chars]


def _build_context(paper: Any, text: str) -> str:
    """
    Build rich extraction context.
    Prepend title so Groq always knows what paper it's reading.
    """
    header = f"""PAPER TITLE: {paper.title or 'Unknown'}
YEAR: {paper.year or 'Unknown'}
SOURCE: {paper.source or 'Unknown'}

"""
    return header + text


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    return json.loads(text)


def extract_with_llm(text: str) -> dict[str, Any]:
    """Send text to LLM and parse the PICO extraction."""
    prompt = EXTRACT_PROMPT.format(text=text[:120_000])
    content = complete_chat(prompt, temperature=0.0)  # temp=0 for consistency

    # Primary: JSON parse
    try:
        payload = _parse_json_loose(content)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed — regex fallback")
        payload: dict[str, Any] = {}

        for key in SCHEMA_FIELDS:
            m = re.search(
                rf'"{re.escape(key)}"\s*:\s*([^,}}\]]+)', content
            )
            if m:
                raw = m.group(1).strip().strip('"')
                try:
                    if raw.lower() == "null":
                        payload[key] = None
                    elif (
                        key in ("population_n",)
                        or "effect" in key
                        or key.startswith("ci_")
                        or key == "p_value"
                    ):
                        payload[key] = float(raw)
                    else:
                        payload[key] = raw
                except ValueError:
                    payload[key] = raw

        for k in list(payload.keys()):
            if not k.startswith("confidence_"):
                payload.setdefault(f"confidence_{k}", 0.3)

    # Cleanup pipeline
    payload = _force_numeric(payload)
    payload = _validate_and_fix_ci(payload)

    return payload


# ── Main extraction runner ────────────────────────────────────────────────────

def run_extraction(paper_ids: list[int] | None = None) -> dict[str, Any]:
    init_db()
    session  = get_session()
    processed = 0
    success   = 0

    try:
        q = session.query(Paper).filter(Paper.include_for_extraction.is_(True))
        if paper_ids:
            q = q.filter(Paper.id.in_(paper_ids))
        papers = q.all()
        logger.info("Starting extraction for %d papers", len(papers))

        for p in papers:
            text = ""

            # 1. Try PDF first (richest source)
            if p.pdf_path:
                text = extract_pdf_text(p.pdf_path)
                if text:
                    logger.info(
                        "Using PDF (%d chars): %s", len(text), p.title[:50]
                    )

            # 2. Fall back to stored abstract/full-text
            if not text or len(text.strip()) < 50:
                text = p.abstract or ""
                if text:
                    logger.info(
                        "Using stored text (%d chars): %s",
                        len(text), p.title[:50]
                    )

            # 3. Nothing available
            if not text or len(text.strip()) < 30:
                logger.warning(
                    "No text available for paper_id=%d: %s",
                    p.id, p.title[:50]
                )
                text = f"Title only: {p.title}"

            # Build context with paper metadata
            full_context = _build_context(p, text)

            # Run LLM extraction
            payload = extract_with_llm(full_context)

            # Log result
            has_effect = payload.get("effect_size") is not None
            has_ci     = (payload.get("ci_lower") is not None
                          and payload.get("ci_upper") is not None)

            logger.info(
                "Paper '%s': effect=%s ci=[%s,%s] n=%s p=%s %s",
                p.title[:45],
                payload.get("effect_size"),
                payload.get("ci_lower"),
                payload.get("ci_upper"),
                payload.get("population_n"),
                payload.get("p_value"),
                "✅" if (has_effect and has_ci) else "⚠️ incomplete",
            )

            if has_effect and has_ci:
                success += 1

            # Validate
            flags  = validate_extraction_payload(payload)
            ci_bad = (
                "ci_lower_gt_ci_upper" in flags
                or "effect_size_outside_ci" in flags
            )
            hitl = bool(flags) or any(
                float(payload.get(f"confidence_{k}", 0) or 0) < 0.75
                for k in SCHEMA_FIELDS
            )

            # Save/update extraction record
            ex = (
                session.query(Extraction)
                .filter(Extraction.paper_id == p.id)
                .order_by(Extraction.id.desc())
                .first()
            )
            if ex is None:
                ex = Extraction(paper_id=p.id)
                session.add(ex)

            # ── Save to actual columns (fixes UI table showing None) ──
            ex.effect_size       = payload.get("effect_size")
            ex.effect_type       = payload.get("effect_type")
            ex.ci_lower          = payload.get("ci_lower")
            ex.ci_upper          = payload.get("ci_upper")
            ex.population_n      = payload.get("population_n")
            ex.p_value           = payload.get("p_value")
            ex.population_desc   = payload.get("population_desc")
            ex.intervention      = payload.get("intervention")
            ex.comparator        = payload.get("comparator")
            ex.outcome_measure   = payload.get("outcome_measure")
            ex.study_design      = payload.get("study_design")
            ex.followup_duration = payload.get("followup_duration")
            ex.ci_quarantine     = ci_bad
            ex.validation_flag   = bool(flags)
            ex.hitl_pending      = hitl
            ex.payload_json           = json.dumps(payload)
            ex.validation_flags_json  = json.dumps(flags)

            session.flush()
            processed += 1
            time.sleep(1)  # rate limit

            logger.info(
                "Extracted %d/%d — %s",
                processed, len(papers), p.title[:60]
            )

        session.commit()
        logger.info(
            "Extraction done: %d processed, %d with valid effect+CI",
            processed, success
        )
        return {
            "extracted":   processed,
            "candidates":  len(papers),
            "with_effect": success,
            "missing":     processed - success,
        }

    except Exception as e:
        logger.error("Extraction failed: %s", str(e))
        session.rollback()
        raise
    finally:
        session.close()