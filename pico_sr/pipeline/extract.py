from __future__ import annotations

import json
import logging
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

EXTRACT_PROMPT = """You are a research data extractor for meta-analysis.
Read this paper carefully and extract ONLY these values.

CRITICAL RULES:
- Return ONLY plain numbers, never objects or dicts
- If value is 0 or 0.0, still return 0 (not null)
- effect_size MUST be a single decimal like 0.45 or -0.32
- ci_lower and ci_upper MUST be single decimals like 0.12 and 0.78
- p_value MUST be a single decimal like 0.032 (if p<0.001 return 0.001)
- population_n MUST be a whole integer like 120
- Do NOT return {{"value": 0.45}} — return just 0.45
- Do NOT return {{"lower": 0.12, "upper": 0.78}} — return each separately

WHERE TO SEARCH:
1. Results section → SMD, Cohen's d, Hedges g, 95% CI, p-value
2. Abstract → any numeric outcomes
3. Methods → sample size (n=, N=, participants)
4. Tables → means/SDs if no effect size given

EFFECT SIZE CONVERSION:
- t-statistic: d = 2t / sqrt(df)
- means/SD: d = (mean1 - mean2) / pooled_SD
- OR: d = ln(OR) * sqrt(3) / pi
- RR: d = ln(RR) * sqrt(3) / pi

Return ONLY this JSON (no markdown, no explanation):
{{
  "population_desc": "string or null",
  "population_n": integer or null,
  "intervention": "string or null",
  "comparator": "string or null",
  "outcome_measure": "string or null",
  "effect_size": number or null,
  "effect_type": "SMD" or "Cohen_d" or "MD" or "RR" or "OR" or "HR" or null,
  "ci_lower": number or null,
  "ci_upper": number or null,
  "p_value": number or null,
  "study_design": "RCT" or "cohort" or "case-control" or "crossover" or null,
  "followup_duration": "string or null",
  "confidence_population_desc": 0.0,
  "confidence_population_n": 0.0,
  "confidence_intervention": 0.0,
  "confidence_comparator": 0.0,
  "confidence_outcome_measure": 0.0,
  "confidence_effect_size": 0.0,
  "confidence_effect_type": 0.0,
  "confidence_ci_lower": 0.0,
  "confidence_ci_upper": 0.0,
  "confidence_p_value": 0.0,
  "confidence_study_design": 0.0,
  "confidence_followup_duration": 0.0
}}

Paper text:
---
{text}
---
"""


def _unwrap(val: Any) -> Any:
    """Unwrap dict-wrapped LLM values. Fixed: explicit None check keeps 0.0."""
    if isinstance(val, dict):
        for key in ("value", "val", "estimate", "number", "lower", "upper"):
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
        m = re.search(r"[<>]?\s*([\d.]+(?:e[+-]?\d+)?)", cleaned, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _unwrap_numeric_fields(payload: dict[str, Any]) -> dict[str, Any]:
    for field in ["effect_size", "ci_lower", "ci_upper", "p_value", "population_n"]:
        if field in payload:
            payload[field] = _unwrap(payload[field])
    return payload


def _force_numeric(payload: dict[str, Any]) -> dict[str, Any]:
    """Force numeric fields to float/int or None. Fixes PyArrow mixed-type crash."""
    for field in ["effect_size", "ci_lower", "ci_upper", "p_value", "population_n"]:
        val = payload.get(field)
        if val is None:
            payload[field] = None
            continue
        val = _unwrap(val)
        if val is not None:
            try:
                payload[field] = float(val)
            except (TypeError, ValueError):
                logger.warning("Cannot convert '%s'='%s' → None", field, val)
                payload[field] = None
        else:
            payload[field] = None
    if payload.get("population_n") is not None:
        try:
            payload["population_n"] = int(payload["population_n"])
        except (TypeError, ValueError):
            payload["population_n"] = None
    return payload


def _validate_ci(payload: dict[str, Any]) -> dict[str, Any]:
    lo = payload.get("ci_lower")
    hi = payload.get("ci_upper")
    es = payload.get("effect_size")
    if lo is not None and hi is not None:
        if lo > hi:
            logger.warning("CI swapped [%s,%s] — fixing", lo, hi)
            payload["ci_lower"], payload["ci_upper"] = hi, lo
            lo, hi = hi, lo
        if es is not None and not (lo <= es <= hi):
            logger.warning("Effect %s outside CI [%s,%s]", es, lo, hi)
    return payload


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


def _build_text_from_abstract(paper: Any) -> str:
    return f"""TITLE: {paper.title or 'Unknown'}

ABSTRACT:
{paper.abstract or 'No abstract available'}

EXTRACTION INSTRUCTIONS:
Abstract only — extract all numeric values present.
Look for: SMD, Cohen d, Hedges g, mean difference, 95% CI,
p-values, sample sizes, group means and SDs.
"""


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    return json.loads(text)


def extract_with_llm(text: str) -> dict[str, Any]:
    prompt = EXTRACT_PROMPT.format(text=text[:120_000])
    content = complete_chat(prompt, temperature=0.1)
    try:
        payload = _parse_json_loose(content)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed — regex fallback")
        payload: dict[str, Any] = {}
        for key in SCHEMA_FIELDS:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*([^,}}\]]+)', content)
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
            if k.startswith("confidence_"):
                continue
            ck = f"confidence_{k}"
            if ck not in payload:
                payload[ck] = 0.3

    payload = _unwrap_numeric_fields(payload)
    payload = _force_numeric(payload)
    payload = _validate_ci(payload)
    return payload


def run_extraction(paper_ids: list[int] | None = None) -> dict[str, Any]:
    init_db()
    session = get_session()
    processed = 0
    try:
        q = session.query(Paper).filter(Paper.include_for_extraction.is_(True))
        if paper_ids:
            q = q.filter(Paper.id.in_(paper_ids))
        papers = q.all()
        logger.info("Starting extraction for %d papers", len(papers))

        for p in papers:
            text = ""
            if p.pdf_path:
                text = extract_pdf_text(p.pdf_path)
                if text:
                    logger.info("PDF (%d chars): %s", len(text), p.title[:50])
            if not text or len(text.strip()) < 50:
                text = _build_text_from_abstract(p)
                logger.info("Abstract fallback: %s", p.title[:50])

            payload = extract_with_llm(text)
            logger.info(
                "Result '%s': effect=%s ci=[%s,%s] n=%s p=%s",
                p.title[:50],
                payload.get("effect_size"),
                payload.get("ci_lower"),
                payload.get("ci_upper"),
                payload.get("population_n"),
                payload.get("p_value"),
            )

            flags = validate_extraction_payload(payload)
            ci_bad = (
                "ci_lower_gt_ci_upper" in flags
                or "effect_size_outside_ci" in flags
            )
            hitl = bool(flags) or any(
                float(payload.get(f"confidence_{k}", 0) or 0) < 0.75
                for k in SCHEMA_FIELDS
            )

            ex = (
                session.query(Extraction)
                .filter(Extraction.paper_id == p.id)
                .order_by(Extraction.id.desc())
                .first()
            )
            if ex is None:
                ex = Extraction(paper_id=p.id)
                session.add(ex)

            # ── KEY FIX: Save to actual columns so UI table shows values ──
            ex.effect_size      = payload.get("effect_size")
            ex.effect_type      = payload.get("effect_type")
            ex.ci_lower         = payload.get("ci_lower")
            ex.ci_upper         = payload.get("ci_upper")
            ex.population_n     = payload.get("population_n")
            ex.p_value          = payload.get("p_value")
            ex.population_desc  = payload.get("population_desc")
            ex.intervention     = payload.get("intervention")
            ex.comparator       = payload.get("comparator")
            ex.outcome_measure  = payload.get("outcome_measure")
            ex.study_design     = payload.get("study_design")
            ex.followup_duration = payload.get("followup_duration")
            ex.ci_quarantine    = ci_bad
            ex.validation_flag  = bool(flags)
            ex.hitl_pending     = hitl
            ex.payload_json          = json.dumps(payload)
            ex.validation_flags_json = json.dumps(flags)

            session.flush()
            processed += 1
            time.sleep(1)
            logger.info("Extracted %d/%d — %s", processed, len(papers), p.title[:60])

        session.commit()
        logger.info("Done: %d/%d papers", processed, len(papers))
        return {"extracted": processed, "candidates": len(papers)}

    except Exception as e:
        logger.error("Extraction failed: %s", str(e))
        session.rollback()
        raise
    finally:
        session.close()