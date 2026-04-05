from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

import feedparser

from pico_sr.db.models import Diff, LivingReviewState, Paper, Run, get_session, init_db
from pico_sr.pipeline.extract import run_extraction
from pico_sr.pipeline.screen import run_screening
from pico_sr.pipeline.search import (
    fetch_pdf_if_oa,
    merge_dedupe,
    normalize_doi,
    normalize_title,
    pubmed_fetch_by_pmids,
)
from pico_sr.pipeline.stats import run_analysis

logger = logging.getLogger(__name__)


def parse_rss_pmids(feed_url: str) -> list[str]:
    parsed = feedparser.parse(feed_url)
    ids: list[str] = []
    for entry in parsed.entries:
        link = getattr(entry, "link", "") or ""
        m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", link)
        if m:
            ids.append(m.group(1))
            continue
        if hasattr(entry, "id"):
            m2 = re.search(r"pubmed/(\d+)", str(entry.id))
            if m2:
                ids.append(m2.group(1))
    return list(dict.fromkeys(ids))


def run_living_review(
    query: str,
    rss_term: str | None = None,
    fetch_pdfs: bool = True,
) -> dict[str, Any]:
    """
    Poll PubMed RSS for new PMIDs vs DB; ingest; screen; extract; analyse; diff vs previous run.
    """
    init_db()
    session = get_session()
    try:
        term = rss_term or query
        feed_url = (
            f"https://pubmed.ncbi.nlm.nih.gov/rss/search/?term={quote(term)}&format=atom"
        )
        pmids = parse_rss_pmids(feed_url)
        rows = session.query(Paper.pmid).filter(Paper.pmid.isnot(None)).all()
        existing_pmids = {r[0] for r in rows if r[0]}
        new_pmids = [p for p in pmids if p and p not in existing_pmids]

        title_rows = session.query(Paper.title).all()
        existing_titles = {normalize_title(r[0] or "") for r in title_rows}

        papers_meta = pubmed_fetch_by_pmids(new_pmids, source="pubmed_rss") if new_pmids else []
        merged = merge_dedupe(papers_meta, existing_titles)

        new_ids: list[int] = []
        for rp in merged:
            nd = normalize_doi(rp.doi)
            if nd and session.query(Paper).filter(Paper.doi == nd).first():
                continue
            pdf_path = None
            if fetch_pdfs and nd:
                pdf_path = fetch_pdf_if_oa(rp.doi)
            row = Paper(
                pmid=rp.pmid,
                doi=nd,
                title=rp.title,
                abstract=rp.abstract,
                year=rp.year,
                source=rp.source,
                pdf_path=pdf_path,
            )
            session.add(row)
            session.flush()
            new_ids.append(row.id)

        session.commit()

        st = session.query(LivingReviewState).first()
        if st is None:
            st = LivingReviewState(query=query)
            session.add(st)
        st.query = query
        st.last_poll_at = datetime.utcnow()
        session.commit()

        if not new_ids:
            return {
                "new_pmids": new_pmids,
                "papers_added": 0,
                "message": "no_new_papers",
                "diff_id": None,
            }

        prev_run = session.query(Run).order_by(Run.id.desc()).first()

        run_screening(paper_ids=new_ids)
        run_extraction(paper_ids=new_ids)

        analysis = run_analysis()
        if analysis.get("error"):
            logger.warning("Analysis: %s", analysis)

        new_run = session.query(Run).order_by(Run.id.desc()).first()

        diff_row = None
        if prev_run and new_run and prev_run.id != new_run.id:
            diff_row = Diff(
                run_old_id=prev_run.id,
                run_new_id=new_run.id,
                delta_pooled_d=(new_run.pooled_d or 0) - (prev_run.pooled_d or 0)
                if new_run.pooled_d is not None and prev_run.pooled_d is not None
                else None,
                delta_ci_lower=(new_run.ci_lower or 0) - (prev_run.ci_lower or 0)
                if new_run.ci_lower is not None and prev_run.ci_lower is not None
                else None,
                delta_ci_upper=(new_run.ci_upper or 0) - (prev_run.ci_upper or 0)
                if new_run.ci_upper is not None and prev_run.ci_upper is not None
                else None,
                delta_i_squared=(new_run.i_squared or 0) - (prev_run.i_squared or 0)
                if new_run.i_squared is not None and prev_run.i_squared is not None
                else None,
                new_papers_added=len(new_ids),
                new_pmids=json.dumps(new_pmids[:200]),
                new_contradictions=0,
                summary_json=json.dumps(
                    {
                        "old": {
                            "pooled_d": prev_run.pooled_d,
                            "ci": [prev_run.ci_lower, prev_run.ci_upper],
                            "forest": prev_run.forest_plot_path,
                        },
                        "new": {
                            "pooled_d": new_run.pooled_d,
                            "ci": [new_run.ci_lower, new_run.ci_upper],
                            "forest": new_run.forest_plot_path,
                        },
                    }
                ),
            )
            session.add(diff_row)

        st = session.query(LivingReviewState).first()
        if st and new_run:
            st.last_forest_path = new_run.forest_plot_path
        session.commit()

        return {
            "new_pmids": new_pmids,
            "papers_added": len(new_ids),
            "diff_id": diff_row.id if diff_row else None,
            "latest_run_id": new_run.id if new_run else None,
            "analysis": analysis,
        }
    finally:
        session.close()
