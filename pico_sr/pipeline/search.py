from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import requests
from rapidfuzz import fuzz

from pico_sr.config import settings
from pico_sr.db.models import Paper, get_session, init_db

logger = logging.getLogger(__name__)

PUBMED_ESearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OPENALEX = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2"


@dataclass
class RawPaper:
    pmid: str | None
    doi: str | None
    title: str
    abstract: str | None
    year: int | None
    source: str


def _ncbi_params(extra: dict) -> dict:
    p = {"tool": "pico_sr", "email": settings.unpaywall_email, **extra}
    if settings.ncbi_api_key:
        p["api_key"] = settings.ncbi_api_key
    return p


def pubmed_search(query: str, retmax: int = 10) -> list[RawPaper]:
    sess = requests.Session()
    esearch = sess.get(
        PUBMED_ESearch,
        params=_ncbi_params(
            {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"}
        ),
        timeout=60,
    )
    esearch.raise_for_status()
    ids = esearch.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    time.sleep(0.11 if settings.ncbi_api_key else 0.35)
    efetch = sess.get(
        PUBMED_EFetch,
        params=_ncbi_params(
            {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            }
        ),
        timeout=120,
    )
    efetch.raise_for_status()
    return _parse_pubmed_efetch_xml(efetch.content, source="pubmed")


def pubmed_fetch_by_pmids(pmids: list[str], source: str = "pubmed") -> list[RawPaper]:
    if not pmids:
        return []
    sess = requests.Session()
    out: list[RawPaper] = []
    batch = 80
    for i in range(0, len(pmids), batch):
        chunk = pmids[i : i + batch]
        time.sleep(0.11 if settings.ncbi_api_key else 0.35)
        efetch = sess.get(
            PUBMED_EFetch,
            params=_ncbi_params(
                {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}
            ),
            timeout=120,
        )
        efetch.raise_for_status()
        out.extend(_parse_pubmed_efetch_xml(efetch.content, source=source))
    return out


def _parse_pubmed_efetch_xml(content: bytes, source: str) -> list[RawPaper]:
    root = ET.fromstring(content)
    out: list[RawPaper] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        abstract_parts = article.findall(".//AbstractText")
        abstract = ""
        if abstract_parts:
            abstract = "\n".join(
                "".join(a.itertext()) for a in abstract_parts if a is not None
            )
        year = None
        for path in (".//PubDate/Year", ".//ArticleDate/Year"):
            y = article.find(path)
            if y is not None and y.text:
                try:
                    year = int(y.text[:4])
                    break
                except ValueError:
                    pass
        doi = None
        for aid in article.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip().lower()
                break
        out.append(
            RawPaper(
                pmid=pmid,
                doi=doi,
                title=title.strip(),
                abstract=abstract or None,
                year=year,
                source=source,
            )
        )
    return out


def openalex_search(query: str, per_page: int = 10, max_pages: int = 1) -> list[RawPaper]:
    sess = requests.Session()
    out: list[RawPaper] = []
    cursor = "*"
    for _ in range(max_pages):
        r = sess.get(
            OPENALEX,
            params={
                "search": query,
                "per_page": min(per_page, 200),
                "cursor": cursor,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        for w in data.get("results", []):
            title = (w.get("display_name") or "").strip()
            if not title:
                continue
            year = w.get("publication_year")
            abstract = None
            ia = w.get("abstract_inverted_index")
            if isinstance(ia, dict) and ia:
                words: list[tuple[int, str]] = []
                for word, positions in ia.items():
                    for pos in positions:
                        words.append((pos, word))
                words.sort(key=lambda x: x[0])
                abstract = " ".join(w for _, w in words)
            doi = None
            ids = w.get("doi")
            if ids:
                doi = ids.replace("https://doi.org/", "").strip().lower()
            pmid = None
            for ext in w.get("ids", {}).values() if isinstance(w.get("ids"), dict) else []:
                if isinstance(ext, str) and ext.startswith("pubmed:"):
                    pmid = ext.split(":")[-1]
            out.append(
                RawPaper(
                    pmid=pmid,
                    doi=doi,
                    title=title,
                    abstract=abstract,
                    year=year,
                    source="openalex",
                )
            )
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return out


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d or None


def normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def merge_dedupe(
    items: list[RawPaper], existing_titles: set[str], threshold: float = 92.0
) -> list[RawPaper]:
    seen_doi: set[str] = set()
    merged: list[RawPaper] = []
    title_keys: list[str] = []
    for rp in items:
        nd = normalize_doi(rp.doi)
        if nd:
            if nd in seen_doi:
                continue
            seen_doi.add(nd)
            merged.append(rp)
            title_keys.append(normalize_title(rp.title))
            continue
        nt = normalize_title(rp.title)
        dup = False
        for prev in title_keys:
            if fuzz.token_sort_ratio(nt, prev) >= threshold:
                dup = True
                break
        if dup:
            continue
        if nt in existing_titles:
            continue
        merged.append(rp)
        title_keys.append(nt)
    return merged


def fetch_pdf_if_oa(doi: str | None) -> str | None:
    doi = normalize_doi(doi)
    if not doi:
        return None
    url = f"{UNPAYWALL}/{doi}"
    r = requests.get(
        url,
        params={"email": settings.unpaywall_email},
        timeout=45,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    best = data.get("best_oa_location")
    if not best:
        return None
    pdf_url = best.get("url_for_pdf") or best.get("url_for_landing_page")
    if not pdf_url:
        return None
    safe = re.sub(r"[^\w\-.]+", "_", doi)[:180]
    dest = settings.pdf_dir / f"{safe}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        pr = requests.get(pdf_url, timeout=120, allow_redirects=True)
        pr.raise_for_status()
        if "pdf" not in (pr.headers.get("content-type") or "").lower() and not pdf_url.lower().endswith(".pdf"):
            if len(pr.content) < 5000:
                return None
        dest.write_bytes(pr.content)
        return str(dest.resolve())
    except Exception as e:
        logger.warning("PDF fetch failed for %s: %s", doi, e)
        return None


def run_search(
    query: str,
    retmax_pubmed: int = 10,
    fetch_pdfs: bool = True,
) -> dict[str, Any]:
    init_db()
    session = get_session()
    try:
        existing = session.query(Paper).all()
        existing_titles = {normalize_title(p.title) for p in existing}
        pubmed = pubmed_search(query, retmax=retmax_pubmed)
        oa = openalex_search(query, per_page=retmax_pubmed, max_pages=1)
        combined = merge_dedupe(pubmed + oa, existing_titles)
        added = 0
        paper_ids: list[int] = []
        for rp in combined:
            nd = normalize_doi(rp.doi)
            q = session.query(Paper)
            if nd:
                existing_p = q.filter(Paper.doi == nd).first()
            else:
                existing_p = None
            if existing_p:
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
            added += 1
            paper_ids.append(row.id)
        session.commit()
        total = session.query(Paper).count()
        if added == 0:
            if len(pubmed) == 0 and len(oa) == 0:
                note = "No results returned from PubMed or OpenAlex for this query. Try broader keywords."
            else:
                note = (
                    "No new rows inserted: hits were already in the database (same DOI) or skipped as duplicates. "
                    f"You already have {total} papers stored from earlier searches."
                )
        else:
            note = None
        return {
            "query": query,
            "papers_added": added,
            "paper_ids": paper_ids,
            "total_in_db": total,
            "note": note,
            "stats": {
                "pubmed_hits": len(pubmed),
                "openalex_hits": len(oa),
                "candidates_after_dedup": len(combined),
            },
        }
    finally:
        session.close()