from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import requests
from rapidfuzz import fuzz

from pico_sr.config import settings
from pico_sr.db.models import Paper, get_session, init_db

logger = logging.getLogger(__name__)

PUBMED_ESearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFetch  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OPENALEX       = "https://api.openalex.org/works"
UNPAYWALL      = "https://api.unpaywall.org/v2"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMC_IDCONV     = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PMC_OAI        = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"


@dataclass
class RawPaper:
    pmid:     str | None
    doi:      str | None
    title:    str
    abstract: str | None
    year:     int | None
    source:   str
    full_text: str | None = field(default=None)  # full text if available


def _ncbi_params(extra: dict) -> dict:
    p = {"tool": "pico_sr", "email": settings.unpaywall_email, **extra}
    if settings.ncbi_api_key:
        p["api_key"] = settings.ncbi_api_key
    return p


# ── Full-text fetchers ────────────────────────────────────────────────────────

def _strip_xml_tags(xml_text: str) -> str:
    """Remove XML/HTML tags and clean whitespace."""
    text = re.sub(r"<[^>]+>", " ", xml_text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_europepmc_fulltext(pmid: str) -> str | None:
    """
    Europe PMC full-text XML — free, no key needed.
    Returns clean plain text or None.
    """
    try:
        r = requests.get(
            f"{EUROPEPMC_BASE}/{pmid}/fullTextXML",
            timeout=60,
        )
        if r.status_code == 200 and len(r.text) > 500:
            text = _strip_xml_tags(r.text)
            if len(text) > 200:
                logger.info(
                    "EuropePMC full text: PMID %s → %d chars", pmid, len(text)
                )
                return text[:100_000]
    except Exception as e:
        logger.debug("EuropePMC failed for PMID %s: %s", pmid, e)
    return None


def fetch_pmc_fulltext(pmid: str) -> str | None:
    """
    Try PMC OAI first, fall back to EuropePMC.
    Returns plain text or None.
    """
    try:
        # Step 1: PMID → PMCID
        r = requests.get(
            PMC_IDCONV,
            params={"ids": pmid, "format": "json"},
            timeout=30,
        )
        if r.status_code != 200:
            return fetch_europepmc_fulltext(pmid)

        records = r.json().get("records", [])
        if not records:
            return fetch_europepmc_fulltext(pmid)

        pmcid = records[0].get("pmcid")
        if not pmcid:
            return fetch_europepmc_fulltext(pmid)

        # Step 2: Fetch full text via OAI
        numeric_id = pmcid.replace("PMC", "")
        r2 = requests.get(
            PMC_OAI,
            params={
                "verb": "GetRecord",
                "identifier": f"oai:pubmedcentral.nih.gov:{numeric_id}",
                "metadataPrefix": "pmc",
            },
            timeout=60,
        )
        if r2.status_code == 200 and len(r2.text) > 500:
            text = _strip_xml_tags(r2.text)
            if len(text) > 200:
                logger.info(
                    "PMC OAI full text: %s → %d chars", pmcid, len(text)
                )
                return text[:100_000]

        # Step 3: Fall back to EuropePMC
        return fetch_europepmc_fulltext(pmid)

    except Exception as e:
        logger.debug("PMC fetch failed for PMID %s: %s", pmid, e)
        return fetch_europepmc_fulltext(pmid)


def fetch_semantic_scholar_text(doi: str) -> str | None:
    """
    Semantic Scholar abstract + tldr.
    Better than nothing when PMC has nothing.
    """
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "title,abstract,tldr"},
            timeout=30,
        )
        if r.status_code == 200:
            data     = r.json()
            abstract = data.get("abstract") or ""
            tldr     = (data.get("tldr") or {}).get("text") or ""
            combined = f"{abstract}\n{tldr}".strip()
            if len(combined) > 100:
                logger.info(
                    "Semantic Scholar text: DOI %s → %d chars", doi, len(combined)
                )
                return combined
    except Exception as e:
        logger.debug("Semantic Scholar failed for DOI %s: %s", doi, e)
    return None


def fetch_pdf_if_oa(doi: str | None) -> str | None:
    """Unpaywall PDF download — last resort."""
    doi = normalize_doi(doi)
    if not doi:
        return None
    try:
        r = requests.get(
            f"{UNPAYWALL}/{doi}",
            params={"email": settings.unpaywall_email},
            timeout=45,
        )
        if r.status_code != 200:
            return None
        best = r.json().get("best_oa_location")
        if not best:
            return None
        pdf_url = best.get("url_for_pdf") or best.get("url_for_landing_page")
        if not pdf_url:
            return None
        safe = re.sub(r"[^\w\-.]+", "_", doi)[:180]
        dest = settings.pdf_dir / f"{safe}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pr = requests.get(pdf_url, timeout=120, allow_redirects=True)
        pr.raise_for_status()
        content_type = (pr.headers.get("content-type") or "").lower()
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            if len(pr.content) < 5000:
                return None
        dest.write_bytes(pr.content)
        logger.info("Unpaywall PDF saved: %s", dest)
        return str(dest.resolve())
    except Exception as e:
        logger.debug("Unpaywall failed for DOI %s: %s", doi, e)
        return None


def get_best_text(pmid: str | None, doi: str | None, abstract: str | None) -> tuple[str | None, str | None]:
    """
    Try all sources to get the richest text.
    Returns (best_text, pdf_path)

    Priority:
    1. PMC full text (best — has results section with numbers)
    2. Europe PMC full text
    3. Semantic Scholar (abstract + summary)
    4. Unpaywall PDF
    5. Original abstract (fallback)
    """
    full_text = None
    pdf_path  = None

    # 1. Try PMC full text
    if pmid:
        full_text = fetch_pmc_fulltext(pmid)
        if full_text:
            logger.info("✅ PMC full text for PMID %s", pmid)
            return full_text, None

    # 2. Try Semantic Scholar
    if doi:
        ss_text = fetch_semantic_scholar_text(doi)
        if ss_text and len(ss_text) > (len(abstract or "") + 100):
            # Only use if it adds more than abstract
            full_text = ss_text
            logger.info("✅ Semantic Scholar text for DOI %s", doi)
            return full_text, None

    # 3. Try Unpaywall PDF
    if doi:
        pdf_path = fetch_pdf_if_oa(doi)
        if pdf_path:
            logger.info("✅ Unpaywall PDF for DOI %s", doi)
            return None, pdf_path  # pdf_path handled by extract.py

    # 4. Fall back to original abstract
    if abstract:
        logger.info("⚠️  Using abstract only for PMID=%s DOI=%s", pmid, doi)
        return abstract, None

    logger.warning("❌ No text found for PMID=%s DOI=%s", pmid, doi)
    return None, None


# ── PubMed search ─────────────────────────────────────────────────────────────

def pubmed_search(query: str, retmax: int = 20) -> list[RawPaper]:
    """
    Search PubMed.
    First tries with pmc[sb] filter (open access only).
    Falls back to full search if no results.
    """
    sess = requests.Session()

    # Try open-access filter first → higher chance of getting full text
    for term in [f"({query}) AND pmc[sb]", query]:
        try:
            esearch = sess.get(
                PUBMED_ESearch,
                params=_ncbi_params({
                    "db": "pubmed", "term": term,
                    "retmax": retmax, "retmode": "json"
                }),
                timeout=60,
            )
            esearch.raise_for_status()
            ids = esearch.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                logger.info(
                    "PubMed search '%s' → %d results", term[:60], len(ids)
                )
                break
        except Exception as e:
            logger.warning("PubMed esearch failed: %s", e)
            ids = []

    if not ids:
        return []

    time.sleep(0.11 if settings.ncbi_api_key else 0.35)

    try:
        efetch = sess.get(
            PUBMED_EFetch,
            params=_ncbi_params({
                "db": "pubmed", "id": ",".join(ids), "retmode": "xml"
            }),
            timeout=120,
        )
        efetch.raise_for_status()
        return _parse_pubmed_efetch_xml(efetch.content, source="pubmed")
    except Exception as e:
        logger.warning("PubMed efetch failed: %s", e)
        return []


def pubmed_fetch_by_pmids(pmids: list[str], source: str = "pubmed") -> list[RawPaper]:
    if not pmids:
        return []
    sess = requests.Session()
    out: list[RawPaper] = []
    for i in range(0, len(pmids), 80):
        chunk = pmids[i: i + 80]
        time.sleep(0.11 if settings.ncbi_api_key else 0.35)
        try:
            efetch = sess.get(
                PUBMED_EFetch,
                params=_ncbi_params({
                    "db": "pubmed", "id": ",".join(chunk), "retmode": "xml"
                }),
                timeout=120,
            )
            efetch.raise_for_status()
            out.extend(_parse_pubmed_efetch_xml(efetch.content, source=source))
        except Exception as e:
            logger.warning("Batch PMIDs fetch failed: %s", e)
    return out


def _parse_pubmed_efetch_xml(content: bytes, source: str) -> list[RawPaper]:
    root = ET.fromstring(content)
    out: list[RawPaper] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el  = article.find(".//PMID")
        pmid     = pmid_el.text if pmid_el is not None else None
        title_el = article.find(".//ArticleTitle")
        title    = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = ""
        if abstract_parts:
            abstract = "\n".join(
                "".join(a.itertext())
                for a in abstract_parts if a is not None
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

        out.append(RawPaper(
            pmid=pmid, doi=doi,
            title=title.strip(),
            abstract=abstract or None,
            year=year, source=source,
        ))
    return out


# ── OpenAlex search ───────────────────────────────────────────────────────────

def openalex_search(query: str, per_page: int = 20, max_pages: int = 1) -> list[RawPaper]:
    sess   = requests.Session()
    out:   list[RawPaper] = []
    cursor = "*"

    for _ in range(max_pages):
        try:
            r = sess.get(
                OPENALEX,
                params={
                    "search":   query,
                    "per_page": min(per_page, 200),
                    "cursor":   cursor,
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("OpenAlex search failed: %s", e)
            break

        for w in data.get("results", []):
            title = (w.get("display_name") or "").strip()
            if not title:
                continue

            year     = w.get("publication_year")
            abstract = None
            ia       = w.get("abstract_inverted_index")
            if isinstance(ia, dict) and ia:
                words: list[tuple[int, str]] = []
                for word, positions in ia.items():
                    for pos in positions:
                        words.append((pos, word))
                words.sort(key=lambda x: x[0])
                abstract = " ".join(wd for _, wd in words)

            doi  = None
            raw_doi = w.get("doi")
            if raw_doi:
                doi = raw_doi.replace("https://doi.org/", "").strip().lower()

            pmid = None
            for ext in (w.get("ids") or {}).values():
                if isinstance(ext, str) and ext.startswith("pubmed:"):
                    pmid = ext.split(":")[-1]

            out.append(RawPaper(
                pmid=pmid, doi=doi, title=title,
                abstract=abstract, year=year, source="openalex",
            ))

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    logger.info("OpenAlex search → %d results", len(out))
    return out


# ── Dedup helpers ─────────────────────────────────────────────────────────────

def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d or None


def normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def merge_dedupe(
    items: list[RawPaper],
    existing_titles: set[str],
    threshold: float = 92.0,
) -> list[RawPaper]:
    seen_doi:   set[str]   = set()
    merged:     list[RawPaper] = []
    title_keys: list[str]  = []

    for rp in items:
        nd = normalize_doi(rp.doi)
        if nd:
            if nd in seen_doi:
                continue
            seen_doi.add(nd)
            merged.append(rp)
            title_keys.append(normalize_title(rp.title))
            continue

        nt  = normalize_title(rp.title)
        dup = any(
            fuzz.token_sort_ratio(nt, prev) >= threshold
            for prev in title_keys
        )
        if dup or nt in existing_titles:
            continue
        merged.append(rp)
        title_keys.append(nt)

    return merged


# ── Main entry point ──────────────────────────────────────────────────────────

def run_search(
    query: str,
    retmax_pubmed: int = 20,
    fetch_pdfs: bool = True,
) -> dict[str, Any]:
    init_db()
    session = get_session()
    try:
        existing        = session.query(Paper).all()
        existing_titles = {normalize_title(p.title) for p in existing}

        logger.info("Searching PubMed and OpenAlex for: %s", query)
        pubmed   = pubmed_search(query, retmax=retmax_pubmed)
        oa       = openalex_search(query, per_page=retmax_pubmed, max_pages=1)
        combined = merge_dedupe(pubmed + oa, existing_titles)

        logger.info(
            "Search results: pubmed=%d openalex=%d combined_new=%d",
            len(pubmed), len(oa), len(combined)
        )

        added     = 0
        paper_ids: list[int] = []
        text_sources = {"pmc": 0, "semantic_scholar": 0, "unpaywall": 0, "abstract": 0, "none": 0}

        for rp in combined:
            nd = normalize_doi(rp.doi)

            # Skip if already in DB
            if nd and session.query(Paper).filter(Paper.doi == nd).first():
                continue

            # Get best available text
            best_text, pdf_path = get_best_text(
                pmid=rp.pmid,
                doi=nd,
                abstract=rp.abstract,
            )

            # Track text source for logging
            if pdf_path:
                text_sources["unpaywall"] += 1
            elif best_text and best_text != rp.abstract:
                if rp.pmid:
                    text_sources["pmc"] += 1
                else:
                    text_sources["semantic_scholar"] += 1
            elif best_text:
                text_sources["abstract"] += 1
            else:
                text_sources["none"] += 1

            row = Paper(
                pmid     = rp.pmid,
                doi      = nd,
                title    = rp.title,
                # Store the richest text we found in abstract field
                # extract.py will use this for extraction
                abstract = best_text or rp.abstract,
                year     = rp.year,
                source   = rp.source,
                pdf_path = pdf_path,
            )
            session.add(row)
            session.flush()
            added += 1
            paper_ids.append(row.id)

            # Be polite to APIs
            time.sleep(0.5)

        session.commit()
        total = session.query(Paper).count()

        logger.info(
            "Search done: added=%d total=%d text_sources=%s",
            added, total, text_sources
        )

        note = None
        if added == 0:
            if not pubmed and not oa:
                note = "No results from PubMed or OpenAlex. Try broader keywords."
            else:
                note = (
                    f"No new papers added — all already in DB. "
                    f"You have {total} papers stored. "
                    "Try a different query or clear the database."
                )

        return {
            "query":        query,
            "papers_added": added,
            "paper_ids":    paper_ids,
            "total_in_db":  total,
            "note":         note,
            "stats": {
                "pubmed_hits":            len(pubmed),
                "openalex_hits":          len(oa),
                "candidates_after_dedup": len(combined),
                "text_sources":           text_sources,
            },
        }
    finally:
        session.close()