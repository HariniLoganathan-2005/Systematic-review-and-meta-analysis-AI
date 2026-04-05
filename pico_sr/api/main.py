from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pico_sr.db.models import Diff, Extraction, Paper, Run, get_session, init_db
from pico_sr.pipeline.extract import run_extraction
from pico_sr.pipeline.living_review import run_living_review
from pico_sr.pipeline.screen import run_screening
from pico_sr.pipeline.search import run_search
from pico_sr.pipeline.stats import run_analysis
from pico_sr.pipeline.scheduler import shutdown_scheduler, start_living_review_job
from pico_sr.llm_client import LLMConfigError, LLMTransportError, health_llm_sync
from pico_sr.reports.export import build_pdf_report

executor = ThreadPoolExecutor(max_workers=2)
ROOT = Path(__file__).resolve().parent.parent.parent


async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(fn, *args, **kwargs))


OLLAMA_SETUP = (
    "Cannot connect to Ollama. Install from https://ollama.com, start the Ollama application "
    "(so the local API is listening), then run: ollama pull llama3.1:8b"
)


async def run_in_thread_llm(fn, *args, **kwargs):
    """Run pipeline in a thread; map LLM failures to clear HTTP errors."""
    try:
        return await run_in_thread(fn, *args, **kwargs)
    except LLMConfigError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "llm_config", "message": str(e)},
        ) from e
    except LLMTransportError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_unavailable",
                "message": e.message,
                "technical": e.technical or str(e),
            },
        ) from e
    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ollama_unavailable",
                "message": OLLAMA_SETUP,
                "technical": str(e),
            },
        ) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    shutdown_scheduler()
    executor.shutdown(wait=False)


app = FastAPI(title="PICO-SR API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/llm")
async def health_llm():
    """Check configured LLM (Ollama or Groq)."""
    body = await run_in_thread(health_llm_sync)
    if not body.get("ok"):
        raise HTTPException(
            status_code=503,
            detail=body,
        )
    return body


@app.get("/health/ollama")
async def health_ollama():
    """Backward-compatible alias: same as /health/llm when using Ollama."""
    return await health_llm()


class SearchBody(BaseModel):
    query: str
    retmax_pubmed: int = 80
    fetch_pdfs: bool = True


class ScreenBody(BaseModel):
    paper_ids: list[int] | None = None
    pico_criteria: str = Field(
        default="Include RCTs and quasi-experimental studies on the intervention vs comparator reporting quantitative outcomes relevant to the review question."
    )


class ExtractBody(BaseModel):
    paper_ids: list[int] | None = None


class AnalyseBody(BaseModel):
    paper_ids: list[int] | None = None


class LivingTriggerBody(BaseModel):
    query: str
    rss_term: str | None = None
    fetch_pdfs: bool = True


class ScheduleBody(BaseModel):
    query: str
    rss_term: str | None = None
    hours: int = 24


class PatchExtractionBody(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    hitl_pending: bool = False


@app.post("/search")
async def search_endpoint(body: SearchBody):
    return await run_in_thread(
        run_search, body.query, body.retmax_pubmed, body.fetch_pdfs
    )


@app.post("/screen")
async def screen_endpoint(body: ScreenBody):
    return await run_in_thread_llm(
        run_screening, body.paper_ids, body.pico_criteria
    )


@app.post("/extract")
async def extract_endpoint(body: ExtractBody):
    return await run_in_thread_llm(run_extraction, body.paper_ids)


@app.post("/analyse")
async def analyse_endpoint(body: AnalyseBody):
    return await run_in_thread_llm(run_analysis, body.paper_ids)


@app.get("/living-review/diff")
async def living_diff():
    session = get_session()
    try:
        d = session.query(Diff).order_by(Diff.id.desc()).first()
        if not d:
            return {}
        summary = {}
        if d.summary_json:
            try:
                summary = json.loads(d.summary_json)
            except json.JSONDecodeError:
                pass
        return {
            "id": d.id,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "delta_pooled_d": d.delta_pooled_d,
            "delta_ci_lower": d.delta_ci_lower,
            "delta_ci_upper": d.delta_ci_upper,
            "delta_i_squared": d.delta_i_squared,
            "new_papers_added": d.new_papers_added,
            "new_pmids": d.new_pmids,
            "summary": summary,
        }
    finally:
        session.close()


@app.post("/living-review/trigger")
async def living_trigger(body: LivingTriggerBody):
    return await run_in_thread_llm(
        run_living_review, body.query, body.rss_term, body.fetch_pdfs
    )


@app.post("/living-review/schedule")
async def living_schedule(body: ScheduleBody):
    await run_in_thread(
        start_living_review_job, body.query, body.rss_term, body.hours
    )
    return {"status": "scheduled", "hours": body.hours}


@app.get("/extractions")
async def list_extractions():
    """Latest extraction per included paper."""
    session = get_session()
    try:
        papers = (
            session.query(Paper)
            .filter(Paper.include_for_extraction.is_(True))
            .order_by(Paper.id.desc())
            .limit(200)
            .all()
        )
        out = []
        for paper in papers:
            ext = (
                session.query(Extraction)
                .filter(Extraction.paper_id == paper.id)
                .order_by(Extraction.id.desc())
                .first()
            )
            pl = ext.payload() if ext else {}
            out.append(
                {
                    "paper_id": paper.id,
                    "title": (paper.title or "")[:120],
                    "effect_size": pl.get("effect_size"),
                    "effect_type": pl.get("effect_type"),
                    "ci_lower": pl.get("ci_lower"),
                    "ci_upper": pl.get("ci_upper"),
                    "population_n": pl.get("population_n"),
                    "hitl_pending": ext.hitl_pending if ext else False,
                }
            )
        return {"extractions": out}
    finally:
        session.close()


@app.get("/papers")
async def list_papers():
    session = get_session()
    try:
        rows = session.query(Paper).order_by(Paper.id.desc()).limit(500).all()
        return {
            "papers": [
                {
                    "id": p.id,
                    "pmid": p.pmid,
                    "doi": p.doi,
                    "title": p.title,
                    "abstract": (p.abstract or "")[:2000],
                    "year": p.year,
                    "source": p.source,
                    "screen_decision": p.screen_decision,
                    "screen_confidence": p.screen_confidence,
                    "screen_reason": p.screen_reason,
                    "uncertain_review": p.uncertain_review,
                    "include_for_extraction": p.include_for_extraction,
                    "pdf_path": p.pdf_path,
                }
                for p in rows
            ]
        }
    finally:
        session.close()


@app.get("/extractions/{paper_id}")
async def get_extraction(paper_id: int):
    session = get_session()
    try:
        ex = (
            session.query(Extraction)
            .filter(Extraction.paper_id == paper_id)
            .order_by(Extraction.id.desc())
            .first()
        )
        if not ex:
            raise HTTPException(404, "No extraction")
        return {
            "paper_id": paper_id,
            "payload": ex.payload(),
            "validation_flags": ex.validation_flags(),
            "hitl_pending": ex.hitl_pending,
        }
    finally:
        session.close()


@app.patch("/extractions/{paper_id}")
async def patch_extraction(paper_id: int, body: PatchExtractionBody):
    session = get_session()
    try:
        ex = (
            session.query(Extraction)
            .filter(Extraction.paper_id == paper_id)
            .order_by(Extraction.id.desc())
            .first()
        )
        if not ex:
            raise HTTPException(404, "No extraction")
        pl = ex.payload()
        pl.update(body.payload)
        ex.payload_json = json.dumps(pl)
        ex.hitl_pending = body.hitl_pending
        session.commit()
        return {"ok": True}
    finally:
        session.close()


@app.get("/runs/latest")
async def latest_run():
    session = get_session()
    try:
        r = session.query(Run).order_by(Run.id.desc()).first()
        if not r:
            return {}
        meta = {}
        if r.meta_json:
            try:
                meta = json.loads(r.meta_json)
            except json.JSONDecodeError:
                pass
        return {
            "id": r.id,
            "pooled_d": r.pooled_d,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "i_squared": r.i_squared,
            "tau_squared": r.tau_squared,
            "egger_p": r.egger_p,
            "forest_plot_path": r.forest_plot_path,
            "n_studies": r.n_studies,
            "meta": meta,
        }
    finally:
        session.close()


@app.get("/forest-image")
async def forest_image(path: str):
    p = Path(path).resolve()
    root = ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(403, "Invalid path")
    if not p.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(p, media_type="image/png")


@app.get("/report/pdf")
async def report_pdf():
    out = await run_in_thread(build_pdf_report)
    if not out.get("path"):
        raise HTTPException(400, out.get("error", "report failed"))
    return FileResponse(
        out["path"], media_type="application/pdf", filename="pico_sr_report.pdf"
    )
