from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from pico_sr.ui import hitl as hitl_mod

st.set_page_config(page_title="PICO-SR", layout="wide")
st.session_state.setdefault("api_base", os.environ.get("PICO_SR_API", "http://127.0.0.1:8000"))

API = st.session_state["api_base"]


def get(path: str, **params):
    return requests.get(f"{API}{path}", params=params, timeout=120)


def post(path: str, json: dict | None = None):
    return requests.post(f"{API}{path}", json=json or {}, timeout=600)


def show_api_response(r: requests.Response) -> None:
    """Show JSON success or parse FastAPI error bodies (including 503 Ollama)."""
    if r.ok:
        st.json(r.json())
        return
    try:
        body = r.json()
    except Exception:
        st.error(r.text or f"HTTP {r.status_code}")
        return
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and detail.get("message"):
        st.error(detail["message"])
        st.json(body)
    elif detail is not None:
        st.error(str(detail))
        if isinstance(body, dict):
            st.json(body)
    else:
        st.json(body)


st.title("PICO-SR — Systematic review & meta-analysis")
st.caption(
    "Backend: FastAPI · LLM: Ollama (local) or Groq (cloud) — set LLM_PROVIDER and GROQ_API_KEY in `.env`"
)

with st.sidebar:
    st.markdown("**LLM check**")
    if st.button("Ping LLM (Ollama or Groq)"):
        h = get("/health/llm")
        if h.ok:
            st.success("LLM backend OK.")
            st.json(h.json())
        else:
            st.error("LLM check failed. For Ollama: start app + `ollama pull`. For Groq: set GROQ_API_KEY + LLM_PROVIDER=groq.")
            try:
                st.json(h.json())
            except Exception:
                st.text(h.text)

tab_search, tab_review, tab_extract, tab_forest, tab_living = st.tabs(
    ["Search", "Review queue", "Extraction table", "Forest plot", "Living review"]
)

with tab_search:
    q = st.text_input("Research question / search query", value="exercise depression randomized trial")
    c1, c2 = st.columns(2)
    with c1:
        retmax = st.number_input("PubMed max results", value=40, min_value=5, max_value=200)
    with c2:
        fetch_pdfs = st.checkbox("Fetch OA PDFs (Unpaywall)", value=False)
    if st.button("Run search"):
        with st.spinner("Searching PubMed + OpenAlex…"):
            r = post("/search", {"query": q, "retmax_pubmed": int(retmax), "fetch_pdfs": fetch_pdfs})
        if r.ok:
            st.json(r.json())
        else:
            st.error(r.text)
    st.divider()
    pico = st.text_area(
        "PICO criteria for screening",
        value="Include RCTs on exercise vs control for depression reporting quantitative outcomes.",
        height=100,
    )
    if st.button("Run screening"):
        r = post("/screen", {"pico_criteria": pico})
        show_api_response(r)
        st.toast("Screening started in the background!", icon="⏳")
        st.success("Screening is now running in the background. The table below will update as papers are processed.")
        
    st.divider()
    c1, c2 = st.columns([1, 5])
    with c1:
        st.button("🔄 Refresh table")
    with c2:
        st.caption("Latest papers in database. Click refresh to see background screening progress.")
        
    r = get("/papers")
    if r.ok:
        papers = r.json().get("papers", [])
        if papers:
            pending_count = sum(1 for p in papers if not p.get("screen_decision"))
            if pending_count > 0:
                st.info(f"⏳ **{pending_count} papers are currently pending screening.** If you started the process, it is running in the background. Refresh periodically.")
            else:
                st.success("✅ All papers have been screened.")
            
            st.dataframe(pd.DataFrame(papers), use_container_width=True)
        else:
            st.info("No papers yet. Run a search to add papers to the database.")
    else:
        st.error(r.text)

with tab_review:
    r = get("/papers")
    if not r.ok:
        st.error("Could not load papers — is the API running?")
    elif not r.json().get("papers"):
        st.info("No papers in database yet. Run a search from the Search tab.")
    else:
        papers = r.json().get("papers", [])
        df = pd.DataFrame(papers)

        if len(df):
            st.dataframe(df, use_container_width=True)
        ids = [p["id"] for p in papers if p.get("uncertain_review") or p.get("screen_decision") == "uncertain"]
        chosen = st.selectbox("Open HITL for paper ID", [""] + [str(i) for i in ids])
        if chosen:
            pid = int(chosen)
            paper = next((p for p in papers if p["id"] == pid), None)
            snippet = ""
            if paper:
                snippet = (paper.get("title") or "") + "\n\n" + (paper.get("abstract") or "")
                
                st.subheader(f"Screening Decision (Current: {paper.get('screen_decision')})")
                if paper.get("screen_reason"):
                    st.info(f"Reason: {paper.get('screen_reason')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Mark as Include", key=f"include_{pid}"):
                        requests.patch(
                            f"{API}/papers/{pid}",
                            json={"screen_decision": "include", "include_for_extraction": True, "uncertain_review": False},
                            timeout=60
                        )
                        st.rerun()
                with col2:
                    if st.button("Mark as Exclude", key=f"exclude_{pid}"):
                        requests.patch(
                            f"{API}/papers/{pid}",
                            json={"screen_decision": "exclude", "include_for_extraction": False, "uncertain_review": False},
                            timeout=60
                        )
                        st.rerun()
                        
            hitl_mod.render_hitl_panel(pid, snippet)

with tab_extract:
    st.caption("Included papers and latest numeric extraction fields.")
    er = get("/extractions")
    if er.ok:
        ex = er.json().get("extractions", [])
        st.dataframe(pd.DataFrame(ex), use_container_width=True)
    else:
        st.error(er.text)

with tab_forest:
    if st.button("Run extraction (included papers)"):
        r = post("/extract", {})
        show_api_response(r)
        st.info("Extraction is processing in the background to avoid session timeouts. Refresh this tab later to see your new data.")
    if st.button("Run analysis"):
        with st.spinner("Pooling effects & drawing forest plot…"):
            r = post("/analyse", {})
        show_api_response(r)
    lr = get("/runs/latest")
    if lr.ok:
        meta = lr.json()
        if meta.get("forest_plot_path"):
            st.subheader("Results")
            st.write(
                f"Pooled d = {meta.get('pooled_d')}, "
                f"95% CI [{meta.get('ci_lower')}, {meta.get('ci_upper')}], "
                f"I² = {meta.get('i_squared')}, τ² = {meta.get('tau_squared')}"
            )
            path = meta["forest_plot_path"]
            try:
                import matplotlib.image as mpimg

                st.image(mpimg.imread(path), use_container_width=True)
            except Exception:
                st.warning("Could not display image from path")
        else:
            st.info("No analysis run yet.")
    import time
    report_url = f"{API}/report/pdf?t={int(time.time())}"
    st.markdown(f"[Download PDF report]({report_url})")

with tab_living:
    lq = st.text_input("Living review query (PubMed RSS)", value="exercise depression meta-analysis")
    if st.button("Trigger living review poll"):
        with st.spinner("Polling RSS & updating pipeline…"):
            r = post("/living-review/trigger", {"query": lq})
        show_api_response(r)
    dr = get("/living-review/diff")
    if dr.ok:
        d = dr.json()
        if d:
            st.subheader("Latest diff")
            st.json(d)
    lr = get("/runs/latest")
    if lr.ok:
        m = lr.json()
        if m.get("forest_plot_path"):
            st.subheader("Current forest plot")
            try:
                import matplotlib.image as mpimg

                st.image(mpimg.imread(m["forest_plot_path"]), use_container_width=True)
            except Exception:
                pass
