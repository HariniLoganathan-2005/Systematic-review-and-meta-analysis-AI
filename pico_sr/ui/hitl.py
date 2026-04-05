from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st


def api_base() -> str:
    return st.session_state.get("api_base", "http://127.0.0.1:8000")


def fetch_extraction(paper_id: int) -> dict[str, Any] | None:
    r = requests.get(f"{api_base()}/extractions/{paper_id}", timeout=120)
    if r.status_code != 200:
        return None
    return r.json()


def save_extraction(paper_id: int, payload: dict, hitl_pending: bool) -> bool:
    r = requests.patch(
        f"{api_base()}/extractions/{paper_id}",
        json={"payload": payload, "hitl_pending": hitl_pending},
        timeout=60,
    )
    return r.status_code == 200


def render_hitl_panel(paper_id: int, pdf_text_snippet: str) -> None:
    data = fetch_extraction(paper_id)
    if not data:
        st.warning("No extraction for this paper yet.")
        return
    payload = data.get("payload") or {}
    flags = data.get("validation_flags") or []
    st.subheader("Human review — extraction")
    if flags:
        st.warning("Validation flags: " + ", ".join(flags))
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Edit payload (JSON)**")
        txt = st.text_area(
            "payload",
            value=json.dumps(payload, indent=2, ensure_ascii=False),
            height=420,
            label_visibility="collapsed",
        )
    with cols[1]:
        st.markdown("**Context (title / abstract snippet)**")
        st.text_area("context", pdf_text_snippet[:12000], height=420, label_visibility="collapsed")
    if st.button("Save corrections", key=f"save_{paper_id}"):
        try:
            new_pl = json.loads(txt)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return
        if save_extraction(paper_id, new_pl, hitl_pending=False):
            st.success("Saved")
        else:
            st.error("Save failed")
