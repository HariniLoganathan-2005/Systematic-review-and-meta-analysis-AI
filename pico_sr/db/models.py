from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pmid: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(512), index=True, nullable=True)
    title: Mapped[str] = mapped_column(Text, default="")
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pdf_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    screen_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    screen_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    screen_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertain_review: Mapped[bool] = mapped_column(Boolean, default=False)
    include_for_extraction: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    extractions: Mapped[list["Extraction"]] = relationship(
        "Extraction", back_populates="paper", cascade="all, delete-orphan"
    )


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )

    # ── Numeric columns ─────────────────────────────────────────────
    # FIX: These were defined but NEVER written by extract.py
    # Now extract.py fills these so the UI table shows real values
    effect_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    effect_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ci_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    population_n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Descriptive columns ─────────────────────────────────────────
    population_desc: Mapped[str | None] = mapped_column(Text, nullable=True)
    intervention: Mapped[str | None] = mapped_column(Text, nullable=True)
    comparator: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome_measure: Mapped[str | None] = mapped_column(Text, nullable=True)
    study_design: Mapped[str | None] = mapped_column(String(64), nullable=True)
    followup_duration: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # ── Validation ──────────────────────────────────────────────────
    ci_quarantine: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    validation_flags_json: Mapped[str] = mapped_column(Text, default="[]")
    hitl_pending: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    paper: Mapped["Paper"] = relationship("Paper", back_populates="extractions")

    def payload(self) -> dict:
        try:
            return json.loads(self.payload_json)
        except json.JSONDecodeError:
            return {}

    @property
    def has_valid_effect(self) -> bool:
        return (
            self.effect_size is not None
            and self.ci_lower is not None
            and self.ci_upper is not None
        )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    pooled_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    i_squared: Mapped[float | None] = mapped_column(Float, nullable=True)
    tau_squared: Mapped[float | None] = mapped_column(Float, nullable=True)
    egger_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    forest_plot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    n_studies: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # FIX: text_report was missing — stats.py tried to save it and crashed
    text_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class Diff(Base):
    __tablename__ = "diffs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    run_old_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    run_new_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    delta_pooled_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_ci_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_ci_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_i_squared: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_papers_added: Mapped[int] = mapped_column(Integer, default=0)
    new_pmids: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_contradictions: Mapped[int] = mapped_column(Integer, default=0)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class LivingReviewState(Base):
    __tablename__ = "living_review_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, default="")
    last_poll_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_forest_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)


_engine = None
_SessionLocal = None


from sqlalchemy import event

def get_engine(database_url: str | None = None):
    global _engine
    if _engine is None:
        from pico_sr.config import settings
        url = database_url or settings.sqlalchemy_url
        _engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
        
        # Enable WAL mode for better concurrency, and increase typical wait times
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal()


def init_db(database_url: str | None = None) -> None:
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    root = Path(__file__).resolve().parent.parent.parent
    (root / "pdfs").mkdir(parents=True, exist_ok=True)
    (root / "output" / "forest").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)