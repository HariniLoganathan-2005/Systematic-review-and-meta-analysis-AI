from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from pico_sr.config import settings

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone="UTC")
    return _scheduler


def start_living_review_job(
    query: str,
    rss_term: str | None = None,
    hours: int = 24,
) -> None:
    from pico_sr.pipeline.living_review import run_living_review

    sched = get_scheduler()

    def job() -> None:
        try:
            run_living_review(query, rss_term=rss_term)
        except Exception as e:
            logger.exception("Living review job failed: %s", e)

    sched.add_job(job, "interval", hours=hours, id="living_review", replace_existing=True)
    if not sched.running:
        sched.start()
    logger.info("Scheduled living review every %s hours", hours)


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
