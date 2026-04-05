from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from pico_sr.config import settings
from pico_sr.db.models import Diff, Run, get_session, init_db


def build_pdf_report() -> dict:
    init_db()
    session = get_session()
    try:
        run = session.query(Run).order_by(Run.id.desc()).first()
        diff = session.query(Diff).order_by(Diff.id.desc()).first()
        if not run:
            return {"error": "no_run"}

        out_dir = Path("output/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "pico_sr_report.pdf"
        c = canvas.Canvas(str(path), pagesize=letter)
        w, h = letter
        y = h - inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, y, "PICO-SR — Meta-analysis report")
        y -= 0.35 * inch
        c.setFont("Helvetica", 10)
        lines = [
            f"Pooled effect (random effects): {run.pooled_d}",
            f"95% CI: [{run.ci_lower}, {run.ci_upper}]",
            f"I² = {run.i_squared}   τ² = {run.tau_squared}",
            f"Studies: {run.n_studies}",
        ]
        for line in lines:
            c.drawString(inch, y, str(line))
            y -= 0.2 * inch
        y -= 0.15 * inch
        if run.forest_plot_path:
            fp = Path(run.forest_plot_path)
            if fp.is_file():
                try:
                    from reportlab.lib.utils import ImageReader

                    img = ImageReader(str(fp))
                    iw, ih = img.getSize()
                    scale = min((w - 2 * inch) / iw, 3 * inch / ih)
                    c.drawImage(
                        img,
                        inch,
                        y - ih * scale,
                        width=iw * scale,
                        height=ih * scale,
                    )
                    y = y - ih * scale - 0.2 * inch
                except Exception:
                    c.drawString(inch, y, "(Forest plot could not be embedded)")
                    y -= 0.25 * inch
        if diff and diff.summary_json:
            y -= 0.2 * inch
            c.setFont("Helvetica-Bold", 11)
            c.drawString(inch, y, "Living review — last diff")
            y -= 0.22 * inch
            c.setFont("Helvetica", 9)
            try:
                s = json.loads(diff.summary_json)
                c.drawString(inch, y, str(s)[:1200])
            except json.JSONDecodeError:
                pass
        c.showPage()
        c.save()
        return {"path": str(path.resolve())}
    finally:
        session.close()
