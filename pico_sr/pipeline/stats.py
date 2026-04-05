from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pico_sr.db.models import Extraction, Paper, Run, get_session, init_db
from pico_sr.config import settings

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float | None:
    """Convert to Python float, return None for NaN/Inf/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _sanitize_dict(d: dict) -> dict:
    """
    Recursively replace NaN/Inf with None so FastAPI can JSON-serialize it.
    This fixes: ValueError: Out of range float values not JSON compliant: nan
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, np.floating):
            f = float(v)
            out[k] = None if (math.isnan(f) or math.isinf(f)) else f
        elif isinstance(v, dict):
            out[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            out[k] = [
                None if (isinstance(x, (float, np.floating))
                         and (math.isnan(float(x)) or math.isinf(float(x))))
                else (float(x) if isinstance(x, np.floating) else x)
                for x in v
            ]
        else:
            out[k] = v
    return out


def _to_cohens_d(effect_size: float, effect_type: str | None) -> float | None:
    et = (effect_type or "").strip().upper()
    try:
        es = float(effect_size)
    except (TypeError, ValueError):
        return None
    if es == 0:
        return 0.0
    if et in ("OR", "ODDS_RATIO"):
        return math.log(es) / 1.8138
    if et in ("RR", "RISK_RATIO", "RELATIVE_RISK"):
        return math.log(abs(es)) / 1.8138 if es > 0 else None
    return es


def _interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:   return "negligible"
    if ad < 0.5:   return "small"
    if ad < 0.8:   return "small-medium"
    if ad < 1.0:   return "medium-large"
    return "large"


def _interpret_i2(i2: float) -> str:
    if i2 < 25:  return "low heterogeneity"
    if i2 < 50:  return "moderate heterogeneity"
    if i2 < 75:  return "substantial heterogeneity"
    return "high heterogeneity"


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(extraction_ids: list[int] | None = None) -> dict[str, Any]:
    init_db()
    session = get_session()
    try:
        q = session.query(Extraction)
        if extraction_ids:
            q = q.filter(Extraction.id.in_(extraction_ids))
        rows = q.all()

        # ── Collect valid studies ─────────────────────────────────────
        studies = []
        skipped = 0
        for row in rows:
            try:
                p = json.loads(row.payload_json or "{}")
            except Exception:
                skipped += 1
                continue

            es = p.get("effect_size")
            lo = p.get("ci_lower")
            hi = p.get("ci_upper")
            et = p.get("effect_type")
            n  = p.get("population_n")

            # Skip if any critical value is missing
            if es is None or lo is None or hi is None:
                skipped += 1
                logger.info(
                    "Skipping paper_id=%s — missing effect/CI (es=%s lo=%s hi=%s)",
                    row.paper_id, es, lo, hi
                )
                continue

            try:
                es = float(es)
                lo = float(lo)
                hi = float(hi)
            except (TypeError, ValueError):
                skipped += 1
                continue

            # Skip NaN/Inf values
            if any(math.isnan(x) or math.isinf(x) for x in (es, lo, hi)):
                skipped += 1
                logger.warning("NaN/Inf in paper_id=%s — skipping", row.paper_id)
                continue

            d = _to_cohens_d(es, et)
            if d is None:
                skipped += 1
                continue

            se = (hi - lo) / 3.92
            if se <= 0:
                skipped += 1
                logger.warning(
                    "SE <= 0 for paper_id=%s (CI=[%s,%s]) — skipping",
                    row.paper_id, lo, hi
                )
                continue

            paper = session.query(Paper).filter(
                Paper.id == row.paper_id
            ).first()
            title = (
                (paper.title[:40] + "...")
                if paper and len(paper.title) > 40
                else (paper.title if paper else f"Study {row.paper_id}")
            )

            studies.append({
                "title": title,
                "d": d,
                "se": se,
                "var": se ** 2,
                "n": int(n) if n else 0,
                "ci_lower": lo,
                "ci_upper": hi,
            })

        logger.info(
            "Analysis: %d valid studies, %d skipped", len(studies), skipped
        )

        if not studies:
            return {
                "error": (
                    f"No studies with valid effect size and CI. "
                    f"({skipped} papers skipped — missing effect_size, "
                    f"ci_lower, or ci_upper). "
                    "Run extraction first and ensure papers report "
                    "numeric outcomes."
                ),
                "skipped": skipped,
            }

        k   = len(studies)
        ds  = np.array([s["d"]   for s in studies], dtype=float)
        vs  = np.array([s["var"] for s in studies], dtype=float)

        # ── Fixed effects ─────────────────────────────────────────────
        w_fe       = 1.0 / vs
        pooled_fe  = float(np.sum(w_fe * ds) / np.sum(w_fe))
        se_fe      = float(math.sqrt(1.0 / float(np.sum(w_fe))))
        ci_fe_low  = pooled_fe - 1.96 * se_fe
        ci_fe_high = pooled_fe + 1.96 * se_fe
        w_fe_pct   = (w_fe / np.sum(w_fe) * 100).tolist()

        # ── Heterogeneity ─────────────────────────────────────────────
        Q  = float(np.sum(w_fe * (ds - pooled_fe) ** 2))
        df = k - 1

        # Safe I² — avoid NaN when Q=0 or k=1
        if Q > 0 and df > 0:
            i2 = max(0.0, (Q - df) / Q * 100)
        else:
            i2 = 0.0

        # Safe τ² — avoid division by zero / NaN
        denom = float(np.sum(w_fe)) - float(np.sum(w_fe ** 2)) / float(np.sum(w_fe))
        if Q > df and denom > 0:
            tau2 = max(0.0, (Q - df) / denom)
        else:
            tau2 = 0.0

        p_Q = float(1 - stats.chi2.cdf(Q, max(df, 1)))

        # ── Random effects (DerSimonian-Laird) ────────────────────────
        w_re       = 1.0 / (vs + tau2)
        pooled_re  = float(np.sum(w_re * ds) / np.sum(w_re))
        se_re      = float(math.sqrt(1.0 / float(np.sum(w_re))))
        ci_re_low  = pooled_re - 1.96 * se_re
        ci_re_high = pooled_re + 1.96 * se_re
        w_re_pct   = (w_re / np.sum(w_re) * 100).tolist()

        # Safe z / p-value — avoid divide-by-zero
        if se_re > 0:
            z     = pooled_re / se_re
            p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
        else:
            z     = 0.0
            p_val = 1.0

        # ── Interpretation ────────────────────────────────────────────
        significant   = ci_re_low > 0 or ci_re_high < 0
        direction     = "positive" if pooled_re > 0 else "negative"
        magnitude     = _interpret_d(pooled_re)
        het_label     = _interpret_i2(i2)
        crosses_zero  = ci_re_low < 0 < ci_re_high

        conclusion_lines = []
        if k == 1:
            conclusion_lines.append(
                "Only 1 study extracted — results are preliminary."
            )
            conclusion_lines.append(
                f"Effect size: {magnitude} (d={pooled_re:.2f})."
            )
            conclusion_lines.append(
                "More studies needed for reliable meta-analysis."
            )
        elif significant:
            conclusion_lines.append(
                f"The intervention shows a {magnitude.upper()} {direction} "
                f"effect (d={pooled_re:.2f})."
            )
            conclusion_lines.append(
                f"CI does not cross zero → significant (p={p_val:.3f})."
            )
            conclusion_lines.append(
                f"Heterogeneity is {het_label} (I²={i2:.1f}%)."
            )
            conclusion_lines.append("Recommendation: SUPPORTS the intervention.")
        else:
            conclusion_lines.append(
                f"Pooled effect is {magnitude} (d={pooled_re:.2f}) "
                f"but CI crosses zero."
            )
            conclusion_lines.append(
                f"NOT statistically significant (p={p_val:.3f})."
            )
            conclusion_lines.append(
                f"Heterogeneity: {het_label} (I²={i2:.1f}%)."
            )
            conclusion_lines.append(
                "Recommendation: INSUFFICIENT evidence."
            )

        total_n = int(sum(s["n"] for s in studies))

        # ── Text report ───────────────────────────────────────────────
        sep = "─" * 47
        report_lines = [
            "┌" + "─" * 47 + "┐",
            "│" + "  PICO-SR Meta-analysis Report  ".center(47) + "│",
            "├" + sep + "┤",
            f"│  Studies included : {k:<26}│",
            f"│  Total patients   : {total_n:<26}│",
            "├" + sep + "┤",
            f"│  Pooled Effect (Cohen's d) = {pooled_re:.2f}",
            f"│  95% CI = [{ci_re_low:.2f}, {ci_re_high:.2f}]"
            + ("  ← crosses zero!" if crosses_zero else "  ← doesn't cross 0!"),
            f"│  p-value = {p_val:.3f}"
            + ("  ← significant!" if significant else "  ← not significant"),
            f"│  I² = {i2:.1f}%  ({het_label})",
            "├" + sep + "┤",
            "│  CONCLUSION:",
        ]
        for line in conclusion_lines:
            report_lines.append(f"│  {line}")
        report_lines.append("└" + "─" * 47 + "┘")
        text_report = "\n".join(report_lines)

        # ── Forest plot ───────────────────────────────────────────────
        plot_path = _draw_forest_plot(
            studies    = studies,
            weights_re = w_re_pct,
            pooled_fe  = pooled_fe,
            ci_fe      = (ci_fe_low, ci_fe_high),
            pooled_re  = pooled_re,
            ci_re      = (ci_re_low, ci_re_high),
            i2=i2, tau2=tau2, Q=Q, df=df, p_Q=p_Q, p_val=p_val,
        )

        # ── Save Run ──────────────────────────────────────────────────
        run_row = Run(
            pooled_d         = _safe_float(pooled_re),
            ci_lower         = _safe_float(ci_re_low),
            ci_upper         = _safe_float(ci_re_high),
            i_squared        = _safe_float(i2),
            tau_squared      = _safe_float(tau2),
            n_studies        = k,
            forest_plot_path = str(plot_path),
            text_report      = text_report,
        )
        session.add(run_row)
        session.commit()

        # ── Build response — sanitize ALL floats before returning ─────
        result = {
            "k"               : k,
            "total_n"         : total_n,
            "pooled_re"       : _safe_float(pooled_re),
            "ci_re"           : [_safe_float(ci_re_low), _safe_float(ci_re_high)],
            "pooled_fe"       : _safe_float(pooled_fe),
            "ci_fe"           : [_safe_float(ci_fe_low), _safe_float(ci_fe_high)],
            "i2"              : _safe_float(i2),
            "tau2"            : _safe_float(tau2),
            "Q"               : _safe_float(Q),
            "p_Q"             : _safe_float(p_Q),
            "p_val"           : _safe_float(p_val),
            "significant"     : significant,
            "conclusion"      : conclusion_lines,
            "text_report"     : text_report,
            "forest_plot_path": str(plot_path),
            "skipped"         : skipped,
        }

        # Final safety net — replace any remaining NaN/Inf
        return _sanitize_dict(result)

    finally:
        session.close()


# ── Forest plot ───────────────────────────────────────────────────────────────

def _draw_forest_plot(
    studies, weights_re,
    pooled_fe, ci_fe,
    pooled_re, ci_re,
    i2, tau2, Q, df, p_Q, p_val,
) -> Path:
    k          = len(studies)
    fig_height = max(6, k * 0.7 + 5)
    fig, ax    = plt.subplots(figsize=(13, fig_height))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    y_positions = list(range(k, 0, -1))

    # Dynamic x-axis — don't clip any study
    all_lows  = [s["ci_lower"] for s in studies] + [ci_re[0]]
    all_highs = [s["ci_upper"] for s in studies] + [ci_re[1]]
    data_min  = min(all_lows)
    data_max  = max(all_highs)
    x_pad     = max((data_max - data_min) * 0.15, 0.1)
    x_left    = data_min - x_pad - 0.9
    x_right   = data_max + x_pad + 1.2

    # Study rows
    for s, y, w in zip(studies, y_positions, weights_re):
        d, lo, hi = s["d"], s["ci_lower"], s["ci_upper"]
        ax.plot([lo, hi], [y, y], color="#2C5F8A",
                linewidth=1.5, solid_capstyle="round")
        marker_size = max(4, min(14, w * 0.8))
        ax.plot(d, y, "s", color="#2C5F8A", markersize=marker_size, zorder=5)
        # Label left
        ax.text(x_left + 0.05, y, s["title"],
                ha="left", va="center", fontsize=8, color="#1a1a1a")
        # Weight right
        ax.text(data_max + x_pad * 0.3, y, f"{w:.1f}%",
                ha="left", va="center", fontsize=8, color="#555")
        # CI text right
        ax.text(data_max + x_pad * 0.7, y,
                f"{d:.2f} [{lo:.2f}, {hi:.2f}]",
                ha="left", va="center", fontsize=8,
                fontfamily="monospace", color="#1a1a1a")

    # Pooled diamond
    d_y = 0
    d_h = 0.35
    diamond = plt.Polygon(
        [[ci_re[0], d_y], [pooled_re, d_y + d_h],
         [ci_re[1], d_y], [pooled_re, d_y - d_h]],
        closed=True, color="#8B0000", zorder=6,
    )
    ax.add_patch(diamond)
    ax.text(x_left + 0.05, d_y, "Pooled (RE)",
            ha="left", va="center", fontsize=9,
            fontweight="bold", color="#8B0000")
    ax.text(data_max + x_pad * 0.7, d_y,
            f"{pooled_re:.2f} [{ci_re[0]:.2f}, {ci_re[1]:.2f}]",
            ha="left", va="center", fontsize=9,
            fontfamily="monospace", fontweight="bold", color="#8B0000")

    # Reference lines
    ax.axvline(pooled_fe, color="#666", linewidth=0.8,
               linestyle="--", alpha=0.5, label="Fixed effect")
    ax.axvline(0, color="#333", linewidth=1.0, linestyle="--", alpha=0.4)

    # Headers
    h_y = k + 1
    ax.text(x_left + 0.05, h_y, "Study",
            ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(data_max + x_pad * 0.3, h_y, "Weight",
            ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(data_max + x_pad * 0.7, h_y, "Effect [95% CI]",
            ha="left", va="center", fontsize=9, fontweight="bold")

    ax.axhline(k + 0.5, color="#ccc", linewidth=0.8)
    ax.axhline(0.5,     color="#ccc", linewidth=0.8)

    # Footer stats
    i2_str   = f"{i2:.1f}%" if not math.isnan(i2) else "N/A"
    tau2_str = f"{tau2:.4f}" if not math.isnan(tau2) else "N/A"
    q_str    = f"{Q:.2f}" if not math.isnan(Q) else "N/A"
    pq_str   = f"{p_Q:.4f}" if not math.isnan(p_Q) else "N/A"
    pv_str   = f"{p_val:.4f}" if not math.isnan(p_val) else "N/A"

    footer = (
        f"I² = {i2_str}   τ² = {tau2_str}   "
        f"Q = {q_str} (df={df}, p={pq_str})   "
        f"Pooled p = {pv_str}"
    )
    mid = (x_left + x_right) / 2
    ax.text(mid, -0.9, footer,
            ha="center", va="center",
            fontsize=8, color="#555", style="italic")
    ax.text(mid - 0.5, -1.5, "← Favours control",
            ha="center", fontsize=8, color="#555")
    ax.text(mid + 0.5, -1.5, "Favours intervention →",
            ha="center", fontsize=8, color="#555")

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-1.8, k + 1.8)
    ax.set_xlabel("Effect size (Cohen's d)", fontsize=9, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    plt.title(
        "PICO-SR — Forest Plot (Random Effects)",
        fontsize=11, fontweight="bold", pad=12, color="#1a1a1a",
    )
    plt.tight_layout()

    out_dir = Path(settings.pdf_dir).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "forest_plot.png"
    plt.savefig(
        plot_path, dpi=150, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close()
    return plot_path