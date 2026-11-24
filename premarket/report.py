"""HTML + Markdown report: one card per ticker (M1: range and warnings).

Card order follows DESIGN.md: range first, direction second, news third,
warnings last. Direction (M3/M4) and news (M4) are not built yet, so M1
cards carry the range, the band check and the warnings line. No buy/sell
wording anywhere.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .garch import Coverage, GarchForecast


@dataclass
class TickerCard:
    ticker: str
    last_close: float
    last_date: str
    forecast: GarchForecast | None
    coverage: Coverage | None
    premarket: float | None
    premarket_asof: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float | None:
        if self.premarket is None:
            return None
        return self.premarket / self.last_close - 1

    @property
    def event_day(self) -> bool:
        """Pre-market gap already larger than one GARCH sigma."""
        if self.gap is None or self.forecast is None:
            return False
        return abs(self.gap) * 100 > self.forecast.sigma_next


def _fmt_ts(iso: str | None, tz: str) -> str:
    if not iso:
        return "—"
    return datetime.fromisoformat(iso).astimezone(ZoneInfo(tz)).strftime("%H:%M %Z")


def card_lines(c: TickerCard) -> dict[str, list[str]]:
    """Plain-text lines per section; shared by Markdown and HTML renderers."""
    if c.premarket is not None:
        head = (f"{c.ticker} — pre-market ${c.premarket:,.2f} "
                f"(gap {c.gap:+.1%} vs close ${c.last_close:,.2f} on {c.last_date})")
    else:
        head = f"{c.ticker} — pre-market n/a (last close ${c.last_close:,.2f} on {c.last_date})"
    out: dict[str, list[str]] = {"head": [head], "VOLATILITY": [], "WARNINGS": []}
    f = c.forecast
    if f:
        out["VOLATILITY"].append(
            f"regime {f.regime} · sigma {f.sigma_next:.2f}% (long-run {f.long_run_sigma:.2f}%) · "
            f"68% band ${f.band68[0]:,.2f}–${f.band68[1]:,.2f} · "
            f"95% band ${f.band95[0]:,.2f}–${f.band95[1]:,.2f}")
        if c.coverage:
            cv = c.coverage
            out["VOLATILITY"].append(
                f"band check: {cv.inside68:.0%} of the last {cv.n_days} days closed inside the 68% band "
                f"(target 68%) · {cv.inside95:.0%} inside the 95% band ({cv.start} → {cv.end}, out-of-sample)")
        p = f.params
        out["VOLATILITY"].append(
            f"model: GJR-GARCH(1,1)-t via {f.engine}, {f.n_obs} days · α {p.alpha:.3f} γ {p.gamma:.3f} "
            f"β {p.beta:.3f} ν {p.nu:.1f} · persistence {p.persistence:.3f}")
        if c.event_day:
            out["VOLATILITY"].append(
                f"EVENT DAY: pre-market gap {c.gap:+.1%} already exceeds 1 sigma "
                f"({f.sigma_next:.2f}%). Band widening from gap base rates arrives in M2.")
    out["WARNINGS"] = c.warnings or ["none"]
    return out


def render_markdown(cards: list[TickerCard], run_meta: dict) -> str:
    lines = [f"# Pre-market report — {run_meta['session']}",
             f"Generated {run_meta['generated_ny']} / {run_meta['generated_local']} · "
             f"forecast for the {run_meta['session']} session · M1 (range only)", ""]
    if run_meta.get("global_warnings"):
        lines += [f"> **Data warning:** {w}" for w in run_meta["global_warnings"]] + [""]
    for c in cards:
        sec = card_lines(c)
        lines.append(f"## {sec['head'][0]}")
        lines.append("```")
        for name in ("VOLATILITY", "WARNINGS"):
            for i, l in enumerate(sec[name]):
                lines.append(f"{name if i == 0 else '':<12} {l}")
            lines.append("")
        lines[-1] = "```"
        lines.append("")
    lines.append("_Probabilities and ranges only; not a recommendation to trade._")
    return "\n".join(lines)


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e4e0;
--high:#b42318;--low:#1f6feb;--normal:#2e7d32;--warn:#9a6700;--warnbg:#fff8e1;--band:#dde7f7;--band95:#eef3fb}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d20;--ink:#ececec;--mute:#9a9aa0;
--line:#2c2c30;--high:#ff7b72;--low:#79b8ff;--normal:#7ee787;--warn:#e3b341;--warnbg:#2b2410;--band:#23324a;--band95:#1b2433}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}.meta{color:var(--mute);font-size:13px;margin-bottom:20px}
.gw{background:var(--warnbg);color:var(--warn);border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:16px}
.card h2{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px;margin-bottom:14px}
.row{display:grid;grid-template-columns:110px 1fr;gap:4px 14px;margin:10px 0}
.lbl{font-size:11px;letter-spacing:.06em;color:var(--mute);padding-top:3px}
.regime{display:inline-block;font-weight:600;font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid currentColor}
.HIGH{color:var(--high)}.LOW{color:var(--low)}.NORMAL{color:var(--normal)}
.kv{display:flex;flex-wrap:wrap;gap:6px 22px;font-variant-numeric:tabular-nums}
.kv b{font-weight:600}.small{color:var(--mute);font-size:13px;font-variant-numeric:tabular-nums}
.bar{position:relative;height:30px;margin:12px 0 4px}
.bar .b95{position:absolute;top:8px;height:14px;background:var(--band95);border-radius:3px}
.bar .b68{position:absolute;top:8px;height:14px;background:var(--band);border-radius:3px}
.bar .tick{position:absolute;top:2px;width:2px;height:26px;background:var(--ink)}
.bar .pm{position:absolute;top:2px;width:2px;height:26px;background:var(--high)}
.axis{display:flex;justify-content:space-between;color:var(--mute);font-size:12px;font-variant-numeric:tabular-nums}
.warn{color:var(--warn)}.event{color:var(--high);font-weight:600}
footer{color:var(--mute);font-size:12px;margin-top:24px}
@media (max-width:560px){.row{grid-template-columns:1fr}.lbl{padding-top:0}}
"""


def _bar_html(c: TickerCard) -> str:
    f = c.forecast
    lo, hi = f.band95
    if c.premarket is not None:
        lo, hi = min(lo, c.premarket), max(hi, c.premarket)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    pos = lambda x: (x - lo) / (hi - lo) * 100
    parts = [
        f'<div class="b95" style="left:{pos(f.band95[0]):.2f}%;width:{pos(f.band95[1]) - pos(f.band95[0]):.2f}%"></div>',
        f'<div class="b68" style="left:{pos(f.band68[0]):.2f}%;width:{pos(f.band68[1]) - pos(f.band68[0]):.2f}%"></div>',
        f'<div class="tick" title="last close" style="left:{pos(c.last_close):.2f}%"></div>',
    ]
    if c.premarket is not None:
        parts.append(f'<div class="pm" title="pre-market" style="left:{pos(c.premarket):.2f}%"></div>')
    axis = (f'<div class="axis"><span>${f.band95[0]:,.2f}</span><span>95% · 68% · '
            f'<span style="color:var(--ink)">▮</span> last close'
            + (' · <span class="event">▮</span> pre-market' if c.premarket is not None else '')
            + f'</span><span>${f.band95[1]:,.2f}</span></div>')
    return f'<div class="bar">{"".join(parts)}</div>{axis}'


def render_html(cards: list[TickerCard], run_meta: dict) -> str:
    e = html.escape
    body = [f"<h1>Pre-market report — {e(run_meta['session'])}</h1>",
            f'<div class="meta">Generated {e(run_meta["generated_ny"])} / {e(run_meta["generated_local"])} · '
            f'forecast for the {e(run_meta["session"])} session · milestone M1 (range only)</div>']
    for w in run_meta.get("global_warnings", []):
        body.append(f'<div class="gw">Data warning: {e(w)}</div>')
    for c in cards:
        f = c.forecast
        if c.premarket is not None:
            sub = (f"pre-market <b>${c.premarket:,.2f}</b> ({c.gap:+.1%}) at "
                   f"{_fmt_ts(c.premarket_asof, 'America/New_York')} · last close ${c.last_close:,.2f} ({c.last_date})")
        else:
            sub = f"pre-market n/a · last close <b>${c.last_close:,.2f}</b> ({e(c.last_date)})"
        body.append(f'<section class="card"><h2>{e(c.ticker)}</h2><div class="sub">{sub}</div>')
        if f:
            p = f.params
            body.append('<div class="row"><div class="lbl">VOLATILITY</div><div>')
            body.append(f'<div class="kv"><span><span class="regime {f.regime}">{f.regime}</span></span>'
                        f'<span>sigma <b>{f.sigma_next:.2f}%</b> <span class="small">(long-run {f.long_run_sigma:.2f}%)</span></span>'
                        f'<span>68% <b>${f.band68[0]:,.2f}–${f.band68[1]:,.2f}</b></span>'
                        f'<span>95% <b>${f.band95[0]:,.2f}–${f.band95[1]:,.2f}</b></span></div>')
            body.append(_bar_html(c))
            if c.coverage:
                cv = c.coverage
                body.append(f'<div class="small">Band check: <b>{cv.inside68:.0%}</b> of the last {cv.n_days} days '
                            f'closed inside the 68% band (target 68%) · {cv.inside95:.0%} inside the 95% band '
                            f'· {cv.start} → {cv.end}, out-of-sample</div>')
            body.append(f'<div class="small">GJR-GARCH(1,1)-t via {f.engine} on {f.n_obs} days · α {p.alpha:.3f} · '
                        f'γ {p.gamma:.3f} · β {p.beta:.3f} · ν {p.nu:.1f} · persistence {p.persistence:.3f}</div>')
            if c.event_day:
                body.append(f'<div class="event">Event day: the pre-market gap {c.gap:+.1%} already exceeds one sigma.</div>')
            body.append('</div></div>')
        body.append('<div class="row"><div class="lbl">DIRECTION · NEWS</div>'
                    '<div class="small">Not built yet (base rates M2, models M3, news + combiner M4).</div></div>')
        warn = "; ".join(e(w) for w in c.warnings) if c.warnings else "none"
        body.append(f'<div class="row"><div class="lbl">WARNINGS</div><div class="{"warn" if c.warnings else "small"}">{warn}</div></div>')
        body.append('</section>')
    body.append('<footer>Probabilities and ranges only; not a recommendation to trade. '
                'Bands are centred on the last close; prices split-adjusted (yfinance).</footer>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Pre-market {e(run_meta["session"])}</title><style>{CSS}</style></head>'
            f'<body><main>{"".join(body)}</main></body></html>')


def write_reports(cards: list[TickerCard], run_meta: dict, reports_dir: str | Path) -> tuple[Path, Path]:
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = run_meta["session"]
    html_path, md_path = d / f"{stem}.html", d / f"{stem}.md"
    html_path.write_text(render_html(cards, run_meta))
    md_path.write_text(render_markdown(cards, run_meta))
    return html_path, md_path


def run_meta_now(session: str, local_tz: str, global_warnings: list[str]) -> dict:
    now = datetime.now(timezone.utc)
    return {"session": session,
            "generated_ny": now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z"),
            "generated_local": now.astimezone(ZoneInfo(local_tz)).strftime("%Y-%m-%d %H:%M %Z"),
            "generated_utc": now.isoformat(),
            "global_warnings": global_warnings}
