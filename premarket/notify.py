"""Push the morning summary (DESIGN.md, M5): Feishu webhook, Telegram, or email.

Channels are enabled by environment variables (put them in PointStock/.env,
which is git-ignored and loaded by the run script):

    PREMARKET_FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/...
    PREMARKET_TELEGRAM_BOT_TOKEN=...      PREMARKET_TELEGRAM_CHAT_ID=...
    PREMARKET_SMTP_HOST=smtp.example.com  PREMARKET_SMTP_PORT=465
    PREMARKET_SMTP_USER=...  PREMARKET_SMTP_PASSWORD=...
    PREMARKET_EMAIL_FROM=...  PREMARKET_EMAIL_TO=a@x.com,b@y.com

No channel configured = nothing is sent. A failed push never fails the run.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage
from pathlib import Path

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4000


def summary_text(cards, meta: dict, report_path: Path | None = None) -> str:
    """Short plain-text summary: one block per ticker, range first, no buy/sell wording."""
    lines = [f"Pre-market {meta['session']} · {meta['generated_ny']}"]
    for w in meta.get("global_warnings", []):
        lines.append(f"⚠ {w}")
    for c in cards:
        f, cb = c.forecast, c.combined
        px = f"${c.premarket:,.2f} ({c.gap:+.1%})" if c.premarket is not None else f"close ${c.last_close:,.2f}"
        lines.append("")
        lines.append(f"{c.ticker} {px}")
        if f:
            b68 = c.event_bands[0] if c.event_bands else f.band68
            lines.append(f"  σ {f.sigma_next:.1f}% {f.regime} · 68% ${b68[0]:,.2f}–${b68[1]:,.2f}"
                         + (" · EVENT DAY" if c.event_day else ""))
        if cb:
            up, down, n = cb.agreement
            lines.append(f"  P(up) {cb.p_final:.0%} (prior {cb.p0:.0%}, {up}↑/{down}↓ of {n})")
        if c.news and c.news.headlines:
            top = max(c.news.headlines, key=lambda h: (h.materiality, abs(h.sentiment)))
            lines.append(f"  news {c.news.score:+.2f} · {top.title[:90]}" if c.news.score is not None
                         else f"  news · {top.title[:90]}")
        for o in getattr(c, "panels", []) or []:
            r = o.record.latest
            if r:
                best = max(r.aggregate.items(), key=lambda kv: kv[1])
                label = next(s["label"] for s in r.scenarios if s["id"] == best[0])
                lines.append(f"  event {o.record.event_type}: most likely '{label}' {best[1]:.0%} "
                             f"(spread {r.disagreement_label})")
        if c.warnings:
            lines.append(f"  warnings: {len(c.warnings)} (see report)")
    if meta.get("ledger_summary"):
        lines += ["", meta["ledger_summary"]]
    if report_path:
        lines += ["", f"Report: {report_path}"]
    lines.append("Probabilities and ranges only; not a trade recommendation.")
    return "\n".join(lines)


def _post_json(url: str, payload: dict, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode() or "{}"
    return json.loads(body)


def send_feishu(webhook: str, text: str) -> None:
    resp = _post_json(webhook, {"msg_type": "text", "content": {"text": text}})
    if resp.get("code", resp.get("StatusCode", 0)) not in (0, None):
        raise RuntimeError(f"Feishu error: {resp}")


def send_telegram(token: str, chat_id: str, text: str) -> None:
    for i in range(0, len(text), TELEGRAM_LIMIT):
        resp = _post_json(f"https://api.telegram.org/bot{token}/sendMessage",
                          {"chat_id": chat_id, "text": text[i:i + TELEGRAM_LIMIT],
                           "disable_web_page_preview": True})
        if not resp.get("ok"):
            raise RuntimeError(f"Telegram error: {resp.get('description')}")


def send_email(env: dict, subject: str, text: str, html_path: Path | None) -> None:
    msg = EmailMessage()
    msg["Subject"], msg["From"] = subject, env["PREMARKET_EMAIL_FROM"]
    msg["To"] = [a.strip() for a in env["PREMARKET_EMAIL_TO"].split(",") if a.strip()]
    msg.set_content(text)
    if html_path and html_path.exists():
        msg.add_alternative(html_path.read_text(), subtype="html")
    port = int(env.get("PREMARKET_SMTP_PORT", "465"))
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(env["PREMARKET_SMTP_HOST"], port, context=ctx, timeout=30) as s:
            s.login(env["PREMARKET_SMTP_USER"], env["PREMARKET_SMTP_PASSWORD"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(env["PREMARKET_SMTP_HOST"], port, timeout=30) as s:
            s.starttls(context=ctx)
            s.login(env["PREMARKET_SMTP_USER"], env["PREMARKET_SMTP_PASSWORD"])
            s.send_message(msg)


def configured_channels(env: dict | None = None) -> list[str]:
    env = env if env is not None else os.environ
    ch = []
    if env.get("PREMARKET_FEISHU_WEBHOOK"):
        ch.append("feishu")
    if env.get("PREMARKET_TELEGRAM_BOT_TOKEN") and env.get("PREMARKET_TELEGRAM_CHAT_ID"):
        ch.append("telegram")
    if all(env.get(k) for k in ("PREMARKET_SMTP_HOST", "PREMARKET_SMTP_USER", "PREMARKET_SMTP_PASSWORD",
                                "PREMARKET_EMAIL_FROM", "PREMARKET_EMAIL_TO")):
        ch.append("email")
    return ch


def push(cards, meta: dict, html_path: Path | None, env: dict | None = None) -> dict[str, str]:
    """Send to every configured channel. Returns {channel: "sent" | "failed: ..."}."""
    env = env if env is not None else dict(os.environ)
    text = summary_text(cards, meta, html_path)
    results = {}
    for ch in configured_channels(env):
        try:
            if ch == "feishu":
                send_feishu(env["PREMARKET_FEISHU_WEBHOOK"], text)
            elif ch == "telegram":
                send_telegram(env["PREMARKET_TELEGRAM_BOT_TOKEN"], env["PREMARKET_TELEGRAM_CHAT_ID"], text)
            elif ch == "email":
                send_email(env, f"Pre-market report {meta['session']}", text, html_path)
            results[ch] = "sent"
        except Exception as exc:
            log.warning("push via %s failed: %s", ch, exc)
            results[ch] = f"failed: {type(exc).__name__}"
    return results
