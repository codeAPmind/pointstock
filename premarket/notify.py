"""Push the morning summary (DESIGN.md, M5): Feishu webhook, Telegram, or email.

Channels are enabled by environment variables (put them in PointStock/.env,
which is git-ignored and loaded by the run script):

    PREMARKET_FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/...
    PREMARKET_TELEGRAM_BOT_TOKEN=...      PREMARKET_TELEGRAM_CHAT_ID=...
    PREMARKET_SMTP_HOST=smtp.example.com  PREMARKET_SMTP_PORT=465
    PREMARKET_SMTP_USER=...  PREMARKET_SMTP_PASSWORD=...
    PREMARKET_EMAIL_FROM=...  PREMARKET_EMAIL_TO=a@x.com,b@y.com

No channel configured = nothing is sent. A failed push never fails the run.

Gmail on this host sometimes drops the SMTP banner ("Connection unexpectedly
closed") and accepts the same message a few seconds later. Email therefore
retries a few times. Auth failures are not retried. Each attempt is appended
to logs/notify.jsonl and the latest per-channel result is kept in
data/runs/<session>/notify.json so a later --resend (or the next scheduled
wake-up) can send only the channels that failed.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4000
EMAIL_ATTEMPTS = 3
EMAIL_BACKOFF_SEC = 3


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


def _smtp_once(env: dict, msg: EmailMessage) -> None:
    port = int(env.get("PREMARKET_SMTP_PORT", "465"))
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(env["PREMARKET_SMTP_HOST"], port, context=ctx, timeout=30) as s:
            s.login(env["PREMARKET_SMTP_USER"], env["PREMARKET_SMTP_PASSWORD"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(env["PREMARKET_SMTP_HOST"], port, timeout=30) as s:
            s.ehlo()
            s.starttls(context=ctx)
            s.ehlo()
            s.login(env["PREMARKET_SMTP_USER"], env["PREMARKET_SMTP_PASSWORD"])
            s.send_message(msg)


def send_email(env: dict, subject: str, text: str, html_path: Path | None,
               attempts: int = EMAIL_ATTEMPTS) -> int:
    """Send one message. Returns the attempt that succeeded (1-based).

    Transient disconnects are retried. SMTPAuthenticationError is not.
    """
    msg = EmailMessage()
    msg["Subject"], msg["From"] = subject, env["PREMARKET_EMAIL_FROM"]
    msg["To"] = [a.strip() for a in env["PREMARKET_EMAIL_TO"].split(",") if a.strip()]
    msg.set_content(text)
    if html_path and Path(html_path).exists():
        msg.add_alternative(Path(html_path).read_text(), subtype="html")
    send_email.attempts = 0
    last: Exception | None = None
    for i in range(1, attempts + 1):
        send_email.attempts = i
        try:
            _smtp_once(env, msg)
        except smtplib.SMTPAuthenticationError:
            raise
        except Exception as exc:
            last = exc
            log.warning("email attempt %d/%d failed: %s", i, attempts, exc)
            if i < attempts:
                time.sleep(EMAIL_BACKOFF_SEC * i)
            continue
        if i > 1:
            log.info("email sent on attempt %d/%d", i, attempts)
        return i
    assert last is not None
    raise last


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


def notify_state_path(data_dir: str | Path, session: str) -> Path:
    return Path(data_dir) / "runs" / session / "notify.json"


def _payload_path(data_dir: str | Path, session: str) -> Path:
    return Path(data_dir) / "runs" / session / "notify_payload.json"


def load_notify_state(data_dir: str | Path, session: str) -> dict | None:
    p = notify_state_path(data_dir, session)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def failed_channels(data_dir: str | Path, session: str) -> list[str]:
    """Channels whose latest recorded status is not sent. No record → nothing pending."""
    state = load_notify_state(data_dir, session)
    if not state:
        return []
    return [ch for ch, info in (state.get("channels") or {}).items() if info.get("status") != "sent"]


def _save_payload(data_dir: str | Path, session: str, subject: str, text: str,
                  html_path: Path | None) -> None:
    p = _payload_path(data_dir, session)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "session": session, "subject": subject, "text": text,
        "html_path": str(html_path) if html_path else "",
    }, indent=2))


def _record(data_dir: str | Path | None, logs_dir: str | Path | None, session: str,
            updates: dict[str, dict], source: str) -> None:
    """Merge channel results into the session state and append one jsonl line."""
    if not updates:
        return
    prev = load_notify_state(data_dir, session) if data_dir else None
    channels = dict((prev or {}).get("channels") or {})
    channels.update(updates)
    rec = {"session": session, "at": datetime.now(timezone.utc).isoformat(),
           "source": source, "channels": channels}
    if data_dir:
        p = notify_state_path(data_dir, session)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=2))
    if logs_dir:
        line = {"session": session, "at": rec["at"], "source": source, "channels": updates}
        log_path = Path(logs_dir) / "notify.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")


def _deliver(ch: str, env: dict, subject: str, text: str, html_path: Path | None) -> dict:
    """One channel. Email retries inside send_email; other channels are a single try."""
    try:
        if ch == "feishu":
            send_feishu(env["PREMARKET_FEISHU_WEBHOOK"], text)
            attempts = 1
        elif ch == "telegram":
            send_telegram(env["PREMARKET_TELEGRAM_BOT_TOKEN"], env["PREMARKET_TELEGRAM_CHAT_ID"], text)
            attempts = 1
        elif ch == "email":
            attempts = send_email(env, subject, text, html_path)
        else:
            raise RuntimeError(f"unknown channel {ch}")
        return {"status": "sent", "attempts": attempts, "error": ""}
    except Exception as exc:
        attempts = getattr(send_email, "attempts", 1) if ch == "email" else 1
        log.warning("push via %s failed: %s", ch, exc)
        return {"status": "failed", "attempts": attempts,
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def _result_strings(detail: dict[str, dict]) -> dict[str, str]:
    out = {}
    for ch, info in detail.items():
        if info["status"] == "sent":
            out[ch] = "sent"
        else:
            name = info["error"].split(":", 1)[0] or "Error"
            out[ch] = f"failed: {name}"
    return out


def push(cards, meta: dict, html_path: Path | None, env: dict | None = None,
         data_dir: str | Path | None = None, logs_dir: str | Path | None = None) -> dict[str, str]:
    """Send to every configured channel. Returns {channel: "sent" | "failed: ..."}."""
    env = env if env is not None else dict(os.environ)
    text = summary_text(cards, meta, html_path)
    subject = f"Pre-market report {meta['session']}"
    if data_dir:
        _save_payload(data_dir, meta["session"], subject, text, html_path)
    detail = {}
    for ch in configured_channels(env):
        detail[ch] = _deliver(ch, env, subject, text, html_path)
    _record(data_dir, logs_dir, meta["session"], detail, "push")
    return _result_strings(detail)


def _load_payload(data_dir: str | Path, reports_dir: str | Path, session: str) -> tuple[str, str, Path | None]:
    p = _payload_path(data_dir, session)
    if p.exists():
        raw = json.loads(p.read_text())
        html = Path(raw["html_path"]) if raw.get("html_path") else None
        return raw["subject"], raw["text"], html
    md = Path(reports_dir) / f"{session}.md"
    html = Path(reports_dir) / f"{session}.html"
    if not md.exists() and not html.exists():
        raise FileNotFoundError(f"no saved report for {session}")
    text = md.read_text() if md.exists() else f"Pre-market report {session}"
    return f"Pre-market report {session}", text, html if html.exists() else None


def resend(session: str, data_dir: str | Path, reports_dir: str | Path,
           logs_dir: str | Path | None = None, env: dict | None = None,
           only_failed: bool = True) -> dict[str, str]:
    """Re-push a saved report. By default only channels that last failed.

    No notify.json yet (a run from before this was recorded): send every
    configured channel. All channels already sent: send nothing.
    """
    env = env if env is not None else dict(os.environ)
    subject, text, html = _load_payload(data_dir, reports_dir, session)
    channels = configured_channels(env)
    if only_failed:
        pending = failed_channels(data_dir, session)
        if load_notify_state(data_dir, session) is not None:
            channels = [c for c in channels if c in pending]
    if not channels:
        return {}
    detail = {ch: _deliver(ch, env, subject, text, html) for ch in channels}
    _record(data_dir, logs_dir, session, detail, "resend")
    return _result_strings(detail)
