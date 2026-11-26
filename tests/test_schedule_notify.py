"""Scheduling guard (holidays, window, once per session) and push summary/channels."""
import json
import smtplib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from premarket import notify as NT
from premarket import schedule as S


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_holidays_and_weekends_skip():
    assert S.should_run("/tmp/x", utc("2026-11-26 13:30"))[0] is False       # Thanksgiving
    assert "market closed" in S.should_run("/tmp/x", utc("2026-11-28 13:30"))[1]   # Saturday
    assert S.is_trading_day("2026-11-27")                                    # half day is a session


def test_window_and_stamp(tmp_path):
    # 13:30 UTC = 08:30 EST on 2026-11-24.
    assert S.should_run(tmp_path, utc("2026-11-24 13:30")) == (True, "pre-market run for 2026-11-24")
    assert S.should_run(tmp_path, utc("2026-11-24 09:00"))[0] is False      # 04:00 ET
    assert S.should_run(tmp_path, utc("2026-11-24 15:00"))[0] is False      # 10:00 ET, market open
    S.mark_done(tmp_path, "2026-11-24")
    assert "already ran" in S.should_run(tmp_path, utc("2026-11-24 13:45"))[1]
    assert S.should_run(tmp_path, utc("2026-11-24 09:00"), force=True)[0] is True


def test_summer_time_window():
    # 12:30 UTC = 08:30 EDT on 2026-06-15 (20:30 Singapore).
    assert S.should_run("/tmp/nonexistent-stamp", utc("2026-06-15 12:30"))[0] is True


def _card():
    f = SimpleNamespace(sigma_next=7.3, regime="HIGH", band68=(155.0, 176.0))
    cb = SimpleNamespace(p_final=0.127, p0=0.14, agreement=(1, 3, 4))
    return SimpleNamespace(ticker="NVDA", premarket=159.0, gap=-0.037, last_close=165.05, forecast=f,
                           combined=cb, event_bands=None, event_day=False, news=None, panels=[],
                           warnings=["w1"])


META = {"session": "2026-11-24", "generated_ny": "2026-11-24 08:30 EST", "global_warnings": [],
        "ledger_summary": "Ledger: no scored forecasts yet."}


def test_summary_text_has_range_and_direction_without_trade_words():
    t = NT.summary_text([_card()], META)
    assert "NVDA $159.00 (-3.7%)" in t and "68% $155.00–$176.00" in t and "P(up) 13%" in t
    low = t.lower()
    assert "not a trade recommendation" in low
    for word in (" buy ", " sell ", "strong buy"):
        assert word not in low.replace("not a trade recommendation", "")


def test_channels_from_env():
    assert NT.configured_channels({}) == []
    env = {"PREMARKET_FEISHU_WEBHOOK": "https://x", "PREMARKET_TELEGRAM_BOT_TOKEN": "t",
           "PREMARKET_TELEGRAM_CHAT_ID": "1"}
    assert NT.configured_channels(env) == ["feishu", "telegram"]


def test_push_isolates_failures(monkeypatch):
    sent = []
    monkeypatch.setattr(NT, "send_feishu", lambda url, text: sent.append(("feishu", text)))
    def boom(*a):
        raise ConnectionError("down")
    monkeypatch.setattr(NT, "send_telegram", boom)
    env = {"PREMARKET_FEISHU_WEBHOOK": "https://x", "PREMARKET_TELEGRAM_BOT_TOKEN": "t",
           "PREMARKET_TELEGRAM_CHAT_ID": "1"}
    res = NT.push([_card()], META, None, env)
    assert res == {"feishu": "sent", "telegram": "failed: ConnectionError"}
    assert sent and "NVDA" in sent[0][1]


def test_push_without_channels_sends_nothing():
    assert NT.push([_card()], META, None, {}) == {}


def test_email_retries_transient_disconnect(monkeypatch):
    calls = {"n": 0}

    def flaky(env, msg):
        calls["n"] += 1
        if calls["n"] < 3:
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

    monkeypatch.setattr(NT, "_smtp_once", flaky)
    monkeypatch.setattr(NT.time, "sleep", lambda s: None)
    env = {"PREMARKET_SMTP_HOST": "smtp.gmail.com", "PREMARKET_SMTP_PORT": "587",
           "PREMARKET_SMTP_USER": "a@b.c", "PREMARKET_SMTP_PASSWORD": "x",
           "PREMARKET_EMAIL_FROM": "a@b.c", "PREMARKET_EMAIL_TO": "d@e.f"}
    assert NT.send_email(env, "subj", "body", None) == 3
    assert calls["n"] == 3


def test_email_auth_failure_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def auth(env, msg):
        calls["n"] += 1
        raise smtplib.SMTPAuthenticationError(535, b"bad")

    monkeypatch.setattr(NT, "_smtp_once", auth)
    monkeypatch.setattr(NT.time, "sleep", lambda s: None)
    env = {"PREMARKET_SMTP_HOST": "smtp.gmail.com", "PREMARKET_SMTP_PORT": "587",
           "PREMARKET_SMTP_USER": "a@b.c", "PREMARKET_SMTP_PASSWORD": "x",
           "PREMARKET_EMAIL_FROM": "a@b.c", "PREMARKET_EMAIL_TO": "d@e.f"}
    with pytest.raises(smtplib.SMTPAuthenticationError):
        NT.send_email(env, "subj", "body", None)
    assert calls["n"] == 1


def test_push_records_email_failure_and_resend(monkeypatch, tmp_path):
    monkeypatch.setattr(NT.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def always_down(env, msg):
        attempts["n"] += 1
        raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

    monkeypatch.setattr(NT, "_smtp_once", always_down)
    env = {"PREMARKET_SMTP_HOST": "smtp.gmail.com", "PREMARKET_SMTP_PORT": "587",
           "PREMARKET_SMTP_USER": "a@b.c", "PREMARKET_SMTP_PASSWORD": "x",
           "PREMARKET_EMAIL_FROM": "a@b.c", "PREMARKET_EMAIL_TO": "d@e.f"}
    data, logs, reports = tmp_path / "data", tmp_path / "logs", tmp_path / "reports"
    res = NT.push([_card()], META, None, env, data_dir=data, logs_dir=logs)
    assert res == {"email": "failed: SMTPServerDisconnected"}
    assert NT.failed_channels(data, "2026-11-24") == ["email"]
    state = json.loads((data / "runs" / "2026-11-24" / "notify.json").read_text())
    assert state["channels"]["email"]["attempts"] == 3
    assert state["channels"]["email"]["status"] == "failed"
    lines = (logs / "notify.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["source"] == "push"

    def ok(env, msg):
        attempts["n"] += 1

    monkeypatch.setattr(NT, "_smtp_once", ok)
    again = NT.resend("2026-11-24", data, reports, logs, env)
    assert again == {"email": "sent"}
    assert NT.failed_channels(data, "2026-11-24") == []
    assert NT.resend("2026-11-24", data, reports, logs, env) == {}
