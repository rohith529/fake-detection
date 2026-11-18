"""Live learning: collect new labelled examples while the site runs, then retrain.

All persistent live data is stored directly in SQLite: data/history.db (live_data table).
Zero dependency on external flat CSV files.

Two sources populate live_data:
  feedback  user-submitted correction approved by admin     (label: 0 or 1, origin: 'feedback')
  rss       fresh headlines from sources.json rated >= 0.90 (label: 1, origin: 'rss')

CLI:
    python backend/learn.py harvest
    python backend/learn.py retrain
    python backend/learn.py status
"""
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import db

ROOT = Path(__file__).resolve().parent.parent
SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text())
MAX_RSS_ROWS = 3000
_lock = threading.Lock()
state = dict(running=False, last="never", error=None)
_log = logging.getLogger(__name__)


def add_feedback(text: str, label: int, ip_hash: str = "") -> bool:
    """Stage user feedback safely in SQLite feedback_staging table for review."""
    text = " ".join(text.split())
    if len(text.split()) < 5 or label not in (0, 1):
        return False
    con = db.get_db()
    try:
        con.execute(
            """INSERT INTO feedback_staging (ts, text, label, ip_hash, status)
               VALUES (datetime('now'), ?, ?, ?, 'pending')""",
            (text, label, ip_hash)
        )
        con.commit()
        return True
    except Exception as ex:
        _log.error("Failed to stage feedback in history.db: %s", ex)
        return False
    finally:
        con.close()


def approve_feedback(feedback_id: int) -> bool:
    """Approve a staged feedback row and promote it directly to live_data table in history.db."""
    con = db.get_db()
    try:
        row = con.execute(
            "SELECT text, label FROM feedback_staging WHERE id = ? AND status = 'pending'",
            (feedback_id,)
        ).fetchone()
        if not row:
            return False

        # Insert into live_data table in history.db
        db.insert_live_rows([dict(
            text=row["text"],
            label=row["label"],
            origin="feedback",
            ts=time.strftime("%Y-%m-%d %H:%M")
        )])

        con.execute("UPDATE feedback_staging SET status = 'approved' WHERE id = ?", (feedback_id,))
        con.commit()
        return True
    finally:
        con.close()


def harvest(per_source: int = 15, min_score: float = 0.90) -> int:
    """Pull recent headlines from highly reliable domains into history.db live_data."""
    import feedparser
    import requests

    seen, _, n_rss = db.get_live_texts_and_counts()
    rows = []
    for domain, score in SOURCES.items():
        if score < min_score or n_rss + len(rows) >= MAX_RSS_ROWS:
            continue
        try:
            r = requests.get(
                "https://news.google.com/rss/search",
                timeout=6,
                params=dict(q=f"site:{domain} when:2d", hl="en-IN", gl="IN", ceid="IN:en"),
                headers={"User-Agent": "VeritasDesk/2.0"}
            )
            for e in feedparser.parse(r.content).entries[:per_source]:
                title = e.title.rsplit(" - ", 1)[0].strip()
                if len(title.split()) >= 5 and title not in seen:
                    seen.add(title)
                    rows.append(dict(
                        text=title,
                        label=1,
                        origin="rss",
                        ts=time.strftime("%Y-%m-%d %H:%M")
                    ))
        except Exception as ex:
            _log.warning("skip %s: %s", domain, ex)

    if rows:
        db.insert_live_rows(rows)
    return len(rows)


def retrain():
    """Retrain model on static datasets + live_data from history.db."""
    import train
    state.update(running=True, error=None)
    try:
        train.main(verbose=False)
        state.update(last=time.strftime("%Y-%m-%d %H:%M"))
        return True
    except Exception as ex:
        state.update(error=str(ex))
        return False
    finally:
        state["running"] = False


def status() -> dict:
    """Fetch live data counts and feedback counts directly from history.db."""
    _, n_live, n_rss = db.get_live_texts_and_counts()
    con = db.get_db()
    try:
        pending_count = con.execute(
            "SELECT count(*) as c FROM feedback_staging WHERE status = 'pending'"
        ).fetchone()["c"]
    except Exception:
        pending_count = 0
    finally:
        con.close()

    return dict(
        live_rows=n_live,
        rss_rows=n_rss,
        pending_feedback=pending_count,
        running=state["running"],
        last=state["last"],
        error=state["error"]
    )


if __name__ == "__main__":
    db.init_db()
    cmd = (sys.argv[1:2] or ["status"])[0].lower()
    if cmd == "harvest":
        n = harvest()
        print(f"Harvested {n} new articles into history.db")
    elif cmd == "retrain":
        ok = retrain()
        print("Retrain " + ("succeeded" if ok else f"failed: {state['error']}"))
    elif cmd == "status":
        s = status()
        print(f"Live data in history.db: {s['live_rows']} total ({s['rss_rows']} RSS), {s['pending_feedback']} pending review.")
        print(f"Last retrain: {s['last']}, running: {s['running']}")
    else:
        print("Usage: python backend/learn.py [harvest|retrain|status]")
