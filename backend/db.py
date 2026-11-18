"""Database module for Veritas Desk.

All persistent runtime data is strictly centralized in data/history.db:
1. history: Past credibility analyses, verdicts, and scores.
2. flags: Unreliable domain frequency watchlist.
3. rss_cache: Persistent Google News RSS responses with 24h TTL.
4. feedback_staging: User feedback submitted from UI, awaiting verification.
5. live_data: Stored feedback and harvested headlines for model retraining (replaces live.csv).

High-concurrency configuration:
- PRAGMA journal_mode = WAL (Write-Ahead Logging)
- PRAGMA busy_timeout = 5000 (Prevents "database is locked" errors)
- PRAGMA synchronous = NORMAL
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data" / "history.db"


def init_db():
    """Run one-time DDL migrations, index creations, and legacy CSV migrations at application startup."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA busy_timeout=5000;")
        con.execute("PRAGMA synchronous=NORMAL;")

        # Application tables
        con.executescript("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                snippet TEXT NOT NULL,
                verdict TEXT NOT NULL,
                credibility INTEGER NOT NULL,
                domain TEXT
            );

            CREATE TABLE IF NOT EXISTS flags (
                domain TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS rss_cache (
                cache_key TEXT PRIMARY KEY,
                articles_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback_staging (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                text TEXT NOT NULL,
                label INTEGER NOT NULL,
                ip_hash TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS live_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT UNIQUE NOT NULL,
                label INTEGER NOT NULL,
                origin TEXT NOT NULL,
                ts TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts);
            CREATE INDEX IF NOT EXISTS idx_history_domain ON history(domain);
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON rss_cache(expires_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback_staging(status);
            CREATE INDEX IF NOT EXISTS idx_live_origin ON live_data(origin);
        """)
        con.commit()

        # Migrate legacy live.csv into live_data table if present
        legacy_csv = DB_FILE.parent / "live.csv"
        if legacy_csv.exists():
            import csv
            try:
                with legacy_csv.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows_to_insert = [
                        (r["text"].strip(), int(r["label"]), r.get("origin", "feedback"), r.get("ts", ""))
                        for r in reader if r.get("text") and str(r.get("label")) in ("0", "1")
                    ]
                    con.executemany(
                        "INSERT OR IGNORE INTO live_data (text, label, origin, ts) VALUES (?, ?, ?, ?)",
                        rows_to_insert
                    )
                con.commit()
                # Archive legacy file so runtime relies strictly on history.db
                bak_file = DB_FILE.parent / "live_migrated.csv.bak"
                if bak_file.exists():
                    bak_file.unlink()
                legacy_csv.rename(bak_file)
            except Exception:
                pass

    finally:
        con.close()


def get_db() -> sqlite3.Connection:
    """Return a configured SQLite connection with row factory enabled."""
    con = sqlite3.connect(DB_FILE, timeout=5.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=5000;")
    return con


# ── Persistent RSS Cache Operations ──────────────────────────────────────────

def cache_get(cache_key: str) -> Optional[list[dict]]:
    """Retrieve unexpired cached articles for cache_key."""
    now = time.time()
    con = get_db()
    try:
        row = con.execute(
            "SELECT articles_json FROM rss_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, now)
        ).fetchone()
        if row:
            return json.loads(row["articles_json"])
        return None
    except Exception:
        return None
    finally:
        con.close()


def cache_set(cache_key: str, articles: list[dict], ttl_seconds: int = 86400):
    """Store articles in persistent SQLite cache with an expiration timestamp."""
    now = time.time()
    expires_at = now + ttl_seconds
    articles_json = json.dumps(articles)
    con = get_db()
    try:
        con.execute(
            """INSERT INTO rss_cache (cache_key, articles_json, created_at, expires_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                   articles_json = excluded.articles_json,
                   created_at = excluded.created_at,
                   expires_at = excluded.expires_at""",
            (cache_key, articles_json, now, expires_at)
        )
        con.commit()
    except Exception:
        pass
    finally:
        con.close()


# ── Live Learning SQLite Operations ──────────────────────────────────────────

def insert_live_rows(rows: list[dict]) -> int:
    """Insert rows into live_data table in history.db."""
    if not rows:
        return 0
    con = get_db()
    try:
        data = [
            (r["text"].strip(), int(r["label"]), r.get("origin", "rss"), r.get("ts", time.strftime("%Y-%m-%d %H:%M")))
            for r in rows
        ]
        cur = con.cursor()
        cur.executemany(
            "INSERT OR IGNORE INTO live_data (text, label, origin, ts) VALUES (?, ?, ?, ?)",
            data
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def get_live_texts_and_counts() -> tuple[set[str], int, int]:
    """Return (seen_texts_set, total_live_count, rss_count) from history.db."""
    con = get_db()
    try:
        rows = con.execute("SELECT text, origin FROM live_data").fetchall()
        seen = {r["text"] for r in rows}
        total = len(rows)
        n_rss = sum(1 for r in rows if r["origin"] == "rss")
        return seen, total, n_rss
    finally:
        con.close()
