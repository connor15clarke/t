# footprints.py
from __future__ import annotations
import sqlite3, time, json
from dataclasses import dataclass
from typing import Optional, Dict, Any

CREATE_PAGES = """
CREATE TABLE IF NOT EXISTS pages (
  url TEXT PRIMARY KEY,
  canonical_url TEXT,
  first_seen INTEGER,
  last_seen INTEGER,
  last_model TEXT,
  screenshot_sha TEXT,
  text_sha_paddle TEXT,
  text_sha_azure TEXT,
  etag TEXT,
  last_modified TEXT,
  notes TEXT
);"""

CREATE_ESCAL = """
CREATE TABLE IF NOT EXISTS escalations (
  ts INTEGER,
  url TEXT,
  from_model TEXT,
  to_model TEXT,
  reason TEXT,
  info TEXT
);"""

CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
  ts INTEGER PRIMARY KEY,
  skipped_nochange INTEGER,
  used_cheap_ocr INTEGER,
  escalated_to_gemini INTEGER
);"""

@dataclass
class PageFP:
  url: str
  screenshot_sha: str = ""
  text_sha_paddle: str = ""
  text_sha_azure: str = ""
  last_model: str = ""
  canonical_url: str = ""
  etag: str = ""
  last_modified: str = ""
  last_seen: int = 0

class Footprints:
    def __init__(self, path: str = "cbnew/footprints.sqlite"):
        self.db = sqlite3.connect(path)
        self.db.execute(CREATE_PAGES); self.db.execute(CREATE_ESCAL); self.db.execute(CREATE_RUNS)
        self.db.commit()

    def get(self, url: str) -> Optional[PageFP]:
        cur = self.db.execute("SELECT url, screenshot_sha, text_sha_paddle, text_sha_azure, last_model, canonical_url, etag, last_modified, last_seen FROM pages WHERE url=?", (url,))
        row = cur.fetchone()
        if not row: return None
        return PageFP(url=row[0], screenshot_sha=row[1] or "", text_sha_paddle=row[2] or "", text_sha_azure=row[3] or "",
                      last_model=row[4] or "", canonical_url=row[5] or "", etag=row[6] or "", last_modified=row[7] or "", last_seen=row[8] or 0)

    def upsert(self, url: str, **kvs):
        now = int(time.time())
        kvs = {k:v for k,v in kvs.items() if v is not None}
        existing = self.get(url)
        if existing:
            sets = ",".join([f"{k}=?" for k in kvs.keys()])
            vals = list(kvs.values()) + [url]
            self.db.execute(f"UPDATE pages SET {sets}, last_seen=? WHERE url=?", (*list(kvs.values()), now, url))
        else:
            cols = ["url","first_seen","last_seen"] + list(kvs.keys())
            qs = ",".join("?" for _ in cols)
            vals = [url, now, now] + list(kvs.values())
            self.db.execute(f"INSERT INTO pages ({','.join(cols)}) VALUES ({qs})", vals)
        self.db.commit()

    def record_escalation(self, url: str, from_model: str, to_model: str, reason: str, info: Dict[str,Any]):
        self.db.execute("INSERT INTO escalations (ts,url,from_model,to_model,reason,info) VALUES (?,?,?,?,?,?)",
                        (int(time.time()), url, from_model, to_model, reason, json.dumps(info)[:2000]))
        self.db.commit()

    def record_run_summary(self, skipped: int, used_ocr: int, escalated: int):
        self.db.execute("INSERT INTO runs (ts, skipped_nochange, used_cheap_ocr, escalated_to_gemini) VALUES (?,?,?,?)",
                        (int(time.time()), skipped, used_ocr, escalated))
        self.db.commit()
