# -*- coding: utf-8 -*-
"""
district_index.py
-----------------

Loads districts for a given state from either:
  • a directory containing "{STATE}.csv" files, or
  • a direct CSV file path.

It now supports a CSV that includes BOTH the district homepage and one or
more pre-resolved Career/Jobs/ATS URLs. We treat the CSV-provided career
URLs as primary entrypoints and use the homepage to build the district
profile. Homepage discovery is only used as a fallback when CSV values are
missing or unusable.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional, List, Dict, Iterable

logger = logging.getLogger("coach_jobs")

@dataclass
class District:
    state: str
    name: str
    homepage: str
    district_email: Optional[str] = None
    about_url: Optional[str] = None
    district_id: Optional[str] = None  # external id (optional)
    # NEW: one or many pre-resolved career/job portal URLs (from CSV)
    career_urls: List[str] = field(default_factory=list)

# ---------- CSV helpers ----------

def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    return p

def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        # default to comma
        return ","

def _read_csv_rows(p: Path) -> Iterable[Dict[str, str]]:
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    sample = text[:2048]
    delim = _sniff_delimiter(sample)
    f = io.StringIO(text)
    reader = csv.DictReader(f, delimiter=delim)
    for row in reader:
        # normalize keys to lower+strip, keep original values (strip)
        norm = {}
        for k, v in (row or {}).items():
            kk = (k or "").strip().lower()
            vv = (v or "").strip()
            norm[kk] = vv
        yield norm

def _first_nonempty(row: Dict[str,str], *keys: str, default: Optional[str]=None) -> Optional[str]:
    for k in keys:
        if not k: 
            continue
        v = row.get(k.strip().lower())
        if v:
            return v
    # fallback: first non-empty field in row
    if default is not None:
        return default
    for v in row.values():
        if v:
            return v
    return None

def _normalize_homepage(u: Optional[str]) -> str:
    if not u:
        return ""
    u = u.strip()
    if not u:
        return ""
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u.lstrip("/")
    return u

def _collect_career_urls(row: Dict[str,str]) -> List[str]:
    # Accept many header variants (case-insensitive). Also accept any header that starts with
    # 'career', 'jobs', 'job', 'hr', 'ats', or 'portal' as a possible link field.
    career_keys = {
        "career_url","careers_url","career","career page","employment_url","employment page",
        "jobs_url","jobs page","job_board_url","job board","ats_url","portal_url","hr_url","hr page",
        "coach_url","coach_search_url"
    }
    out: List[str] = []
    for k, v in row.items():
        if not v:
            continue
        if k in career_keys or k.startswith(("career","jobs","job","hr","ats","portal")):
            # allow multiple per cell separated by ; or |
            parts = re.split(r"[;|]+", v)
            for part in parts:
                u = (part or "").strip()
                if u and not u.lower().startswith(("http://","https://")):
                    u = "https://" + u.lstrip("/")
                if u and u not in out:
                    out.append(u)
    return out

def _iter_from_file(p: Path, state_abbrev: str) -> Iterable[District]:
    for row in _read_csv_rows(p):
        name = _first_nonempty(row, "district","district_name","name","school district")
        homepage = _normalize_homepage(_first_nonempty(
            row,
            "homepage","homepage_url","home page","website","site","url",
            "district_homepage","district homepage","www"
        ))
        district_email = _first_nonempty(row, "district_email","email","contact_email","hr_email")
        about_url = _first_nonempty(row, "about_url","about page","about")
        district_id = _first_nonempty(row, "district_id","lea_id","id")
        career_urls = _collect_career_urls(row)

        yield District(
            state=state_abbrev,
            name=(name or "").strip() or "Unknown District",
            homepage=homepage or "",
            district_email=(district_email or None),
            about_url=(about_url or None),
            district_id=(district_id or None),
            career_urls=career_urls,
        )

def _find_state_csv(base_path: Path, state_abbrev: str) -> Optional[Path]:
    state_abbrev = (state_abbrev or "").strip().upper()
    if not state_abbrev:
        return None
    # Prefer exact '{STATE}.csv'
    exact = base_path / f"{state_abbrev}.csv"
    if exact.exists():
        return exact
    # case-insensitive search
    cands = list(base_path.glob("*.csv"))
    for p in cands:
        if p.name.lower() == f"{state_abbrev.lower()}.csv":
            return p
    # fuzzy: match e.g., 'Top School Districts - CO.csv'
    rx = re.compile(rf"(^|[/\-_ ]){re.escape(state_abbrev.lower())}(\.|\b).*\.csv$", re.I)
    fuzzy = [p for p in cands if rx.search(p.name.lower())]
    if fuzzy:
        fuzzy.sort(key=lambda x: (len(x.name), x.name.lower()))
        return fuzzy[0]
    return None

async def iter_districts(state_abbrev: str, base_dir: str = "data/districts") -> AsyncIterator[District]:
    """
    Yields District rows for a given state from a directory or a direct CSV path.
    """
    state_abbrev = (state_abbrev or "").strip().upper()
    base_path = _resolve(base_dir)
    logger.info("district_index: base path = %s (exists=%s, is_file=%s, is_dir=%s)",
                base_path, base_path.exists(), base_path.is_file(), base_path.is_dir())

    if base_path.exists():
        if base_path.is_file():
            # Direct CSV passed in
            logger.info("district_index: using CSV file %s", base_path.name)
            for d in _iter_from_file(base_path, state_abbrev):
                yield d
            return
        if base_path.is_dir():
            best = _find_state_csv(base_path, state_abbrev)
            if best:
                logger.info("district_index: resolved %s", best.name)
                for d in _iter_from_file(best, state_abbrev):
                    yield d
                return
            try:
                names = sorted([p.name for p in base_path.glob("*.csv")])
            except Exception:
                names = []
            logger.warning("district_index: no '%s.csv' under %s. CSVs here: %s",
                           state_abbrev, base_path, names)
            return

    logger.warning("district_index: path not found or unsupported: %s", base_path)
    return

__all__ = ["District", "iter_districts"]
