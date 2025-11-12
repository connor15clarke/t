# -*- coding: utf-8 -*-
"""
jobs_xml.py
===========

Create, append to, and sweep a structured XML file of coaching jobs.

- Uses lxml for robust pretty-printing and optional XSD validation.
- All fields are written as elements; optional fields may be empty strings.

XML Shape:

<Jobs generatedAt="..." schemaVersion="1.0">
  <Job id="..."
       state="TX"
       district="Austin ISD"
       districtId="..."
       active="true"
       lastSeen="2025-10-03T00:00:00Z">
    <JobTitle>Head Football Coach</JobTitle>
    <JobDescription>...</JobDescription>
    <JobType>full-time</JobType>
    <Location>LBJ Early College High School</Location>
    <City>Austin</City>
    <State>TX</State>
    <Country>USA</Country>
    <ZipCode>78723</ZipCode>
    <Experience>3+ years varsity coaching</Experience>
    <SalaryRange>$65,000 - $78,000</SalaryRange>
        <Benefits>...</Benefits>
    <PostingDate>2025-10-03</PostingDate>
    <ClosingDate>2025-10-31</ClosingDate>
<JobUrl>https://...</JobUrl>
    <ApplyUrl>https://...</ApplyUrl>
    <CoachSearchUrl>https://...</CoachSearchUrl>
    <EmployerEmail>principal@school.org</EmployerEmail>
    <EmployerFullName>LBJ Early College High School</EmployerFullName>
    <CompanyDescription>District overview...</CompanyDescription>
    <CompanyEmail>hr@austinisd.org</CompanyEmail>
    <CompanyName>Austin Independent School District</CompanyName>
  </Job>
</Jobs>
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Iterable
from uuid import uuid4
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import re

try:
    from lxml import etree as ET
except Exception:
    
    import xml.etree.ElementTree as ET  # type: ignore

# Only selected fields are written as elements (field mask). Optional fields may be empty strings.

ALL_FIELD_KEYS = [
    "job_title","job_description","job_type","sport","location","city","state","country","zip_code",
    "experience","salary_range","benefits","posting_date","closing_date","job_url","apply_url","coach_search_url","employer_email","employer_full_name",
    "company_description","company_email","company_name"
]
TAG_BY_KEY = {
    "job_title": "JobTitle",
    "job_description": "JobDescription",
    "job_type": "JobType",
    "sport": "Sport",
    "location": "Location",
    "city": "City",
    "state": "State",
    "country": "Country",
    "zip_code": "ZipCode",
    "experience": "Experience",
    "salary_range": "SalaryRange",
    "benefits": "Benefits",
    "posting_date": "PostingDate",
    "closing_date": "ClosingDate",
    "job_url": "JobUrl",
    "apply_url": "ApplyUrl",
    "coach_search_url": "CoachSearchUrl",
    "employer_email": "EmployerEmail",
    "employer_full_name": "EmployerFullName",
    "company_description": "CompanyDescription",
    "company_email": "CompanyEmail",
    "company_name": "CompanyName",
}

JOB_KEYS = {"jobid","jid","id","rid","req","reqid","requisition","requisitionid",
            "postingid","positionid","vacancyid","oppid"}
TRACKERS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content",
            "gclid","fbclid","mc_cid","mc_eid","msclkid"}
@dataclass
class JobXMLRecord:
    job_title: str
    job_description: str
    job_type: str  # full-time | part-time | remote | hybrid | seasonal | stipend
    sport: str
    location: str
    city: str
    state: str
    country: str
    zip_code: str
    experience: str
    salary_range: str
    benefits: str
    posting_date: str
    closing_date: str
    job_url: str
    apply_url: str
    coach_search_url: str
    employer_email: str
    employer_full_name: str
    company_description: str
    company_email: str
    company_name: str

    # book-keeping
    district: Optional[str] = None
    district_id: Optional[str] = None
    active: bool = True
    last_seen: Optional[str] = None  # ISO timestamp when observed


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_date_str(s: str) -> str:
    """
    Try to coerce a free-form date into YYYY-MM-DD. If parsing fails, return the original string.
    Accepts formats like '2025-10-03', '10/03/2025', 'Oct 3, 2025', 'October 03 2025'.
    """
    from datetime import datetime
    s2 = (s or "").strip()
    if not s2:
        return ""
    # Try a few common formats
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s2, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return s2
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

class JobsXML:
    def __init__(self, path: str, schema_version: str = "1.0", field_mask: Optional[set[str]] = None):
        self.path = path
        self.schema_version = schema_version
        self._tree = None  # type: ignore
        self._root = None  # type: ignore
        self._field_mask = set(field_mask or []) or set(ALL_FIELD_KEYS)

    def _ensure_tree(self):
        if self._tree is not None:
            return
        try:
            # try parse existing
            self._tree = ET.parse(self.path)
            self._root = self._tree.getroot()
        except Exception:
            # create new
            self._root = ET.Element("Jobs", attrib={
                "generatedAt": _now_iso(),
                "schemaVersion": self.schema_version
            })
            self._tree = ET.ElementTree(self._root)

    def append_jobs(self, jobs: Iterable[JobXMLRecord]) -> None:
        self._ensure_tree()
        for job in jobs:
            jid = str(uuid4())
            elem = ET.SubElement(self._root, "Job", attrib={
                "id": jid,
                "state": (job.state or ""),
                "district": (job.district or job.company_name or ""),
                "districtId": (job.district_id or ""),
                "active": "true" if job.active else "false",
                "lastSeen": job.last_seen or _now_iso(),
            })
            for key in ALL_FIELD_KEYS:
                tag = TAG_BY_KEY[key]
                # If the key was selected/pulled, write the value; otherwise emit an empty tag.
                if key in self._field_mask:
                    value = (getattr(job, key) if hasattr(job, key) else (job.get(key) if isinstance(job, dict) else "")) or ""
                else:
                    value = ""
                child = ET.SubElement(elem, tag)
                if key in {"posting_date","closing_date"} and value:
                    try:
                        value = _normalize_date_str(str(value))
                    except Exception:
                        pass
                child.text = value

    def mark_seen(self, job_id: str, active: Optional[bool] = None) -> None:
        self._ensure_tree()
        for job in self._root.findall("Job"):
            if job.get("id") == job_id:
                if active is not None:
                    job.set("active", "true" if active else "false")
                job.set("lastSeen", _now_iso())
                break

    def write(self) -> None:
        self._ensure_tree()
        try:
            # lxml pretty print path
            ET.indent(self._tree, space="  ")  # type: ignore[attr-defined]
        except Exception:
            pass
        self._tree.write(self.path, encoding="utf-8", xml_declaration=True)

    # --- Canonicalization utilities (shared) ---
    def canonicalize_url(self, u: str) -> str:
        """
        Canonicalize a job/apply URL by stripping trackers and keeping only job-identifying params.
        Mirrors the logic used for de-dup keys.
        """
        try:
            p = urlparse(u)
        except Exception:
            return (u or "").strip()
        keep = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            kl = k.lower()
            if kl in TRACKERS:
                continue
            if kl in JOB_KEYS or ("job" in kl and any(ch.isdigit() for ch in v)):
                keep.append((k, v))
        frag = p.fragment if re.search(r"(job|posting|vacancy|req|rid|id)", (p.fragment or "").lower()) else ""
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(keep, doseq=True), frag))

    def seen_apply_urls(self) -> set[str]:
        """
        Return the set of canonical Apply URLs already present in this XML.
        """
        self._ensure_tree()
        out: set[str] = set()
        for job in self._root.findall("Job"):
            au = (job.findtext("ApplyUrl") or "").strip()
            if au:
                out.add(self.canonicalize_url(au))
        return out

    def mark_seen_by_apply_url(self, apply_url: str, *, active: Optional[bool] = None) -> bool:
        self._ensure_tree()

        target = self.canonicalize_url((apply_url or "").strip())
        for job in self._root.findall("Job"):
            au = self.canonicalize_url((job.findtext("ApplyUrl") or "").strip())
            if au == target:
                if active is not None:
                    job.set("active", "true" if active else "false")
                job.set("lastSeen", _now_iso())
                return True
        return False
