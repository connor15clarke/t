

# cbnew — Coaching Jobs Scraper

Scrapes **K‑12 athletic coaching jobs** from district career pages and writes them to structured XML. The scraper favors **cheap checks first** and **escalates only when needed**:

1. **PaddleOCR** (local, free) →
2. **Azure AI Vision (Image Analysis v4 “read”)** →
3. **Gemini Computer‑Use** (browser agent for hard cases)

This “cheap‑first” design slashes rescrape costs and tracks page **fingerprints** so unchanged pages are skipped. The project measures three numbers after each run:
**A)** pages skipped as “no change,” **B)** pages handled by cheap OCR, **C)** pages that escalated to Gemini. 

---

## Project layout

```
cbnew/
├─ imagescrape1.py        # Orchestrator; discovery & rescrape flows (Gemini agent + browser) 
├─ district_index.py      # Loads districts from state CSV(s), career URLs as entry points
├─ jobs_xml.py            # XML writer + canonicalization + seen/update helpers
├─ vision_ocr.py          # PaddleOCR + Azure Image Analysis v4 (“read”) wrappers
├─ vision_router.py       # Cheap-first escalation (Paddle → Azure → Gemini)
├─ footprints.py          # SQLite store for page fingerprints + escalation logs
├─ job_sites.py           # ATS heuristics; closed/expired page cues, parsing helpers
├─ job_sweeper.py         # Daily sweeper to mark inactive/closed jobs
│
├─ data/                  # State CSV files with district career URLs (input)
└─ out/                   # XML outputs by state: AAAAA{STATE}.xml
   └─ debug/              # Optional: model/browser debug dumps (if enabled)
```

* **Entry points & CSVs:** The loader treats CSV‑provided **career URLs** as **primary entry points** and uses the **homepage** only as a fallback. Multiple “career” column names are accepted (e.g., `career_url`, `jobs_url`, `hr_url`, etc.). 
* **XML writing:** The writer produces a consistent XML schema (job fields as elements), canonicalizes Apply URLs, and lets you **mark jobs seen** or **update by Apply URL** on rescrape. 
* **Vision pipeline:** PaddleOCR + Azure “read” (v4) are used on job **detail page screenshots**; if confidence/length thresholds aren’t met or content changed, the page **escalates to Gemini**.
* **Footprints DB:** Stores screenshot/text hashes, last model used, escalations, and run summaries so rescrapes can skip unchanged pages. 
* **Orchestrator:** Discovery (from CSV career URLs) and Rescrape (from known Apply URLs) are both implemented in `imagescrape1.py`. The Gemini **Computer‑Use** model ID defaults to `gemini-2.5-computer-use-preview-10-2025`. 
* **ATS helpers + Sweeper:** Heuristics for common ATS and a daily sweeper to mark stale/closed jobs inactive.

---

## Requirements

* **Python 3.10+** recommended
* System packages for **Playwright** (Chromium) and **Pillow**
* Python libs:

  ```bash
  pip install google-genai python-dotenv playwright pillow lxml paddleocr requests beautifulsoup4
  ```

  Then install browsers:

  ```bash
  python -m playwright install
  ```

> **Notes**
>
> * **PaddleOCR** pulls additional deps (e.g., numpy).
> * Azure step uses simple **REST** via `requests` (no Azure SDK required). 
> * The browser runs **headful by default** (set `PLAYWRIGHT_HEADLESS=1` to run headless). 

---

## Configuration

Create a `.env` file in `cbnew/`:

```ini
GOOGLE_API_KEY=YOUR_VERTEX_OR_GOOGLE_GENAI_KEY

# Azure Image Analysis v4 (“read”)
AZURE_VISION_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
AZURE_VISION_KEY=YOUR_AZURE_KEY

# Optional runtime knobs
PLAYWRIGHT_HEADLESS=1
ACTION_BUDGET_START=25
DEBUG_DUMP_DIR=cbnew/out/debug
FOOTPRINTS_DB=cbnew/footprints.sqlite
VISION_ORDER=paddle,azure,gemini
```

Key defaults come from `imagescrape1.py` (model ID, action budget, etc.). 

---

## Data inputs (CSV)

Place state CSVs in `cbnew/data/` (or point `--data_dir` elsewhere). The loader accepts many header variants; key fields include:

* `district`, `name`, `homepage` (optional)
* **Career/Jobs URLs** (one or many): accepts any column named like `career_url`, `careers_url`, `jobs_url`, `hr_url`, `ats_url`, `coach_search_url`, or **any column starting with** `career`, `jobs`, `job`, `hr`, `ats`, `portal`. Multiple URLs can be `;` or `|` separated. 

**Tip:** You can also pass a **single CSV path** to `--data_dir`; the loader will use it directly. 

---

## Outputs (XML)

XML files are written per state to `cbnew/out/` with the name `AAAAA{STATE}.xml` (e.g., `AAAAACO.xml`). Each job is a `<Job>` element with child elements for fields like `JobTitle`, `ApplyUrl`, `PostingDate`, etc. Dates are normalized when possible to `YYYY-MM-DD`. 

On rescrape, **existing jobs are matched by canonicalized Apply URL**. The writer supports:

* `seen_apply_urls()` → the set of known Apply URLs
* `mark_seen_by_apply_url(url)` → update `lastSeen` / `active`
* `update_fields_by_apply_url(url, fields)` → in‑place refresh of values if content changed 

---

## Quickstart

### 1) Discovery run (find jobs from the district career pages)

> Starts from the **career URLs** listed in your CSV and lets the Gemini agent navigate to each job’s detail page and call `extract_fields`. 

```bash
# Run as a module so relative imports work
python -m cbnew.imagescrape1 CO --data_dir cbnew/data --log-level INFO

# Limit to first N districts while testing
python -m cbnew.imagescrape1 CO --data_dir cbnew/data --limit 5 -v
```

Outputs land in `cbnew/out/AAAAACO.xml`.

---

### 2) Rescrape run (cheap‑first; escalate only when needed)

> Revisits **known Apply URLs** from the XML, takes **one screenshot** per job detail page, and uses the **Vision Router**: Paddle → Azure → Gemini **only if** OCR confidence/length is low or content changed. Produces the three headline numbers: skip/OCR/Gemini.

```bash
# All three stages in default order
python -m cbnew.imagescrape1 CO --rescrape --log-level INFO

# Just Paddle
python -m cbnew.imagescrape1 CO --rescrape --preset paddle

# Just Azure
python -m cbnew.imagescrape1 CO --rescrape --preset azure

# Just Gemini
python -m cbnew.imagescrape1 CO --rescrape --preset gemini

# Paddle + Azure (no Gemini)
python -m cbnew.imagescrape1 CO --rescrape --preset paddle+azure

# Paddle + Gemini (skip Azure)
python -m cbnew.imagescrape1 CO --rescrape --preset paddle+gemini --vision-order paddle,gemini

# Azure + Gemini (skip Paddle)
python -m cbnew.imagescrape1 CO --rescrape --preset azure+gemini --vision-order azure,gemini
```

The router logs per‑URL escalations in **SQLite** (`cbnew/footprints.sqlite`) and prints a run summary at the end. Tables:

* `pages(url, screenshot_sha, text_sha_paddle, text_sha_azure, last_model, ...)`
* `escalations(ts, url, from_model, to_model, reason, info)`
* `runs(ts, skipped_nochange, used_cheap_ocr, escalated_to_gemini)` 

---

## How the escalation works

1. **Screenshot** a job detail page
2. **PaddleOCR** extracts text; if confident & long enough **and** hasn’t changed vs. last time → **mark seen** (cheap).
3. Otherwise **Azure “read”** tries again. If still insufficient or changed →
4. **Gemini Computer‑Use** extracts fields by calling your `extract_fields` function exactly once on the detail page.

Thresholds are configurable in `vision_router.py` (e.g., min chars, min confidence). 

---

## Logs & Debug

* **Console logs** respect `--log-level` (or `-v` for DEBUG). 
* **Model/browser dumps:** set `DEBUG_DUMP_DIR=cbnew/out/debug` or pass `--debug-dump-dir cbnew/out/debug`. The orchestrator appends **FunctionResponse** screenshots and optional “state updates” after extracts to keep the agent on task. 
* **Footprints DB:** inspect with `sqlite3 cbnew/footprints.sqlite` (see tables above). 

---

## Daily Sweeper (optional)

The sweeper re‑checks existing jobs and marks them **inactive** if the Apply page shows **“filled/closed/not found”** cues or otherwise looks inactive—using the same general cues as the ATS helpers. Run it against a specific XML file, e.g.:

```bash
python cbnew/job_sweeper.py --xml cbnew/out/AAAAACO.xml
```

It updates `active="false"` and `lastSeen` in‑place. 

---

## ATS & field helpers

* `job_sites.py` provides **closed‑page cues**, basic salary/ZIP extraction, and vendor hints to derive Apply URLs. These heuristics complement the browser agent’s extraction. 

---

## Environment variables (summary)

* `GOOGLE_API_KEY` — required (Gemini). **Model** default: `gemini-2.5-computer-use-preview-10-2025`. 
* `AZURE_VISION_ENDPOINT`, `AZURE_VISION_KEY` — required to enable the Azure OCR stage. 
* `PLAYWRIGHT_HEADLESS` — `1` to run headless; default is **headful**. 
* `ACTION_BUDGET_START` — safety brake on per‑site action count (default `25`). 
* `DEBUG_DUMP_DIR` — where to store per‑turn dumps (recommended: `cbnew/out/debug`). 
* `FOOTPRINTS_DB` — path to SQLite fingerprints DB (default `cbnew/footprints.sqlite`). 
* `VISION_ORDER` — override the pipeline order (comma‑separated). 

---

## FAQ

**Q: Do I need both Paddle and Azure?**
A: No. You can run any combination using `--preset` or the `--enable-*` flags. The router respects your order (e.g., `--vision-order azure,gemini`). 

**Q: Where does the system remember what changed?**
A: In `footprints.sqlite` (`pages` table holds screenshot/text hashes per URL). If the **screenshot hash matches**, the page is skipped as “no change.” 

**Q: How are duplicates avoided?**
A: The XML writer **canonicalizes Apply URLs** and uses them as the de‑dup key; rescrape updates by Apply URL instead of appending duplicates. 

**Q: Where do I see the three numbers after a rescrape?**
A: In the console at the end of the run and optionally recorded in `runs` (footprints DB):
A) skipped_nochange, B) used_cheap_ocr, C) escalated_to_gemini. 

---

## Roadmap (matches the two‑week plan)

* **Week 1:** measurement (A/B/C), rescrape mode, cheap OCR, fingerprints, trial/tuning.
* **Week 2:** daily sweeper, date sanity, sampling QA, scheduling, health report, rollout. 

---

## Troubleshooting

* **`GOOGLE_API_KEY not found`** → Set it in `.env`. 
* **Azure OCR disabled** → Ensure `AZURE_VISION_ENDPOINT` and `AZURE_VISION_KEY` are set; otherwise the router will skip the Azure stage. 
* **Playwright errors** → Run `python -m playwright install` and consider `PLAYWRIGHT_HEADLESS=1` on CI. 
* **PaddleOCR import error** → `pip install paddleocr` (it may take time to build/download). 
* **Sweeper import error** when run as a module → run `python cbnew/job_sweeper.py` (direct), since `job_sweeper.py` uses non‑relative imports. 

---

---

### References (repo internal)

* **Orchestrator & Gemini agent:** imagescrape1.py. 
* **District CSV loader:** district_index.py. 
* **XML writer & helpers:** jobs_xml.py. 
* **Vision OCR wrappers:** vision_ocr.py. 
* **Escalation router:** vision_router.py. 
* **Footprints DB:** footprints.py. 
* **ATS helpers:** job_sites.py. 
* **Daily sweeper:** job_sweeper.py. 
* **Two‑week operating plan:** notes. 

---
