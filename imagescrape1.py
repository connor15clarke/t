
import os
import csv
import base64
import argparse
import asyncio
import time
import logging
from io import BytesIO
from typing import List, Dict, Any, Optional, Iterable

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page, Playwright

from google import genai
from google.genai import types
from google.genai.types import Content, Part, FunctionResponse

from PIL import Image

# Import your provided XML writer (package-relative when run as module: python -m cbnew.imagescrape_verbose_mapped)
from .jobs_xml import JobsXML, JobXMLRecord
from .district_index import District, iter_districts

# ------------------------------
# Config / Constants
# ------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in a .env file.")

SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
MODEL_ID = "gemini-2.5-computer-use-preview-10-2025"
ACTION_BUDGET_START = int(os.getenv("ACTION_BUDGET_START", "25"))  # Safety brake to prevent infinite loops
BROWSER_ACTION_CALLS = {
    "navigate", "click_at", "type_text_at", "scroll_document", "hover_at",
    "go_back", "go_forward", "wait_5_seconds", "key_combination"
}

# Default set of fields the model should extract from job pages.
DEFAULT_EXTRACT_FIELDS = [
    "job_title","job_description","job_type","location","city","state","zip_code",
    "experience","salary_range","benefits","job_url","apply_url","employer_email",
    "posting_date","closing_date","sport",
]

# Guidance for the agent to use the right scrolling primitive on nested UIs.
AGENT_HINTS = (
    "If the job list is inside a scrollable panel, prefer calling 'scroll_at' and target the panel area. "
    "Use 'scroll_document' only for full-page scrolling."
    "If scrolling appears stuck, try 'key_combination' with PageUp/Home/PageDown/End."
)

# Logger
logger = logging.getLogger("imagescrape")


# ------------------------------
# Utility / Helpers
# ------------------------------

def setup_logging(level: str = "INFO", verbose: bool = False):
    """
    Configure root logging once per process.
    """
    # If --verbose is set, override to DEBUG
    if verbose:
        level = "DEBUG"
    lvl = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.debug("Logging configured. Level=%s", logging.getLevelName(lvl))

def denorm_x(x: int) -> int:  # 0..999 -> pixels
    px = int(x / 1000 * SCREEN_WIDTH)
    logger.debug("Denorm X: %s -> %s (px)", x, px)
    return px


def denorm_y(y: int) -> int:
    py = int(y / 1000 * SCREEN_HEIGHT)
    logger.debug("Denorm Y: %s -> %s (px)", y, py)
    return py


def has_function_calls(response) -> bool:
    cand = response.candidates[0]
    has_any = any(getattr(p, "function_call", None) for p in (cand.content.parts or []))
    logger.debug("Model returned function calls? %s", has_any)
    if has_any:
        for i, p in enumerate(cand.content.parts or []):
            fc = getattr(p, "function_call", None)
            if fc:
                logger.debug("  Call %d: name=%s args=%s", i, fc.name, dict(fc.args or {}))
    else:
        # If no function calls, log any text the model returned (truncated).
        texts = [getattr(p, "text", "") for p in (cand.content.parts or []) if getattr(p, "text", "")]
        if texts:
            logger.debug("Model text: %s", (texts[0][:400] + "...") if len(texts[0]) > 400 else texts[0])
    return has_any


# --- Robust scrolling helpers ---

def _do_scroll_document(page: Page, direction: str = "down", magnitude: int = 800):
    """
    Scroll the main document with wheel; if no movement, fall back to keyboard.
    Returns a dict with before/after and whether a fallback was used.
    """
    # Before offsets
    before = page.evaluate("({x: window.scrollX, y: window.scrollY})")
    fallback_used = False
    dx, dy = 0, 0
    if direction in ("up", "down"):
        dy = -abs(magnitude) if direction == "up" else abs(magnitude)
        page.mouse.wheel(0, dy)
    else:
        dx = -abs(magnitude) if direction == "left" else abs(magnitude)
        page.evaluate(f"window.scrollBy({dx}, 0)")

    after = page.evaluate("({x: window.scrollX, y: window.scrollY})")
    moved = (before != after)

    if not moved:
        # Keyboard fallback (some sites trap wheel events)
        key = "PageUp" if direction == "up" else "PageDown"
        if direction in ("left", "right"):
            key = "Home" if direction == "left" else "End"
        page.keyboard.press(key)
        fallback_used = True
        after2 = page.evaluate("({x: window.scrollX, y: window.scrollY})")
        moved = (before != after2)
        after = after2

    logger.debug("scroll_document: dir=%s mag=%s before=%s after=%s moved=%s fallback=%s",
                 direction, magnitude, before, after, moved, fallback_used)
    return {"direction": direction, "magnitude": magnitude, "before": before, "after": after,
            "moved": moved, "fallback_used": fallback_used}


def _do_scroll_at(page: Page, nx: int, ny: int, direction: str = "down", magnitude: int = 800):
    """
    Scroll the nearest scrollable ancestor of the element at normalized (nx, ny) in 0..999 space.
    Returns details of the target container and before/after offsets. Falls back to PageUp/PageDown.
    """
    dx, dy = 0, 0
    if direction in ("up", "down"):
        dy = -abs(magnitude) if direction == "up" else abs(magnitude)
    else:
        dx = -abs(magnitude) if direction == "left" else abs(magnitude)

    # Execute in-page JS to find a scrollable container and scroll it.
    result = page.evaluate(
        '''
        ({nx, ny, dx, dy}) => {
          const px = Math.round(nx/1000 * window.innerWidth);
          const py = Math.round(ny/1000 * window.innerHeight);
          let el = document.elementFromPoint(px, py);

          const isScrollable = (e) => {
            if (!e) return false;
            const s = getComputedStyle(e);
            const yOk = (s.overflowY === 'auto' || s.overflowY === 'scroll') && e.scrollHeight > e.clientHeight;
            const xOk = (s.overflowX === 'auto' || s.overflowX === 'scroll') && e.scrollWidth > e.clientWidth;
            return yOk || xOk;
          };

          let target = el;
          while (target && !isScrollable(target)) target = target.parentElement;
          if (!target) target = document.scrollingElement || document.documentElement;

          const before = {x: target.scrollLeft || 0, y: target.scrollTop || 0};
          target.scrollBy({left: dx, top: dy});
          const after = {x: target.scrollLeft || 0, y: target.scrollTop || 0};

          const r = target.getBoundingClientRect ? target.getBoundingClientRect() : {top:0,left:0,bottom:0,right:0,width:0,height:0};
          return {
            target: {
              tag: target.tagName || 'UNKNOWN',
              id: target.id || '',
              class: (target.className && String(target.className)) || '',
              rect: {top: r.top, left: r.left, bottom: r.bottom, right: r.right, width: r.width, height: r.height},
              scrollWidth: target.scrollWidth || 0,
              scrollHeight: target.scrollHeight || 0,
              clientWidth: target.clientWidth || 0,
              clientHeight: target.clientHeight || 0
            },
            before, after
          };
        }
        ''',
        {"nx": int(nx), "ny": int(ny), "dx": dx, "dy": dy}
    )

    before = result.get("before", {})
    after = result.get("after", {})
    moved = before != after
    fallback_used = False

    if not moved:
        # As a last resort, try PageUp/PageDown which various virtual scrollers listen to
        key = "PageUp" if direction == "up" else "PageDown"
        if direction in ("left", "right"):
            key = "Home" if direction == "left" else "End"
        page.keyboard.press(key)
        fallback_used = True

        # Re-check the same target again to see if anything moved.
        result2 = page.evaluate(
            '''
            ({nx, ny}) => {
              const px = Math.round(nx/1000 * window.innerWidth);
              const py = Math.round(ny/1000 * window.innerHeight);
              let el = document.elementFromPoint(px, py);
              let t = el;
              const isScrollable = (e) => {
                if (!e) return false;
                const s = getComputedStyle(e);
                const yOk = (s.overflowY === 'auto' || s.overflowY === 'scroll') && e.scrollHeight > e.clientHeight;
                const xOk = (s.overflowX === 'auto' || s.overflowX === 'scroll') && e.scrollWidth > e.clientWidth;
                return yOk || xOk;
              };
              while (t && !isScrollable(t)) t = t.parentElement;
              if (!t) t = document.scrollingElement || document.documentElement;
              return {x: t.scrollLeft || 0, y: t.scrollTop || 0};
            }
            ''',
            {"nx": int(nx), "ny": int(ny)}
        )
        after = result2
        moved = before != after

    tgt = result.get("target", {})
    logger.debug(
        "scroll_at: (%s,%s norm) -> dir=%s mag=%s; target=%s#%s.%s before=%s after=%s moved=%s fallback=%s",
        nx, ny, direction, magnitude, tgt.get("tag"), tgt.get("id"), tgt.get("class"), before, after, moved, fallback_used
    )

    out = {
        "direction": direction,
        "magnitude": magnitude,
        "norm_xy": {"x": int(nx), "y": int(ny)},
        "target": tgt,
        "before": before,
        "after": after,
        "moved": moved,
        "fallback_used": fallback_used
    }
    return out

def _safe_dump(obj) -> str:
    """
    Best-effort, safe string for debug dumps.
    """
    try:
        if hasattr(obj, "to_dict"):
            import json
            return json.dumps(obj.to_dict())
        return repr(obj)
    except Exception as e:
        return f"<unserializable: {e}>"


def execute_function_calls(candidate, page: Page, debug: bool = False, dump_dir: Optional[str] = None):
    """
    Execute every function_call returned this turn.
    Returns: (results, extracts)
      results  -> List[(name, result_dict)] for FunctionResponse   (STRICT 1:1 per call)
      extracts -> List[dict] of job fields from our custom 'extract_fields' calls
    """
    results, extracts = [], []

    logger.debug("Executing function calls on page: %s", page.url)
    for idx, part in enumerate(candidate.content.parts or []):
        fc = getattr(part, "function_call", None)
        if not fc:
            continue

        name = fc.name
        args = dict(fc.args or {})
        logger.info("→ CALL %s (%d): args=%s", name, idx, args)

        payload = None  # we will set exactly one payload per function call

        try:
            if name == "open_web_browser":
                logger.debug("  open_web_browser: already open (noop)")
                payload = {"ok": True, "noop": True}

            elif name == "navigate":
                url = args.get("url") or args.get("uri") or ""
                logger.info("  navigate → %s", url)
                if url:
                    page.goto(url, wait_until="load", timeout=60000)
                payload = {"ok": True, "url": page.url}

            elif name == "click_at":
                x = denorm_x(int(args["x"]))
                y = denorm_y(int(args["y"]))
                logger.info("  click_at (%s, %s) pixels", x, y)
                page.mouse.click(x, y)
                payload = {"ok": True, "clicked_px": {"x": x, "y": y}, "url": page.url}

            elif name == "type_text_at":
                x = denorm_x(int(args["x"]))
                y = denorm_y(int(args["y"]))
                txt = args.get("text", "") or ""
                press_enter = bool(args.get("press_enter", False))
                logger.info("  type_text_at (%s, %s) text=%r enter=%s", x, y, txt[:80], press_enter)
                page.mouse.click(x, y)
                page.keyboard.type(txt)
                if press_enter:
                    page.keyboard.press("Enter")
                payload = {"ok": True, "typed_len": len(txt), "enter": press_enter, "url": page.url}

            elif name == "scroll_document":
                # Support both dy or explicit direction; default to down
                dy = int(args.get("dy", args.get("delta_y", args.get("magnitude", 800))))
                direction = args.get("direction") or ("down" if dy >= 0 else "up")
                logger.info("  scroll_document dir=%s (dy=%s)", direction, dy)
                info = _do_scroll_document(page, direction=direction, magnitude=abs(dy))
                payload = {"ok": True, **info, "url": page.url}

            elif name == "scroll_at":
                # Normalized coordinates 0..999
                x = int(args.get("x"))
                y = int(args.get("y"))
                direction = args.get("direction") or ("down" if int(args.get("delta_y", args.get("magnitude", 800))) >= 0 else "up")
                magnitude = int(args.get("magnitude", args.get("delta_y", 800)))
                logger.info("  scroll_at (%s,%s norm) dir=%s mag=%s", x, y, direction, magnitude)
                info = _do_scroll_at(page, x, y, direction=direction, magnitude=abs(magnitude))
                payload = {"ok": True, **info, "url": page.url}

            elif name == "hover_at":
                x = denorm_x(int(args["x"]))
                y = denorm_y(int(args["y"]))
                logger.info("  hover_at (%s, %s)", x, y)
                page.mouse.move(x, y)
                payload = {"ok": True, "hovered_px": {"x": x, "y": y}, "url": page.url}

            elif name == "go_back":
                logger.info("  go_back()")
                page.go_back(wait_until="load", timeout=60000)
                payload = {"ok": True, "url": page.url}

            elif name == "go_forward":
                logger.info("  go_forward()")
                page.go_forward(wait_until="load", timeout=60000)
                payload = {"ok": True, "url": page.url}

            elif name == "wait_5_seconds":
                logger.info("  wait_5_seconds()")
                time.sleep(5)
                payload = {"ok": True}

            elif name == "key_combination":
                keys = args.get("keys") or []
                logger.info("  key_combination: %s", keys)
                for combo in keys:
                    page.keyboard.press(str(combo))
                payload = {"ok": True, "keys": keys, "url": page.url}

            elif name == "search":
                # Some model variants may call 'search' with a query; support as noop+ack
                q = args.get("query", "")
                logger.info("  search(query=%r) -- not implemented; acknowledging.", q)
                payload = {"ok": True, "note": "search noop", "query": q}

            elif name == "extract_fields":
                # Custom function: model supplies parsed job fields
                if isinstance(args, dict):
                    logger.info(
                        "  extract_fields received: %s",
                        {k: (v[:120] + '...' if isinstance(v, str) and len(v) > 120 else v)
                         for k, v in args.items()}
                    )
                    extracts.append(args)
                payload = {"ok": True, "received_fields": sorted(list(args.keys()))}

            else:
                logger.warning("  Unimplemented function: %s", name)
                payload = {"warning": f"unimplemented: {name}"}

            # Post-action stabilization (ignore timeouts)
            try:
                page.wait_for_load_state("load", timeout=6000)
            except Exception:
                pass

            logger.debug("  after %s → url=%s", name, page.url)

        except Exception as e:
            logger.exception("  Error executing %s: %s", name, e)
            payload = {"error": str(e)}

        # STRICT 1:1 — exactly one FunctionResponse per Function Call
        if payload is None:
            payload = {"ok": True}
        results.append((name, payload))

    # Optional per-turn dump
    if dump_dir:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, "last_candidate.txt"), "w", encoding="utf-8") as f:
                f.write(_safe_dump(candidate))
        except Exception as e:
            logger.debug("Failed to dump candidate: %s", e)

    return results, extracts


def make_function_response_parts(page: Page, results):
    """
    Create FunctionResponse objects (one per executed call) with a screenshot
    embedded inside each via FunctionResponsePart.inline_data.
    """
    png = page.screenshot(type="png")
    url = page.url
    out = []
    for name, result in results:
        payload = {"url": url, **result}
        fr = FunctionResponse(
            name=name,
            response=payload,
            parts=[types.FunctionResponsePart(
                inline_data=types.FunctionResponseBlob(mime_type="image/png", data=png)
            )]
        )
        logger.debug("← RESP %s: %s", name, payload)
        out.append(fr)
    return out


async def collect_districts_for_state(state: str, base_dir: str) -> List[District]:
    """
    Helper to run async iter_districts and collect results into a list.
    """
    districts = []
    try:
        async for district in iter_districts(state, base_dir=base_dir):
            districts.append(district)
    except Exception as e:
        logger.exception("Failed to load districts for %s: %s", state, e)
    return districts


# ------------------------------
# 1) Browser "Hands"
# ------------------------------

class BrowserAgent:
    """
    Playwright wrapper.
    """
    def __init__(self, playwright: Playwright, headless: Optional[bool] = None):
        if headless is None:
            headless = os.getenv("PLAYWRIGHT_HEADLESS", "0").strip() not in {"0", "false", "no"}
        logger.debug("Launching Chromium (headless=%s)", headless)
        self.browser = playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context(viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
        self.page = self.context.new_page()
        logger.info("Browser ready @ %sx%s", SCREEN_WIDTH, SCREEN_HEIGHT)

    def goto(self, url: str):
        logger.info("Navigating to: %s", url)
        self.page.goto(url, wait_until="load", timeout=60000)
        self.page.wait_for_timeout(1200)

    def screenshot_bytes(self) -> bytes:
        return self.page.screenshot(type="png")

    # (Legacy) base64 screenshot if needed elsewhere
    def take_screenshot(self) -> str:
        logger.debug("Taking screenshot (base64)...")
        screenshot_bytes = self.page.screenshot(type="png")
        img = Image.open(BytesIO(screenshot_bytes))
        img.thumbnail((1024, 768))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def close(self):
        logger.debug("Closing browser.")
        try:
            self.context.close()
        finally:
            self.browser.close()
        logger.info("Browser closed.")


# ------------------------------
# 2) Gemini "Brain"
# ------------------------------

class VisionAgent:
    """
    Holds the Gemini client, tool config, and rolling contents history.
    """
    def __init__(self, api_key: Optional[str] = None, dump_dir: Optional[str] = None, fields_to_extract: Optional[List[str]] = None):
        self.client = genai.Client(api_key=api_key)
        self.model_name = MODEL_ID
        self.dump_dir = dump_dir

        # Build dynamic function schema based on fields_to_extract (or defaults)
        fields = list(fields_to_extract or DEFAULT_EXTRACT_FIELDS)
        # Always include a few critical fields for downstream writing
        for must in ["apply_url", "employer_full_name", "job_title","state","company_name","city","location","sport"]:
            if must not in fields:
                fields.append(must)
        properties = {k: {"type": "string"} for k in fields}

        self.config = types.GenerateContentConfig(
            tools=[
                # Computer Use tool (browser environment)
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=["drag_and_drop", "open_web_browser"]
                    )
                ),
                # Custom function the model can call on detail pages
                types.Tool(
                    function_declarations=[{
                        "name": "extract_fields",
                        "description": "On a job detail page, send extracted fields for XML.",
                        "parameters": {
                            "type": "object",
                            "properties": properties
                        }
                    }]
                )
            ]
        )
        self.contents: List[Content] = []
    def reset(self):
        self.contents = []
        logger.debug("Conversation reset.")

    def seed_with_goal_and_screenshot(self, goal_text: str, screenshot_png: bytes):
        logger.info("Seeding conversation with goal (truncated): %r", goal_text[:140])
        self.contents.append(
            Content(role="user", parts=[
                Part(text=goal_text),
                Part(text=AGENT_HINTS),
                Part.from_bytes(data=screenshot_png, mime_type="image/png"),
            ])
        )

    def ask(self):
        logger.debug("Calling model %s with %d prior turns.", self.model_name, len(self.contents))
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=self.contents,
            config=self.config,
        )
        cand = resp.candidates[0]
        self.contents.append(cand.content)
        logger.debug("Model responded with %d part(s).", len(cand.content.parts or []))

        if self.dump_dir:
            try:
                os.makedirs(self.dump_dir, exist_ok=True)
                with open(os.path.join(self.dump_dir, f"turn_{len(self.contents)}.txt"), "w", encoding="utf-8") as f:
                    f.write(_safe_dump(resp))
            except Exception as e:
                logger.debug("Failed to dump model turn: %s", e)

        return resp

    def append_function_responses(self, frs: List[FunctionResponse], extracts_this_turn: List[dict]):
        """
        Append function responses. If extracts occurred, also append a 
        text-based state update to remind the agent what it just did.
        """
        logger.debug("Appending %d FunctionResponse(s) to conversation.", len(frs))
        
        # Start with all the normal function response parts (and their screenshots)
        all_parts = [Part(function_response=fr) for fr in frs]

       
        # If we successfully extracted jobs, add a text part to this turn
        # to update the agent's "memory."
        if extracts_this_turn:
            # Get just the job titles, or "N/A"
            titles = [
                e.get('job_title', 'N/A') for e in extracts_this_turn 
                if e and e.get('job_title')
            ]
            
            if titles:
                state_update_text = (
                    f"\n--- STATE UPDATE ---\n"
                    f"You just successfully extracted {len(titles)} job(s): {', '.join(titles)}\n"
                    f"You are now back on the previous page (or the job page). "
                    f"Do not try to extract these specific jobs again. "
                    f"Continue searching for *new* 'coach' or 'athletic' jobs."
                )
                
                # Add this text to the list of parts for this turn
                all_parts.append(Part(text=state_update_text))
                logger.info("Injecting state update for %d extracted job(s).", len(titles))
    
        # Append the single "user" content turn, which now contains
        # all the function responses AND our new state update text.
        self.contents.append(
            Content(role="user", parts=all_parts)
        )

# ------------------------------
# 3) Orchestrator / Main
# ------------------------------

async def collect_districts_for_state(state: str, base_dir: str) -> List[District]:
    """Reads a user's CSV file; only rows with a career_url are returned."""
    districts = []
    async for d in iter_districts(state, base_dir=base_dir):
        districts.append(d)
    return districts

def main():
    parser = argparse.ArgumentParser(
        description="Scrape coaching jobs from district career pages for given states."
    )
    # --- inputs & common flags ---
    parser.add_argument('states', metavar='STATE', type=str, nargs='+',
                        help='One or more state abbreviations to process (e.g., "CO" "TX" "AZ")')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base directory containing state CSV files (default: "data")')
    parser.add_argument('-l', '--limit', type=int, default=None,
                        help='Limit to the first N districts/URLs (e.g., --limit 5)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Logging level (default: INFO)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Shortcut for --log-level DEBUG')
    parser.add_argument('--debug-dump-dir', type=str, default=os.getenv("DEBUG_DUMP_DIR", "").strip() or None,
                        help='Optional directory to dump model turns and candidates.')
    parser.add_argument('--fields', type=str, default=None,
                        help='Comma-separated field keys for extraction/writing '
                             '(e.g., "job_title,job_description,posting_date,closing_date")')

    # --- escalation + rescrape flags (wired up below) ---
    parser.add_argument('--rescrape', action='store_true',
                        help='Rescrape mode: revisit known Apply URLs and escalate only if content changed.')
    parser.add_argument('--footprints-db', type=str, default=os.getenv("FOOTPRINTS_DB", "cbnew/footprints.sqlite"),
                        help='SQLite path for per-URL fingerprints & run summaries.')
    parser.add_argument('--vision-order', type=str, default=os.getenv("VISION_ORDER", "paddle,azure,gemini"),
                        help='Escalation order, comma-separated (e.g., "paddle,azure,gemini").')
    parser.add_argument('--enable-paddle', action='store_true', help='Enable PaddleOCR stage')
    parser.add_argument('--enable-azure', action='store_true', help='Enable Azure OCR stage')
    parser.add_argument('--enable-gemini', action='store_true', help='Enable Gemini stage')
    parser.add_argument('--preset', type=str, default=None,
                        help='One of: paddle | azure | gemini | paddle+azure | paddle+gemini | azure+gemini | all')

    args = parser.parse_args()
    if args.verbose:
        args.log_level = 'DEBUG'

    # --- interpret presets (matrix testing you requested) ---
    if args.preset:
        p = args.preset.lower()
        args.enable_paddle = 'paddle' in p
        args.enable_azure  = 'azure'  in p
        args.enable_gemini = 'gemini' in p
    else:
        # default: if no flags, enable all; otherwise respect explicit flags
        if not any([args.enable_paddle, args.enable_azure, args.enable_gemini]):
            args.enable_paddle = args.enable_azure = args.enable_gemini = True

    order = [s.strip() for s in (args.vision_order.split(",") if args.vision_order else []) if s.strip()]

    # --- field mask from CLI or defaults ---
    fields_mask = [f.strip() for f in (args.fields.split(',') if args.fields else []) if f.strip()] or DEFAULT_EXTRACT_FIELDS

    setup_logging(level=args.log_level, verbose=args.verbose)
    logger.info("Args: %s", vars(args))

    # ----------------------------
    # RESCRAPE MODE (cheap-first OCR; escalate only when needed)
    # ----------------------------
    if args.rescrape:
        # Local imports so you don’t have to change file-level imports:
        from .vision_router import VisionRouter, RouterConfig  # escalation orchestrator (Paddle→Azure→Gemini)
        from .footprints import Footprints                    # per-URL text/image fingerprints

        # Init footprints DB + router
        fp = Footprints(args.footprints_db)
        router = VisionRouter(
            fpdb=fp,
            cfg=RouterConfig(
                enable_paddle=args.enable_paddle,
                enable_azure=args.enable_azure,
                enable_gemini=args.enable_gemini,
                order=order or ["paddle", "azure", "gemini"]
            )
        )

        # Open one browser for the whole run (fastest)
        with sync_playwright() as pw:
            browser = BrowserAgent(pw)
            try:
                for state_abbrev in args.states:
                    state = state_abbrev.strip().upper()
                    # Open existing state XML (or create if missing)
                    out_path = f"cbnew/out/AAAAA{state}.xml"
                    writer = JobsXML(
                        path=out_path,
                        field_mask=set(fields_mask) | {
                            'job_url', 'coach_search_url', 'company_name',
                            'employer_full_name', 'employer_email', 'sport', 'job_title', 'state'
                        }
                    )

                    seen_apply = list(writer.seen_apply_urls())  # canonicalized Apply URLs already in XML  :contentReference[oaicite:3]{index=3}
                    if args.limit:
                        seen_apply = seen_apply[: args.limit]
                    logger.info("Rescrape %s: %d known Apply URLs.", state, len(seen_apply))

                    for apply_url in seen_apply:
                        try:
                            browser.goto(apply_url)
                            shot = browser.screenshot_bytes()

                            # Final stage (Gemini) extractor: single-turn "detail page" pull, no navigation
                            def _extract_with_gemini() -> Dict[str, Any]:
                                goal = (
                                    "You are on a single job detail page. "
                                    "Extract fields by calling `extract_fields` ONCE, then STOP. "
                                    "Do not navigate or click."
                                )
                                local_agent = VisionAgent(
                                    api_key=GOOGLE_API_KEY,
                                    dump_dir=args.debug_dump_dir,
                                    fields_to_extract=fields_mask
                                )
                                local_agent.reset()
                                local_agent.seed_with_goal_and_screenshot(goal, shot)
                                resp = local_agent.ask()
                                fields: Dict[str, Any] = {}
                                if has_function_calls(resp):
                                    cand = resp.candidates[0]
                                    # Execute only what the model asks (usually just extract_fields)
                                    _results, extracts = execute_function_calls(
                                        cand, browser.page, debug=args.verbose, dump_dir=args.debug_dump_dir
                                    )
                                    if extracts:
                                        fields = extracts[0]
                                return fields

                            decision = router.check_or_escalate(
                                url=apply_url,
                                screenshot_bytes=shot,
                                on_need_gemini=_extract_with_gemini
                            )
                            status = decision.get("status")
                            fields = decision.get("fields") or {}
                            changed = bool(decision.get("changed"))

                            if status == "skipped":
                                # screenshot identical; just bump lastSeen
                                writer.mark_seen_by_apply_url(apply_url, active=True)
                                writer.write()
                                continue

                            if status in {"paddle", "azure"} and not changed:
                                # Cheap OCR confirmed "no important change" → mark seen
                                writer.mark_seen_by_apply_url(apply_url, active=True)
                                writer.write()
                                continue

                            # status == "gemini" (or changed with OCR): try to update fields; fall back to mark_seen
                            updated = False
                            if fields:
                                try:
                                    # If you added the helper we discussed, use it; otherwise the except keeps it safe.  :contentReference[oaicite:4]{index=4}
                                    if hasattr(writer, "update_fields_by_apply_url"):
                                        updated = writer.update_fields_by_apply_url(apply_url, fields)
                                except Exception as e:
                                    logger.debug("update_fields_by_apply_url failed: %s", e)

                            if not updated:
                                writer.mark_seen_by_apply_url(apply_url, active=True)

                            writer.write()

                        except Exception as e:
                            logger.exception("Rescrape error for %s: %s", apply_url, e)

            finally:
                browser.close()

        # Print & persist Week‑1 counters (A/B/C) you asked for  :contentReference[oaicite:5]{index=5}
        s = router.summary()
        logger.info("Run summary — A) skipped(no change): %d | B) cheap OCR: %d | C) escalated to Gemini: %d",
                    s.get("skipped_nochange", 0), s.get("used_cheap_ocr", 0), s.get("escalated_to_gemini", 0))
        # If your Footprints class includes this helper, record the rollup; otherwise skip quietly.
        try:
            fp.record_run_summary(s["skipped_nochange"], s["used_cheap_ocr"], s["escalated_to_gemini"])
        except Exception:
            pass
        return  # end rescrape mode

    # ----------------------------
    # DISCOVERY MODE (your existing flow)
    #   Start from CSV career URLs, let the Computer‑Use agent navigate,
    #   and write new Job records as before.  
    # ----------------------------
    brain = VisionAgent(api_key=GOOGLE_API_KEY, dump_dir=args.debug_dump_dir, fields_to_extract=fields_mask)

    logger.info("Loading district data...")
    all_districts: List[District] = []
    for state_abbrev in args.states:
        state = state_abbrev.strip().upper()
        logger.info("Loading districts for %s from '%s'...", state, args.data_dir)
        districts_for_state = asyncio.run(collect_districts_for_state(state, args.data_dir))
        if districts_for_state:
            logger.info("  Found %d districts for %s.", len(districts_for_state), state)
            all_districts.extend(districts_for_state[: args.limit] if args.limit else districts_for_state)
        else:
            logger.warning("  No districts found for %s.", state)

    if not all_districts:
        logger.error("No districts found. Exiting.")
        return

    xml_writers: Dict[str, JobsXML] = {}

    with sync_playwright() as p:
        for district in all_districts:
            district_name = district.name

            if not district.career_urls:
                logger.warning("Skipping %s (State: %s): No career URLs.", district_name, district.state)
                continue

            for career_url in district.career_urls:
                logger.info("--- Processing District: %s (%s) ---", district_name, district.state)
                logger.info("Career URL (coach search): %s", career_url)

                browser = None
                try:
                    browser = BrowserAgent(p)
                    brain.reset()

                    goal = f"""
                        ### 1. ROLE & OBJECTIVE
                        You are an autonomous web-browsing agent.
                        Your objective is to find all 'coach' and 'athletic' job postings for the employer "{district_name}".
                        Your starting point is: {career_url}

                        ### 2. CORE TASK & RULES
                        You must follow this exact workflow:
                        1.  **NAVIGATE:** Scroll through the job listings on the career page to find coaching-related jobs. You may need to use the search bar. 
                        2.  **CLICK:** Click on each individual 'coach' job link to open its *job detail page*.
                        3.  **EXTRACT:** Once on a detail page, you MUST call the `extract_fields` function.
                        4.  **REPEAT:** Continue until all coaching-related jobs have been extracted.

                        ### 3. CONSTRAINTS (IMPORTANT!)
                        * **FOCUS:** ONLY interested in "coach" or "athletics" jobs. Ignore all others.
                        * **EXTRACTION RULE:** You MUST ONLY call `extract_fields` when you are on a *detail page* for a single job.

                        ### 4. DATA EXTRACTION RULES (CRITICAL)
                        When calling `extract_fields`, pay close attention to the following fields:

                        * **`job_url`:** This MUST be the direct URL to the job description page you are currently viewing.
                        * **`apply_url`:** This is the link/button that takes the user away from the site to a *separate* application portal (e.g., AppliTrack, TalentEd). Capture this if present.
                        * **`company_name` / `district`:** This is always the School District: **"{district_name}"** (Do NOT extract this from the job description).
                        * **`employer_full_name`:** Extract the **specific High School or Campus name** if mentioned in the job title or description (e.g., "North High School").
                        """

                    browser.goto(career_url)
                    brain.seed_with_goal_and_screenshot(goal, browser.screenshot_bytes())

                    actions_remaining = ACTION_BUDGET_START
                    turn = 0
                    while actions_remaining > 0:
                        turn += 1
                        logger.info("Turn %d | actions remaining: %d", turn, actions_remaining)
                        resp = brain.ask()

                        if not has_function_calls(resp):
                            logger.info("Agent finished for %s (no more function calls).", district_name)
                            break

                        cand = resp.candidates[0]

                        results, extracts = execute_function_calls(
                            cand,
                            browser.page,
                            debug=args.verbose,
                            dump_dir=args.debug_dump_dir
                        )

                        for data in (extracts or []):
                            # --- REQUIRED FIELD MAPPING  ---
                            # company_name  = district name from CSV
                            # coach_search_url (acts as "company URL") = general coach search URL (career_url)
                            # employer_full_name = school name only if specified (data['location']); else ""
                            # employer_email = value if extracted else ""
                            employer_full_name = (data.get("location") or "").strip()
                            employer_email = (data.get("employer_email") or "").strip()
                            company_name = district_name
                            coach_search = career_url
                            job_state = (data.get("state") or district.state or "Unknown").strip()
                            logger.debug(
                                "Mapping summary: company_name=%r, coach_search_url=%r, employer_full_name=%r, employer_email=%r, district=%r, district_id=%r",
                                company_name, coach_search, employer_full_name, employer_email, district_name, (district.district_id or "")
                            )

                            job_record = JobXMLRecord(
                                job_title=data.get("job_title", "N/A"),
                                job_description=data.get("job_description", "N/A"),
                                job_type=data.get("job_type", "N/A"),
                                sport=data.get("sport", ""),    
                                location=data.get("location", ""),
                                city=data.get("city", ""),
                                state=job_state,
                                country="USA",
                                zip_code=(data.get("zip_code") or ""),
                                experience=data.get("experience", ""),
                                salary_range=data.get("salary_range", ""),
                                benefits=data.get("benefits", ""),
                                posting_date=data.get("posting_date", ""),
                                closing_date=data.get("closing_date", ""),
                                job_url=data.get("job_url", ""),
                                apply_url=data.get("apply_url", browser.page.url),
                                coach_search_url=coach_search,
                                employer_email=employer_email,
                                employer_full_name=employer_full_name,
                                company_description=f"{district_name} coaching jobs",
                                company_email="",
                                company_name=company_name,
                                district=district_name,
                                district_id=district.district_id or "",
                            )

                            if job_state not in xml_writers:
                                output_path = f"cbnew/out/AAAAA{job_state}.xml"
                                logger.info("Creating new XML file for state '%s': %s", job_state, output_path)
                                xml_writers[job_state] = JobsXML(
                                    path=output_path,
                                    field_mask=set(fields_mask) | {'job_url','coach_search_url','company_name','employer_full_name','employer_email','company_description','sport','job_title','state'}
                                )

                            writer = xml_writers[job_state]
                            writer.append_jobs([job_record])
                            writer.write()
                            logger.debug("XML write complete for state %s.", job_state)

                        if extracts:
                            actions_remaining = ACTION_BUDGET_START
                            logger.info("Found %d coaching job(s). Action budget reset to %d.",
                                        len(extracts), ACTION_BUDGET_START)
                            
                        # Respond back to the model with function results (unchanged)
                        frs = make_function_response_parts(browser.page, results)
                        brain.append_function_responses(frs, extracts)

                        # Consume budget based on browser actions executed this turn.
                        spent = sum(1 for (name, _res) in results if name in BROWSER_ACTION_CALLS)
                        if spent:
                            actions_remaining -= spent
                            logger.debug("Spent %d action(s); %d remaining.", spent, actions_remaining)
                    else:
                        logger.warning("Action budget exhausted for %s; moving on.", district_name)

                except Exception as e:
                    logger.exception("Error while processing %s (%s): %s", district_name, career_url, e)

                finally:
                    if browser:
                        browser.close()

    logger.info("--- Scraping complete. Results saved to coaching_jobs_[STATE].xml files ---")

if __name__ == "__main__":
    main()
