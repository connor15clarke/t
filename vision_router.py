# vision_router.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List
from vision_ocr import PaddleClient, AzureVisionClient, OCRText, text_sha, img_sha
from footprints import Footprints
import logging

logger = logging.getLogger("vision_router")

@dataclass
class RouterConfig:
    enable_paddle: bool = True
    enable_azure: bool = True
    enable_gemini: bool = True
    order: List[str] = None  # e.g., ["paddle","azure","gemini"]
    min_chars_for_confidence: int = 300      # low text => escalate
    min_confidence: float = 0.65             # OCR low conf => escalate

    def __post_init__(self):
        if not self.order:
            self.order = ["paddle","azure","gemini"]

class VisionRouter:
    def __init__(self, fpdb: Footprints, cfg: RouterConfig):
        self.fp = fpdb
        self.cfg = cfg
        self.paddle = PaddleClient() if cfg.enable_paddle else None
        self.azure = AzureVisionClient() if cfg.enable_azure else None

        # 3 headline numbers (Week 1 Day 1 metric targets)
        self.skipped_nochange = 0
        self.used_cheap_ocr = 0
        self.escalated_to_gemini = 0

    def _needs_escalation(self, o: OCRText, prev_text_sha: str) -> bool:
        # Not enough text, or low confidence, or text hash changed vs. previous → escalate
        if not (o and (o.text or "").strip()):
            return True
        if len(o.text) < self.cfg.min_chars_for_confidence:
            return True
        if o.confidence < self.cfg.min_confidence:
            return True
        if prev_text_sha and text_sha(o.text) != prev_text_sha:
            # Changed content: escalate to confirm & re-extract
            return True
        return False

    def check_or_escalate(
        self,
        url: str,
        screenshot_bytes: bytes,
        *,
        previous_fp: Optional[Dict[str, Any]] = None,
        on_need_gemini: Optional[Callable[[], Dict[str, Any]]] = None   # callback that runs your Gemini extraction
    ) -> Dict[str, Any]:
        """
        Returns: dict(status="skipped|paddle|azure|gemini", changed=bool, fields=dict)
        """
        prev = self.fp.get(url)
        prev_paddle_sha = (prev.text_sha_paddle if prev else "")
        prev_azure_sha  = (prev.text_sha_azure  if prev else "")
        shot_sha = img_sha(screenshot_bytes)

        # If screenshot identical, skip immediately
        if prev and prev.screenshot_sha == shot_sha:
            self.skipped_nochange += 1
            self.fp.upsert(url, screenshot_sha=shot_sha)
            logger.info("[SKIP:nochange] %s", url)
            return {"status":"skipped", "changed": False, "fields": {}}

        ocr_used = False
        last_stage = None
        text_shas = {}

        for stage in self.cfg.order:
            last_stage = stage
            if stage == "paddle" and self.cfg.enable_paddle:
                o = self.paddle.run(screenshot_bytes)
                ocr_used = True
                text_shas["paddle"] = text_sha(o.text)
                self.fp.upsert(url, screenshot_sha=shot_sha, text_sha_paddle=text_shas["paddle"], last_model="paddle")
                if not self._needs_escalation(o, prev_paddle_sha):
                    self.used_cheap_ocr += 1
                    logger.info("[OK:paddle] %s conf=%.2f chars=%d", url, o.confidence, len(o.text))
                    return {"status":"paddle", "changed": False, "fields": {}}
                else:
                    self.fp.record_escalation(url, "paddle", "next", "low_conf_or_changed", {"conf": o.confidence, "chars": len(o.text)})

            if stage == "azure" and self.cfg.enable_azure:
                o = self.azure.run(screenshot_bytes)
                ocr_used = True
                text_shas["azure"] = text_sha(o.text)
                self.fp.upsert(url, screenshot_sha=shot_sha, text_sha_azure=text_shas["azure"], last_model="azure")
                if not self._needs_escalation(o, prev_azure_sha):
                    self.used_cheap_ocr += 1
                    logger.info("[OK:azure] %s conf=%.2f chars=%d", url, o.confidence, len(o.text))
                    return {"status":"azure", "changed": False, "fields": {}}
                else:
                    self.fp.record_escalation(url, "azure", "next", "low_conf_or_changed", {"conf": o.confidence, "chars": len(o.text)})

            if stage == "gemini" and self.cfg.enable_gemini:
                self.escalated_to_gemini += 1
                if on_need_gemini:
                    fields = on_need_gemini() or {}
                else:
                    fields = {}
                self.fp.upsert(url, screenshot_sha=shot_sha, last_model="gemini")
                logger.info("[EXTRACT:gemini] %s", url)
                return {"status":"gemini", "changed": True, "fields": fields}

        # If we got here without returns:
        if ocr_used:
            # We used OCR but still didn't pass thresholds, treat as changed w/o fields
            logger.info("[CHANGED:ocr] %s", url)
            return {"status": last_stage or "ocr", "changed": True, "fields": {}}
        # No stages enabled
        logger.warning("[NOOP] Vision pipeline disabled for %s", url)
        return {"status":"noop", "changed": True, "fields": {}}

    def summary(self) -> Dict[str,int]:
        return {
            "skipped_nochange": self.skipped_nochange,
            "used_cheap_ocr": self.used_cheap_ocr,
            "escalated_to_gemini": self.escalated_to_gemini
        }
