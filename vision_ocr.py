# vision_ocr.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from io import BytesIO
import os, json, hashlib, time, requests
from PIL import Image

@dataclass
class OCRText:
    text: str
    confidence: float
    model: str
    extra: Dict[str, Any]

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    # collapse whitespace + strip junk punctuation commonly added by OCR
    return " ".join("".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch in "-_/.:@")
                    .split())

def text_sha(s: str) -> str:
    return hashlib.sha256(_normalize_text(s).encode("utf-8")).hexdigest()

def img_sha(img_bytes: bytes) -> str:
    return hashlib.sha1(img_bytes).hexdigest()  # cheap, good enough for change checks

class PaddleClient:
    """Cheap OCR. $0. Local. Good for 'did this page change?'."""
    def __init__(self, lang="en"):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:
            raise ImportError("PaddleOCR not installed: pip install paddleocr") from e
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def run(self, image_bytes: bytes) -> OCRText:
        import numpy as np
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        t0 = time.time()
        result = self._ocr.ocr(arr, cls=True)
        ms = int((time.time() - t0) * 1000)

        lines, confs = [], []
        # result is list[ list[ [bbox, (text, score)], ... ] ]
        for block in result or []:
            for line in block or []:
                txt = line[1][0] if line and line[1] else ""
                score = float(line[1][1]) if line and line[1] else 0.0
                if txt:
                    lines.append(txt)
                    confs.append(score)

        text = "\n".join(lines).strip()
        conf = sum(confs)/len(confs) if confs else 0.0
        return OCRText(text=text, confidence=conf, model="paddle", extra={"lines": len(lines), "latency_ms": ms})

class AzureVisionClient:
    """
    Mid-tier OCR (v4 Image Analysis 'read' feature).
    Endpoint example:
      {ENDPOINT}/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read&language=en
    """
    def __init__(self, endpoint: str | None = None, key: str | None = None):
        self.endpoint = (endpoint or os.getenv("AZURE_VISION_ENDPOINT") or "").rstrip("/")
        self.key = key or os.getenv("AZURE_VISION_KEY") or ""
        if not self.endpoint or not self.key:
            raise ValueError("AZURE_VISION_ENDPOINT / AZURE_VISION_KEY not set")

    def run(self, image_bytes: bytes) -> OCRText:
        url = f"{self.endpoint}/computervision/imageanalysis:analyze"
        params = {"api-version": "2024-02-01", "features": "read", "language": "en"}
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/octet-stream"
        }
        t0 = time.time()
        r = requests.post(url, params=params, headers=headers, data=image_bytes, timeout=45)
        ms = int((time.time() - t0) * 1000)
        r.raise_for_status()
        data = r.json()

        # v4 'read' returns readResult with blocks->lines->words (words may include confidence)
        lines, confs = [], []
        read = (data or {}).get("readResult", {}) or {}
        for blk in read.get("blocks", []) or []:
            for ln in blk.get("lines", []) or []:
                txt = (ln.get("text") or "").strip()
                if txt:
                    lines.append(txt)
                # avg confidence from words if present
                w_confs = [float(w.get("confidence", 1.0)) for w in ln.get("words", []) or [] if "confidence" in w]
                if w_confs:
                    confs.extend(w_confs)

        text = "\n".join(lines).strip()
        conf = sum(confs)/len(confs) if confs else (0.9 if text else 0.0)
        return OCRText(text=text, confidence=conf, model="azure", extra={"lines": len(lines), "latency_ms": ms, "raw_len": len(json.dumps(data))})
