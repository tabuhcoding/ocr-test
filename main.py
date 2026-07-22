"""ndaseal-util — OCR sidecar.

Hybrid Vietnamese OCR over HTTP for ndaseal-services:

    POST /ocr   multipart field "image" → {engine, text, lines}

RapidOCR (the PP-OCR detection+recognition models running on ONNXRuntime)
detects the text boxes and recognises each line once; lines that look like MRZ
(the OCR-B strip on the CCCD back) or carry long digit runs keep RapidOCR's
reading, every other line is re-recognised by VietOCR (a transformer trained on
Vietnamese), which restores diacritics far better on real card photos.

Why RapidOCR and not paddleocr: paddlepaddle's CPU inference predictor crashes
sporadically on macOS with "No allocator found for the place, Place(undefined:0)",
a process-level corruption that survives object rebuilds and forced respawns.
RapidOCR runs the exact same PP-OCR models through ONNXRuntime, which has no such
bug — so the whole subprocess-isolation / respawn dance is gone and both engines
load in-process (ONNXRuntime and torch coexist cleanly, unlike paddle + torch).

The Go backend treats this as one OCREngine implementation (ocr.engine:
"remote"); all CCCD field/MRZ parsing stays on the Go side.
"""

import logging
import re

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

logging.getLogger("rapidocr").setLevel(logging.WARNING)

app = FastAPI(title="ndaseal-util", version="0.5.0")

# ---------------------------------------------------------------- detection
# RapidOCR = PP-OCR detect + recognise on ONNXRuntime. Stable in-process, so no
# subprocess isolation is needed (that only existed to contain the paddle bug).

_rapid = None


def _get_rapid():
    global _rapid
    if _rapid is None:
        from rapidocr_onnxruntime import RapidOCR

        _rapid = RapidOCR()
    return _rapid


def _run_rapid(img):
    """Detect + recognise; return (box, text, score) tuples. box is 4 [x,y]."""
    result, _ = _get_rapid()(img)
    lines = []
    for box, text, score in result or []:
        lines.append((box, text, float(score)))
    return lines


# ---------------------------------------------------------------- torch side

_viet = None


def _get_viet():
    global _viet
    if _viet is None:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name("vgg_seq2seq")  # lighter than transformer
        cfg["device"] = "cpu"
        cfg["predictor"]["beamsearch"] = False
        _viet = Predictor(cfg)
    return _viet


@app.on_event("startup")
def warm_up():
    _get_rapid()
    _get_viet()


# ---------------------------------------------------------------- pipeline

# MRZ-ish: long, spaceless A-Z/0-9/< with filler or many digits — keep
# RapidOCR's reading, VietOCR would "correct" it into Vietnamese words.
_MRZ_CHARSET = re.compile(r"^[A-Z0-9<]+$")
_DIGIT_RUN = re.compile(r"\d{6,}")


def _looks_like_mrz(text):
    s = text.replace(" ", "")
    if s.count("<") >= 2:
        return True
    return (
        len(s) >= 20
        and bool(_MRZ_CHARSET.match(s))
        and sum(c.isdigit() for c in s) >= 6
    )


def _keep_detected(text):
    """Long digit runs (CCCD number): VietOCR mangles them, RapidOCR doesn't."""
    return _looks_like_mrz(text) or bool(_DIGIT_RUN.search(text))


def _crop(img, box, pad=0.06):
    """Perspective-crop one detected quad, slightly expanded — tight boxes
    clip ascenders/diacritics, which is exactly what VietOCR needs to see."""
    pts = np.array(box, dtype=np.float32)
    center = pts.mean(axis=0)
    pts = center + (pts - center) * (1 + pad)
    h_img, w_img = img.shape[:2]
    pts[:, 0] = pts[:, 0].clip(0, w_img - 1)
    pts[:, 1] = pts[:, 1].clip(0, h_img - 1)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if w < 4 or h < 4:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, m, (w, h))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
def ocr(image: UploadFile = File(...)):
    data = image.file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="not a decodable image")

    lines = []
    for box, dtext, score in _run_rapid(img):
        text, source = dtext, "rapidocr"
        if not _keep_detected(dtext):
            crop = _crop(img, box)
            if crop is not None:
                vtext = _get_viet().predict(
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                )
                if vtext and vtext.strip():
                    text, source = vtext.strip(), "vietocr"
        lines.append({"text": text, "confidence": score, "box": box, "source": source})

    print(f"OCR: {len(lines)} lines, {sum(l['confidence'] for l in lines)/len(lines):.2f} avg confidence")

    return {
        "engine": "rapidocr-det+vietocr-rec",
        "text": "\n".join(l["text"] for l in lines),
        "lines": lines,
    }
