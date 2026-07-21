"""ndaseal-util — OCR sidecar.

Hybrid Vietnamese OCR over HTTP for ndaseal-services:

    POST /ocr   multipart field "image" → {engine, text, lines}

PaddleOCR detects the text boxes and recognises each line once; lines that
look like MRZ (the OCR-B strip on the CCCD back) or carry long digit runs
keep PaddleOCR's reading, every other line is re-recognised by VietOCR
(transformer trained on Vietnamese), which restores diacritics far better on
real card photos.

Process layout: paddlepaddle and torch corrupt each other's device context
when loaded into one process on macOS ("No allocator found for the place"),
so PaddleOCR runs in a dedicated spawned subprocess and VietOCR stays in the
web process. Models load lazily so the spawned child does not pull torch in.

The Go backend treats this as one OCREngine implementation (ocr.engine:
"remote"); all CCCD field/MRZ parsing stays on the Go side.
"""

import logging
import re
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

logging.getLogger("ppocr").setLevel(logging.WARNING)

app = FastAPI(title="ndaseal-util", version="0.3.0")

# ---------------------------------------------------------------- paddle side
# Everything below runs inside the single spawned worker process.

_paddle = None


def _paddle_init():
    global _paddle
    from paddleocr import PaddleOCR

    _paddle = PaddleOCR(lang="vi", use_angle_cls=True, show_log=False)


def _paddle_ocr(img):
    """Runs in the worker; returns picklable (box, text, score) tuples.

    paddlepaddle 3.0 on macOS sporadically corrupts its device context after
    repeated predictions ("No allocator found for the place") — rebuilding
    the predictor clears it, so retry once with a fresh instance.
    """
    try:
        result = _paddle.ocr(img, cls=True)
    except RuntimeError:
        _paddle_init()
        result = _paddle.ocr(img, cls=True)
    lines = []
    for page in result or []:
        for box, (text, score) in page or []:
            lines.append((box, text, float(score)))
    return lines


_paddle_proc = ProcessPoolExecutor(max_workers=1, initializer=_paddle_init)

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
    _paddle_proc.submit(int).result()  # forces the worker + PaddleOCR init
    _get_viet()


# ---------------------------------------------------------------- pipeline

# MRZ-ish: long, spaceless A-Z/0-9/< with filler or many digits — keep
# PaddleOCR's reading, VietOCR would "correct" it into Vietnamese words.
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


def _keep_paddle(text):
    """Long digit runs (CCCD number): VietOCR mangles them, Paddle doesn't."""
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
    for box, ptext, score in _paddle_proc.submit(_paddle_ocr, img).result():
        text, source = ptext, "paddle"
        if not _keep_paddle(ptext):
            crop = _crop(img, box)
            if crop is not None:
                vtext = _get_viet().predict(
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                )
                if vtext and vtext.strip():
                    text, source = vtext.strip(), "vietocr"
        lines.append({"text": text, "confidence": score, "box": box, "source": source})

    return {
        "engine": "paddle-det+vietocr-rec",
        "text": "\n".join(l["text"] for l in lines),
        "lines": lines,
    }
