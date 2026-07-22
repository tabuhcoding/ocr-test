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

That same allocator bug also corrupts the worker's paddle context at the
PROCESS level after repeated predictions — rebuilding the PaddleOCR object in
place does not clear it. So on that error we discard the whole worker process
and respawn a fresh one, then retry the request once (see _run_paddle).

The Go backend treats this as one OCREngine implementation (ocr.engine:
"remote"); all CCCD field/MRZ parsing stays on the Go side.
"""

import logging
import re
import threading
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

logging.getLogger("ppocr").setLevel(logging.WARNING)

app = FastAPI(title="ndaseal-util", version="0.4.0")

# ---------------------------------------------------------------- paddle side
# Everything below runs inside the single spawned worker process.

_paddle = None


def _pin_cpu():
    """Root-cause fix for the sporadic macOS crash
        RuntimeError: (NotFound) No allocator found for the place, Place(undefined:0)
    The 'undefined' place means paddle failed to resolve the device, so matmul
    can't find an allocator. Forcing an explicit CPU place — before the
    predictor is built, and again per prediction as insurance — makes the place
    concrete and the allocator resolvable."""
    import os

    # naive_best_fit is the stable CPU allocator; auto_growth has been linked
    # to the undefined-place crash on macOS. Pin BLAS/OMP to one thread too —
    # the crash is a multi-threaded CPU-matmul race.
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import paddle

    paddle.set_device("cpu")


def _paddle_init():
    global _paddle
    _pin_cpu()
    from paddleocr import PaddleOCR

    # The sporadic macOS crash "No allocator found for the place,
    # Place(undefined:0)" comes from the CPU inference predictor, not the
    # dynamic graph, so it is fixed HERE at predictor-build time — not by
    # set_device:
    #   use_gpu=False   → no undefined GPU place (PaddleOCR defaults use_gpu=True,
    #                     but this box has no GPU).
    #   cpu_threads=1   → serialise CPU matmul; the crash is a 10-thread race
    #                     (PaddleOCR defaults cpu_threads=10).
    #   enable_mkldnn=False → keep the default, plain CPU kernels.
    _paddle = PaddleOCR(
        lang="vi",
        use_angle_cls=True,
        show_log=False,
        use_gpu=False,
        cpu_threads=1,
        enable_mkldnn=False,
    )


def _paddle_ocr(img):
    """Runs in the worker; returns picklable (box, text, score) tuples."""
    _pin_cpu()  # re-assert the CPU place for this prediction
    result = _paddle.ocr(img, cls=True)
    lines = []
    for page in result or []:
        for box, (text, score) in page or []:
            lines.append((box, text, float(score)))
    return lines


# The paddle worker pool, its generation counter and a lock guarding respawns.
# FastAPI serves the sync endpoint from a threadpool, so several /ocr calls can
# race here; the generation counter lets a failing request tell whether another
# thread has already respawned the pool it used.
_paddle_lock = threading.Lock()
_paddle_gen = 0
_paddle_proc = None


def _spawn_pool():
    pool = ProcessPoolExecutor(max_workers=1, initializer=_paddle_init)
    pool.submit(int).result()  # force the worker to fork + load PaddleOCR
    return pool


def _ensure_pool():
    global _paddle_proc
    if _paddle_proc is None:
        with _paddle_lock:
            if _paddle_proc is None:
                _paddle_proc = _spawn_pool()
    return _paddle_proc, _paddle_gen


def _respawn(seen_gen):
    """Replace the worker process, unless another thread already did it."""
    global _paddle_proc, _paddle_gen
    with _paddle_lock:
        if seen_gen != _paddle_gen:
            return  # someone else already respawned since this request started
        old = _paddle_proc
        _paddle_proc = _spawn_pool()
        _paddle_gen += 1
    if old is not None:
        old.shutdown(wait=False, cancel_futures=True)


def _run_paddle(img):
    """Run OCR in the worker. Single-threaded CPU inference (use_gpu=False,
    cpu_threads=1) avoids the paddle place crash; should it still occur,
    respawn the worker process and retry, with a short settle between tries,
    before surfacing a clean 503."""
    last = None
    for attempt in range(4):
        pool, gen = _ensure_pool()
        try:
            return pool.submit(_paddle_ocr, img).result()
        except (RuntimeError, BrokenExecutor, OSError) as e:
            last = e
            _respawn(gen)
            time.sleep(0.3 * (attempt + 1))  # let any transient state settle
    raise HTTPException(
        status_code=503,
        detail=f"OCR engine unavailable (paddle place error): {last}",
    )

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
    _ensure_pool()  # fork the worker + load PaddleOCR
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
    for box, ptext, score in _run_paddle(img):
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
