"""Hybrid RapidOCR + VietOCR pipeline for Vietnamese ID card images."""

from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ndaseal_util.config import settings

logging.getLogger("rapidocr").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_rapid = None
_viet = None

_MRZ_CHARSET = re.compile(r"^[A-Z0-9<]+$")
_DIGIT_RUN = re.compile(r"\d{6,}")


def warm_up() -> None:
    """Load OCR models early so the first request does not pay model latency."""
    get_rapid()
    get_viet()


def get_rapid():
    global _rapid
    if _rapid is None:
        from rapidocr_onnxruntime import RapidOCR

        _rapid = RapidOCR()
    return _rapid


def get_viet():
    global _viet
    if _viet is None:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name("vgg_seq2seq")
        cfg["device"] = "cpu"
        cfg["predictor"]["beamsearch"] = False
        _viet = Predictor(cfg)
    return _viet


def read_image(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def recognize_image(img: np.ndarray) -> dict[str, Any]:
    lines = []
    for box, detected_text, score in run_rapid(img):
        text, source = detected_text, "rapidocr"
        if not keep_detected_text(detected_text):
            crop = crop_box(img, box)
            if crop is not None:
                viet_text = get_viet().predict(
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                )
                if viet_text and viet_text.strip():
                    text, source = viet_text.strip(), "vietocr"
        lines.append(
            {
                "text": text,
                "confidence": score,
                "box": normalize_box(box),
                "source": source,
            }
        )

    if lines:
        avg_confidence = sum(line["confidence"] for line in lines) / len(lines)
        logger.info(
            "OCR recognized %s lines, %.2f average confidence",
            len(lines),
            avg_confidence,
        )
    else:
        logger.info("OCR recognized 0 lines")

    return {
        "engine": settings.ocr_engine,
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
    }


def run_rapid(img: np.ndarray) -> list[tuple[list[list[float]], str, float]]:
    """Detect and recognize text lines with RapidOCR."""
    result, _ = get_rapid()(img)
    return [(box, text, float(score)) for box, text, score in result or []]


def keep_detected_text(text: str) -> bool:
    """Keep RapidOCR output for MRZ and long digit runs that VietOCR may mangle."""
    return looks_like_mrz(text) or bool(_DIGIT_RUN.search(text))


def normalize_box(box: list[list[float]]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in box]


def looks_like_mrz(text: str) -> bool:
    stripped = text.replace(" ", "")
    if stripped.count("<") >= 2:
        return True
    return (
        len(stripped) >= 20
        and bool(_MRZ_CHARSET.match(stripped))
        and sum(char.isdigit() for char in stripped) >= 6
    )


def crop_box(
    img: np.ndarray, box: list[list[float]], pad: float = 0.06
) -> np.ndarray | None:
    """Perspective-crop one detected quadrilateral with padding for diacritics."""
    pts = np.array(box, dtype=np.float32)
    center = pts.mean(axis=0)
    pts = center + (pts - center) * (1 + pad)

    image_height, image_width = img.shape[:2]
    pts[:, 0] = pts[:, 0].clip(0, image_width - 1)
    pts[:, 1] = pts[:, 1].clip(0, image_height - 1)

    width = int(
        max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    )
    height = int(
        max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    )
    if width < 4 or height < 4:
        return None

    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, matrix, (width, height))
