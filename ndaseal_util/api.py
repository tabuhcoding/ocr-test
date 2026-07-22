"""FastAPI application for the OCR sidecar."""

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile

from ndaseal_util.config import settings
from ndaseal_util.ocr.pipeline import read_image, recognize_image, warm_up

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.on_event("startup")
def on_startup() -> None:
    if settings.warm_up_on_startup:
        logger.info("warming up OCR models")
        warm_up()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
def ocr(image: UploadFile = File(...)):
    img = read_image(image.file.read())
    if img is None:
        raise HTTPException(status_code=400, detail="not a decodable image")
    return recognize_image(img)

