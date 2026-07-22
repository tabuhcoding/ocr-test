"""Runtime configuration for the OCR sidecar."""

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "ndaseal-util"
    app_version: str = "0.5.0"
    ocr_engine: str = "rapidocr-det+vietocr-rec"
    host: str = os.getenv("OCR_HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", os.getenv("OCR_PORT", "8090")))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    warm_up_on_startup: bool = _env_bool("OCR_WARM_UP", True)


settings = Settings()

