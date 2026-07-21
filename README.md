# ndaseal-util

OCR sidecar cho `ndaseal-services`, pipeline hybrid:

- **PaddleOCR** detect vùng chữ + đọc các dòng MRZ / dòng nhiều chữ số
  (số CCCD) — phần nó đọc chuẩn nhất.
- **VietOCR** (vgg_seq2seq) đọc lại từng crop chữ Việt — phục hồi dấu tốt hơn
  hẳn trên ảnh chụp thật.

Backend Go gọi sang đây khi cấu hình `ocr.engine: remote` (`OCR_ENGINE=remote`,
`OCR_REMOTE_URL=http://localhost:8090`); toàn bộ việc bóc tách field CCCD/MRZ
vẫn nằm bên Go.

## Chạy

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --port 8090
```

Lần chạy đầu tự tải model: PaddleOCR (~20MB, về `~/.paddleocr/`) và VietOCR
vgg_seq2seq (~90MB, về `~/.cache/vietocr/` hoặc `/tmp`).

## API

- `GET /health` → `{"status": "ok"}`
- `POST /ocr` — multipart field `image` (JPEG/PNG/WebP)
  → `{"engine": "paddleocr-vi", "text": "...", "lines": [{text, confidence, box}]}`
