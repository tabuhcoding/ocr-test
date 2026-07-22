# ndaseal-util

OCR sidecar cho `ndaseal-services`.

Pipeline hybrid:

- **RapidOCR / ONNXRuntime** detect vùng chữ và đọc sơ bộ từng dòng.
- **VietOCR** (`vgg_seq2seq`) đọc lại các crop chữ Việt để phục hồi dấu tốt hơn.
- Các dòng MRZ hoặc dòng có chuỗi số dài giữ kết quả RapidOCR để tránh VietOCR
  sửa nhầm số CCCD/MRZ.

Backend Go gọi sang service này khi cấu hình `ocr.engine: remote`
(`OCR_ENGINE=remote`, `OCR_REMOTE_URL=http://localhost:8090`). Việc bóc tách
field CCCD/MRZ vẫn nằm bên Go.

## Structure

```text
.
├── ndaseal_util/
│   ├── api.py            # FastAPI routes
│   ├── config.py         # env-based settings
│   └── ocr/pipeline.py   # RapidOCR + VietOCR pipeline
├── main.py               # compatibility entrypoint for uvicorn main:app
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.sh
```

## Chạy

### Local

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
./run.sh            # nghe 127.0.0.1:8090
PORT=9000 ./run.sh  # đổi cổng
```

Lần chạy đầu tự tải model RapidOCR và VietOCR vào cache của user hiện tại.

### Docker

```bash
docker compose up --build
```

Service expose `8090`. Compose mount volume cache model để các lần start sau
không tải lại từ đầu.

Build image riêng:

```bash
docker build -t ndaseal-util:local .
docker run --rm -p 8090:8090 ndaseal-util:local
```

## Cấu hình

| Biến môi trường | Mặc định | Mô tả |
| --- | --- | --- |
| `OCR_HOST` | `127.0.0.1` local, `0.0.0.0` Docker | Host bind uvicorn |
| `OCR_PORT` / `PORT` | `8090` | Port service |
| `LOG_LEVEL` | `info` | Logging level |
| `OCR_WARM_UP` | `true` | Load model lúc startup |

## API

- `GET /health` → `{"status": "ok"}`
- `POST /ocr` — multipart field `image` (JPEG/PNG/WebP)
  → `{"engine": "rapidocr-det+vietocr-rec", "text": "...", "lines": [{text, confidence, box, source}]}`
