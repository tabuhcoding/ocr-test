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

Lần đầu (cài môi trường):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Khởi động service — **dùng `run.sh`** (đã set `OMP_NUM_THREADS=1` để tránh lỗi
paddle trên macOS):

```bash
./run.sh            # nghe 127.0.0.1:8090
PORT=9000 ./run.sh  # đổi cổng
```

Lần chạy đầu tự tải model: PaddleOCR (~20MB, về `~/.paddleocr/`) và VietOCR
vgg_seq2seq (~90MB, về `~/.cache/vietocr/` hoặc `/tmp`).

### Lưu ý ổn định (macOS + paddle 3.0)

Paddle 3.0 trên macOS thỉnh thoảng crash `No allocator found for the place,
Place(undefined:0)` — đây là **race của CPU inference đa luồng**, không phải do
cách start. Đã cố định trong `main.py`: `use_gpu=False`, `cpu_threads=1`,
`OMP_NUM_THREADS=1`, và nếu vẫn hy hữu dính thì worker tự respawn + retry, cuối
cùng trả **503** rõ ràng thay vì 500. Đừng chạy nhiều instance trên cùng cổng.

## API

- `GET /health` → `{"status": "ok"}`
- `POST /ocr` — multipart field `image` (JPEG/PNG/WebP)
  → `{"engine": "paddleocr-vi", "text": "...", "lines": [{text, confidence, box}]}`
