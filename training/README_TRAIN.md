# 🧠 Tự train bộ não JARVIS — ngay trên máy bạn, độc lập 100%

Kỹ thuật **distillation (chưng cất tri thức)**: model lớn (Claude — "thầy") viết bộ dữ liệu mẫu, bạn dùng nó dạy một model nhỏ (Qwen2.5-0.5B — "trò") bắt chước phong cách JARVIS. Kết quả: một model **của riêng bạn**, chạy **offline trên máy bạn**, không API key, không provider, không cần cả Colab.

```
🧠 Claude viết dataset.jsonl ──▶ 🎓 Train trên GPU máy bạn ──▶ 📦 jarvis-model/
                                                                     │
                                    server nội bộ (serve_brain.py) ◀─┘
                                              │
                                   cắm vào Jarvis (localhost)
```

> **Máy bạn:** GTX 1650 4GB — vừa đủ. Model 0.5B khi train chỉ ngốn ~1.3 GB VRAM, phản hồi tức thì khi chạy.

---

## Bước 1 — Tạo bộ dữ liệu (~10 giây)

```powershell
python training/build_dataset.py
```
→ gộp các `part_*.jsonl` (Claude đã viết) thành `training/dataset.jsonl` (~330 mẫu).

## Bước 2 — Train ngay trên máy (~10-15 phút)

```powershell
python training/train_local.py
```
Lần đầu tự tải model gốc (~1 GB). Xong sẽ có `training/jarvis-model/` — bộ não của bạn.

> Thư viện cần: `torch` (đã có sẵn bản CUDA), `transformers`, `peft`, `accelerate` — đã cài đủ.

## Bước 3 — Chạy bộ não (mở cửa sổ riêng, để nguyên)

```powershell
python training/serve_brain.py
```
Nó mở server nội bộ tại `http://localhost:8080/v1` (chuẩn OpenAI). Không internet, không key.

## Bước 4 — Cắm vào Jarvis

Trong `config.py` sửa 2 dòng:
```python
AI_BASE_URL = "http://localhost:8080/v1"
AI_MODEL = "jarvis"
```
(không cần `.env` — server local không đòi key)

Xong! Tắt Jarvis cũ ("Hey Jarvis" → "goodbye"), chạy lại `python jarvis.py`. Giờ mọi câu hỏi tự do đều do **model của chính bạn** trả lời, hoàn toàn offline.

---

## Làm model thông minh hơn

- **Thêm dữ liệu**: nhờ Claude viết thêm `part_*.jsonl` về chủ đề bạn hay hỏi → chạy lại Bước 1-2. Dữ liệu càng nhiều & đa dạng, "trò" càng giỏi.
- **Lên model 1.5B**: thông minh hơn nhưng trọng số ~3 GB, cần train kiểu 4-bit (bitsandbytes) hoặc GPU 6 GB+. Nhờ Claude dựng nếu muốn.
- **Giữ 2 não**: để model local làm mặc định, cắm thêm Groq/Claude làm "não dự phòng" khi cần kiến thức sâu.

## Các file trong thư mục này

| File | Vai trò |
|---|---|
| `part_*.jsonl` | Dữ liệu thô Claude viết (5 chủ đề) |
| `build_dataset.py` | Gộp + kiểm tra → `dataset.jsonl` |
| `train_local.py` | Train LoRA trên GPU máy bạn |
| `merge_model.py` | Gộp adapter vào model gốc (chạy lại nếu bước gộp lỗi) |
| `jarvis-lora/` | Adapter LoRA (nhẹ, ~34 MB) |
| `jarvis-model/` | Bộ não hoàn chỉnh, chạy được (~1 GB) |
| `serve_brain.py` | Server nội bộ chạy bộ não |

---

## Sự đánh đổi (nói thẳng)

- ✅ **Được**: độc lập tuyệt đối — bộ não trong ổ cứng bạn, offline, đúng chất JARVIS (giọng, "sir", ngắn gọn, hiểu Việt trả lời Anh).
- ⚠️ **Mất**: kiến thức chuyên sâu của model 0.5B không bằng model khổng lồ. Nó trò chuyện duyên và làm tốt việc trợ lý, nhưng không giải toán cao cấp như Claude. Đây là cái giá của sự tự chủ hoàn toàn.
