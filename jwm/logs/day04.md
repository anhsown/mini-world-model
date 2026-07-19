# JWM — Nhật ký dự án · NGÀY 4 (2026-07-19)

> V2 chứng minh anti-shortcut có tác dụng trên ảnh tổng hợp nhưng chưa chuyển
> sang tài liệu thật; hôm nay thiết kế JWM-Read v3 và pipeline T4×2 để sửa từ gốc.

## 1. Kết luận benchmark v2

- Synthetic blind-control: vision gain **+0,136** — model đã bắt đầu nhìn ảnh.
- Real document blind-control: **−0,0002** — vẫn không dùng ảnh thật.
- VietDocVQA: CER median **0,99**, exact **0%**.
- MTVQA-VI: CER median **1,00**, exact **0%**.
- V2 tốt hơn v1 ở khả năng dừng EOS và OCR synthetic ngắn, nhưng domain transfer
  và document reasoning vẫn chết.

## 2. Học từ LocateAnything

Không clone Qwen/MoonViT 3B. Chỉ lấy các nguyên lý phù hợp scale của JWM:
spatial reasoning trước token merge, giữ tỉ lệ ảnh tài liệu, tọa độ rời rạc
0–1000 và dự đoán bốn tọa độ song song.

## 3. Kiến trúc v3

- 1024×768; patch-16 → local attention trên lưới 64×48 → merge 2×2 sau reasoning
  → 768 visual tokens.
- Reasoner d512/L10, MoE 16 experts top-2 + shared expert; 119,06M tổng,
  82,77M trainable khi đóng băng generator.
- Tokenizer grapheme tiếng Việt có byte fallback.
- Thêm CTC OCR, four-query coordinate head, noisy teacher forcing và loss ảnh
  đúng so với ảnh tráo.
- Loss thật được tách đúng: synthetic có QA+CTC+box; tài liệu thật chỉ QA và
  visual contrast, không giả vờ đáp án QA là transcript toàn trang.

## 4. Dataset và kiểm định giả thuyết

Split theo page, không theo conversation. Sáu giả thuyết data phải pass trước
train: nhãn random duy nhất, box hợp lệ, không truncate, chữ nhìn thấy, zero
page leakage, ảnh thật đọc được. Probe trên dataset thật: **6/6 pass**, domain
gap thống kê **0,05003**, leakage **0**.

## 5. Trainer Kaggle T4×2

- DDP hai GPU, global batch 12, FP16, gradient accumulation, cosine LR.
- 4 stage có gate bằng free-running CER, CTC-CER, box IoU và vision gain.
- Gate fail thì extend cùng stage; hết budget thì dừng và xuất blocked artifact,
  tuyệt đối không tự chạy tiếp như v2.
- Atomic checkpoint mỗi 250 bước; rerun cell tự resume.
- Gradient audit bắt được AMP scale 65536 gây NaN ở CTC; đổi 1024 → **0 NaN/Inf**.
- Smoke end-to-end đã xuất model, metrics, history, validation và status.
- 51/51 test Reader + math pass.

Notebook bàn giao: `jwm/kaggle/jwm_read_t4x2_v3.ipynb`. Thời gian dự kiến base
7–9 giờ, worst-case khoảng 11 giờ nếu phải mở rộng stage.

