# JWM — Nhật ký dự án / Project Logs

Chuỗi nhật ký theo ngày của dự án bộ não world-model cho JARVIS.
Daily log series for the JARVIS world-model brain project.

| Ngày / Day | Ngày tháng | Tiếng Việt | English | Tóm tắt một dòng |
|---|---|---|---|---|
| **Day 1** | 2026-07-16 | [day01.md](day01.md) | [day01_en.md](day01_en.md) | Paper → kiến trúc JWM + kiểm định 2 vòng → dataset validated → v1 train + trials → 30fps vision → tái cấu trúc 7-stage → saga debug 68M (5 vòng loại trừ, không bug) → **jwm_v3.pt 31M: QA 58.8%, mIoU +42%, ECE 0.025 — vượt v1 toàn diện** |
| **Day 2** | 2026-07-17 | [day02.md](day02.md) | [day02_en.md](day02_en.md) | Đọc Inkling (975B MoE) → Inkling-mini: MoE reasoner 74M/31M active → A/B WIN (60.8% vs 57.2%) → **jwm_v4.pt: QA 65.6%, IoU@0.5 0.360, trials IoU +51%** → push GitHub; caveat: latency batch-1 + ECE cần Day 3 |
| **Day 3** | 2026-07-18 | [day03.md](day03.md) | [day03_en.md](day03_en.md) | QLoRA 2B trên 4GB chết → "kiến trúc của chúng ta" → **JWM-Read** 768px/576 token/168M train Kaggle T4 (tok_acc 0.80) → benchmark 3 bộ (Ladder + VietDocVQA + MTVQA-VI): **exact 0%, blind-control Δ=0.000 — model không dùng ảnh, shortcut learning do curriculum chữ-đoán-được**; Day 4: data ngẫu nhiên chống-shortcut, gate bằng CER tự sinh, KV cache |

## Quy ước / Conventions

- Mỗi ngày một cặp file: `dayNN.md` (vi) + `dayNN_en.md` (en)
- Mỗi log: các *giai đoạn* với quyết định – bằng chứng – bài học; bảng saga cho chuỗi debug
- Vấn đề mở của ngày N là dòng đầu của ngày N+1
