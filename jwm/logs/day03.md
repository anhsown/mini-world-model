# JWM — Nhật ký dự án · NGÀY 3 (2026-07-18)

> JWM-Read: dạy JWM đọc chữ Việt, tài liệu, cảnh OOD — **trên chính kiến trúc của
> mình**, train trên Kaggle T4 sau khi chứng minh 4GB local không gánh nổi.
> *(Mục lục chuỗi nhật ký: [README.md](README.md))*

---

## Mở màn — di sản Day 2

`jwm_v4.pt` (73.94M MoE) đang là shadow brain. Backlog: latency 8.3s/trial (thiếu
KV cache), ECE 0.084 cần recalibrate, FD dưới copy-baseline. Hôm nay user đưa bài
mới: dataset tài liệu tiếng Việt 64.5K trang — "clone về và train cho models của
chúng ta".

## 1. Ngã rẽ QLoRA — thí nghiệm thất bại có giá trị

Kế hoạch đầu: QLoRA Qwen3-VL-2B trên GTX 1650. Hai vòng smoke đều chết:
- v1 OOM: peft upcast lm_head 151K-vocab lên fp32 (~1.2GB); LoRA target lan cả
  vision tower
- v2 (LLM-only modules, manual kbit): hết OOM cứng nhưng Windows WDDM tràn sang
  shared memory — "9.6GB VRAM" trên card 4.3GB → **112s/bước** → 10K bước ≈ 13 ngày
- User chốt hai điều: (a) "chúng ta đang train trên kiến trúc của chúng ta mà" —
  không đi đường model ngoài; (b) "vậy dùng kaggle T4 đi"

Bài học: trên 4GB + WDDM, mô hình 2B kể cả 4-bit là ảo tưởng tốc độ; và mục tiêu
dự án là kiến trúc tự viết, không phải fine-tune model người khác.

## 2. Tái kiến trúc mắt — JWM-Read (`reader_scale`)

Câu hỏi user: "tái kiến trúc để JWM đọc được chữ, tài liệu, cảnh OOD được không?"
Trả lời bằng thiết kế, không mode mới — READ chỉ là QA-mode với mắt to hơn:
- **768px** input (từ 64px), patch-16 + **gộp 2×2** (Inkling-style hierarchical
  MLP stem, 2 tầng) → 24×24 = **576 token thị giác**, img_tok_dim 3072
- Config mới: `patch_merge`, `vision_mlp_layers`; d512/L10/MoE 32 experts giữ
  nguyên topology đã thắng Day 2 → **167.9M tổng / ~91M active**
- Smoke local: forward+backward 768px batch 2 = **1.82GB VRAM** → T4 chạy batch 16
- Metric mới: **CER** (Levenshtein trên ký tự unicode — sai 1 dấu = 1 lỗi, không
  phải 2-3 lỗi byte) + `eval_read` per-level

## 3. Dữ liệu — lazy là bắt buộc, không phải lựa chọn

64.516 trang × 768² uint8 ≈ 50GB tensor — không precompute nổi → `read_data.py`:
- **Synthetic vô hạn**: render chữ Việt PIL 4 cấp (L1 từ to 70-130px → L4 đoạn
  văn 20-36px), nền giấy, camera_degrade tham số pin từ Day 1
- **Doc thật**: JSONL 64.5K record đa lượt → tách lượt đơn; đáp án gốc dài (>90%
  vượt 224 byte) → fallback **lấy câu trả lời trực tiếp đầu tiên** nếu tự đứng
  được → giữ **21.641 cặp** (từ ~7K nếu lọc thô — bài học: đừng truncate giữa
  chừng, supervision cụt dạy model nói cụt)
- `LazyReadBatcher` render/load lúc batch + `PrefetchBatcher` (thread nền che
  CPU render sau GPU step); `train_stage` thêm tham số `batcher=` để tiêm
- 11 test mới (40 tổng, all pass): shape stem gộp, tính chất CER, render
  deterministic theo seed, letterbox giữ tỉ lệ

## 4. Kaggle T4 — hạ tầng ngoài nhà lần đầu

- Notebook tự chứa `jwm/kaggle/jwm_read_t4.ipynb`: clone GitHub repo → tải
  dataset HF → giải nén tars → 3 stage curriculum (1.5K/4K/6K bước, batch 16)
  → eval CER → save; **checkpoint atomic mỗi 500 bước**, resume theo stage
- Va vấp thực tế đủ bộ: GPU xám vì tài khoản chưa verify (đổi tài khoản), repo
  private không clone được (push sang `anhsown/mini-world-model` public), file
  output bay vì Persistence chưa bật + session hết hạn (học được **Save & Run
  All (Commit)** — chạy nền cloud, output vĩnh viễn)
- Notebook giữ local, không push (quy tắc riêng tư); source push 2 remote:
  celesnity (`32e658f`) + anhsown

## 5. Run hoàn tất — jwm_read_v1.pt (671MB)

- Stage 0: tok_acc 0→**0.62**/1500 bước · Stage 1: 0.51→**0.71**/4000 · Stage 2
  (50% doc thật): →**~0.80**/6000, 0.7 it/s, không sự cố resume
- `moe_aux` phẳng ~0.090 = router cân tải, không expert chết; warning
  `float(la)` đã audit — chỉ là metric, gradient aux vẫn chảy (dòng 333)
- Eval Kaggle: CER doc 0.78; CER synthetic **nghịch đảo độ khó** (L1 29.9 →
  L4 0.82) — dấu hiệu đầu tiên của chuyện lớn hơn (xem §6)

## 6. Benchmark — model không đọc, và bằng chứng tận tay

User: "chạy thử benchmark xem điểm chết ở đâu." Research: MTVQA (ByteDance,
public) dùng được; ViTextVQA/ViOCRVQA/5CD-AI đều gated; VinText cần detection.
Dựng 3 benchmark + 1 đối chứng (`scripts/bench_read_v1.py`, `bench_read_blind.py`):

**Ladder synthetic (108 mẫu, seed 2026):**

| Bài | CER md | Exact | Chứa | Dừng EOS |
|---|---|---|---|---|
| T0 1 ký tự 200px | 52.0 | 0 | 0.17≈hên | 0 |
| T1 1 từ (120→28px) | 12-15 | 0 | **0** | 0 |
| T2 dòng / T3 đoạn | 2.3 / 0.82 | 0 | 0 | 0 |
| T4 từ 80px + nhiễu | 14.8 | 0 | 0 | 0 |

**VietDocVQA held-out (40 trang):** CER md 0.70, exact 0, dừng **0.80**.
**MTVQA-VI (50 mẫu):** CER md 2.36, exact 0, chứa 0 — trả lời đúng *kiểu* câu
hỏi (hỏi thương hiệu → "Thương hiệu của... là") nhưng nội dung bịa hoàn toàn.

**Đối chứng blind (quyết định):** tok_acc teacher-forced với ảnh đúng vs ảnh
TRÁO: synth 0.6068 vs 0.6073, doc 0.7878 vs 0.7879 — **Δ ≈ 0.000, model không
hề dùng ảnh khi sinh chữ**. tok_acc 0.78 của run là 100% học vẹt ngôn ngữ.

**Chẩn đoán gốc rễ — shortcut learning do lỗi thiết kế curriculum**: v3/v4 buộc
phải nhìn vì đáp án (màu/hình/vị trí) không đoán nổi bằng ngôn ngữ; JWM-Read
train trên **từ tiếng Việt thật** — đoán được rất tốt bằng ngôn ngữ thuần →
gradient chọn đường dễ, kẹt cực tiểu học vẹt suốt 11.5K bước. Tín hiệu thị giác
duy nhất: phân loại *loại ảnh* (dừng 80% trên doc thật vs 0% trên synthetic).
CER "đẹp dần theo độ dài" (52→0.82) là ảo ảnh mẫu số, không phải năng lực.

## 7. Sự cố ngoài lề — session đồng nghiệp hiện trong app

## 7. Sự cố ngoài lề — session đồng nghiệp hiện trong app

Tài khoản Claude công ty dùng chung; đồng nghiệp bật remote control → session của
họ hiện trong danh sách của mọi máy cùng account. Đã xác minh session dự án này
**thuần local, không remote control** (không nhãn hover, không có trên
claude.ai/code). Khuyến nghị đã gửi: tách account riêng từng người.

## Agenda Day 4 — sửa gốc rễ, thi lại cùng đề

1. **Dữ liệu chống-shortcut**: curriculum lõi = chuỗi ký tự/từ NGẪU NHIÊN
   (ngôn ngữ không đoán nổi → muốn có điểm phải nhìn); chữ thật quay lại sau
2. **Gate chuyển stage bằng CER sinh-tự-do** — tok_acc đã lừa ta trọn một run
3. EOS: tăng mật độ đáp án ngắn + upweight token dừng
4. KV cache cho generate_answer (benchmark 108 mẫu mất 40 phút — cần 10×)
5. Train v2 trên Kaggle → chạy lại đúng 3 benchmark seed cố định, so delta
6. Backlog cũ: batched expert GEMM, ECE recalibration, FD vs copy-baseline
