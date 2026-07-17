# JWM — Nhật ký dự án · NGÀY 2 (2026-07-17)

> Inkling-mini: clone topology MoE của Thinking Machines xuống quy mô micro làm
> reasoner mới → **jwm_v4.pt, kỷ lục mọi metric hiểu + grounding**.
> *(Mục lục chuỗi nhật ký: [README.md](README.md))*

---

## Mở màn — di sản Day 1

Day 1 khép với `jwm_v3.pt` (31M dense, QA 58.8%, mIoU 0.285) và câu hỏi mở: làm sao
có dung lượng 68M-class khi ngân sách bước train chỉ đủ cho 28M?

## 1. Đọc kiến trúc Inkling (`thinkingmachines/Inkling`)

Từ config.json thật: 975B/41B active, 66 layer, **fine-grained MoE kiểu DeepSeek**
(256 experts hidden=d/2, top-6 + 2 shared, sigmoid gate, dense layer đầu), 55/66
layer local-attention cửa sổ 512, MTP 8 tầng, vision = MLP phân tầng patch 40px.
Không có dataset public ("None public yet"). Không thể chạy local: NVFP4 ~550GB —
lượng tử hóa không bắc nổi cầu 137× (đã tính cho user).

## 2. Inkling-mini — trả lời đúng câu hỏi Day 1

Cấy **đúng phần quý nhất** (topology MoE) vào reasoner tower của JWM, giữ nguyên
mọi thứ đã chứng minh (một-biến-mỗi-lần):
- 32 experts hidden d/2=192, top-4 + 1 shared, sigmoid→top-k→chuẩn hóa, dense layer 0
- Switch aux loss α=0.01; generator tower giữ dense
- **73.94M tổng / 30.6M active** — dung lượng vượt bản 68M từng thất bại, chi phí
  bước = bản 28M đã chứng minh (87.5% tốc độ dense)
- 29/29 test (thêm: sparsity, aux, gradient generator không rò vào MoE)

## 3. A/B đối chứng (một biến, cùng data/LR/batch/seed)

| | dense (baseline v3) | **MoE** |
|---|---|---|
| r1 (3000 bước) | 54.5% | **58.0%** |
| r2 (800 bước) | 57.2% | **60.8% → WIN** |

Router khỏe hoàn hảo: entropy 3.28-3.40/3.47, **0 expert chết** cả 7 layer.

## 4. g1→g5 trên reasoner MoE → jwm_v4.pt (129.6 phút)

Bug bắt kịp trước khi chạy: `init_generator_from_reasoner` copy FFN MoE→dense sẽ
vỡ shape → sửa MoE-aware (chỉ copy attention+norm khi loại FFN khác nhau).

**Bảng tốt nghiệp (test set):**

| Metric | v1 | v3 | **v4-MoE** |
|---|---|---|---|
| QA exact-match | 56.4% | 58.8% | **65.6%** |
| — what_held | 57% | 57% | **68.8%** |
| — where | 55% | 68% | **72.3%** |
| Grounding IoU@0.5 (4-step) | 0.184 | 0.268 | **0.360** |
| Grounding mIoU | 0.201 | 0.285 | **0.355** |
| FD beats-copy | 27% | 31% | **45.8%** |
| T2I neg-probe | — | 0.75 | 0.708 |

**Trials (74):** mean IoU 0.247 → **0.373 (+51%)**; IoU@0.5 0.275 → **0.375**.

## 5. Caveat trung thực (agenda Day 3)

1. **Latency inference batch=1**: 8.3s/trial (v3: 1.6s) — hai thủ phạm:
   (a) vòng lặp expert-major 32 vòng/layer chưa tối ưu cho batch nhỏ,
   (b) `generate_answer` rebuild cả chuỗi mỗi byte — **chưa có KV cache** (tối ưu
   này lợi cho cả dense lẫn MoE, ước ~10×).
2. **ECE 4-step 0.084** (v3: 0.052) — calibration hơi yếu đi; wrong_abstain tăng
   (7 vs 2), precision-khi-assert lần đầu <100% (8/9). Cần Platt per-generation
   + xem lại threshold.
3. FD vẫn dưới copy-baseline (20.77 vs 21.39) dù beats-copy tăng mạnh.

## 6. Việc khác trong ngày

- **Push GitHub**: https://github.com/celesnity/mini-world-model (44 file, allowlist
  staging — loại trừ notebook/data/checkpoint vì quyền riêng tư + dung lượng).
- Soạn message tiếng Anh tổng kết ngày cho user.

## Thống kê Day 2

Inkling đọc + mini thiết kế + MoE implement + A/B 218 phút + pipeline 130 phút
+ trials — trọn trong một ngày, trên một chiếc GTX 1650.
