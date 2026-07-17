# Inkling-mini — MoE reasoner cho JWM (thí nghiệm Day 2)

> Đọc kiến trúc thật của `thinkingmachines/Inkling` (từ config.json) và thu nhỏ
> có chọn lọc xuống GTX 1650 4GB. Mục tiêu: **dung lượng tham số 68M-class với
> chi phí bước train 28M-class** — đánh thẳng vào kết luận saga Day 1.

---

## 1. Kiến trúc Inkling thật (đọc từ config.json)

| Thành phần | Giá trị thật | Ghi chú |
|---|---|---|
| Backbone | decoder-only, 66 layers, hidden 6144 | |
| Attention | 64 Q heads / 8 KV heads (GQA 8:1), head_dim 128 | log-scaled attention, floor 128K, α 0.1 |
| Hybrid attention | **55/66 layer LOCAL** (sliding window 512) + 11 GLOBAL | pattern lặp: local tại 0,2,4-7 mỗi nhóm 8 |
| Position | relative PE extent 1024, dim 16; context **1M token** | |
| **MoE** | **256 routed experts, top-6, +2 shared** | sigmoid gate (không softmax), route_scale 8.0 |
| Expert size | intermediate **3072 = d/2** (fine-grained, kiểu DeepSeek) | dense MLP intermediate 24576 = 4d |
| Dense layers | layer đầu dùng dense MLP (không MoE) | ổn định routing sớm |
| MTP | 8 next-n prediction layers | tăng hiệu quả train |
| Vision | hierarchical MLP 4 lớp, patch **40px**, temporal patch 2 frame | không ViT attention! |
| Audio | 80 mel bins, quantize 16-vocab, delta-mel | |
| Tổng/Active | 975B / 41B (4.2%) | |

**Bài học cấu trúc đáng giá nhất:** experts NHỎ (d/2) và NHIỀU (256), kích hoạt
thưa (2.3%) — dung lượng khổng lồ, compute mỗi token gần như không đổi.

## 2. Nguyên tắc thu nhỏ

1. **Đổi MỘT biến mỗi lần** (bài học saga Day 1): bản mini v1 chỉ thay **FFN của
   reasoner tower** bằng MoE. Attention MRoPE hai-tower, generator dense, mọi
   objective — GIỮ NGUYÊN như v3 đã chứng minh 58.8%.
2. Bỏ những gì vô nghĩa ở quy mô của ta: hybrid local/global (chuỗi 200 token),
   context 1M, GQA (đầu ta đã nhỏ), audio encoder (JARVIS có ASR riêng).
3. Để dành cho thí nghiệm sau: hierarchical patch encoder (ứng viên chữa shape),
   MTP (tăng tốc học byte-level).

## 3. Bản mini — thông số

| Thành phần | Inkling | **Inkling-mini (JWM v4 reasoner)** | Tỷ lệ giữ |
|---|---|---|---|
| hidden | 6144 | 384 (= pipeline_scale v3) | 1/16 |
| layers | 66 | 8 | — |
| Expert hidden | d/2 = 3072 | **d/2 = 192** | ✓ giữ nguyên tỷ lệ |
| Routed experts | 256 | **32** | 1/8 |
| Top-k | 6 (2.3%) | **4 (12.5%)** | thưa vừa phải cho model nhỏ |
| Shared experts | 2 | **1** | ✓ |
| Gate | sigmoid + route_scale | **sigmoid → top-k → chuẩn hóa tổng=1** | ✓ đơn giản hóa an toàn |
| Dense layer đầu | có | **layer 0 dense** | ✓ |
| Load balancing | (nội bộ) | aux loss kiểu Switch, α=0.01 | chuẩn mở |

**Toán tham số (mỗi layer reasoner, SwiGLU 3·d·h):**
- 1 expert: 3·384·192 = 221K
- MoE layer: 32 routed + 1 shared = 33 × 221K ≈ **7.3M** (dense v3: 3·384·1024 ≈ 1.18M → dung lượng ×6.2)
- **Active mỗi token**: (4 routed + 1 shared) × 221K ≈ 1.11M ≈ dense v3 (1.18M) → **tốc độ bước ≈ v3**

**Tổng thể model:**
- Reasoner tower (7 MoE + 1 dense): ~56M | Generator tower (dense, giữ nguyên): ~26M
- **Tổng ≈ 86M — dung lượng lớn hơn cả bản 68M từng thất bại**
- **Active/bước ≈ 31M — đúng chi phí của bản 28M đã chứng minh**
- VRAM train (fp32 + Adam + grads + activations b48) ≈ 2.6-3.0GB ✓

## 4. Kế hoạch kiểm chứng (một biến, đối chứng sạch)

- Baseline: v3 r1/r2 đã có số (54.5% / 57.2%)
- Thí nghiệm: **r1-MoE 3000 bước + r2-MoE 800 bước** — cùng data, cùng LR, cùng batch 48,
  chỉ khác reasoner FFN → so `val_qa_acc`
- Điều kiện thành công: r2-MoE > 57.2% và tốc độ bước ≥ 85% của v3
- Nếu thắng: chạy tiếp g1→g5 với MoE reasoner → jwm_v4.pt
- Test bổ sung: router health (không expert nào chết, entropy tải), active-param đếm thực

## 5. File liên quan

- `jwm/moe.py` — MoEFFN (sigmoid router, top-k, shared expert, aux loss)
- `jwm/config.py` — cờ `reasoner_moe` + siêu tham số MoE
- `jwm/configs.py` — `pipeline_scale_moe()`
- `tests/test_jwm_math.py` — test router/sparsity/aux
- `scripts/exp_moe_reasoner.py` — thí nghiệm đối chứng r1+r2 (không đụng checkpoint v3)
