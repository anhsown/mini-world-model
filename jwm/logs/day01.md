# JWM — Nhật ký dự án · NGÀY 1 (2026-07-16)

> Từ paper Cosmos 3 → bộ não world-model chạy thật trên GTX 1650 4GB,
> tích hợp vào JARVIS với trial logging đầy đủ.
> *(Chuỗi nhật ký nhiều ngày — xem [README.md](README.md) để có mục lục.)*

---

## Giai đoạn 0 — Nền tảng kiến thức

Nghiên cứu kỹ paper **NVIDIA Cosmos 3** (Omnimodal World Models for Physical AI) qua 4 phần:
kiến trúc (dual-tower MoT, MRoPE, rectified flow, action tokens), dữ liệu (2 nhánh, curation
đa tầng), huấn luyện (curriculum nhiều giai đoạn), hạ tầng (SILA, packing, caching).

## Giai đoạn 1 — Kiến trúc JWM (v1, 10.7M params)

**Package `jwm/`** — viết từ đầu, mọi công thức khớp 1-1 với đặc tả [DESIGN.md](DESIGN.md):

| File | Nội dung |
|---|---|
| `mathx.py` | 3D MRoPE (float coords, temporal modulation δt=TPS_base/TPS), rectified flow (xσ, v*, Euler, shift schedule), logit-normal/mode sampling, rot6d↔SO(3), IoU, ECE, PSNR, sqrt-len normalize |
| `layers.py` | Dual-tower MoT block (reasoner causal / generator bidirectional), two-way flat attention (2 lần SDPA), AdaLN-zero, reasoner K/V detach + cache |
| `model.py` | 4 mode: QA (AR) / GROUND (bbox action + clean latent conditioning) / FD (dự đoán frame kế) / T2I (thêm ở v3); confidence head P(IoU≥0.5); tower-copy init |
| `tokenizer.py` | Byte-level 262 vocab (an toàn tiếng Việt) |

**Kiểm định 2 vòng đúng yêu cầu:**
- **27 property test** (`tests/test_jwm_math.py`): RoPE bất biến vị trí tương đối, flow khôi phục
  x₀ giải tích, AR *không bao giờ* thấy DM (perturbation + gradient), AdaLN-zero = identity,
  CE alignment ghim bằng mutation-proof overfit test, cached-sampler tương đương số học, CFG giải tích.
- **Workflow đối kháng 30 agents** đọc code thật: 25 phát hiện → 5 xác nhận → sửa hết
  (reasoner K/V caching ~8× rẻ hơn khi sample, CFG cho FD, 3 lỗ hổng test).

**Phê phán Cosmos 3** ([COSMOS3_CRITIQUE.md](COSMOS3_CRITIQUE.md)) — workflow 38 agents
(research web + 4 lăng kính + phản biện): 29 điểm yếu đề xuất → **7 sống sót**. Nặng nhất:
confidence tự khai JSON "mù kiến trúc" với sample của generator. **3 cải tiến đã hiện thực:**
1. Confidence head học + Platt calibration, đo bằng ECE (thay JSON tự khai)
2. Ablation boundary-embedding vs gap cố định 15000
3. Reflection pass lúc inference (reasoner tái kiểm output generator ở mức pipeline)

## Giai đoạn 2 — Dataset SDG (validate với real case)

`sdg.py` — SDG-JarvisSim: cảnh procedural (6 hình dạng × 6 màu, vật lý chuyển động, che khuất),
nền ghép từ **frame camera thật của JARVIS**, camera-degradation model **auto-tune đến khi
5 thống kê Wasserstein** (độ sáng, tương phản, gradient, sharpness, entropy màu) **dưới ngưỡng
so với frame thật** — đúng yêu cầu "dựng lại đến khi valid". Judge 3 trục
(faithfulness/completeness/correctness) + dedup scene-hash + structured caption chuẩn Cosmos.

## Giai đoạn 3 — Train v1 qua notebook quan sát được

`train_world_brain.ipynb` (21 cell) chạy live trong JupyterLab (điều khiển Run-All qua
`window.jupyterapp`), user quan sát trực tiếp. Kết quả v1 (10.7M, test set):

| Metric | v1 | Ghi chú |
|---|---|---|
| QA exact-match | **56.4%** (count 81%, exist 64%) | |
| Grounding mIoU / IoU@0.5 | 0.20 / 0.18 | bão hòa ở 10.7M |
| **ECE sau Platt** | **0.040** (từ 0.414) | đạt mục tiêu <0.05 của critique |
| FD PSNR (min-over-k) | 20.4 vs copy-baseline 21.4 | |
| Latency ground 4-step | **95ms** | realtime được |

Kèm: Stage 2.5 inject vào notebook đang sống; script `calibrate_confidence.py`.

## Giai đoạn 4 — Gắn vào JARVIS + 66 trial có log

- `core/world_brain.py`: 3 chế độ `off/shadow/primary` (mặc định **shadow** — chạy song song
  Qwen3-VL, chỉ ghi log); reflection pass; tự chọn checkpoint mới nhất (v3>v2>v1).
- Hook `_shadow_world_brain` trong `vision.py` (không bao giờ được phép phá vision turn).
- `scripts/run_world_brain_trials.py`: **mỗi trial ghi đủ 9 trường** — audio, transcript, frame,
  predicted region, answer, confidence, latency, ground truth, failure category.
- **66 trials** (40 synthetic GT-chính-xác + 26 audio thật qua faster-whisper trên frame thật
  có bbox annotate tay): synthetic — model assert 5 lần **đúng cả 5 (precision 100%)**, abstain
  đúng 34/40; real OOD — confidence ≈ 0.00-0.02, **không bao giờ hallucinate sự tự tin**.

## Giai đoạn 5 — Vision mode 30fps (yêu cầu user)

Điều tra: xử lý chỉ tốn ~3ms (không phải nghẽn) → **camera V380 trần cứng 21.5fps@720p**
(MJPG đã mặc định; 480p rơi về YUY2 11.5fps tệ hơn; exposure bị driver bỏ qua; chỉ 1 camera).
Giải pháp trong `vision.py`: tách **capture thread** (nhịp camera tự nhiên) + **display thread
30Hz** với **cross-fade frame interpolation** (FRC chuẩn), HUD nhận BGR trực tiếp, auto-degrade
width, telemetry 2 con số trung thực. **Đo thực tế: 30.0fps push, 28.3 khung hình riêng biệt/s,
p95 4.3ms.** 46/46 test pass.

## Giai đoạn 6 — v2 28M (bị thay thế) + shutdown-safety

Scale lên 28M monolithic; giữa chừng user cần tắt máy → dừng có kiểm soát, lưu partial từ RAM
kernel (bài học: 2 kernel cùng lúc, interrupt phải đúng kernel id; queue run-all bị hủy theo
interrupt). Từ đó **mọi training đều checkpoint định kỳ** (`ckpt_fn` trong trainer). v2 sau đó
bị v3 thay thế.

## Giai đoạn 7 — v3: Tái cấu trúc đúng paper + 68.65M

User yêu cầu: mỗi giai đoạn train một file riêng + data 2 nhánh + 0.5B.
- **0.5B bất khả thi trên 4GB** (riêng optimizer ~8GB) → AskUserQuestion → chốt **~80M**
  (thực tế 68.65M: d512/L10/h16, VRAM peak 3.01GB, 0.66 it/s).
- **`jwm/data_builders/`** (2 nhánh, 5 loại): reasoner_pretrain (40K) + reasoner_sft (12K) |
  generator_image T2I (8K) + generator_video FD (6K) + generator_action bbox (14K).
  Audio của paper được thay bằng Video có chủ đích (JARVIS có ASR/TTS riêng). Post-tier =
  top-quantile theo quality score (sửa từ ngưỡng cố định bị vô nghĩa 99.6%).
- **`jwm/stages/`** (7 giai đoạn nối checkpoint): r1→r2 (reasoner) → g1 (train+freeze ConvAE,
  **copy tower reasoner→generator** đúng nghi thức Cosmos §4) → g2 (action vào) → g3 (T2I post)
  → g4 (I2V post) → g5 (policy 4-step + Platt → `jwm_v3.pt` triển khai). `run_pipeline.py`
  chạy thống nhất, **mọi stage shutdown-safe**.
- Model thêm mode **T2I** + metric tự nhất quán (reasoner của chính nó kiểm tra ảnh sinh ra).
- **Dashboard live** `scripts/pipeline_dashboard.py` (localhost:8877, tự refresh 5s).

## Giai đoạn 8 — Saga debug reasoner v3 (đang diễn ra)

| Vòng | Hành động | Kết quả | Bài học |
|---|---|---|---|
| 1 | r1 1800 bước, 14K data | val QA 35% (v1: 56%) | — |
| 1b | +800 bước r2 | 38.4% → **37.2%** | thêm bước vô ích |
| — | Phân tích per-kind | count 67%, **what_held 23.5%** | **exact-match ≈ tok_acc^độ_dài** (0.95²⁷≈25% ✓) |
| 2 | Data ×3 (52K), r1 3200 bước | **38%** | data thêm không đủ |
| — | Chẩn đoán 4-số + dump ảnh | gap train/val = **0.0001**; train-exact cũng chỉ 35%; ảnh: **màu đúng, HÌNH sai hệ thống** (tròn→vuông, tam giác→vuông) | không phải memorization; optimization kẹt (plateau 3000 bước) + tín hiệu shape loãng |
| 3 (hiện tại) | Curriculum tách thuộc tính (câu hỏi shape-only/color-only, đáp án ~10 byte) + chờ verdict LR từ r2 (1.2e-4) | đang chạy | — |

Phát hiện phụ: `serve_brain.py` cũ chiếm ~1GB VRAM **suốt mọi đợt train** → đã tắt.

## Thống kê tổng

- **~40 file** code/test/doc mới; 27 property test; 2 workflow đối kháng (68 agents)
- 66 trials logged; 3 thế hệ model (v1 hoàn chỉnh, v2 thay thế, v3 đang train)
- Hạ tầng: staged pipeline shutdown-safe, dashboard live, monitor nền, trial harness
- Vision 5fps → 30fps thật trên camera 21.5fps

## EPILOGUE — Saga khép lại lúc 02:16 (rạng sáng ngày 2)

Saga debug kết thúc bằng chuỗi loại trừ hoàn chỉnh: thêm bước ✗ → data ×3 ✗ → LR thấp ✗ →
scale/init ✗ (thí nghiệm 3 nhánh: 28M/68M/68M-scaled đều 0.92 @400 bước) → nhãn ✗ (soi mắt)
→ probe data cũ 0.92 @400 → **kết luận: không có bug**. Sự thật hai lớp: (1) 68M cần ngân
sách QA-polish vượt khả năng một đêm của GTX 1650 (68M@3200 bước → tok-acc 0.94; 28M@~2500
→ 0.98); (2) camera autotune vòng 3 trôi lên noise 10.37 (+44% so với v1) làm shape khó thêm.

**Đơn thuốc cuối (user chọn qua AskUserQuestion):** pipeline chạy ở scale 28M đã chứng minh
(`pipeline_scale()`), camera ghim về tham số v1, data rebuild kèm curriculum tách thuộc tính
(câu hỏi shape-only/color-only), batch 48. 68M trở thành đề tài nghiên cứu Day 2.

**Kết quả — pipeline 7 giai đoạn hoàn tất trong 224 phút, `jwm_v3.pt` (31M) xuất xưởng:**

| Metric (test) | v1 (10.7M) | **v3 (31M)** |
|---|---|---|
| QA exact-match | 56.4% | **58.8%** (where 68%, exist 77%, count 81%) |
| Grounding mIoU / IoU@0.5 (4-step) | 0.201 / 0.184 | **0.285 / 0.268** (+42%) |
| ECE hiệu chuẩn (val) | 0.040 | **0.025** |
| T2I self-consistency (mode mới) | — | 0.48 pos / 0.75 neg |
| Latency ground 4-step | 95ms | 104ms |

**Trials (74):** synthetic ok-rate 12.5%→**20%**, mean IoU 0.205→**0.247**, precision khi
assert vẫn **100%** (8/8); real OOD vẫn luôn abstain đúng (conf 0.02-0.06). Não v3 được
`world_brain` tự nhận, chạy shadow mode trong JARVIS.

Nhật ký r1 từng giai đoạn: r1 54.5% → r2 57.2% → g2 grounding 0.275 → g3 T2I 0.67 →
g5 ECE 0.025. Mọi giai đoạn đều shutdown-safe, quan sát qua dashboard localhost:8877.

## Mở sang Day 2

1. **Thí nghiệm ngân sách 68M**: xác định số bước QA-polish cần để d512/L10 đạt tok-acc 0.98
2. Cân nhắc promote WORLD_BRAIN_MODE shadow → primary cho synthetic-domain scenes
3. FD vẫn chưa vượt copy-baseline (20.1 vs 21.4) — hướng cải thiện: motion lớn hơn, nhiều frame
