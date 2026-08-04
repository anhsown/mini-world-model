# JWM — Tổng kết 6 ngày xây dựng mini-world-model

**Giai đoạn:** 2026-07-16 → 2026-07-21 · **Phần cứng chính:** GTX 1650 4GB + Kaggle T4 / T4×2
**Repo công khai:** `github.com/anhsown/mini-world-model` · `github.com/celesnity/mini-world-model`
**Nguồn:** tổng hợp từ [day01](day01.md)–[day06](day06.md), các file `metrics_*.json` và benchmark artifact thật.

---

## 1. Tóm tắt điều hành

Sáu ngày này xây từ con số không một world model kiểu Cosmos 3 thu nhỏ, chạy được trên card 4GB, rồi mở rộng sang hai năng lực khó hơn: **đọc chữ** và **hình học vật lý**.

Kết quả chia làm hai nửa rõ rệt:

| Giai đoạn | Ngày | Kết quả |
|---|---|---|
| **Xây lõi + scale up** | 1–2 | ✅ **Thành công** — 4 thế hệ model, mọi chỉ số tăng đơn điệu, `jwm_v4.pt` ship và chạy shadow trong JARVIS |
| **Mở sang đọc chữ** | 3–4 | ❌ **Thất bại** — model không hề dùng ảnh, phát hiện bằng blind-control |
| **Mở sang hình học vật lý** | 5–6 | ❌ **Blocked** — 4 lần build liên tiếp, lần cuối chỉ qua 2/7 causal gate |

**Thành tựu lớn nhất không phải là model, mà là quy trình kiểm định.** Ba lần liên tiếp, các chỉ số huấn luyện thông thường báo tiến bộ trong khi hệ thống kiểm soát nhân quả chứng minh model không học được gì thật. Nếu chỉ nhìn training loss, cả ba đều đã được công bố nhầm là thành công.

---

## 2. Dòng thời gian

| Ngày | Chủ đề | Sản phẩm | Trạng thái |
|---|---|---|---|
| **1** | Paper Cosmos 3 → kiến trúc JWM → v1 → tái cấu trúc 7-stage → v3 | `jwm_v1.pt` 10.7M, `jwm_v3.pt` 31M | ✅ Ship |
| **2** | Đọc Inkling → cấy MoE vào reasoner → A/B → v4 | `jwm_v4.pt` 73.94M | ✅ Ship, shadow mode |
| **3** | JWM-Read: dạy đọc chữ Việt, train Kaggle T4 | `jwm_read_v1.pt` 167.9M | ❌ Không đọc được |
| **4** | JWM-Read v3: chống shortcut, DDP T4×2 | checkpoint `blocked` | ❌ Dừng ở gate s0 |
| **5** | Eye Physical v1 → v2 → v3 → v3.1 | 4 checkpoint `blocked` | ❌ Fail OOD |
| **6** | Eye v3.2.1 Robust Causal Geometry | `jwm_eye_v321_blocked.pt` 381.9M | ❌ 2/7 gate |

---

## 3. Chi tiết từng ngày

### Ngày 1 — Từ paper đến model chạy thật

Đọc kỹ Cosmos 3 (kiến trúc, dữ liệu, huấn luyện, hạ tầng) rồi viết lại từ đầu package `jwm/` với mọi công thức khớp 1-1 đặc tả trong `DESIGN.md`.

**Kiểm định hai vòng:**
- 27 property test: bất biến vị trí tương đối của RoPE, flow khôi phục x₀ giải tích, AR **không bao giờ** thấy DM (kiểm bằng cả perturbation lẫn gradient), AdaLN-zero = identity, CE alignment ghim bằng mutation-proof overfit test.
- Workflow đối kháng 30 agent đọc code thật: 25 phát hiện → 5 xác nhận → sửa hết.

**Phê phán Cosmos 3** (`COSMOS3_CRITIQUE.md`, workflow 38 agent): 29 điểm yếu đề xuất → **7 sống sót**. Ba cải tiến đã hiện thực trong JWM: confidence head học được + Platt calibration đo bằng ECE (thay confidence tự khai JSON), ablation boundary-embedding vs gap cố định, reflection pass lúc inference.

**Dataset SDG-JarvisSim:** cảnh procedural ghép trên frame camera thật của JARVIS, camera-degradation **auto-tune đến khi 5 thống kê Wasserstein** (độ sáng, tương phản, gradient, sharpness, entropy màu) đạt ngưỡng so với frame thật.

**Saga debug 68M — chuỗi loại trừ hoàn chỉnh.** Bản 68.65M cho val QA chỉ 35–38% trong khi v1 10.7M đạt 56%. Sáu vòng loại trừ:

| Vòng | Giả thuyết | Kết quả |
|---|---|---|
| 1 | Thiếu bước train | +800 bước → **giảm** 38.4% → 37.2% |
| 2 | Thiếu dữ liệu | ×3 data → 38%, không đổi |
| 3 | Learning rate | LR thấp → 35.2%, tệ hơn |
| 4 | Thiếu scaled init | 3 nhánh 28M/68M/68M-scaled → **đều 0.92 @400 bước** |
| 5 | Nhãn sai | soi ảnh tận mắt → nhãn đúng, **màu đúng nhưng HÌNH sai hệ thống** |
| 6 | Probe data cũ | 0.92 @400 → **không có bug ở đâu cả** |

Phát hiện then chốt: **exact-match ≈ tok_acc^độ_dài_đáp_án**. Với đáp án 27 byte, 0.95²⁷ ≈ 25%, còn 0.98²⁷ ≈ 58%. Nghĩa là muốn exact-match tốt phải đẩy tok-acc lên 0.98, mà 68M cần ngân sách vượt khả năng một đêm trên GTX 1650.

Phát hiện phụ: một tiến trình `serve_brain.py` cũ chiếm ~1GB VRAM **suốt mọi đợt train** trước đó.

**Kết quả:** pipeline 7 giai đoạn (r1→r2→g1→g5) chạy 224 phút, xuất `jwm_v3.pt` 31M.

### Ngày 2 — Inkling-mini, ngày duy nhất mọi thứ đều thắng

Đọc config thật của `thinkingmachines/Inkling`: 975B/41B active, 66 layer, MoE fine-grained kiểu DeepSeek (256 expert hidden d/2, top-6 + 2 shared, sigmoid gate, dense layer đầu). Xác định **không thể chạy local** — NVFP4 ~550GB, lượng tử hoá không bắc nổi cầu 137×.

Cấy đúng phần quý nhất (topology MoE) vào reasoner tower, giữ nguyên mọi thứ đã chứng minh — **một biến mỗi lần**:

| | dense (v3) | **MoE** |
|---|---|---|
| r1 pretrain 3000 bước | 54.5% | **58.0%** |
| r2 SFT 800 bước | 57.2% | **60.8%** |

Router khoẻ hoàn hảo: entropy 3.28–3.40 trên 3.47, **0 expert chết** cả 7 layer. Bắt được một bug trước khi chạy: `init_generator_from_reasoner` copy FFN MoE→dense sẽ vỡ shape, đã sửa thành MoE-aware.

**`jwm_v4.pt` — 73.94M tổng / 30.6M active**, dung lượng vượt bản 68M từng thất bại nhưng chi phí mỗi bước bằng bản 28M đã chứng minh.

### Ngày 3 — JWM-Read, và bài học đắt nhất của cả dự án

Bắt đầu bằng một ngã rẽ thất bại có giá trị: QLoRA Qwen3-VL-2B trên GTX 1650 chết hai lần (OOM do peft upcast lm_head 151K-vocab lên fp32; rồi Windows WDDM tràn sang shared memory → 112s/bước → 10K bước ≈ 13 ngày). User chốt hai điều định hướng cả dự án: *"chúng ta đang train trên kiến trúc của chúng ta mà"* và *"vậy dùng Kaggle T4 đi"*.

**Kiến trúc JWM-Read:** 768px, patch-16 + gộp 2×2 (hierarchical MLP stem kiểu Inkling) → 576 token thị giác, 167.9M tổng / ~91M active. Train Kaggle T4, 3 stage, 11.5K bước, đạt **tok_acc 0.80**.

Rồi benchmark, và mọi thứ sụp:

| Benchmark | Kết quả |
|---|---|
| Ladder synthetic 108 mẫu | Exact **0%** mọi tier; T0 một ký tự 200px: containment 0.17 ≈ mức hên |
| VietDocVQA 40 trang | CER median 0.70, exact **0%** |
| MTVQA-VI 50 mẫu | CER median 2.36, exact **0%** — trả lời đúng *kiểu* câu hỏi nhưng nội dung bịa hoàn toàn |

**Đối chứng blind — phép đo quyết định.** So tok_acc teacher-forced khi đưa **ảnh đúng** với khi đưa **ảnh tráo**:

```
synthetic : 0.6068  vs  0.6073     Δ = -0.0005
document  : 0.7878  vs  0.7879     Δ = -0.0001
```

**Δ ≈ 0.000. Model không hề dùng ảnh khi sinh chữ.** Toàn bộ tok_acc 0.80 là học vẹt ngôn ngữ thuần tuý.

**Chẩn đoán gốc rễ — shortcut learning do lỗi thiết kế curriculum.** v3/v4 buộc phải nhìn vì đáp án (màu, hình, vị trí) không đoán nổi bằng ngôn ngữ. JWM-Read train trên **từ tiếng Việt thật** — đoán được rất tốt bằng mô hình ngôn ngữ → gradient chọn đường dễ, kẹt trong cực tiểu học vẹt suốt 11.5K bước.

Chỉ số CER "đẹp dần theo độ dài" (T0 52.0 → T3 0.82) là **ảo ảnh mẫu số**, không phải năng lực.

### Ngày 4 — Sửa gốc rễ, vẫn chưa qua

Thiết kế lại v3: 1024×768, local attention trên lưới 64×48, merge 2×2 **sau** reasoning, 119.06M. Thêm CTC OCR, four-query coordinate head, noisy teacher forcing, và **loss đối chiếu ảnh đúng vs ảnh tráo** ngay trong hàm mục tiêu.

Sáu giả thuyết dữ liệu phải pass trước khi train: nhãn random duy nhất, box hợp lệ, không truncate, chữ nhìn thấy được, zero page leakage, ảnh thật đọc được → **6/6 pass**, domain gap 0.05003, leakage 0.

Trainer T4×2 DDP với **gate chuyển stage bằng free-running CER** thay vì tok_acc — sửa đúng thứ đã lừa cả một run ở Ngày 3. Gradient audit bắt được AMP scale 65536 gây NaN ở CTC, đổi xuống 1024 → 0 NaN/Inf.

Kaggle dừng an toàn tại `s0_glyph_bootstrap` bước 3.200, trạng thái `blocked_by_metric_gate`: **CTC-CER = 1.000** trong khi gate yêu cầu ≤ 0.72.

Nhưng có tiến bộ thật: blind-image gap từ −0.0002 (v2) lên **+0.554**, ảnh đúng thắng 63.2% số batch. **Model đã mã hoá tín hiệu thị giác** — nhưng CTC blank-collapse 99.73% khiến decoder không chuyển được tín hiệu đó thành văn bản.

### Ngày 5 — Eye Physical, bốn lần build trong một ngày

| Bản | Điều đã làm | Kết quả |
|---|---|---|
| **v1** | Warm-start từ v4, geometric context memory | Pass gate nội bộ, **fail OOD**: thua constant/identity prior ở cả 6 metric trên TUM dynamic; ảnh đen còn cho depth *tốt hơn* ảnh thật |
| **v2** | Local pairwise cost volume, relative SE(3), cycle, counterfactual ranking | 4 arm A–D, arm B thắng, nhưng **1/6 causal gate** — pose vẫn thua identity prior |
| **v3** | Tìm ra lỗi camera của v2: adapter có intrinsics nhưng **collator làm rơi trường này**. Dựng CTPG-Eye, differentiable BA | Nổ số học: `grad=NaN` ở step 175, gradient norm 67549, `track_epe=Infinity` |
| **v3.1** | FP32 BA, Levenberg damping, track theo lưới nội vùng, DDP finite governor | 1/7 gate. Root cause: **invalid rigid-flow bị nội suy trước khi mask** → track EPE tăng phi vật lý tới **483,729 px** |

Một chi tiết đáng giữ lại: v1 chứng minh được depth **thật sự dùng ảnh** (sai cảnh làm AbsRel xấu 4.78×, ảnh đen xấu 12.13×) trong khi pose thì không (sai cảnh chỉ làm ATE xấu 1.058×). Cùng một model, một năng lực có nhân quả thị giác và một năng lực thì không — chỉ phát hiện được nhờ chạy control riêng cho từng đầu ra.

### Ngày 6 — Eye v3.2.1, dừng đúng lúc

Sửa flow-mask bằng validity-normalized interpolation, thêm robust heteroscedastic Laplace loss cho tracking, temporal compatibility head, negative window phá thứ tự thời gian, và định nghĩa lại **bảy promotion gate**.

Training bị adaptive controller dừng ở step **1.000** với quyết định `stop_overfit`: training loss vẫn tốt lên trong khi held-out OOD score suy giảm. Best score ở step 400 (0.20374), tại step 1.000 còn −0.03962.

**Kết quả 7 gate:**

| Gate | Đo được | Ngưỡng | |
|---|---:|---:|---|
| Depth thắng fixed prior | 0.838× | ≥1.20× | FAIL |
| Pose thắng moving identity | 1.015× | ≥1.20× | FAIL |
| BA cải thiện residual và pose | pose gain 22.63× | — | **PASS** |
| Phát hiện wrong temporal window | ≈0.000 | ≥0.15 | FAIL |
| Tracking usable + calibrated | 0.978 | ≥0.80 | **PASS** |
| Phát hiện reverse time | 1.004× | ≥1.10× | FAIL |
| Phát hiện wrong intrinsics | 1.015× | ≥1.15× | FAIL |

Việc sửa flow-mask **đã thành công** — track EPE không còn bùng nổ, và tracking/calibration trở thành năng lực mạnh nhất của checkpoint (PCK@3 = 0.995, ECE@3 = 0.058). Nhưng `temporal_compatibility ≈ 0.499` cho thấy temporal head đang ở đúng mức đoán ngẫu nhiên, và `dynamic F1 ≈ 0.03` xác nhận dynamic-scene understanding vẫn collapse.

---

## 4. Kiến trúc thay đổi qua từng ngày — cái gì, vì sao, ý tưởng

Mỗi thay đổi dưới đây đều xuất phát từ **một thất bại đo được**, không phải từ trực giác. Cột "Ý tưởng" là nguyên lý được dùng để sửa.

### Bảng dung lượng theo thời gian

| | Ngày | Model | Tổng | Active / trainable | Ảnh vào |
|---|---|---|---|---|---|
| 1 | 1 | JWM v1 | 10.7M | 10.7M | 64×64 |
| 2 | 1 | JWM v2 | 28M | 28M | 64×64 |
| 3 | 1 | *(thử 68.65M — thất bại)* | 68.65M | 68.65M | 64×64 |
| 4 | 1 | **JWM v3** | 31.01M | 31.01M | 64×64 |
| 5 | 2 | **JWM v4 MoE** | 73.94M | **30.6M** | 64×64 |
| 6 | 3 | JWM-Read v1 | 167.9M | ~91M | **768×768** |
| 7 | 4 | JWM-Read v3 | 119.06M | 82.77M | **1024×768** |
| 8 | 5 | Eye Physical v2 | 86.77M | **12.91M** | chuỗi frame |
| 9 | 5 | Eye v3.1 | 79.75M | — | chuỗi frame |
| 10 | 6 | **Eye v3.2.1** | **381.93M** | — | chuỗi frame |

Đáng chú ý: dung lượng **không tăng đơn điệu**. Sau mỗi thất bại, model bị thu nhỏ lại (68.65M → 31M ở Ngày 1; 167.9M → 119M ở Ngày 4) vì bài học là *ngân sách bước train, không phải số tham số*, mới là ràng buộc thật.

---

### Ngày 1 — Đặt nền: một model, ba cách xếp token

**Kiến trúc gốc.** Dual-tower MoT: reasoner attention nhân quả, generator attention hai chiều, nối một chiều bằng K/V của reasoner đã detach. Ảnh 64×64 patch-8 → 64 token cho nhánh hiểu; ConvAE đóng băng nén xuống 8×8×8 → gộp 2×2 → 16 token cho nhánh sinh. Tokenizer byte-level vocab 262.

**Ý tưởng nền tảng:** *không tạo mode mới, chỉ đổi cách xếp token*. Ba năng lực dùng chung một model:

| Mode | Chuỗi token | Sinh ra |
|---|---|---|
| QA | `[BOS, IMG×64, BOQ, câu hỏi, BOA]` | byte đáp án, tự hồi quy |
| GROUND | QA + `[DM: bbox nhiễu]` | khử nhiễu bbox 4 chiều |
| FD | `[BOS, IMG×64, BOQ, motion][DM: z_t sạch, z_{t+1} nhiễu]` | latent frame kế tiếp |

Vì sao byte-level: tiếng Việt có dấu an toàn tuyệt đối, không cần train tokenizer, vocab nhỏ hợp model nhỏ. **Cái giá phải trả lộ ra ngay trong ngày**: `exact_match ≈ tok_acc^độ_dài`, mà đáp án byte thì dài.

**Thay đổi 1 — bbox thành "action token".** bbox `(cx,cy,w,h)` chuẩn hoá [0,1] → affine sang [−1,1], có projection riêng vào/ra. *Ý tưởng:* giữ đúng vai trò action modality của Cosmos §2.1.3, chỉ thu về một domain là screen-space bbox.

**Thay đổi 2 — confidence head học được thay confidence tự khai.** Cosmos 3 để model tự viết độ tin cậy ra JSON. Bản critique gọi đây là điểm yếu nặng nhất: *"mù kiến trúc"* — model không có cách nào biết generator của chính nó sẽ sample ra gì. JWM thay bằng head dự đoán trực tiếp `P(IoU ≥ 0.5)` + Platt calibration, đo bằng ECE. *Kết quả:* ECE 0.414 → 0.040.

**Thay đổi 3 — tái cấu trúc 7 stage.** Từ notebook một khối thành `r1→r2→g1→g2→g3→g4→g5`, mỗi stage một file, nối bằng checkpoint. *Ý tưởng:* bám đúng nghi thức Cosmos §4 — đặc biệt `g1` vừa train-rồi-đóng-băng ConvAE, vừa **copy tower reasoner sang generator**. *Lợi ích phụ ngoài dự tính:* mọi stage shutdown-safe, cứu dự án nhiều lần sau đó.

**Thay đổi 4 — thu nhỏ ngược từ 68.65M về 31M.** Sau 6 vòng loại trừ, kết luận: d512/L10 cần ngân sách QA-polish vượt khả năng một đêm GTX 1650 (68M@3200 bước → tok-acc 0.94; 28M@~2500 → 0.98). *Ý tưởng sửa:* thay vì thêm tham số, **tách thuộc tính trong curriculum** — thêm câu hỏi chỉ-hình và chỉ-màu với đáp án ~10 byte. Đáp án ngắn thì `tok_acc^len` không còn giết exact-match.

---

### Ngày 2 — MoE: giải bài toán "dung lượng lớn, ngân sách nhỏ"

**Vấn đề cần giải:** Ngày 1 chứng minh 68M-class thất bại vì thiếu ngân sách bước, còn 28M-class thì thiếu dung lượng.

**Ý tưởng lấy từ Inkling:** experts **nhỏ** (`hidden = d/2`) và **nhiều** (256), kích hoạt thưa 2.3%. Dung lượng khổng lồ nhưng compute mỗi token gần như không đổi. Đây chính là thứ tách rời hai đại lượng đang mâu thuẫn.

**Toán quyết định** (mỗi layer reasoner, SwiGLU 3·d·h, d=384):

```
1 expert            : 3 × 384 × 192  = 221K
MoE layer (32 + 1)  : 33 × 221K      ≈ 7.30M     ← dung lượng ×6.2 so với dense
Dense layer v3      : 3 × 384 × 1024 ≈ 1.18M
Active mỗi token    : (4 + 1) × 221K ≈ 1.11M     ← chi phí ≈ dense (1.18M)
```

Dung lượng gấp 6.2 lần, chi phí mỗi bước gần như không đổi. Đúng thứ cần.

**Nguyên tắc thu nhỏ — chỉ đổi một biến.** Bản mini chỉ thay **FFN của reasoner tower**. Attention MRoPE hai tower, generator dense, mọi objective — giữ nguyên như v3 đã chứng minh 58.8%. Cố tình **bỏ** những thứ vô nghĩa ở quy mô này: hybrid local/global attention (chuỗi chỉ 200 token), context 1M, GQA (đầu đã nhỏ), audio encoder (JARVIS có ASR riêng).

| Thành phần | Inkling | JWM v4 |
|---|---|---|
| Expert hidden | d/2 = 3072 | **d/2 = 192** ✓ giữ tỷ lệ |
| Routed experts | 256 | 32 |
| Top-k | 6 (2.3%) | 4 (12.5%) — thưa vừa phải cho model nhỏ |
| Shared expert | 2 | 1 |
| Gate | sigmoid + route_scale | sigmoid → top-k → chuẩn hoá tổng = 1 |
| Layer 0 | dense | dense ✓ (ổn định routing sớm) |

**Bug bắt được nhờ thay đổi này:** `init_generator_from_reasoner` copy FFN từ MoE sang dense sẽ vỡ shape. Sửa thành MoE-aware — chỉ copy attention + norm khi loại FFN khác nhau.

---

### Ngày 3 — Mắt to hơn, nhưng đặt sai chỗ

**Ý tưởng:** đọc chữ **không cần mode mới**. READ chỉ là QA-mode với mắt lớn hơn — đúng tinh thần "một model, ba cách xếp token" của Ngày 1.

| | v4 | JWM-Read v1 |
|---|---|---|
| Ảnh vào | 64×64 | **768×768** |
| Patch | 8 | 16 + **gộp 2×2** |
| Token thị giác | 64 | **576** |
| Vision stem | Linear đơn | **MLP phân tầng 2 lớp** (kiểu Inkling) |

Vì sao MLP phân tầng thay ViT attention: Inkling dùng đúng cách này cho vision, và ở 576 token thì attention toàn cục tốn hơn nhiều so với lợi ích.

**Thay đổi phụ — metric.** Đổi từ đếm byte sang **CER trên ký tự unicode**: sai một dấu tiếng Việt tính 1 lỗi, không phải 2–3 lỗi byte.

**Vì sao vẫn thất bại:** lỗi không nằm ở kiến trúc mà ở **curriculum**. Train trên từ tiếng Việt thật → đoán được bằng prior ngôn ngữ → gradient chọn đường dễ. Blind-control Δ ≈ 0.000. Bài học: *kiến trúc đúng không cứu được dữ liệu cho phép gian lận*.

---

### Ngày 4 — Đưa ràng buộc "phải nhìn" vào hàm mục tiêu

`READ_V3.md` chẩn đoán v2 bằng ba câu cụ thể: **nén 2×2 trước khi suy luận không gian**, **chỉ tối ưu cross-entropy QA tự hồi quy**, và **tokenizer byte khiến target tiếng Việt dài làm loãng giám sát thị giác**. Ba thay đổi tương ứng:

**1. Đảo thứ tự nén và suy luận.**
```
v1/v2 : patch → GỘP 2×2 → suy luận không gian
v3    : patch-16 → local window attention trên lưới 64×48 → GỘP 2×2 → 768 token
```
*Ý tưởng:* quan hệ giữa các nét chữ phải được giải quyết **trước** khi nén, vì nén làm mất đúng thông tin phân biệt glyph.

**2. Tokenizer grapheme có byte fallback.** Ký tự tiếng Việt tổ hợp sẵn dùng một token; unicode lạ rơi về UTF-8 byte. *Ý tưởng:* rút ngắn target để `tok_acc^len` bớt khắc nghiệt, mà không mất tính không mất mát.

**3. Ba head mới + ràng buộc thị giác trong loss.**

```
L = L_QA + λ_ctc·L_CTC + λ_box·Σ CE(coord_i) + λ_vis·max(0, m + L_đúng − L_tráo) + L_MoE
```

Số hạng thứ tư là điểm mấu chốt: **phạt trực tiếp nếu loss với ảnh đúng không thấp hơn loss với ảnh tráo**. Ngày 3 chỉ *đo* sự phụ thuộc thị giác sau khi train xong; Ngày 4 **bắt buộc** nó ngay trong hàm mục tiêu.

Coordinate head lấy nguyên lý từ LocateAnything (toạ độ rời rạc 1001 bin, bốn query song song) nhưng **không clone stack Qwen/MoonViT 3B** — chỉ giữ nguyên lý hợp với quy mô.

**4. Loss tách theo nguồn dữ liệu.** Chỉ mẫu synthetic mới nhận CTC transcript và box chính xác; tài liệu thật chỉ nhận QA + contrast ảnh tráo. *Ý tưởng:* không bao giờ gán nhầm đáp án QA thành transcript toàn trang — một dạng nhãn sai tinh vi mà v2 mắc phải.

---

### Ngày 5 — Bốn lần sửa, mỗi lần bịt một đường gian lận

`EYE_PHYSICAL_V2.md` gọi tên ba shortcut của v1: **hồi quy pose tuyệt đối theo từng frame**, **depth chuẩn hoá theo anchor**, và **thiếu đa dạng cảnh động/thật**.

**v1 → v2: bỏ khả năng gian lận, không chỉ phạt nó.**

| v1 | v2 | Ý tưởng |
|---|---|---|
| Hồi quy pose tuyệt đối từ một frame | **Cost volume cặp frame** `C_t(p,δ) = ⟨F_t(p), F_{t−1}(p+δ)⟩/√d` | Muốn biết chuyển động thì buộc phải so hai frame |
| Pose tuyệt đối | **SE(3) tương đối**, tích phân từ `T₀ = I` | Model **không thể** thấy chỉ số frame tuyệt đối hay quỹ đạo đích |
| Depth chuẩn hoá anchor | **Tách shape/scale**: `d_metric = d_relative · exp(Ψ_scale(...))` | Anchor-scale che giấu lỗi scale |
| — | **Dynamic masking** `q_t(p)` loại vùng tự chuyển động khỏi thống kê ego-motion | Người đi qua không được tính là camera dịch chuyển |
| — | **Cycle thuận-nghịch phải bằng identity** | Bịt đường học một chiều rồi bịa chiều kia |

**v2 → v3: intrinsics là bắt buộc.** Tìm ra lỗi thật của v2 — adapter *có* intrinsics nhưng **collator làm rơi trường này**. V3 bắt buộc K theo từng frame, timestamp float64, quy ước projection, rigid flow và provenance của dynamic mask. Thêm differentiable robust BA.

**v3 → v3.1: sửa số học.** BA bắt buộc FP32, Levenberg damping theo Hessian, giới hạn SE(3), rollback đơn điệu, truncated solver-gradient; chọn track theo lưới nội vùng thay vì top-k (top-k dồn vào biên ảnh).

---

### Ngày 6 — v3.2.1: tách hình học khỏi hình chiếu

Nhảy lên **381.93M** (d576, 16 layer MoT, 18 head × 32 chiều) — trong đó geometry chỉ 14.66M, còn 341.38M là MoT backbone.

| Thay đổi | Vì sao | Ý tưởng |
|---|---|---|
| **Safe identity initialization** | Global initializer của JWM **ghi đè** zero-delta pose và track head | Prior identity phải được khôi phục **sau** init toàn cục, và có test bảo vệ bất biến này |
| **Depth-ray factorization** | Một head depth đang phải gánh cả khoảng cách lẫn hình học nhìn | Ghép metric depth với **camera ray đã hiệu chuẩn** + ray residual có chặn — tách khoảng cách cảnh khỏi hình chiếu |
| **Causal scene registers** | Full space-time attention không kham nổi, memory không chặn | 16 register chỉ tổng hợp frame hiện tại và quá khứ; token tại `t` **không được** attend tới `> t` |
| **Bidirectional track cycle** | Track trôi và sinh correspondence giả | Warp track thuận về nguồn, phạt sai số chu trình |
| **Calibrated confidence** | Confidence không khớp thứ evaluator đo | Cho confidence dự đoán **đúng sự kiện đo được** `P(EPE ≤ ngưỡng)` |
| **Scale-preserving warm start** | Muốn tận dụng v4 mà chiều đã khác | Copy không gian con 384×8 của v4 vào model 576×16; **layer chèn thêm là residual identity**, kênh mới giữ initializer, còn mắt mới thì cố ý reset |
| **Validity-normalized interpolation** | Rigid-flow không hợp lệ bị nội suy **trước** khi mask → sentinel tràn vào vùng hợp lệ → EPE 483,729 px | Nội suy phải có trọng số theo tính hợp lệ |
| **Temporal compatibility head + negative window** | Không gate nào ép model quan tâm chiều thời gian | Thêm nhiệm vụ phân biệt cửa sổ thời gian thật/giả |

**Nguyên tắc bao trùm được ghi thẳng vào tài liệu:** *training loss không được phép promote một checkpoint.* Chỉ 7 gate trên dữ liệu thật held-out mới quyết định — depth, pose, BA, wrong-window, reverse-time, wrong-intrinsics, tracking quality.

---

### Ba nguyên lý xuyên suốt

**1. Một model, nhiều cách xếp token** — QA/GROUND/FD/T2I và cả READ đều không phải mode mới, chỉ là cách sắp xếp khác. Giữ được sự đơn giản này suốt 6 ngày.

**2. Bịt đường gian lận trong *kiến trúc*, đừng chỉ phạt trong *loss*.** Eye v2 không cho model thấy chỉ số frame tuyệt đối; Read v3 đưa contrast ảnh tráo vào hàm mục tiêu. Hai lần đều mạnh hơn việc chỉ đo sau khi train xong.

**3. Đổi một biến mỗi lần.** MoE thắng đáng tin vì mọi thứ khác giữ nguyên. Ngược lại, Read v1 đổi cùng lúc độ phân giải, stem, dữ liệu và metric — nên khi thất bại đã mất một ngày mới truy được nguyên nhân.

---

### Các thành phần code chính

| Thành phần | Nội dung |
|---|---|
| `mathx.py` | 3D MRoPE với temporal modulation, rectified flow (xσ, v*, Euler, shift schedule), rot6d↔SO(3), IoU, ECE, PSNR, CER |
| `layers.py` | Dual-tower MoT block, two-way flat attention, AdaLN-zero, reasoner K/V detach + cache |
| `moe.py` | Fine-grained MoE: 32 expert hidden d/2, top-4 + 1 shared, sigmoid→top-k→chuẩn hoá, Switch aux loss |
| `model.py` | 4 mode QA / GROUND / FD / T2I; confidence head P(IoU≥0.5); tower-copy init |
| `stages/` | 7 giai đoạn rời, nối bằng checkpoint, mọi stage shutdown-safe |
| `data_builders/` | 2 nhánh × 5 loại dữ liệu, post-tier theo top-quantile quality score |
| `vision_v3.py` | Vision stem Read v3: local window attention trước, merge 2×2 sau |
| `geometric_eye_v3.py` | CTPG-Eye: ray-conditioned pyramid, sparse recurrent tracks, metric pointmap, differentiable robust BA |

**Tích hợp JARVIS:** `core/world_brain.py` với 3 chế độ `off/shadow/primary` (mặc định shadow — chạy song song Qwen3-VL, chỉ ghi log, không bao giờ được phép phá vision turn).

---

## 5. Kết quả định lượng

### Dòng model chính (từ `metrics_v*.json`)

| Metric | v1 (10.7M) | v3 (31.01M) | **v4 (73.94M MoE)** |
|---|---|---|---|
| QA exact-match (test) | 0.564 | 0.588 | **0.656** |
| — what_held | 0.570 | 0.570 | **0.688** |
| — where | 0.553 | 0.681 | **0.723** |
| — exist | 0.641 | 0.769 | **0.795** |
| Grounding IoU@0.5 (4-step) | 0.184 | 0.268 | **0.360** |
| Grounding mIoU (4-step) | 0.201 | 0.285 | **0.355** |
| ECE hiệu chuẩn (val) | 0.0396 | **0.0248** | 0.0468 ⚠️ |
| FD beats-copy | 27% | 27.1% | **45.8%** |
| FD PSNR vs copy-baseline | — | 20.07 / 21.39 | 20.77 / 21.39 ⚠️ |
| Latency `qa_answer` | — | **182.9 ms** | **1788.9 ms** ⚠️ |
| Latency ground 4-step | 95 ms | 103.9 ms | 251.8 ms |

Ba chỗ có mũi tên cảnh báo là **hồi quy thật** khi lên v4: calibration xấu đi gần gấp đôi, FD vẫn dưới copy-baseline, và latency sinh đáp án chậm **9.8×** (do vòng lặp expert-major batch-1 chưa tối ưu + `generate_answer` chưa có KV cache).

### Trials trong JARVIS

| | v1 | v3 | v4 |
|---|---|---|---|
| Số trial | 66 | 74 | 74 |
| Synthetic ok-rate | 12.5% | 20% | — |
| Mean IoU | 0.205 | 0.247 | **0.373** (+51%) |
| Precision khi assert | **100%** (5/5) | **100%** (8/8) | 88.9% (8/9) ⚠️ |
| Hành vi trên real OOD | conf 0.00–0.02, luôn abstain | conf 0.02–0.06, luôn abstain | — |

Hành vi trên OOD chính là cải tiến so với Cosmos 3 mà `COSMOS3_CRITIQUE.md` đề xuất: **model không bao giờ hallucinate sự tự tin** — nó abstain thay vì đoán bừa.

---

## 6. Cái gì thành công

1. **Kiến trúc dual-tower chạy được ở quy mô micro.** Toàn bộ công thức Cosmos 3 (MRoPE 3D, rectified flow, MoT, AdaLN-zero, tower-copy init) hoạt động đúng ở 10–74M tham số trên card 4GB.
2. **MoE là thắng lợi rõ ràng và đo được.** A/B một biến: +3.5 điểm ở pretrain, +3.6 ở SFT, router khoẻ, 0 expert chết. Giải đúng bài toán "cần dung lượng 68M nhưng chỉ đủ ngân sách bước cho 28M".
3. **Confidence calibration.** ECE từ 0.414 xuống 0.025 (v3) — vượt mục tiêu <0.05 mà chính bản critique đặt ra.
4. **Hành vi abstain trên OOD.** Precision 100% khi dám khẳng định, ở cả v1 lẫn v3.
5. **Hạ tầng shutdown-safe.** Mọi stage checkpoint định kỳ, resume được — sinh ra từ một lần phải tắt máy giữa chừng ở Ngày 1, và cứu dự án nhiều lần sau đó trên Kaggle.

## 7. Cái gì thất bại, và vì sao

| Thất bại | Nguyên nhân gốc | Đã kiểm chứng bằng |
|---|---|---|
| **JWM-Read không đọc** | Curriculum dùng từ tiếng Việt thật → đoán được bằng ngôn ngữ → gradient chọn đường dễ | Blind-control: Δ ≈ 0.000 |
| **JWM-Read v3 blocked** | CTC blank-collapse 99.73% — tín hiệu thị giác có (gap +0.554) nhưng decoder không chuyển thành chữ | Gate free-running CER |
| **Eye pose không học** | Motion distribution thiên về gần-identity; loss không buộc pose vượt identity prior | Identity-prior control: 0.898–1.015× |
| **Eye temporal collapse** | Không có supervision đủ mạnh cho chiều thời gian | Reverse-time control: 1.004× |
| **Eye v3 nổ số học** | Point top-k dồn vào biên + đạo hàm xuyên linear solve kém điều kiện | grad NaN @ step 175 |
| **Eye track EPE 483,729 px** | Invalid rigid-flow bị nội suy **trước** khi mask → sentinel tràn vào vùng hợp lệ | Phân tích thủ công |
| **FD dưới copy-baseline** | Chưa giải quyết suốt 6 ngày | 20.77 vs 21.39 |
| **Latency v4 chậm 9.8×** | Vòng lặp expert-major batch-1 + thiếu KV cache | 1788.9 ms |

---

## 8. Bài học phương pháp — phần giá trị nhất

**1. Chỉ số huấn luyện nói dối, control nhân quả thì không.**
Ba lần liên tiếp — Read v1, Eye v1, Eye v3.2.1 — chỉ số nội bộ báo tiến bộ trong khi model không học được gì thật. Chỉ có blind-image control, identity-prior control, reverse-time control và wrong-intrinsics control mới phát hiện ra. Mỗi lần đều rẻ hơn nhiều so với chi phí tin nhầm.

**2. Đo bằng thứ mình sẽ dùng lúc suy luận.**
`tok_acc` teacher-forced lừa trọn một run 11.5K bước. Gate phải đặt trên **free-running CER**, tức đúng chế độ model chạy thật.

**3. Exact-match tụt theo hàm mũ của độ dài đáp án.**
`exact ≈ tok_acc^len`. Muốn 27 byte đúng hết thì tok_acc phải ≥0.98, không phải 0.95. Nhiều "model dốt" thật ra chỉ là đáp án quá dài.

**4. Curriculum quyết định model có nhìn hay không.**
Nếu đáp án đoán được bằng prior ngôn ngữ, model sẽ đoán chứ không nhìn. Muốn buộc nó nhìn thì đáp án phải **không đoán nổi nếu không nhìn**.

**5. Thêm bước, thêm dữ liệu, hạ LR — cả ba đều vô ích nếu chẩn đoán sai.**
Saga Ngày 1 mất 6 vòng để kết luận "không có bug ở đâu cả". Chẩn đoán 4 số (train/val gap, train-exact, tok-acc plateau, dump ảnh) rẻ hơn nhiều so với việc thử từng cách sửa.

**6. Dừng đúng lúc là một tính năng.**
Adaptive controller dừng ở step 1.000 vì OOD score suy giảm trong khi train loss vẫn tốt. Không có nó, ta đã đốt thêm nhiều giờ để có checkpoint tệ hơn.

**7. Một biến mỗi lần.**
Thắng lợi MoE chỉ đáng tin vì mọi thứ khác được giữ nguyên: cùng data, LR, batch, seed, cùng recipe với baseline v3.

---

## 9. Tài sản để lại

**Checkpoint** (`jwm/checkpoints/`): `jwm_v1.pt` 43MB · `jwm_v3.pt` 124MB · `jwm_v4.pt` 296MB · `exp_moe_reasoner.pt` + toàn bộ stage checkpoint g1–g5. Ngoài ra `jwm_read_v1.pt` 671MB và các bản `*_blocked.pt` của dòng Eye ở thư mục gốc.

**Benchmark artifact** (`jwm/benchmarks/`): eye_physical_v1 full / smoke / TUM controlled, ở cả dạng JSON lẫn Markdown.

**Tài liệu thiết kế** (14 file): `DESIGN.md`, `COSMOS3_CRITIQUE.md` (33KB — 7 điểm yếu Cosmos 3 đã kiểm chứng), `INKLING_MINI.md`, `EYE_V3_RESEARCH.md` (22KB), `READ_V3.md`, `GEOMETRY_DATA.md`.

**Test:** 28 file test, đỉnh điểm 142 test pass toàn workspace ở Ngày 6.

---

## 10. Việc còn dở

Xếp theo giá trị trên công sức:

1. **KV cache cho `generate_answer`** — ước tính 10×, có lợi cho cả dense lẫn MoE. Đây là lý do latency v4 là 1788.9 ms.
2. **Batched expert GEMM** thay vòng lặp expert-major 32 vòng/layer ở batch nhỏ.
3. **Recalibrate ECE của v4** — Platt per-generation + xem lại threshold; hiện 0.0468 val so với 0.0248 của v3.
4. **FD vượt copy-baseline** — treo suốt 6 ngày. Hướng đề xuất: motion lớn hơn, nhiều frame hơn.
5. **JWM-Read v4** với curriculum ký tự ngẫu nhiên (không đoán được bằng ngôn ngữ) + ROI/line CTC 1D thay CTC toàn trang 2D.
6. **Eye: temporal supervision** — `temporal_compatibility` đang ở 0.499, tức đoán ngẫu nhiên. Không sửa được cái này thì không có gate nào về thời gian pass được.
7. **Cân nhắc promote `WORLD_BRAIN_MODE` shadow → primary** cho riêng synthetic-domain scene, nơi v4 đã đạt QA 65.6%.

**Ghi chú tính toàn vẹn dữ liệu:** file `metrics_v1.json` ghi test `qa_acc = 0.536` và `ece = 0.1556`, trong khi nhật ký và khối `reference` trong `metrics_v3/v4.json` ghi 0.564 và 0.0396. Chênh lệch này là do `metrics_v1.json` là ảnh chụp **trước** Stage 2.5 và trước Platt calibration. Khi trích dẫn v1 nên dùng 0.564 / 0.0396 và nêu rõ đó là số sau hiệu chuẩn.

---

## 11. Kết luận

Sáu ngày cho ra **một world model 74M chạy được, tự hiệu chuẩn, biết từ chối trả lời khi không chắc** — và ba hướng mở rộng đều bị chặn lại bởi chính hệ thống kiểm định của dự án.

Việc bị chặn không phải thất bại của quy trình mà là quy trình đang làm đúng việc. Ba lần model trông như đang tiến bộ mà thực ra không học được gì; cả ba lần đều bị bắt trước khi kịp gắn vào JARVIS. Đối chiếu với các báo cáo benchmark ở `research/report/`, nơi Qwen2-VL-2B và Cosmos 3 Nano **đều thua baseline tầm thường** ở đúng nhiệm vụ quan trọng nhất, thì tiêu chuẩn kiểm định dựng lên trong 6 ngày này là tài sản dùng lại được — và đang được dùng lại cho PIADE.
