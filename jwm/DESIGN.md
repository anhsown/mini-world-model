# JWM — Jarvis World Model

**Bộ não world-model cho JARVIS, kiến trúc Cosmos-3-style thu nhỏ, train được trên GTX 1650 4GB.**

> Tài liệu này là "đặc tả chính thức": mọi công thức ở đây phải khớp 1-1 với code trong `jwm/`
> và phải có test tương ứng trong `tests/test_jwm_math.py`.

---

## 0. Mục tiêu & ràng buộc

| | |
|---|---|
| Vai trò | Bộ não thị giác-ngôn ngữ của JARVIS: nhận frame camera + câu hỏi (vi/en) → trả lời + vùng ảnh (bbox) + confidence đã hiệu chuẩn + dự đoán tiến hóa cảnh (forward dynamics) |
| Phần cứng | GTX 1650 4GB (Turing, fp16 OK, bf16 KHÔNG), i5 4-core |
| Quy mô | ~15M tham số transformer (dual-tower), ảnh 64×64, chuỗi ≤ 256 token |
| Nguyên tắc | Giữ ĐÚNG các quyết định kiến trúc của Cosmos 3 (dual-tower MoT, token arrangement, MRoPE + temporal modulation, rectified flow, action token, confidence) — chỉ thu nhỏ chiều, không đổi bản chất |

**Ba chế độ (mode) — cùng một model, chỉ khác cách xếp token:**

1. **QA** (VLM mode): `[AR: BOS, IMG×64, BOQ, q_bytes, BOA] → sinh a_bytes` (autoregressive)
2. **GROUND** (inverse-dynamics-style): `[AR như trên][DM: b̃box]` → denoise bbox 4-chiều = "vùng ảnh được hỏi"
3. **FD** (forward dynamics): `[AR: BOS, IMG×64, BOQ, motion_text][DM: z_t (clean 16 tok), z̃_{t+1} (noisy 16 tok)]` → denoise latent frame kế tiếp

---

## 1. Encoders

### 1.1 Text — byte-level tokenizer
UTF-8 bytes (0..255) + special: `PAD=256, BOS=257, EOS=258, BOQ=259, BOA=260, BOG=261` → vocab **262**.
Lý do: tiếng Việt có dấu an toàn tuyệt đối, không cần train tokenizer, vocab nhỏ hợp model nhỏ.

### 1.2 Vision hiểu (ViT-lite, thuộc chuỗi AR)
Frame RGB 64×64×3, patch 8×8 → 64 token. Patch embed = Linear(8·8·3=192 → d).
Cộng **modality embedding học được** `e_img ∈ R^d` (Cosmos §2.1: mỗi modality phi ngôn ngữ một vector riêng).

### 1.3 Vision sinh (conv-VAE, thuộc chuỗi DM) — ĐÓNG BĂNG sau pretrain
Encoder: 64×64×3 → 8×8×`z_ch`(=8) (nén không gian 8×). Decoder đối xứng.
Latent 8×8×8 → patch-merge 2×2 → **16 token**, mỗi token 32-dim → Linear(32 → d).
Train riêng bằng MSE tái tạo (~2 phút), sau đó freeze — đúng vai trò Wan-VAE frozen trong Cosmos.

### 1.4 Action token — bbox như một "hành động"
bbox = `(cx, cy, w, h)` chuẩn hóa [0,1] → affine sang **[-1,1]**: `x = 2·b − 1`.
Input projection `W_in ∈ R^{d×4}`, output projection `W_out ∈ R^{4×d}` (tương ứng domain-aware
projections của Cosmos §2.1.3; ở đây 1 domain "screen-space bbox").
Cộng modality embedding `e_act`. Một token duy nhất.
*(Module `rot6d` cho embodiment SE(3) tương lai vẫn được viết + test đầy đủ trong `mathx.py`.)*

---

## 2. Token arrangement (đúng 3 quy tắc Cosmos §2.2.1)

```
[  AR subsequence  ][        DM subsequence        ]
   BOS IMG…IMG BOQ q… (BOA a…)   [clean cond tokens][noisy tokens]
   └── tower Reasoner ──┘        └──── tower Generator ────┘
```
1. AR trước DM. 2. Trong DM: clean trước noisy. 3. Thứ tự modality: vision → action.

## 3. Dual-tower MoT block

Mỗi layer có **hai bộ tham số đầy đủ** (reasoner / generator): RMSNorm, QKV, O, SwiGLU-FFN.

**RMSNorm:** `RMS(x) = x / sqrt(mean(x²) + ε) · γ`, ε = 1e-6.

**SwiGLU:** `FFN(x) = W_down( SiLU(W_gate x) ⊙ W_up x )`, hidden = 672 (≈ 8/3·d, bội 32).

### 3.1 Dual-stream joint attention (Cosmos eq. 7–8)

```
O_AR = Attn_causal(Q_AR, K_AR, V_AR)                        # reasoner: chỉ nhìn AR, causal
O_DM = Attn_full (Q_DM, [K_AR; K_DM], [V_AR; V_DM])          # generator: nhìn tất cả, 2 chiều
```
- Bất biến cứng: **token AR không bao giờ phụ thuộc token DM** (test đạo hàm/perturbation bắt buộc).
- Hiện thực bằng **2 lần gọi SDPA** mỗi layer (two-way flat attention, Cosmos §5.2.2).

### 3.2 Điều kiện hóa σ cho generator — AdaLN-zero (DiT)
Cosmos ghi `v_θ(x_σ, σ, c)` nhưng không nêu cách đưa σ vào; ta dùng chuẩn tốt nhất hiện hành:
`emb = MLP(sinusoidal(σ))` → mỗi layer generator: `(shift₁, scale₁, gate₁, shift₂, scale₂, gate₂) = Linear(emb)`
```
h = h + gate₁ ⊙ Attn( RMS(h)·(1+scale₁) + shift₁ )
h = h + gate₂ ⊙ FFN ( RMS(h)·(1+scale₂) + shift₂ )
```
`gate` khởi tạo **0** (zero-init) → lúc bắt đầu train, generator là hàm đồng nhất, ổn định.
Token **clean** trong DM nhận σ=0. Mỗi modality trong DM có σ riêng (per-modality time sampling).

## 4. MRoPE — 3D RoPE + absolute temporal modulation

Head dim 32 → nửa dim 16 tần số, chia section **(t=8, h=4, w=4)**.
Tần số: `θ_i = base^(−i/section)` với base=10000 (từng section đánh chỉ số riêng).
Góc của token có tọa độ (t,h,w): `angles = [t·θ^(t)_0..7, h·θ^(h)_0..3, w·θ^(w)_0..3]` → cos/sin, áp dụng kiểu rotate-half.

**Cấp tọa độ (Cosmos §2.4.1):**

| Token | t | h | w |
|---|---|---|---|
| text (AR) | p (đơn điệu) | p | p |
| IMG patch (AR, frame f, hàng r, cột c) | t_text_cuối + 1 + f | r | c |
| DM latent frame f, hàng r, cột c | G + f·δt | r | c |
| DM bbox token | G | 0 | 0 |

- **G = gap AR→DM.** Cosmos dùng hằng 15000; ta đặt mặc định **G=64** (tỷ lệ với thang chuỗi 256 của ta so với 74K của Cosmos) và **ablate G ∈ {0, 64} + boundary embedding học được** (cải tiến #1).
- **Temporal modulation (Cosmos eq. 9):** `δt = TPS_base / TPS`. Ta đặt TPS_base = 5 (camera JARVIS ~5fps hiển thị); frame FD lấy ở 2fps ⇒ δt = 2.5. Đây là phép kiểm chứng cơ chế, đúng công thức gốc.
- Tọa độ t là **float** (do modulation) — RoPE nhận float tự nhiên.

**Thuộc tính phải test:** với text thuần, `⟨RoPE(q,p), RoPE(k,p+Δ)⟩` chỉ phụ thuộc Δ (bất biến vị trí tương đối, sai số < 1e-4).

## 5. Objectives

### 5.1 AR (reasoner) — CE chuẩn hóa căn bậc hai (Cosmos §4.1.1)
Loss chỉ tính trên các vị trí **sau BOA** (answer bytes + EOS):
`L_AR = Σ_sample ( Σ_tok CE ) / sqrt(N_tok_sample)`, trung bình theo batch.

### 5.2 DM (generator) — rectified flow (Cosmos §4.2)
```
x_σ = σ·ε + (1−σ)·x₀,   ε ~ N(0, I)
v*  = ε − x₀
L_DM = mean‖v̂_θ(x_σ, σ, c) − v*‖²        (mask: chỉ token noisy; token clean bị loại khỏi loss)
```
- **Per-modality σ**: bbox và latent rút σ độc lập trong cùng sample.
- **Sampling σ**: logit-normal `σ = sigmoid(z), z~N(0,1)` cho bbox & latent (đúng lựa chọn
  của Cosmos cho action/image; ta không có video dài nên không cần mode-sampling — vẫn cài
  `mode_sample` của SD3 để ablate: `t = 1 − u − s·(cos²(πu/2) − 1 + u)`).
- **Shift** khi *inference*: `σ(t̄) = s·t̄ / (1 + (s−1)·t̄)`, t̄ = 1−t. Kiểm chứng: σ(t=0)=1, σ(t=1)=0, đơn điệu giảm, s>1 dồn về nhiễu cao.
- **Loss scale**: `L = L_AR + λ_bbox·L_bbox + λ_lat·L_lat` với **λ_bbox = 10** (như action ×10 của Cosmos — MSE 4 chiều quá nhỏ so với latent 512 chiều), λ_lat = 1.
- **Text dropout 10%** (thay câu hỏi bằng chuỗi rỗng) → hỗ trợ CFG lúc inference.

### 5.3 Confidence head — CẢI TIẾN so với Cosmos (confidence tự khai trong JSON)
Tại σ bất kỳ trong training, ước lượng one-step: `x̂₀ = x_σ − σ·v̂`.
Nhãn: `y = 1[IoU(x̂₀→bbox, bbox_gt) ≥ 0.5]`.
Head: `p = sigmoid(MLP(h_bbox_token))` (h đã "biết" σ qua AdaLN), loss BCE, trọng số 0.05.
Inference: sau khi sample xong (σ≈0), đọc `p` → confidence **đã hiệu chuẩn**, đo bằng **ECE**.
Abstain khi `p < τ` (τ chọn trên validation).

## 6. Sampling

**Euler (rectified flow):** lưới t̄ đều → σ_k qua shift; bước:
`x_{σ_{k+1}} = x_{σ_k} − (σ_k − σ_{k+1})·v̂(x_{σ_k}, σ_k, c)`
Test giải tích: với v* thật (hằng theo đường), 1 bước từ σ về 0 khôi phục đúng x₀: `x₀ = x_σ − σ·v*`.

**CFG:** `v̂ = v_uncond + g·(v_cond − v_uncond)`, g mặc định 1.0 (tắt) / 2.0 (bật), 2 forward/bước.
**Số bước:** 50 (chuẩn), **4 (chế độ policy-style)** — báo cáo cả hai (đối chiếu latency/chất lượng như Cosmos policy).
**Reasoner caching:** K_AR, V_AR tính **một lần**, tái dùng cho mọi bước denoise (đúng Cosmos §5.3.1).

## 7. Reflection pass (cải tiến #3 — inference-time)
Sau khi GROUND ra bbox: crop vùng bbox (phóng to), đưa lại như IMG mới vào **chế độ QA**:
"vùng này có phải …?" → nếu reasoner phủ nhận → hạ confidence / abstain.
Đây là vòng phản hồi DM→AR mà kiến trúc gốc cấm ở mức attention — thực hiện ở mức *pipeline* nên không phá causal integrity.

## 8. Khác biệt có chủ đích so với Cosmos 3 (khai báo trung thực)

| Cosmos 3 | JWM | Lý do |
|---|---|---|
| Audio modality | Không (ASR/TTS đã có ở tầng Jarvis) | Não không cần sinh audio |
| Video N frame | FD 1-bước (z_t → z_{t+1}) | 4GB VRAM; đủ để kiểm chứng cơ chế world-model |
| Flat packing 74K | Bucket theo độ dài + pad, budget 4096 tok/batch | Batch nhỏ, packing phẳng không đáng độ phức tạp |
| Tokenizer BPE 150K | Byte-level 262 | Model 15M không gánh nổi embedding lớn |
| Khởi tạo từ Qwen3-VL | Train từ đầu | Không có VLM 15M tương thích; điểm yếu đã biết |
| Confidence tự khai JSON | Head hiệu chuẩn P(IoU≥0.5) + ECE | Cải tiến có đo lường |
| Gap 15000 cố định | G=64 + ablation boundary-embedding | Cải tiến có đo lường |

## 9. Ngân sách tham số (d=256, L=6, heads=8)

| Thành phần | Ước lượng |
|---|---|
| Embedding text 262×256 + head (tied) | 0.07M |
| Patch embed + modality emb | 0.05M |
| Reasoner tower 6 layer (attn 4d² + FFN 3·d·672) | ~4.9M |
| Generator tower 6 layer + AdaLN (6d·d_emb) | ~6.1M |
| VAE conv (riêng, frozen) | ~0.4M |
| bbox proj + conf head + latent proj | ~0.1M |
| **Tổng transformer** | **~11.3M** |

fp16 + Adam fp32 state ≈ 11.3M×(2+4+4+4) ≈ 160MB — thừa chỗ trong 4GB kể cả activation.

## 10. Kế hoạch kiểm định (bắt buộc pass trước khi train)

1. `test_rope_relative_invariance` — RoPE/MRoPE bất biến vị trí tương đối
2. `test_mrope_allocation` — bảng tọa độ §4 đúng từng loại token; temporal modulation δt đúng
3. `test_rectified_flow_identities` — x_σ, v*, one-step recovery, Euler nhiều bước hội tụ x₀ với v* thật
4. `test_shift_schedule` — biên σ(0)=1, σ(1)=0, đơn điệu, s>1 đẩy nhiễu cao
5. `test_sigma_samplers` — logit-normal & mode-sample trong [0,1], phân bố đúng dạng
6. `test_rot6d_roundtrip` — R→6D→R, trực giao, det=+1; SVD projection
7. `test_attention_isolation` — perturb DM không đổi output AR; causal trong AR; DM đổi khi AR đổi
8. `test_adaln_zero_init` — tại init, generator block ≈ identity
9. `test_iou_ece` — IoU case chuẩn; ECE với phân bố biết trước
10. `test_end_to_end_shapes` — forward 3 mode chạy, loss hữu hạn, backward không NaN
