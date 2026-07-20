# JWM — NGÀY 5 · Eye Physical v1: pass nội bộ, fail OOD

## Kết quả huấn luyện

- Warm-start từ `jwm_v4.pt`; 212 tensor ngữ nghĩa được nạp, vision stem và Geometric Context Memory được train mới.
- `g0_exact_geometry`: 1.800 steps, `Depth AbsRel=0.0540`, `ATE=0.0169`, **pass**.
- `g1_real_rgbd_adapt`: 3.000 steps, `Depth AbsRel=0.3333`, `ATE=0.0702`, **pass** gate nội bộ.
- Checkpoint cuối có 292 tensor, không NaN/Inf; SHA-256 `75E260BD43E46E75AA0649703B8C10EB3DB4836337039CECE1B67EA9FD5C71C2`.

## Benchmark độc lập

Tên benchmark: **JWM-Eye-Physical Independent Geometry + Blind Controls**.

| Tập / control | Depth AbsRel ↓ | ATE ↓ | Abs. rotation ↓ | RPE trans. ↓ | RPE rotation ↓ |
|---|---:|---:|---:|---:|---:|
| Procedural test, ảnh đúng | **0.03894** | **0.01510** | 1.13669° | **0.00442** | 0.56676° |
| Procedural, sai cảnh | 0.18626 | 0.01598 | 1.13695° | 0.00485 | 0.56679° |
| Procedural, constant/identity prior | 0.77238 | 0.03188 | **1.11483°** | 0.01318 | **0.55742°** |
| TUM fr3/walking_xyz OOD, ảnh đúng | 0.40807 | 0.03843 | 2.82815° | 0.00982 | 1.15841° |
| TUM OOD, ảnh đen | 0.32924 | 0.03859 | 2.83377° | 0.00996 | 1.15776° |
| TUM OOD, sai cửa sổ | 0.41607 | 0.03909 | 2.82986° | 0.01038 | 1.15769° |
| TUM OOD, constant/identity prior | **0.29444** | **0.03699** | **2.80012°** | **0.00921** | **1.15544°** |

### Diễn giải

- Trên procedural held-out, depth thật sự dùng ảnh: sai cảnh làm AbsRel xấu `4.78×`, ảnh đen làm xấu `12.13×`.
- Pose chưa dùng đủ bằng chứng thị giác: sai cảnh chỉ làm ATE xấu `1.058×`; rotation gần như không đổi và thua identity prior.
- Trên TUM dynamic OOD, model thua constant/identity prior ở cả sáu metric; ảnh đen còn cho depth tốt hơn ảnh thật.
- Real adaptation cải thiện procedural so với stage 0 (`Depth AbsRel` giảm khoảng 28%, `ATE` giảm khoảng 22%), nhưng không tạo được khả năng tổng quát hóa ra cảnh người thật chuyển động.
- Throughput end-to-end trên GTX 1650 khi JARVIS cùng chạy là `20.44 FPS`, chưa đạt mục tiêu 30 FPS.

## Quyết định

**BLOCKED — không promote và chưa gắn checkpoint này vào JARVIS.** Checkpoint pass gate huấn luyện nhưng fail external OOD và fail causal vision-dependence gate cho pose.

Các nguyên nhân cần kiểm chứng ở vòng sau: procedural shortcut; tỷ lệ và độ đa dạng real RGB-D quá thấp; motion distribution thiên về gần-identity; loss/gate chưa buộc pose vượt identity prior; anchor-scale depth metric che giấu một phần lỗi scale. Không tăng steps trên cùng mixture trước khi các giả thuyết này được ablate.

## Artifact

- `jwm/benchmarks/eye_physical_v1_full.json`
- `jwm/benchmarks/eye_physical_v1_full.md`
- `jwm/benchmarks/eye_physical_v1_tum_walking_xyz_controlled.json`
- `scripts/bench_eye_physical.py`

## Eye Physical v2 — corrective build (đang chờ train)

- Thay absolute-pose shortcut bằng local pairwise cost volume, relative SE(3)
  prediction và tích phân quỹ đạo từ `T0 = I`.
- Tách relative-depth/metric-scale, thêm masked valid-depth pooling, dynamic
  masking, forward/reverse cycle và wrong-image counterfactual ranking.
- Dữ liệu mới gồm procedural exact, TartanAir, TUM RGB-D và Bonn Dynamic; mọi
  nguồn phải pass kiểm định metric scale, SO(3), motion và scene-split trước
  khi được đưa vào train.
- Dựng ablation A–D cùng initialization/cùng sample order; arm thắng được nối
  liên tục vào E0 → E1 → E2. Promotion yêu cầu vượt fixed prior và pass sáu
  causal controls, nếu fail chỉ xuất checkpoint `blocked`.
- Full scale: `86.77M` tổng / `12.91M` trainable; AMP smoke 8 frame trên GTX
  1650 pass, peak `1606.3 MiB`; toàn bộ `107` tests pass.
- Notebook Kaggle: `jwm/kaggle/jwm_eye_physical_v2_t4x2_day05.ipynb`.

## Eye Physical v2 — kết quả pilot T4×2

Admission dữ liệu đã **PASS** cho toàn bộ train/validation/test: TUM, Bonn và
TartanAir đều hợp lệ; scene split không rò rỉ. Pilot chạy bốn arm A–D, mỗi arm
800 optimizer steps với cùng initialization và sample order.

| Arm | Depth AbsRel ↓ | Depth δ1 ↑ | ATE metric ↓ | Gate đạt / 6 |
|---|---:|---:|---:|---:|
| A — pairwise base | 0.3068 | 0.4871 | 0.1730 | 1 |
| **B — + SE(3) cycle** | **0.3051** | 0.4949 | 0.0748 | **1** |
| C — + dynamic mask | 0.3063 | 0.5055 | 0.0773 | 1 |
| D — + counterfactual | 0.3086 | **0.5163** | **0.0738** | 1 |

Arm B thắng theo composite score, nhưng không arm nào vượt đủ causal gate nên
full E0→E1→E2 bị chặn. Trên held-out real OOD TUM+Bonn của arm B:

| Causal check | Tỷ lệ đo được | Yêu cầu | Kết quả |
|---|---:|---:|---|
| Fixed-depth prior / model AbsRel | 1.126× | ≥1.20× | FAIL |
| Identity prior / model ATE | 0.898× | ≥1.20× | FAIL |
| Black / normal depth error | 1.459× | ≥1.25× | PASS |
| Wrong-window / normal depth error | 1.153× | ≥1.25× | FAIL |
| Wrong-window / normal ATE | 1.155× | ≥1.25× | FAIL |
| Reverse-time / normal motion RPE | 1.067× | ≥1.10× | FAIL |

Depth đã dùng bằng chứng ảnh ở mức rõ ràng (ảnh đen làm lỗi tăng 46%), nhưng
chưa vượt fixed prior đủ biên và chưa phụ thuộc đúng cặp frame. Pose vẫn thua
identity prior; đảo thời gian chỉ làm RPE xấu 6.7%, cho thấy temporal direction
và ego-motion chưa được học đủ. Checkpoint pilot có trạng thái
`blocked_by_ood_gate`, 86,871,076 parameters, không NaN/Inf, SHA-256
`423230C5CFA61F09B937F5507BFC5261A3B3664BC6DC71D7D0544C087D1AAFAE`.

**Quyết định: BLOCKED.** Không gắn checkpoint pilot vào JARVIS và không chạy
full curriculum từ nó. Vòng tiếp theo phải sửa objective/sampling cho temporal
direction, hard camera motion và wrong-window ranking trước khi train lại.

## Eye Physical v3 — hoàn tất build tới gate 6

- Tìm đúng lỗi camera của v2: adapter có intrinsics nhưng collator làm rơi trường
  này. V3 bắt buộc K theo từng frame, timestamp float64, projection convention,
  rigid flow và provenance của dynamic mask.
- Dựng CTPG-Eye: ray-conditioned pyramid, sparse recurrent tracks, dynamic/static
  split, metric pointmap, SE(3), differentiable robust BA và bounded memory.
- 5/5 probe cơ chế pass; full regression 123/123 test pass. Full-graph local
  256px × 6 frame dùng 0.825 GiB peak allocated trong smoke ngắn.
- Dựng trainer adaptive: OOD slope tự quyết định continue/reduce-LR/advance/stop;
  không promote nếu depth/pose chưa thắng prior hoặc causal controls chưa pass.
- Notebook T4×2 đã tạo tại `jwm/kaggle/jwm_eye_physical_v3_t4x2_day05.ipynb`.
  Full training vẫn chờ data admission và exact 100-step profile trên Kaggle.

## Eye Physical v3 pilot — numerical gate chặn ở step 200

- Data admission pass nhưng g0 phát nổ: `grad=NaN` ở step 175, gradient norm
  `67549` ở step 200 và `track_epe=Infinity`.
- Adaptive controller phát `stop_unstable`; chỉ xuất
  `jwm_eye_v3_blocked.pt`, không resume và không deploy.
- Root cause được định vị ở point top-k dồn vào biên, fallback track rỗng không
  finite-safe và đạo hàm xuyên linear solve của unrolled BA kém điều kiện.

## Eye Physical v3.1 Stability — corrective package

- Chọn track point theo lưới nội vùng, clamp track trong ảnh và báo riêng valid-track ratio.
- BA bắt buộc FP32, Levenberg damping theo Hessian, giới hạn SE(3), rollback đơn điệu
  và truncated solver-gradient; loss/evaluator không còn tạo NaN/Infinity khi track rỗng.
- G0 LR giảm `3e-4 → 1e-4`; thêm governor đồng bộ DDP để skip non-finite step,
  hạ loss scale/LR và block sau ba lỗi liên tiếp.
- Profile mới chạy 250 bước trên đúng Procedural+TartanAir thay vì procedural-only;
  log đủ depth/track/valid/BA/gradient.
- Local exact-seed canary 100 bước pass: valid-track khoảng `0.93–0.95`, không có
  NaN/Inf; toàn bộ `127/127` tests pass.
- Notebook mới: `jwm/kaggle/jwm_eye_physical_v31_t4x2_day05.ipynb`.

## Eye v3.1 — DDP adaptive-LR hotfix

- Run Kaggle dừng ở stage 1 step 2400 vì rank 1 gọi
  `acknowledge_lr_decay()` dù chỉ rank 0 có observation. Hotfix giới hạn mutation
  controller ở rank 0, nhưng vẫn áp dụng cùng LR factor trên cả hai GPU; run có
  thể resume từ `resume.pt` mà không train lại từ đầu.

## Benchmark Eye v3.1 sau full training

- Checkpoint `jwm_eye_v31_blocked.pt` hợp lệ: 79.75M tham số, không có tensor
  NaN/Inf; data admission pass, không scene leakage và mechanism probe pass 5/5.
- Held-out causal/OOD trên TartanAir, TUM và Bonn chỉ pass **1/7 gates**. Depth
  học được (`AbsRel=0.466`, gain so với prior `1.798x`), nhưng pose chỉ tốt hơn
  identity `1.067x`, BA giảm residual `12.49%` và các control temporal/K gần 1x.
- Independent procedural recheck pass **2/7 gates**: Depth AbsRel `0.2057`, ATE
  `0.1604 m`, track EPE `0.2807 px`, nhưng BA `0%` và confidence chỉ `2.94e-5`.
- Root cause chính: invalid rigid-flow bị nội suy trước khi mask, làm real-data
  track EPE tăng phi vật lý tới `483729 px`; confidence collapse khiến BA mất
  correspondence, còn reverse/wrong-window/wrong-K chưa được train trước khi
  controller chặn stage 1. Không tăng steps cho v3.1; chuyển sang thiết kế v3.2.
