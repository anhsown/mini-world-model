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
