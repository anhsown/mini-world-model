# JWM — NGÀY 6 · Eye v3.2.1 Robust Causal Geometry

## Điểm xuất phát

- Benchmark gần nhất chỉ vượt **3/7 causal/OOD gates**.
- Các điểm chết còn lại nằm ở pose, quan hệ nhân quả theo thời gian, độ bền tracking và khả năng tổng quát hóa trên cảnh động.
- Phân tích sâu phát hiện một lỗi dữ liệu nghiêm trọng: rigid-flow không hợp lệ được nội suy trước khi mask, khiến sentinel tràn vào vùng hợp lệ, tạo EPE hàng trăm nghìn pixel và gradient spike.

## Research và quyết định thiết kế

- Đối chiếu các hướng hiện đại về metric geometry, point tracking, optical flow, differentiable BA và dynamic-scene reconstruction.
- Không tăng steps trên kiến trúc cũ vì lỗi chính thuộc về target construction, uncertainty, temporal supervision và evaluator contract.
- Chuyển sang **JWM-Eye v3.2.1**, tập trung vào causal geometry có thể kiểm chứng thay vì chỉ giảm training loss.

## Những thay đổi đã hoàn thành

- Sửa resize flow bằng validity-normalized interpolation để sentinel không còn làm nhiễm nhãn hợp lệ.
- Tracking sử dụng robust heteroscedastic Laplace loss, dự đoán confidence, visibility và uncertainty scale riêng.
- Căn chỉnh confidence target với đúng ngưỡng evaluator `P(EPE ≤ 3 px)`; báo cáo riêng calibration tại 1 px và 3 px.
- Thêm temporal compatibility head và negative window phá thứ tự thời gian thật sự.
- Cân bằng dynamic focal loss và buộc dataset phải có dynamic-positive supervision.
- BA chỉ sử dụng correspondence có confidence, visibility và static probability phù hợp; gate BA yêu cầu cải thiện cả residual lẫn ATE.
- Pose gate chỉ đánh giá trên các window có chuyển động đủ lớn để identity prior không tạo kết quả giả.
- Định nghĩa lại bảy promotion gates cho depth, pose, BA, temporal pairing, tracking quality, reverse time và wrong intrinsics.

## Kiểm định phần mềm

- Full workspace: **142 tests passed**.
- Public export repository: **121 tests passed**, 4 warning không chặn.
- Exact-graph CPU smoke test chạy hữu hạn, không xuất hiện NaN/Inf.
- Source đã được push bằng tài khoản `anhsown` tại commit `6c6ccfe`.

## Training v3.2.1

- Notebook: `jwm/kaggle/jwm_eye_v321_robust_causal_t4x2.ipynb`.
- Nền tảng: Kaggle T4×2.
- Training bị adaptive controller dừng ở `g0_calibrated_tracks`, step **1.000**, với quyết định `stop_overfit`: training loss tiếp tục tốt lên trong khi held-out OOD score suy giảm.
- Best controller score xuất hiện ở step **400** (`0.20374`); score tại step 1.000 còn `-0.03962`.
- Training không đi tiếp sang các stage temporal/dynamic sau vì promotion contract đã chặn đúng.

## Kết quả training và benchmark

| Hạng mục | Kết quả |
|---|---|
| Checkpoint cuối | `jwm_eye_v321_blocked.pt` |
| Model | 381.927M parameters, 639 tensors, 1.527 GB |
| Steps thực chạy | 1.000, dừng tại stage g0 |
| Checkpoint integrity | Không có tensor NaN/Inf |
| Final real Depth AbsRel ↓ | 0.48775 |
| Final real Depth δ1 ↑ | 0.26889 |
| Final real ATE ↓ | 0.03282 m |
| Track EPE / P90 ↓ | 0.57980 / 1.06852 px |
| Track PCK@3 ↑ | 0.99469 |
| Track ECE@3 ↓ | 0.05842 |
| Dynamic F1 ↑ | 0.02993 |
| Causal/OOD gates | **2/7** |
| Quyết định | **BLOCKED — không deploy vào JARVIS** |

### Chi tiết bảy gates

| Gate | Đo được | Ngưỡng | Kết quả |
|---|---:|---:|---|
| Depth thắng fixed prior | 0.838× | ≥1.20× | FAIL |
| Pose thắng moving identity | 1.015× | ≥1.20× | FAIL |
| BA cải thiện residual và pose | residual 0.99992; pose gain 22.63× | đạt cả hai điều kiện | **PASS** |
| Phát hiện wrong temporal window | gap ≈ 0.000 | ≥0.15 | FAIL |
| Tracking usable + calibrated | quality 0.978 | ≥0.80 | **PASS** |
| Phát hiện reverse time | 1.004× | ≥1.10× | FAIL |
| Phát hiện wrong intrinsics | 1.015× | ≥1.15× | FAIL |

### Kết luận kỹ thuật

- Sửa flow-mask và uncertainty đã thành công: track EPE không còn bùng nổ tới hàng trăm nghìn pixel; tracking và calibration hiện là năng lực mạnh nhất của checkpoint.
- Model vẫn thua fixed-depth prior, chỉ ngang identity-pose prior và gần như không phân biệt đúng/sai thứ tự frame hoặc intrinsics.
- `temporal_compatibility≈0.49939` cho thấy temporal head đang ở gần mức đoán ngẫu nhiên.
- `dynamic F1≈0.03` xác nhận dynamic-scene understanding vẫn collapse.
- Kết quả 2/7 không so trực tiếp hoàn toàn với 3/7 trước đó vì v3.2.1 đã thay evaluator contract; tuy nhiên theo gate mới, model rõ ràng chưa đủ điều kiện promotion.

SHA-256 checkpoint: `6967192C246BD52D22B530DB7F4DF350C67AE2B11429901EF7D841165966D4B7`.

Checkpoint được giữ lại để phân tích failure và warm-start có chọn lọc; không được gắn trực tiếp vào JARVIS.
