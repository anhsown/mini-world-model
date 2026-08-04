# JWM — Kế hoạch NGÀY 5 · Sau khi JWM-Read v3 train xong

> Mục tiêu Day 5: xác minh JWM-Read v3 thật sự đọc pixel tài liệu thật, quyết định
> checkpoint có đủ điều kiện trở thành foveal encoder của JWM-Eye hay không, rồi
> dựng nền móng JWM-Eye Physical theo hướng object-centric + future latent modeling.

## Nguyên tắc không thương lượng

- Có checkpoint không đồng nghĩa model đã học đúng.
- Không dùng training loss hoặc teacher-forced token accuracy làm bằng chứng duy nhất.
- Không chuyển sang JWM-Eye Physical nếu shuffled/blank image không làm kết quả giảm.
- Không đưa dataset vào train trước khi các giả thuyết dữ liệu pass.
- Không cập nhật cả Reasoner và Generator cùng lúc trong bước đầu của Eye Physical.
- Không push checkpoint, dataset, ảnh camera cá nhân hoặc trial media lên Git.

## Đầu vào bắt buộc từ Kaggle

Copy nguyên thư mục output hoặc ít nhất các file sau vào local artifact folder:

1. `jwm_read_v3.pt` hoặc `jwm_read_v3_blocked.pt`;
2. `metrics_read_v3.json` và các `metrics_s*.json`;
3. `history_read_v3.json`;
4. `dataset_validation_v3.json`;
5. `training_status_v3.json`.

Ghi thêm Kaggle version, git commit, GPU type, wall time và SHA-256 của từng file.

---

## Pha 0 — Artifact integrity và khả năng tái lập

### Việc làm

- Xác minh `training_status_v3.json` là `complete` hay `blocked_by_metric_gate`.
- Load checkpoint trên CPU trước, sau đó GPU.
- Kiểm tra config, tokenizer, parameter count, missing/unexpected keys.
- Quét toàn bộ tensor để tìm NaN/Inf.
- Đối chiếu global step với history và stage reports.
- Kiểm tra checkpoint inference cho cùng kết quả khi dùng cùng seed.
- Lưu manifest và hash; không sửa checkpoint gốc.

### Gate D5.0

Pass khi checkpoint đọc được, tensor hữu hạn, config đúng v3, history liên tục và
không có mismatch im lặng. Fail thì dừng benchmark và sửa artifact/export trước.

### Đầu ra

- `artifacts/jwm_read_v3/manifest.json`
- `reports/day05_checkpoint_audit.json`

---

## Pha 1 — Benchmark Reader v3 độc lập

### 1.1 Synthetic OCR ladder

- Random L1: glyph/từ ngắn.
- Random L2: cụm từ.
- Random L3: dòng dài.
- Random L4: đoạn nhiều dòng và layout phức tạp.
- Biến thể font chưa train, kích thước chữ, màu, blur, JPEG, noise, perspective,
  low-light, crop, occlusion và background clutter.

Metrics: free-running CER, normalized edit similarity, exact match, CTC-CER,
EOS accuracy, box mIoU và coordinate error.

### 1.2 Real-document benchmark

- VietDocVQA held-out theo page/document.
- MTVQA-VI held-out.
- Một tập hard-OOD riêng: ảnh camera tài liệu, góc nghiêng, màn hình, biển báo,
  chữ nhỏ và cảnh có nhiều vùng chữ.

Metrics: CER, ANLS, exact match, answer containment/F1 và latency. Exact match
không được dùng một mình vì câu trả lời dài bị phạt quá mạnh.

### 1.3 Causal vision controls

Với cùng question, chạy:

1. đúng ảnh;
2. ảnh bị shuffle trong batch;
3. ảnh trắng/đen;
4. crop không chứa đáp án;
5. crop đúng vùng chứa đáp án.

Đo correct-image win rate, loss gap, answer-change rate và bootstrap 95% CI.
Model chỉ được coi là “nhìn ảnh thật” khi đúng ảnh thắng các control một cách có
ý nghĩa thống kê, không chỉ hơn vài phần nghìn do nhiễu.

### 1.4 So sánh v1 → v2 → v3

Chạy cùng code, cùng split, cùng prompt normalization và cùng seed. Báo cáo cả
absolute score và relative improvement; không so các số đến từ harness khác nhau.

### Gate D5.1 — Reader acceptance

Các điều kiện tối thiểu:

- toàn bộ metric gate trong trainer v3 đã pass;
- real-document correct-image win rate > 0.60 và CI thấp hơn vẫn > 0.50;
- real-document vision gap dương rõ ràng;
- median CER trên ít nhất hai real/OOD suites cải thiện tối thiểu 15% tương đối
  so với v2 (`0.99/1.00`);
- shuffled/blank image làm kết quả giảm, đúng crop tốt hơn crop sai;
- không có degeneration kiểu lặp vô hạn, EOS sớm hàng loạt hoặc output rỗng.

Đây là promotion gate, không phải tuyên bố Reader đã hoàn thiện.

### Ba nhánh quyết định

- **PASS:** archive v3 làm `JWM-Eye foveal checkpoint`; sang Pha 3.
- **PARTIAL:** model dùng ảnh nhưng real CER còn yếu; chạy Pha 2 và một đợt
  domain-adaptation ngắn, không full-train lại ngay.
- **FAIL:** blind control gần zero hoặc âm; không xây Eye trên checkpoint này.
  Quay lại kiến trúc/data Reader bằng bằng chứng từ Pha 2.

### Đầu ra

- `reports/read_v3_benchmark_day05.json`
- `reports/read_v3_predictions_day05.jsonl`
- confusion/failure table và biểu đồ v1–v2–v3

---

## Pha 2 — Failure localization

Phân loại từng lỗi vào một nguyên nhân kiểm thử được:

| Nhóm lỗi | Probe bắt buộc | Hành động nếu xác nhận |
|---|---|---|
| Vision blindness | đúng/shuffle/blank/crop control | tăng visual contrast và hard-negative pairing |
| Glyph failure | CTC đúng nhưng AR sai hoặc cả hai sai | kiểm tra tokenizer, scale chữ, local stem |
| Layout failure | box IoU thấp, OCR cục bộ đúng | tăng region/layout supervision |
| Language shortcut | prediction giống nhau giữa ảnh | tăng random/OOD labels, counterfactual pages |
| Domain gap | synthetic tốt, real chết | style transfer, real crops, side-tuning/domain adapter |
| Decoder exposure bias | teacher-forced tốt, free-run chết | scheduled/noisy teacher forcing, EOS calibration |
| MoE collapse | expert load lệch/dead experts | router temperature/aux loss/top-k ablation |
| Truncation | answer chạm max length | đo coverage rồi đổi length/tokenization có kiểm soát |

Mỗi giả thuyết chỉ được giữ nếu có ablation một biến và cùng evaluation split.

---

## Pha 3 — Đặc tả JWM-Eye Physical v1

### 3.1 Dual-rate vision

**Peripheral fast path**

- Nhận toàn bộ stream camera ở 30 FPS, độ phân giải thấp.
- Phát hiện motion/change, looming/collision cue và saliency.
- Mục tiêu local deployment: throughput ≥ 30 FPS, không block camera capture.

**Foveal semantic path**

- Tái sử dụng JWM-Read v3 cho keyframe/crop có độ phân giải cao.
- Chạy theo sự kiện hoặc khoảng 2–5 Hz, không chạy full 1024×768 ở mọi frame.
- Đảm nhiệm OCR, fine-grained object/property và vùng được Reasoner yêu cầu.

### 3.2 Object-centric temporal state

Mỗi object slot duy trì:

```text
id, semantic_embedding, bbox/mask, depth, SE(3) pose,
velocity, visibility/occlusion, confidence, timestamp
```

- Dùng temporal association để giữ ID qua frame.
- Hỗ trợ object permanence khi vật thể bị che ngắn hạn.
- Tách camera ego-motion khỏi object motion.
- Biểu diễn quan hệ trong ego-, world- và object-centric frames.

### 3.3 Future latent modeling

Thêm các future tokens cho nhiều horizon, ví dụ `t+1`, `t+4`, `t+8`:

```text
z_t = EyeStudent(o_t)
z_target = stop_gradient(EyeEMA(o_t+k))
z_hat = FutureHead(z_t, state_t, optional_action)
L_future = 1 - cosine(z_hat, z_target)
```

- EMA/frozen target encoder chống representation collapse.
- Thêm variance/covariance hoặc temporal-negative controls nếu latent collapse.
- Generator pixel-video vẫn tồn tại nhưng chỉ dùng cho rollout/audit chi tiết.

### 3.4 Latent action preparation

- Học action code từ cặp/triple frame không nhãn.
- Tách latent action khỏi native robot command.
- Chưa cho latent action điều khiển thiết bị thật trong Day 5.
- Chuẩn bị inverse/forward cycle: `state + action → future` và
  `state + future → action`.

### 3.5 Multi-task loss

```text
L_eye = λread L_read
      + λground L_ground
      + λtrack L_track
      + λdepth L_depth
      + λpose L_pose
      + λfuture L_future
      + λchange L_change
      + λcal L_calibration
      + λmoe L_moe
```

Loss dùng task mask theo annotation hiện có và chuẩn hóa theo số label hợp lệ để
dataset lớn không vô tình nuốt các nhiệm vụ nhỏ.

---

## Pha 4 — Dataset contract cho Eye Physical

### Các nhánh dữ liệu

1. **Retention:** 10–15% OCR/document + scene QA hiện có.
2. **Real temporal:** video camera/egocentric, tracking, depth, pose và dynamic scenes.
3. **Robot/action:** DROID/Open-X subset hoặc trajectory tương đương có timestamp.
4. **Synthetic/simulation:** motion, collision, occlusion, camera movement và edge cases.
5. **Counterfactual controls:** reversed/shuffled frames, zero/shuffled actions,
   wrong crops và background-only clips.

Dataset ứng viên chỉ được tải sau khi kiểm tra license, dung lượng, annotation và
khả năng tạo held-out split. RoboSpatial, tracking/segmentation, depth/pose và
robot trajectories được lấy theo subset nhỏ có chủ đích trước khi scale.

### Giả thuyết phải kiểm định trước train

- video đọc được, timestamp tăng đơn điệu, FPS và frame gaps đúng;
- không để các frame lân cận/cùng clip lọt qua train–val–test;
- track ID và bbox/mask hợp lệ qua thời gian;
- depth, pose, intrinsics/extrinsics có đơn vị và convention thống nhất;
- action nằm đúng khoảng chuyển tiếp `o_t → o_t+1`;
- real/sim/synthetic được báo metric và sampling ratio riêng;
- event/motion distribution không bị static clip áp đảo;
- OCR retention không bị mất coverage;
- counterfactual labels thực sự khác và không rò shortcut;
- local camera data là private và bị chặn bởi `.gitignore`/export allowlist.

### Active data policy

Không tăng data ngẫu nhiên. Sau mỗi benchmark, cluster failure theo object,
motion, lighting, occlusion, camera và action; chỉ thu/sinh thêm vào vùng yếu.

---

## Pha 5 — Training curriculum dự kiến cho Eye Physical

| Stage | Module được train | Data chính | Gate |
|---|---|---|---|
| E0 temporal bootstrap | fast path + temporal encoder | frame pairs/triples | temporal order/use gap |
| E1 object state | slots + track/ground/depth | real + synthetic object video | HOTA/IDF1, mIoU, depth |
| E2 spatial frames | ego/world/object transforms | posed scenes + RoboSpatial subset | relation/metric accuracy |
| E3 future latent | future tokens + EMA target | unlabeled real/ego video | latent retrieval + no collapse |
| E4 latent action | action tokenizer + inverse head | human/robot video | action-use and cycle consistency |
| E5 joint adaptation | Reasoner adapters, Generator/Policy | balanced real/sim/action | closed-loop + retention |

Trong E0–E4, generator tower và phần Reader đã được chấp nhận sẽ đóng băng hoặc
chỉ mở adapter có kiểm soát. Chỉ E5 mới cân nhắc joint training.

---

## Pha 6 — Metrics và controls bắt buộc

### Perception

- detection/grounding mAP hoặc mIoU;
- tracking HOTA, IDF1, ID switches;
- depth AbsRel và δ1;
- pose ADD-S/translation/rotation error nếu có nhãn;
- OCR CER/ANLS retention.

### Temporal/world understanding

- correct-order so với shuffled/reversed-frame gap;
- future-latent cosine/retrieval accuracy;
- object permanence sau occlusion;
- trajectory/velocity error;
- action-conditioned future so với zero/shuffled-action control;
- inverse–forward cycle consistency.

### Deployment

- camera capture 30 FPS không drop;
- fast-path throughput ≥30 FPS;
- foveal throughput và event-trigger rate;
- p50/p95 latency, VRAM, RAM và power;
- confidence ECE, risk recall và abstention precision.

### Closed-loop

Mọi trial tiếp tục ghi audio, transcript, frame, predicted region, answer,
confidence, latency, ground truth và failure category; bổ sung clip/track ID,
predicted state, action, observed outcome, model version, data version và seed.

---

## Pha 7 — Ablation matrix

Chạy cùng seed/split/budget:

1. frame-only Reader baseline;
2. + fast temporal path;
3. + object slots;
4. + future latent objective;
5. + foveated event routing;
6. + latent action objective.

Một module chỉ được giữ nếu cải thiện metric mục tiêu mà không làm giảm retention,
calibration hoặc latency vượt ngân sách.

---

## Hạ tầng và notebook Day 5

- Viết unit/property tests trước long run.
- Smoke 20–100 optimizer steps trên local/T4 trước full run.
- T4×2 DDP, AMP, gradient clipping, atomic checkpoint và auto-resume.
- Preflight disk ≥4 GiB; không giữ đồng thời tar và extracted copy.
- Mỗi stage ghi JSON history, gate report và sample visualization.
- Notebook phải chạy được bằng `Save Version → Save & Run All`, Persistence
  `Files only`, và output cuối được copy ra Kaggle Output.

Thời gian full Eye training chỉ chốt sau benchmark 100-step vì sequence/video
length quyết định throughput. Không dùng ước lượng mù trước profiling.

---

## Deliverables cuối Day 5

### Nếu Reader v3 PASS

- benchmark v1/v2/v3 + blind controls;
- checkpoint manifest và promotion decision;
- `jwm/EYE_V1.md`;
- prototype fast/foveal/object/future-latent modules;
- dataset validator + compact data manifest;
- unit tests và 100-step profiling report;
- Kaggle T4×2 notebook cho stage E0/E1;
- `day05.md` và `day05_en.md` ghi kết quả thật.

### Nếu PARTIAL/FAIL

- failure localization report;
- một ablation sửa đúng nguyên nhân;
- notebook Reader v3.1 hoặc blocked decision;
- tuyệt đối không tự tuyên bố đã chuyển sang Eye Physical.

## Tiêu chí hoàn thành Day 5

Day 5 hoàn thành khi có một quyết định bằng metric có thể tái lập:

1. **Promote JWM-Read v3** làm foveal checkpoint và có Eye Physical smoke pass; hoặc
2. **Block promotion**, xác định được failure mechanism và chuẩn bị đúng một
   thí nghiệm v3.1 để bác bỏ/xác nhận nó.

Không đánh dấu hoàn thành chỉ vì notebook chạy hết cell.

