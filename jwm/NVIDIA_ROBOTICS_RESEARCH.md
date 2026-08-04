# NVIDIA Robotics Research Notes for JWM

Updated: 2026-07-19

Mục tiêu của tài liệu này là chuyển các kết quả nghiên cứu thành giả thuyết thiết kế có thể kiểm định cho JWM, không sao chép nguyên kiến trúc hoặc coi kết quả của paper là mặc định đúng với quy mô phần cứng của dự án.

## 1. GR00T N1 / N1.5 — tách reasoning và continuous action

GR00T dùng VLM để mã hóa hình ảnh/ngôn ngữ và một Diffusion Transformer để xử lý state cùng noisy actions. N1.5 đóng băng VLM trong cả pretraining và finetuning, chuẩn hóa adapter giữa vision và language, đồng thời thêm FLARE future-latent objective.

Áp dụng cho JWM:

- Giữ kiến trúc hai nhánh Reasoner và Generator/Policy.
- Sau khi JWM-Eye đạt benchmark, đóng băng phần perception ổn định trước khi train action head.
- Chuẩn hóa visual/text/action token trước cross-attention để giảm lệch thang đo.
- Không dùng action loss làm tín hiệu duy nhất; thêm future-state objective.

Nguồn: https://research.nvidia.com/publication/2025-03_nvidia-isaac-gr00t-n1-open-foundation-model-humanoid-robots

Nguồn: https://research.nvidia.com/labs/gear/gr00t-n1_5/

## 2. FLARE — world modeling trong latent space

FLARE thêm future tokens vào policy và buộc chúng khớp với embedding của quan sát tương lai. Cách này tránh chi phí tái tạo mọi pixel nhưng vẫn ép policy học hậu quả của hành động.

Ứng dụng ưu tiên cao cho JWM:

```text
z_t = Eye(o_t)
z_future_hat = FutureHead(z_t, instruction, action)
target = stop_gradient(Eye_target(o_t+k))
L_future = 1 - cosine(z_future_hat, target)
```

- Dùng latent target encoder EMA hoặc frozen để tránh representation collapse.
- Học nhiều horizon ngắn/trung bình thay vì chỉ một frame kế tiếp.
- Generator pixel/video chỉ chạy khi cần visualization, audit hoặc rollout chi tiết.

Nguồn: https://research.nvidia.com/labs/gear/flare/

## 3. Latent Action Pretraining — học hành động từ video không có action label

LAPA dùng VQ-VAE để lượng tử hóa thay đổi giữa các frame thành discrete latent actions, pretrain VLA dự đoán latent action từ quan sát và instruction, rồi dùng lượng nhỏ robot data để ánh xạ latent action sang native robot control.

Áp dụng cho JWM:

- Thêm action tokenizer học từ `(o_t, o_t+k)` trước khi có robot thật.
- Video con người và video egocentric có thể dùng để học motion/action prior.
- Latent action không được coi là control command thật cho đến khi qua embodiment adapter và safety controller.

Nguồn: https://research.nvidia.com/publication/2025-04_latent-action-pretraining-videos

## 4. DreamGen — video world model thành neural trajectories

DreamGen finetune video model cho embodiment, sinh video theo ảnh đầu + instruction, dùng latent-action model hoặc inverse-dynamics model để suy ra pseudo action, rồi huấn luyện policy bằng neural trajectories.

Áp dụng cho JWM:

- Synthetic trajectory phải qua các gate: instruction following, temporal consistency, contact/physics, inverse-forward cycle consistency và uncertainty.
- Co-train với real data; không để pseudo trajectories thay thế hoàn toàn dữ liệu thật.
- Ưu tiên sinh failure/edge cases mà replay buffer đang thiếu.

Nguồn: https://research.nvidia.com/labs/gear/dreamgen/

## 5. Spatial perception — RoboSpatial, FoundationPose, FoundationStereo

RoboSpatial nhấn mạnh ba hệ tọa độ ego-, world- và object-centric. FoundationPose cho thấy 6D pose estimation/tracking trên vật thể mới có thể generalize bằng synthetic scale + contrastive learning. FoundationStereo dùng một triệu stereo pair synthetic, self-curation và side-tuning từ monocular foundation priors để giảm sim-to-real gap.

Áp dụng cho JWM-Eye:

- Mỗi object slot cần `id, semantic embedding, bbox/mask, depth, SE(3) pose, velocity, visibility, uncertainty, timestamp`.
- Grounding data phải ghi rõ reference frame.
- Depth/pose/track phải có temporal consistency loss, không chỉ per-frame supervision.
- Synthetic depth/pose data phải self-curate trước khi train và luôn benchmark trên real held-out domains.

Nguồn: https://research.nvidia.com/publication/2025-06_robospatial-teaching-spatial-understanding-2d-and-3d-vision-language-models

Nguồn: https://research.nvidia.com/publication/2024-06_foundationpose-unified-6d-pose-estimation-and-tracking-novel-objects

Nguồn: https://research.nvidia.com/publication/2025-06_foundationstereo-zero-shot-stereo-matching

## 6. Simulation, sim-real co-training và adaptive data

Isaac Lab tích hợp GPU physics, multi-frequency sensors, domain randomization, RL/imitation và human demonstrations. Sim-and-real co-training cho thấy dữ liệu mô phỏng và thật nên được học cùng nhau. AdaDemo mở rộng demonstrations theo đúng failure mode hiện tại của policy.

Áp dụng cho JWM:

- Simulator phải mô phỏng sensor ở nhiều frequency thay vì giả định tất cả modalities đồng bộ một tốc độ.
- Batch sampler cần giữ tỷ lệ real/sim/synthetic ổn định và theo dõi metric riêng từng domain.
- Data acquisition phải active: thu thêm dữ liệu ở failure cluster, không chỉ tăng ngẫu nhiên số sample.

Nguồn: https://research.nvidia.com/labs/prl/publication/isaaclab2025/

Nguồn: https://research.nvidia.com/labs/lpr/publication/maddukuri2025simandreal/

Nguồn: https://research.nvidia.com/publication/2024-12_adademo-data-efficient-demonstration-expansion-generalist-robotic-agent

## 7. Continual improvement — ASPIRE và ENPIRE

ASPIRE lưu multimodal execution traces, sửa code-as-policy từ lỗi, kiểm định lại và đưa bản sửa đã xác minh vào skill library. ENPIRE tổ chức vòng lặp reset, rollout, verification và evolution trên robot thật. Cả hai củng cố hướng lifelong-learning của JARVIS, nhưng cũng chỉ ra rằng safety reset, success detection, calibration, compute cost và stale memory vẫn là vấn đề chưa giải quyết hoàn toàn.

Áp dụng cho JARVIS:

- Log mọi trial theo schema đã định và thêm model/data/code version.
- Tách debug seeds và held-out evaluation seeds.
- Chỉ lưu skill khi tái thực thi pass; version skill và phát hiện stale/redundant entries.
- Không cho live learner trực tiếp ghi đè core weights hoặc bỏ qua safety controller.

Nguồn: https://research.nvidia.com/labs/gear/aspire/

Nguồn: https://research.nvidia.com/labs/gear/enpire/

## 8. JWM-Eye architecture hypothesis

```text
30 FPS camera
  ├─ Fast peripheral stream: motion/change/depth-risk
  ├─ Foveal stream: high-resolution selected regions
  └─ Temporal object encoder
          ↓
Object slots + ego/world/object coordinate transforms
          ↓
Future latent predictor (FLARE-style)
          ↓
Latent action tokenizer / inverse dynamics
          ↓
Reasoner ↔ Generator/Policy
          ↓
Safety controller + calibrated uncertainty
```

Ưu tiên triển khai sau JWM-Read:

1. Object permanence và tracking trên video.
2. Depth + camera ego-motion + coordinate frames.
3. Future latent alignment với blind/temporal controls.
4. Latent action discovery từ video không nhãn.
5. Action-conditioned rollout và inverse-forward cycle consistency.
6. Active failure-driven data collection.

## 9. Validation contract

Mọi thay đổi chỉ được giữ nếu vượt qua:

- vision-use controls: exact image so với shuffled/blank image;
- temporal-use controls: đúng thứ tự frame so với shuffled/reversed frames;
- action-use controls: đúng action so với shuffled/zero action;
- sim-to-real score theo từng domain;
- OOD object/environment/lighting/camera;
- retention benchmark để phát hiện catastrophic forgetting;
- calibration, latency và memory budget;
- closed-loop task success, không chỉ teacher-forced loss.

