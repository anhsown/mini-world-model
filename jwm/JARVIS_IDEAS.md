# JARVIS — Long-term Architecture Ideas

Tài liệu sống để lưu các giả thuyết và hướng thiết kế trước khi chuyển thành đặc tả triển khai. Mỗi ý tưởng phải được kiểm định bằng ablation, benchmark và kiểm tra hồi quy trước khi tích hợp vào model chính.

## 1. Lifelong learning

- Mục tiêu không phải loại bỏ training, mà loại bỏ nhu cầu thường xuyên full-train thủ công.
- Tách ba tốc độ học: working/episodic memory tức thời; semantic memory có truy xuất; adapter/skill consolidation định kỳ.
- Core model ổn định. Dữ liệu sống chỉ được chọn qua novelty, uncertainty, prediction error và redundancy gate.
- Cập nhật trọng số phải diễn ra trong sandbox, có replay chống quên, benchmark hồi quy, versioning và rollback.
- Kiến thức đọc được lưu kèm nguồn, thời gian, độ tin cậy và mâu thuẫn; không dùng một nguồn đơn lẻ để tự sửa core weights.
- Dữ liệu vật lý tự giám sát có dạng `(observation_t, action_t, observation_t+1)` để học dynamics và quan hệ nhân quả.

## 2. JWM-Eye inspired by human vision

- Mắt người không xử lý toàn bộ cảnh ở độ phân giải cao đồng đều. Fovea xử lý chi tiết; ngoại vi xử lý thô nhưng rất nhạy với chuyển động và bất ngờ.
- Cảm nhận “toàn cảnh liên tục” được tạo bởi chuyển động mắt chủ động, ký ức ngắn hạn, theo dõi đối tượng và dự đoán của não.
- Thiết kế JWM-Eye nên kết hợp: multi-resolution/foveated vision, frame stream và event/motion stream, object-centric tracking, temporal state memory, predictive coding, saliency/uncertainty routing và active gaze.
- Hệ thống chạy đa tốc độ: capture 30 FPS; motion/tracking nhẹ ở tốc độ cao; semantic reasoning nặng trên keyframe hoặc khi có sự kiện.
- Đại diện chính của cảnh nên là trạng thái đối tượng và quan hệ đang thay đổi, không phải lưu và suy luận lại toàn bộ pixel ở mọi frame.

## 3. Research routing

- Các kết luận từ NVIDIA robotics/world-model research và giả thuyết áp dụng cho JWM được duy trì trong `NVIDIA_ROBOTICS_RESEARCH.md`.
- Hướng ưu tiên: object-centric spatial state, future latent alignment, latent-action discovery, sim-real co-training, active failure-driven data và verified continual learning.
