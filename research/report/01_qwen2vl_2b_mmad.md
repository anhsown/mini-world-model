# Report — Qwen2-VL-2B × MMAD

## Mục 1: Thông tin cơ bản

| Mục | Nội dung |
|---|---|
| **Tên model test** | Qwen2-VL-2B-Instruct (`Qwen/Qwen2-VL-2B-Instruct`) |
| **Dataset test trên** | MMAD (ICLR 2025) — industrial visual anomaly benchmark |
| **Quy mô** | 39,670 câu MCQ / 8,366 ảnh / 38 category / 4 nguồn / 7 task |
| **Ngày test** | 2026-07-29 → 2026-07-30 |
| **Người test** | Sơn |
| **Manifest SHA-256** | `ff6bf9547135bb9b58dcafcbfafb73e0cfbf804edea0b6dee57fa4dbbf5fb5d6` |

---

## Mục 2: Setup

| Mục | Nội dung |
|---|---|
| **Chạy ở đâu** | Kaggle, GPU T4 (free tier) |
| **Chi phí** | 0 đ — không thuê server |
| **Bản model** | Qwen2-VL-2B-Instruct, 2B params |
| **Backend** | Transformers FP16, attention SDPA, batch = 4, có OOM fallback |
| **Zero-shot?** | **CÓ — hoàn toàn zero-shot.** Không fine-tune, không probe, không few-shot example nào trong prompt |
| **Có train gì không?** | **KHÔNG.** Không XGBoost, không linear probe, không head nào được huấn luyện |
| **Sinh output** | Deterministic (greedy). Parser chỉ chấp nhận đúng một chữ cái A–D không nhập nhằng |
| **Notebook** | `research/mmad_model_benchmark/models/qwen2_vl/Qwen2VL_2B_MMAD_Full.ipynb` |
| **Resume** | Checkpoint JSONL theo `sample_id` ổn định, không dùng vị trí |

Ghi chú: đây là setting zero-shot theo protocol nội bộ, **không phải** bản tái hiện setting 1-shot và domain-knowledge của paper MMAD gốc. Vì vậy không so trực tiếp được với leaderboard chính thức.

---

## Mục 3: Kiểm tra leak

Vì đây là zero-shot thuần, **không tồn tại tập train của mình** → không có khái niệm cycle/participant trùng giữa train/test, và embedding-similarity train↔test không áp dụng. Các kiểm tra thay thế đã làm:

| Kiểm tra | Kết quả | Trạng thái |
|---|---|---|
| Tên file query có lộ nhãn không | Đổi hết thành `sample_XXXX` trước khi đưa vào prompt | ✅ PASS |
| Nhãn source / class / defect có xuất hiện trong prompt không | Không — chỉ có ảnh + câu hỏi + 4 lựa chọn | ✅ PASS |
| Annotation gốc có bị chèn vào prompt không | Không — chỉ dùng để chấm điểm | ✅ PASS |
| Parser có "đoán hộ" model không | Không — chỉ nhận A–D rõ ràng, còn lại tính parse_failure | ✅ PASS |
| Model có thấy MMAD lúc pretrain không | **CHƯA KIỂM TRA ĐƯỢC** | ⚠️ REVIEW |

**Kết luận:** Ở mức protocol, dữ liệu đủ độc lập — model không nhận được manh mối nào ngoài pixel và câu hỏi. Rủi ro còn lại là **nhiễm bẩn từ pretraining**: MMAD là dataset public trên HuggingFace từ ICLR 2025, Qwen2-VL có thể đã gặp ảnh MVTec-AD/VisA trong dữ liệu huấn luyện. Không có cách kiểm chứng từ phía mình. Rủi ro này áp dụng như nhau cho mọi model open-source được so sánh, nên **không làm sai lệch phép so sánh giữa các model**, chỉ làm con số tuyệt đối lạc quan hơn thực tế.

---

## Mục 4: Kết quả chính

### 4.1 Toàn bộ MMAD

MMAD là MCQ 2–4 lựa chọn, phân bố đáp án hơi lệch (A 30.4% / B 30.1% / C 20.3% / D 19.2%). Metric chính báo cáo là **Macro-F1 trên nhãn đáp án** kèm micro accuracy theo protocol công bố của MMAD.

| | Micro accuracy | Macro-F1 | n |
|---|---|---|---|
| **Qwen2-VL-2B** | **64.73%** (CI 95% 64.26–65.20) | **0.6499** | 39,669 |
| Baseline majority-letter (luôn trả lời A) | 30.38% | 0.1165 | 39,669 |
| Baseline random-choice (kỳ vọng) | 30.21% | — | 39,669 |
| **Chênh lệch vs majority** | **+34.35 điểm** (gấp 2.13×) | **+0.5334** | |

Baseline được tính bằng code riêng trên **đúng 39,669 record model đã trả lời**, không mượn từ nguồn khác.

### 4.2 Theo nguồn dữ liệu

| Nguồn | n | Accuracy | CI 95% |
|---|---|---|---|
| MVTec-AD | 8,336 | **75.25%** | 74.31–76.17 |
| GoodsAD | 13,088 | 64.23% | 63.40–65.04 |
| VisA | 10,621 | 63.93% | 63.01–64.84 |
| MVTec-LOCO | 7,624 | **55.22%** | 54.10–56.33 |

Chênh 20 điểm giữa nguồn dễ nhất và khó nhất. **Hệ quả quan trọng cho việc so sánh:** bất kỳ model nào chỉ chạy trên MVTec-AD sẽ có con số cao giả tạo. Xem `02_cosmos3_nano_mmad.md`.

### 4.3 Theo task

| Task | n | Accuracy |
|---|---|---|
| Object Classification | 3,155 | 93.82% |
| Object Analysis | 9,160 | 78.30% |
| Defect Analysis | 4,782 | 72.90% |
| Defect Description | 4,710 | 63.74% |
| Anomaly Detection | 8,297 | 58.31% |
| Defect Localization | 4,877 | 46.73% |
| Defect Classification | 4,688 | **41.42%** |

### 4.4 Anomaly Detection — subtask quan trọng nhất về mặt vận hành

Đây là dữ liệu mất cân bằng (60.83% câu là "có lỗi"), nên **không dùng accuracy**:

| | Precision | Recall | F1 | Miss rate | Overkill rate |
|---|---|---|---|---|---|
| **Qwen2-VL-2B** | 0.7428 | **0.4807** | **0.5837** | 51.93% | 25.85% |
| Baseline "luôn nói có lỗi" | 0.6083 | 1.0000 | **0.7564** | 0% | 100% |
| **Chênh lệch** | +0.1345 | **−0.5193** | **−0.1727** | | |

> ⚠️ **Trên riêng subtask Anomaly Detection, Qwen2-VL-2B THUA baseline ngu.** F1 0.584 so với 0.756; accuracy 58.31% so với 60.83%. Model bỏ sót hơn một nửa số ca lỗi thật.

Chi tiết theo nguồn (mức bỏ sót rất khác nhau):

| Nguồn | Precision | Recall | F1 | Miss rate |
|---|---|---|---|---|
| MVTec-AD | 0.850 | 0.834 | 0.841 | 16.6% |
| VisA | 0.650 | 0.689 | 0.669 | 31.1% |
| GoodsAD | 0.733 | 0.254 | 0.377 | 74.6% |
| MVTec-LOCO | 0.722 | 0.167 | 0.272 | **83.3%** |

### 4.5 Độ trễ

Median 0.687 s/câu, p95 0.773 s, mean 0.691 s. Toàn bộ 39,670 câu chạy trong khoảng 7.6 giờ GPU T4.

---

## Mục 5: Kiểm tra shortcut

MMAD là **ảnh tĩnh đơn**, không phải video → **test static-frame không áp dụng**.

Phép kiểm tương đương cho ảnh tĩnh là **blind-image control** (thay ảnh thật bằng ảnh nhiễu hoặc ảnh không liên quan, giữ nguyên câu hỏi). Nếu điểm không giảm → model đang trả lời bằng kiến thức ngôn ngữ chứ không nhìn ảnh.

| Kiểm tra | Trạng thái |
|---|---|
| Blind-image control | ❌ **CHƯA LÀM** |
| Answer-position bias | ✅ Đã đo |

**Answer-position bias đã đo được:**

| Đáp án | Tỉ lệ trong ground truth | Tỉ lệ model chọn | Lệch |
|---|---|---|---|
| A | 30.38% | 31.10% | +0.72% |
| B | 30.11% | 27.02% | **−3.09%** |
| C | 20.31% | 19.57% | −0.74% |
| D | 19.20% | 22.30% | **+3.10%** |

Lệch tối đa 3.1 điểm — có thiên lệch vị trí nhẹ nghiêng về D, nhưng chưa đủ lớn để giải thích kết quả. Không phải shortcut nghiêm trọng.

**Kết luận mục 5:** chưa đủ căn cứ khẳng định model thật sự "nhìn" ảnh. Blind-image control là việc bắt buộc phải bổ sung — với model 2B và độ chênh giữa các task rất lớn (93.8% vs 41.4%), khả năng một phần điểm đến từ prior ngôn ngữ là có thật.

---

## Mục 6: Ví dụ lỗi cụ thể

Toàn bộ lấy từ tập test đầy đủ (không có tập train). Mỗi ca ghi rõ model đoán gì / đáp án thật / phân loại nguyên nhân.

### Lỗi 1 — `mmad_full_00064` (MVTec-AD / bottle / Anomaly Detection)
- **Câu hỏi:** "Is there any defect in the object?" — A) No. B) Yes.
- **Đáp án thật:** A — "No." (ảnh này là hàng bình thường)
- **Model đoán:** B — "Yes."
- **Loại lỗi:** **Lý do khác — thiên lệch trả lời.** Model báo động giả trên hàng tốt. Khớp với overkill_rate 25.85% toàn cục. Ca `mmad_full_00065` cùng category lặp lại y hệt.

### Lỗi 2 — `mmad_full_08311` (MVTec-AD / bottle / Defect Classification)
- **Câu hỏi:** "What is the type of the defect?" — A) Breakage. B) Stain. C) Discoloration. D) Crack.
- **Đáp án thật:** A — "Breakage."
- **Model đoán:** D — "Crack."
- **Loại lỗi:** **Nhầm lẫn giữa 2 nhãn quá giống nhau.** "Breakage" (vỡ hẳn) và "Crack" (nứt) khác nhau ở mức độ, đòi hỏi phân biệt độ sâu/mức phá huỷ. Đây là mẫu lỗi chi phối task Defect Classification (41.42% — task yếu nhất).

### Lỗi 3 — `mmad_full_12991` (MVTec-AD / bottle / Defect Localization)
- **Câu hỏi:** "Where is the defect?" — A) Bottom center. B) Left side. C) Right side. D) Top center.
- **Đáp án thật:** A — "Bottom center of the image."
- **Model đoán:** D — "Top center of the image."
- **Loại lỗi:** **Lý do khác — yếu định vị không gian.** Model chọn đúng trục (center) nhưng ngược chiều dọc hoàn toàn. Đây không phải nhầm nhãn giống nhau mà là hỏng khả năng grounding toạ độ — đúng với Defect Localization 46.73%.

### Lỗi 4 — `mmad_full_30520` (MVTec-AD / bottle / Object Analysis)
- **Câu hỏi:** "Where is the location of the object's opening?" — A) Bottom. B) Side. C) Top. D) Closed.
- **Đáp án thật:** C — "At the top of the object."
- **Model đoán:** A — "At the bottom of the object."
- **Loại lỗi:** **Lý do khác — yếu định vị không gian.** Cùng dạng lỗi trên/dưới với Lỗi 3, nhưng ở câu hỏi về vật thể bình thường chứ không phải về lỗi. Xác nhận đây là điểm yếu hệ thống của model, không phải do defect khó nhìn.

### Lỗi 5 — `mmad_full_27502` (MVTec-AD / grid / Object Classification)
- **Câu hỏi:** "What kind of product is in the image?" — A) Wallpaper. B) Plastic sheeting. C) Ceramic tile. D) Textured fabric.
- **Đáp án thật:** D — "Textured fabric."
- **Model đoán:** B — "Plastic sheeting."
- **Loại lỗi:** **Nhiễu/mất chi tiết do xử lý.** Phân biệt vải dệt với tấm nhựa phụ thuộc vào kết cấu sợi ở tần số không gian cao — thứ dễ mất nhất khi resize ảnh về độ phân giải đầu vào của model.

### Lỗi 6 — `mmad_full_22618` (MVTec-AD / bottle / Defect Analysis)
- **Câu hỏi:** "What may be the effect of the defect?" — A) Change in liquid flavor or contamination. B) Unsellable. C) Affect ability to hold liquid. D) Leaking contents.
- **Đáp án thật:** A — "It could cause a change in liquid flavor or contamination."
- **Model đoán:** D — "It could lead to leaking contents."
- **Loại lỗi:** **Nhầm lẫn giữa 2 nhãn quá giống nhau.** Cả 4 lựa chọn đều là hệ quả hợp lý của một chai bị lỗi; muốn chọn đúng phải biết *loại* lỗi cụ thể trên ảnh. Model suy luận hệ quả chung chung thay vì căn cứ vào bằng chứng thị giác.

### Phân bố lỗi theo category (Pareto — 5 category chiếm 32.3% tổng lỗi)

| Nguồn | Category | Số lỗi | % tổng lỗi |
|---|---|---|---|
| GoodsAD | drink_bottle | 1,248 | 8.92% |
| GoodsAD | food_bottle | 876 | 6.26% |
| GoodsAD | food_package | 824 | 5.89% |
| MVTec-LOCO | screw_bag | 814 | 5.82% |
| MVTec-LOCO | juice_bottle | 758 | 5.42% |

---

## Mục 7: Kết luận cuối cùng

**Dataset: MMAD (toàn bộ 39,670 câu)**

- ☑ **Model này THẮNG RÕ RÀNG so với baseline đơn giản** — 64.73% vs 30.38% majority-letter, chênh +34.35 điểm, khoảng tin cậy không chồng lấn.
- ☐ Model này NGANG BẰNG baseline đơn giản
- ☐ Model này THUA baseline đơn giản

**Kết luận phụ bắt buộc phải ghi kèm — subtask Anomaly Detection (n = 8,297):**

- ☐ THẮNG RÕ RÀNG
- ☐ NGANG BẰNG
- ☑ **THUA baseline đơn giản** — F1 0.584 vs 0.756 của baseline "luôn nói có lỗi"; miss rate 51.93%.

Hai kết luận này không mâu thuẫn: Qwen2-VL-2B là **model mô tả và nhận dạng vật thể tốt**, nhưng **không dùng được làm cổng phát hiện bất thường** vì bỏ sót quá nửa số ca lỗi. Nếu mục tiêu công ty là B3 (anomaly & fault localization), con số 64.73% không phản ánh đúng năng lực thực tế.

---

## Mục 8: Lưu ý / giới hạn

**Mẫu test có đủ lớn không?**
Có. 39,669 record parse-valid, CI 95% chỉ ±0.47 điểm. Đây là bộ đầy đủ, không phải subset. Mọi kết luận ở mức toàn dataset đều vững về mặt thống kê.

**Có phát hiện leak không? Kết quả Mục 4 có đáng tin?**
Không phát hiện leak ở mức protocol — tên file đã trung tính, nhãn không lọt vào prompt, parser không đoán hộ. Kết quả Mục 4 đáng tin **ở mức so sánh tương đối** giữa các model chạy cùng manifest. Con số tuyệt đối có thể lạc quan do khả năng nhiễm bẩn pretraining, không kiểm chứng được.

**Có điều gì chưa kiểm tra mà làm thêm sẽ chắc chắn hơn?**

1. **Blind-image control** — bắt buộc. Không có nó thì chưa chứng minh được model thật sự dùng ảnh. Chi phí thấp: chạy lại một mẫu ~500 câu với ảnh bị xáo trộn.
2. **Chưa lưu probability/logits** — nên không claim được gì về calibration (ECE) hay khả năng abstain. Cần bật ở lần chạy sau, đây là yêu cầu của Cosmos-style confidence.
3. **Chưa chạy setting 1-shot và domain-knowledge** của MMAD gốc → không đối chiếu được với leaderboard chính thức và với mốc GPT-4o ~74.9%.
4. **Chưa phân tích ảnh hưởng độ phân giải** — nghi ngờ ở Lỗi 5 (mất kết cấu do resize) chưa được kiểm chứng bằng thí nghiệm đổi độ phân giải đầu vào.
5. **Chưa có human baseline nội bộ** trên cùng subset để biết trần thực tế của bộ dữ liệu này.

---

## Nguồn dữ liệu và mã nguồn

| Loại | Đường dẫn |
|---|---|
| Prediction thô | `research/mmad_model_benchmark/models/qwen2_vl/qwen2_vl_mmad_full_results/predictions.jsonl` |
| Bảng chấm điểm | `.../qwen2_vl_mmad_full_results/predictions_scored.csv` |
| Metrics | `.../qwen2_vl_mmad_full_results/metrics.json` |
| Phân tích sâu + hình | `.../qwen2_vl_mmad_full_results/deep_analysis/` |
| Hình tổng quan | `.../qwen2_vl_mmad_full_results/full_mmad_analysis.png` |
| Notebook | `research/mmad_model_benchmark/models/qwen2_vl/Qwen2VL_2B_MMAD_Full.ipynb` |
| Repo public | `https://github.com/anhsown/mini-world-model` |

**Hình nên gắn kèm khi trình bày:**
`deep_analysis/02_task_source_accuracy_ci.png`, `deep_analysis/05_anomaly_by_source.png`, `deep_analysis/07_error_pareto_latency.png`.
