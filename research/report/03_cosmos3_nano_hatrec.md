# Report — Cosmos 3 Nano Reasoner × HATREC

## Mục 1: Thông tin cơ bản

| Mục | Nội dung |
|---|---|
| **Tên model test** | Cosmos 3 Nano Reasoner (NVIDIA), qua endpoint chính thức NVIDIA Build |
| **Dataset test trên** | HATREC — video lắp ráp công nghiệp, phân loại 7 công đoạn |
| **Quy mô** | 546 video / 78 cycle / 7 lớp cân bằng (78 video mỗi lớp) |
| **Đặc điểm video** | 720×1280, 30 fps, dài trung bình 3.46 s (min 1.57 s, max 7.30 s) |
| **Độ phủ đã chạy** | 546/546 = **100%** |
| **Ngày test** | 2026-07-27 → 2026-07-28 |
| **Người test** | Sơn |

**7 nhãn công đoạn:** 0 = Assembling the spring · 1 = Placing the white plastic part · 2 = Screwing-1 · 3 = Inflating the valve · 4 = Placing the black plastic part · 5 = Screwing-2 · 6 = Fixing the cable

---

## Mục 2: Setup

| Mục | Nội dung |
|---|---|
| **Chạy ở đâu** | Máy local, tự động hoá trình duyệt tới NVIDIA Build |
| **Chi phí** | 0 đ — free tier, không thuê server |
| **Bản model** | Cosmos 3 Nano Reasoner, endpoint chính thức, **không lượng tử hoá** |
| **Zero-shot?** | **CÓ — hoàn toàn zero-shot.** Không fine-tune, không few-shot example |
| **Có train gì không?** | **KHÔNG.** Không probe, không classifier, không head huấn luyện |
| **Đầu vào** | Video đầy đủ (không trích frame), upload trực tiếp lên UI |
| **Prompt** | Prompt video công nghiệp có cấu trúc, yêu cầu 9 mục bắt buộc, kết thúc bằng `MOST LIKELY HATREC TASK` + `CONFIDENCE AND LIMITATIONS` |
| **Runner** | `research/hatrec_cosmos3/run_ui.py` |
| **Latency trung bình** | 16.49 s/video |

**Lưu ý về hai lần chấm điểm:** lần chấm đầu (`outputs/ui_evaluation.json`) chỉ parse được 135/546 output → accuracy 18.52%. Parser sau đó được cải tiến, phục hồi được **449/546** output → các con số trong báo cáo này lấy từ lần chấm sâu (`outputs/analysis/deep_audit_summary.json`). Con số 18.52% từng lưu hành **đã lỗi thời**, không dùng nữa.

---

## Mục 3: Kiểm tra leak

| Kiểm tra | Kết quả | Trạng thái |
|---|---|---|
| **Tên file có lộ nhãn không** | Tên gốc `Cycle_0_task_3.mp4` **chứa nhãn**. Runner copy sang tên trung tính `sample_000123.mp4` trong thư mục tạm rồi mới upload, và **verify tên đã upload** (`upload_verified: true`) | ✅ PASS |
| Cycle/participant trùng giữa train/test | **Không áp dụng** — zero-shot, không có tập train | — |
| Embedding similarity train↔test | **Không áp dụng** — model không được huấn luyện trên dữ liệu này | — |
| Video hỏng / trùng lặp | 546/546 video hợp lệ, phân bố lớp cân bằng tuyệt đối (78 mỗi lớp) | ✅ PASS |
| Nhãn có lọt vào prompt không | Không — prompt chỉ mô tả nhiệm vụ và liệt kê 7 lựa chọn | ✅ PASS |

**Kết luận:** Với Cosmos 3, **không có đường leak nào**. Model chưa từng thấy HATREC, tên file đã trung tính hoá và được xác minh, nhãn không xuất hiện trong prompt. Dữ liệu đủ độc lập để tin kết quả Mục 4.

⚠️ Lưu ý riêng cho dataset: HATREC có các quan ngại về hiệu lực đã phát hiện khi làm việc với V-JEPA 2 (các cycle rất giống nhau về mặt thị giác, similarity 0.99). Điều đó **ảnh hưởng tới model có huấn luyện**, không ảnh hưởng tới đánh giá zero-shot này. Chi tiết ở `04_vjepa2_hatrec.md`.

---

## Mục 4: Kết quả chính

Dataset cân bằng hoàn hảo (7 lớp × 78), nên chance = 14.29%. Metric chính: **Macro-F1**.

### 4.1 Kết quả tổng

| | Accuracy | Macro-F1 | n |
|---|---|---|---|
| **Cosmos 3 Nano — end-to-end** (output không parse được tính là sai) | **23.63%** (CI 20.15–27.29) | **0.2162** | 546 |
| **Cosmos 3 Nano — có điều kiện** (chỉ tính output parse được) | **28.73%** (CI 24.72–32.96) | 0.2394 | 449 |
| Baseline majority-class | 14.29% | **0.0357** | 546 |
| Baseline random-choice | 14.29% | — | 546 |
| **Chênh lệch (end-to-end vs majority)** | **+9.34 điểm** | **+0.1805** (gấp 6.1×) | |

Khoảng tin cậy end-to-end [20.15%, 27.29%] **không chứa** mốc baseline 14.29% → chênh lệch có ý nghĩa thống kê.

Cận trên lạc quan nhất (giả sử toàn bộ 97 output không parse được đều đúng): **41.39%**. Kể cả trong kịch bản tốt nhất này, model vẫn sai gần 6/10 lần.

Macro precision 0.2712 · Macro recall 0.2363.

### 4.2 Theo từng lớp — phát hiện quan trọng nhất

| Nhãn | Tên công đoạn | Support | Đúng | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 4 | Placing black plastic | 78 | 42 | 0.326 | 0.538 | **0.406** |
| 1 | Placing white plastic | 78 | 19 | 0.543 | 0.244 | 0.336 |
| 2 | Screwing-1 | 78 | 37 | 0.236 | 0.474 | 0.315 |
| 5 | Screwing-2 | 78 | 22 | 0.253 | 0.282 | 0.267 |
| 6 | Fixing cable | 78 | 8 | 0.500 | 0.103 | 0.170 |
| 0 | Assembling the spring | 78 | **1** | 0.042 | 0.013 | **0.020** |
| 3 | **Inflating the valve** | 78 | **0** | 0.000 | 0.000 | **0.000** |

> ❗ **Sụp đổ lớp (class collapse).** Lớp 3 "Inflating the valve" đúng **0/78**, và trong toàn bộ 449 output parse được, model chỉ **một lần duy nhất** dự đoán lớp này. Lớp 0 "Assembling the spring" đúng 1/78.

Phân bố dự đoán (449 output parse được) so với phân bố thật (mỗi lớp phải ~14.3%):

| Nhãn dự đoán | Số lần | Tỉ lệ |
|---|---|---|
| 2 — Screwing-1 | 157 | **34.97%** |
| 4 — Placing black plastic | 129 | 28.73% |
| 5 — Screwing-2 | 87 | 19.38% |
| 1 — Placing white plastic | 35 | 7.80% |
| 0 — Assembling the spring | 24 | 5.35% |
| 6 — Fixing cable | 16 | 3.56% |
| 3 — Inflating the valve | **1** | **0.22%** |

Model dồn 83% dự đoán vào 3 lớp. Total-variation distance so với phân bố thật = 0.402.

### 4.3 Độ tin cậy — không hiệu chỉnh

| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| ECE | **0.5552** | Sai lệch hiệu chỉnh cực lớn |
| Brier score | 0.5188 | |
| Confidence hay gặp nhất | 0.9 (167 lần), 0.95 (57 lần) | Model nói "chắc 90%" trong khi đúng 24% |

> Model **tự tin sai một cách hệ thống**. Đây là điểm đối lập trực tiếp với yêu cầu "calibrated confidence + abstention" mà dự án đặt ra.

### 4.4 Tỉ lệ hoàn thành — nút thắt hạ tầng

| Trạng thái | Số lượng | Tỉ lệ |
|---|---|---|
| `complete` (hoàn tất bình thường) | 125 | 22.9% |
| `partial_timeout` (bị cắt giữa chừng) | 405 | **74.2%** |
| `timeout_no_output` (không ra chữ nào) | 16 | 2.9% |
| **Parse được nhãn** | **449** | **82.2%** |

Tuân thủ định dạng: đủ 9 mục = 65.93%; có tag bằng chứng = 36.26%.

**77% số lần chạy bị timeout ở mức nào đó.** Đây vừa là giới hạn của UI, vừa là dấu hiệu model "nghĩ" quá dài cho một video 3.5 giây.

---

## Mục 5: Kiểm tra shortcut

HATREC là **video** → mục này **có áp dụng**.

| Kiểm tra | Trạng thái | Ghi chú |
|---|---|---|
| Static-frame (lấy 1 frame lặp lại) | ❌ **CHƯA LÀM** | Bắt buộc bổ sung |
| Temporal shuffle / đảo ngược thứ tự frame | ❌ **CHƯA LÀM** | |
| Che vật thể / can thiệp nền | ❌ **CHƯA LÀM** | |

**Chưa có số static-frame nên chưa kết luận được theo tiêu chí <5% / >20–30% của template.**

Tuy nhiên có một chỉ báo gián tiếp đã đo được từ phân tích ngữ nghĩa nội dung sinh ra:

| Chỉ số | Giá trị |
|---|---|
| **Tỉ lệ bằng chứng mang tính thời gian** (mean temporal evidence ratio) | **0.1764** |
| Tỉ lệ kết luận có được lập luận đỡ | 0.8223 |
| Tỉ lệ khẳng định chắc chắn nhưng không có căn cứ | 0.6886 |
| Tỉ lệ có nêu giới hạn nhận thức rõ ràng | 0.8590 |

Chỉ **17.6%** bằng chứng model viện dẫn là về chuyển động/trình tự thời gian; hơn 82% là mô tả tĩnh (vật thể, màu sắc, bố cục bàn làm việc). Đây là **dấu hiệu cho thấy model chủ yếu đọc khung hình tĩnh**, phù hợp với việc nó nhầm lẫn nặng giữa Screwing-1 và Screwing-2 — hai công đoạn gần như không phân biệt được nếu không theo dõi trình tự thời gian.

> ⚠️ Lưu ý về diễn giải: các chỉ số này mô tả **văn bản model sinh ra**, không phải hành vi nội tại của model. Muốn kết luận chắc chắn vẫn phải chạy static-frame control.

---

## Mục 6: Ví dụ lỗi cụ thể

Tất cả lấy từ 546 video test (không có tập train).

### Lỗi 1 — `Cycle_0/Cycle_0_task_3`
- **Đáp án thật:** 3 — Inflating the valve
- **Model đoán:** 0 — Assembling the spring
- **Confidence model tự khai:** **0.9** | runtime: thinking_too_long, 16.42 s
- **Loại lỗi:** **Model hallucinate + sụp đổ lớp.** Model tự tin 90% vào một nhãn hoàn toàn khác. Kết hợp với việc lớp 3 chỉ được dự đoán 1/449 lần trên toàn bộ dataset → model **không có biểu diễn nào cho công đoạn bơm van**, nó thay bằng công đoạn quen thuộc hơn.

### Lỗi 2 — `Cycle_10/Cycle_10_task_3`
- **Đáp án thật:** 3 — Inflating the valve | **Model đoán:** 1 — Placing white plastic
- **Confidence:** 0.6 | runtime: normal (14.29 s — chạy trọn vẹn, không timeout)
- **Loại lỗi:** **Model hallucinate.** Đây là ca chạy *bình thường*, không bị cắt, model có đủ thời gian và vẫn sai. Chứng minh lỗi ở lớp 3 **không phải do timeout** mà do bản thân năng lực nhận dạng.

### Lỗi 3 — `Cycle_0/Cycle_0_task_0`
- **Đáp án thật:** 0 — Assembling the spring | **Model đoán:** 2 — Screwing-1
- **Confidence:** 0.9
- **Loại lỗi:** **Nhầm lẫn giữa 2 nhãn quá giống nhau + thiên lệch về lớp trội.** Lắp lò xo và vặn vít đều là thao tác tay nhỏ trên cùng cụm chi tiết. Screwing-1 là nhãn model dự đoán nhiều nhất (34.97%) → model rơi về lớp mặc định khi không chắc.

### Lỗi 4 — `Cycle_1/Cycle_1_task_6`
- **Đáp án thật:** 6 — Fixing cable | **Model đoán:** *không có output* (0 ký tự)
- **Trạng thái:** `timeout_no_output`
- **Loại lỗi:** **Thiếu thông tin đầu ra — giới hạn hạ tầng.** Model không trả lời gì. Trong chấm end-to-end ca này tính là sai. Có 16 ca như vậy.

### Lỗi 5 — `Cycle_0/Cycle_0_task_5`
- **Đáp án thật:** 5 — Screwing-2 | **Model đoán:** bị cắt giữa chừng sau 4,882 ký tự
- **Trạng thái:** `partial_timeout`
- **Loại lỗi:** **Thiếu thông tin đầu vào/đầu ra — model "nghĩ" quá dài.** Model viết gần 5,000 ký tự phân tích cho một video 5.9 giây rồi bị cắt trước khi kết luận. Có 405 ca partial_timeout (74.2%).

### Lỗi 6 — `Cycle_11/Cycle_11_task_0`
- **Đáp án thật:** 0 — Assembling the spring | **Model đoán:** 4 — Placing black plastic
- **Confidence:** 0.6
- **Loại lỗi:** **Nhầm lẫn giữa 2 nhãn quá giống nhau.** Cả hai đều là thao tác đặt/lắp chi tiết nhỏ; phân biệt được phải nhìn *vật thể nào* đang cầm — chi tiết dễ mất ở độ phân giải sau nén.

### Ma trận nhầm lẫn — các cặp nhầm nhiều nhất

| Thật → Đoán | Số lần |
|---|---|
| 0 (Assembling spring) → 1 (White plastic) | 21 |
| 5 (Screwing-2) → 1 (White plastic) | 16 |
| 5 (Screwing-2) → 2 (Screwing-1) | 14 |
| 2 (Screwing-1) → 1 (White plastic) | 13 |
| 0 (Assembling spring) → 2 (Screwing-1) | 10 |

Cặp `Screwing-2 → Screwing-1` (14 lần) là bằng chứng trực tiếp: hai công đoạn chỉ khác nhau ở **vị trí trong trình tự thời gian**, và model không phân biệt được.

---

## Mục 7: Kết luận cuối cùng

**Dataset: HATREC (546 video, phân loại 7 công đoạn)**

- ☑ **Model này THẮNG RÕ RÀNG so với baseline đơn giản** — Macro-F1 0.2162 vs 0.0357 (gấp 6.1×); accuracy 23.63% vs 14.29%; CI 95% [20.15%, 27.29%] không chứa mốc baseline.
- ☐ Model này NGANG BẰNG baseline đơn giản
- ☐ Model này THUA baseline đơn giản

> ⚠️ **"Thắng baseline" ở đây KHÔNG có nghĩa là "dùng được".** Model sai 76% số lần, hỏng hoàn toàn 2/7 lớp, tự tin 90% khi đoán sai, và 77% số lần chạy bị timeout. Về mặt vận hành đây là **không khả dụng**. Ô đánh dấu ở trên chỉ phản ánh việc model có tín hiệu thị giác thật vượt trên mức đoán mò — không phải khuyến nghị triển khai.

---

## Mục 8: Lưu ý / giới hạn

**Mẫu test có đủ lớn không?**
Có, ở mức tổng: 546 video, phủ 100% dataset, các lớp cân bằng tuyệt đối. Ở mức từng lớp thì 78 mẫu/lớp là chấp nhận được nhưng khoảng tin cậy per-class khá rộng. Điểm yếu thật sự không phải cỡ mẫu mà là **chỉ 82.2% output parse được**.

**Có phát hiện leak không? Kết quả Mục 4 có đáng tin?**
Không phát hiện leak. Tên file đã trung tính hoá và verify, nhãn không lọt vào prompt, model chưa từng huấn luyện trên HATREC. Kết quả Mục 4 **đáng tin** trong phạm vi đã đo. Cảnh báo duy nhất: 17.8% output không parse được có thể không phân bố ngẫu nhiên (lớp 5 Screwing-2 có tới 6 ca timeout so với 1 ca của lớp 0), nên con số per-class có thể lệch nhẹ.

**Có điều gì chưa kiểm tra mà làm thêm sẽ chắc chắn hơn?**

1. **Static-frame control** — bắt buộc, và là việc còn thiếu quan trọng nhất. Với temporal evidence ratio chỉ 0.176, khả năng cao model không dùng chuyển động. Cần biết chắc trước khi kết luận Cosmos "hiểu video".
2. **Temporal shuffle control** — đảo thứ tự frame. Nếu điểm không đổi → xác nhận model bỏ qua thông tin thời gian.
3. **Điều tra lớp 3 (Inflating the valve)** — 0/78 và chỉ được dự đoán 1 lần. Cần kiểm tra xem đây là do prompt mô tả nhãn không rõ, hay do model thật sự không nhận ra công đoạn này. Sửa mô tả nhãn trong prompt rồi chạy lại 78 video của lớp này là thí nghiệm rẻ.
4. **Giảm timeout** — 74.2% partial_timeout đang bóp méo kết quả. Nên thêm chỉ thị giới hạn độ dài suy luận, hoặc tăng timeout, rồi chạy lại.
5. **Chưa so với model video chuyên dụng nào khác trên cùng split** ngoài V-JEPA 2 (vốn có setup khác hẳn — có huấn luyện probe). Chưa có so sánh zero-shot ↔ zero-shot công bằng.
6. **Chưa có human baseline** — video chỉ 3.5 giây, cần biết người thường phân biệt được 7 công đoạn này chính xác đến đâu.

---

## Nguồn dữ liệu và mã nguồn

| Loại | Đường dẫn |
|---|---|
| Kết quả từng video | `research/hatrec_cosmos3/outputs/analysis/hatrec_cosmos3_per_video.csv` |
| Tổng hợp chấm sâu | `.../outputs/analysis/deep_audit_summary.json` |
| Chỉ số per-class | `.../outputs/analysis/deep_per_class_metrics.csv` |
| Phân loại lỗi | `.../outputs/analysis/deep_error_taxonomy.csv` |
| Phân tích ngữ nghĩa | `.../outputs/analysis/semantic_content_summary.json`, `semantic_cue_confusion.csv` |
| Báo cáo thô từng video | `.../outputs/ui_reports/Cycle_*/` (78 thư mục) |
| Log chạy | `.../outputs/ui_run_546.log` |
| Runner | `research/hatrec_cosmos3/run_ui.py` |
| Audit dataset | `.../outputs/dataset_audit.json` |
| Notebook phân tích | `research/hatrec_cosmos3/HATRec_Cosmos3_Insight_Analysis.ipynb` |

**Hình nên gắn kèm khi trình bày:**
`outputs/analysis/deep_confusion_matrices.png`, `outputs/analysis/per_class_prf1.png`, `outputs/analysis/confidence_calibration.png`, `outputs/analysis/completion_status.png`.
