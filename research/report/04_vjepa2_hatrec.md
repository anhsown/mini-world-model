# Report — V-JEPA 2 ViT-L × HATREC

> 🚨 **Cảnh báo đọc trước:** model đạt **100% (84/84)** trên tập test. Con số này **không được dùng để khẳng định năng lực**. Hai phép kiểm soát bắt buộc để loại trừ shortcut tĩnh vẫn **chưa chạy**, và độ tương đồng thị giác test↔train là **0.9922**. Báo cáo này ghi nhận kết quả đồng thời giải thích vì sao chưa tin được.

## Mục 1: Thông tin cơ bản

| Mục | Nội dung |
|---|---|
| **Tên model test** | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), ~300M params |
| **Dataset test trên** | HATREC — video lắp ráp công nghiệp, phân loại 7 công đoạn |
| **Quy mô** | 546 video / 78 cycle / 7 lớp cân bằng |
| **Chia tập** | 54 cycle train / 12 cycle val / 12 cycle test → 378 / 84 / 84 dòng |
| **Ngày test** | 2026-07-30 |
| **Người test** | Sơn |

---

## Mục 2: Setup

| Mục | Nội dung |
|---|---|
| **Chạy ở đâu** | Kaggle, GPU **T4×2** (notebook assert đúng 2 GPU) |
| **Chi phí** | 0 đ — free tier, không thuê server |
| **Bản model** | V-JEPA 2 ViT-L 300M, checkpoint `facebook/vjepa2-vitl-fpc64-256` |
| **Số frame** | 64 frame/clip (cấu hình chính); có ablation 14 frame |
| **Giải mã video** | `decord` (cài trong notebook), `transformers>=5.0.0` |
| **Notebook** | `research/hatrec_cosmos3/VJEPA2_HATREC_LeakageSafe.ipynb` |

### ⚠️ ĐÂY KHÔNG PHẢI ZERO-SHOT

| Mục | Nội dung |
|---|---|
| **Zero-shot?** | **KHÔNG** |
| **Có train gì không?** | **CÓ — linear probe.** `sklearn.linear_model.LogisticRegression`, C = 0.01, huấn luyện trên embedding của 378 clip thuộc 54 cycle train |
| **Encoder** | **Đóng băng** — V-JEPA 2 không được cập nhật trọng số |
| **Cái được huấn luyện** | Chỉ lớp phân loại tuyến tính phía trên embedding |

Ghi rõ điều này vì V-JEPA 2 về bản chất là **encoder biểu diễn/dự đoán**, không phải model ngôn ngữ. Nó **không tự sinh được câu trả lời trắc nghiệm hay lời giải thích tự do** — muốn ra nhãn thì bắt buộc phải gắn probe. Vì vậy kết quả ở đây **không so trực tiếp được** với các con số zero-shot của Cosmos 3 hay Qwen2-VL trên cùng dataset.

**Ánh xạ năng lực đã khai báo trong notebook:**
- **B1** (phân loại trạng thái/công đoạn) — có claim, metric chính = macro-F1
- **B2** — *không* claim, vì các clip đã được cắt sẵn theo công đoạn
- **B10** — *không* claim, vì cycle giữ lại chỉ là kiểm soát tổng quát hoá, không phải tập OOD thật để đo khả năng abstain

---

## Mục 3: Kiểm tra leak

Đây là model **có huấn luyện**, nên mục này áp dụng đầy đủ và là phần quan trọng nhất của báo cáo.

| Kiểm tra | Giá trị | Trạng thái |
|---|---|---|
| **Cycle nào trùng giữa train và test không** | Chia theo cycle: 54 / 12 / 12, **không cycle nào xuất hiện ở hai tập** | ✅ PASS |
| **Tên file có lộ nhãn không** | Đã trung tính hoá **ở mức vật lý** (`physical_filename_neutralization: true`) | ✅ PASS |
| **Trùng lặp byte tuyệt đối giữa các video** | 0 | ✅ PASS |
| **Số mẫu có cosine > 0.999** | **0** | ✅ PASS |
| **Cosine trung bình test → train gần nhất** | **0.9922** | ⚠️ **REVIEW** |
| **Cosine p95 test → train gần nhất** | **0.9959** | ⚠️ **REVIEW** |

### Kết luận Mục 3: dữ liệu **KHÔNG đủ độc lập** để tin kết quả

Về mặt **quy trình**, mọi kiểm soát đều pass — không cycle trùng, không file lộ nhãn, không trùng byte, không mẫu nào vượt ngưỡng gần-trùng 0.999.

Nhưng về mặt **nội dung**, cosine trung bình 0.9922 nghĩa là mỗi clip test có một clip train gần như đồng nhất trong không gian biểu diễn. Cả 546 video được quay ở **cùng một trạm làm việc, cùng bố cục, cùng bộ dụng cụ, cùng thao tác lặp lại**. Việc chia theo cycle ngăn được leak *chính xác cùng một clip*, nhưng **không tạo ra được sự đa dạng thật** giữa train và test.

Nói cách khác: chia tập là hợp lệ, còn dataset thì không đủ đa dạng để phép chia đó có ý nghĩa. 100% ở đây có thể chỉ chứng minh rằng **7 công đoạn tách tuyến tính được trong không gian embedding của V-JEPA 2** — chứ không chứng minh model hiểu hành động hay chuyển động.

---

## Mục 4: Kết quả chính

Dataset cân bằng (7 lớp), chance = 14.29%. Metric chính: **Macro-F1**.

### 4.1 Kết quả tổng

| | Accuracy | Macro-F1 | n |
|---|---|---|---|
| **V-JEPA 2 ViT-L + linear probe** | **100.00%** (Wilson CI 95.63–100.00) | **1.0000** | 84 |
| Baseline majority-class (blind) | 14.29% | **0.0357** | 84 |
| Baseline nhãn xáo trộn (shuffled labels) | 9.52% | 0.0995 | 84 |
| Baseline random-choice | 14.29% | — | 84 |
| **Chênh lệch vs majority** | **+85.71 điểm** | **+0.9643** | |

Cluster bootstrap macro-F1 CI 95% (bootstrap theo cycle, không theo dòng): **[1.0, 1.0]**.

Cả hai negative control đều rơi về mức ngẫu nhiên → **pipeline chấm điểm không có lỗi**. Con số 100% là thật trong khuôn khổ thí nghiệm này, không phải bug đánh giá.

### 4.2 Theo từng lớp

| Nhãn | Công đoạn | Support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0 | Assembling spring | 12 | 1.0 | 1.0 | 1.0 |
| 1 | White plastic | 12 | 1.0 | 1.0 | 1.0 |
| 2 | Screwing-1 | 12 | 1.0 | 1.0 | 1.0 |
| 3 | Inflating valve | 12 | 1.0 | 1.0 | 1.0 |
| 4 | Black plastic | 12 | 1.0 | 1.0 | 1.0 |
| 5 | Screwing-2 | 12 | 1.0 | 1.0 | 1.0 |
| 6 | Fixing cable | 12 | 1.0 | 1.0 | 1.0 |

Đáng chú ý: **Screwing-1 và Screwing-2 phân biệt hoàn hảo** — đúng cặp mà Cosmos 3 nhầm nhiều nhất (14 lần). Cần giải thích: hai công đoạn này khác nhau về vị trí trong trình tự, nhưng cũng có thể khác nhau về chi tiết tĩnh (vị trí tay, trạng thái cụm lắp ráp tại thời điểm đó) — mà probe tuyến tính hoàn toàn có thể bắt được từ đặc trưng tĩnh.

### 4.3 Các cấu hình thí nghiệm

| Cấu hình | Loại | Accuracy | Macro-F1 |
|---|---|---|---|
| Cycle holdout — 64 frame | **chính** | 1.0 | 1.0 |
| Cycle holdout — 14 frame nhìn thấy | ablation | **1.0** | **1.0** |
| Random clip — 64 frame | so sánh rủi ro leak | 1.0 | 1.0 |
| Majority blind | negative control | 0.1429 | 0.0357 |
| Nhãn xáo trộn | negative control | 0.0952 | 0.0995 |

> ⚠️ **Ablation 14 frame cũng đạt 100%.** Giảm từ 64 xuống 14 frame — tức bỏ đi 78% thông tin thời gian — **không làm giảm điểm chút nào**. Đây là tín hiệu cảnh báo mạnh về shortcut tĩnh, xem Mục 5.

### 4.4 Hiệu chỉnh độ tin cậy

| Chỉ số | Giá trị |
|---|---|
| ECE | 0.0185 |
| Multiclass Brier | 0.00086 |

Hiệu chỉnh rất tốt — nhưng trên một bài toán mà model đúng 100% thì chỉ số này gần như không mang thông tin.

---

## Mục 5: Kiểm tra shortcut

HATREC là **video** → mục này **có áp dụng**, và đây là chỗ báo cáo thất bại.

| Kiểm tra | Trạng thái | Ghi chú trong file chẩn đoán |
|---|---|---|
| **Static-frame (1 frame lặp lại 64 lần)** | ❌ **CHƯA LÀM** | `MISSING — Required to separate static cues from motion` |
| **Temporal shuffle / đảo thứ tự frame** | ❌ **CHƯA LÀM** | `MISSING — Required to test order sensitivity` |
| Che vật thể / dụng cụ, can thiệp nền | ❌ **CHƯA LÀM** | |
| Giữ riêng operator/trạm/sản phẩm | ❌ **CHƯA LÀM** | Chưa có metadata |
| Chênh lệch 64 frame vs 14 frame | ⚠️ **INCONCLUSIVE** | `0.0 — No degradation with 14 visible frames` |

**Chưa có số static-frame nên không áp được tiêu chí <5% / >20–30% của template.**

Nhưng bằng chứng gián tiếp đã đủ để nghiêng về giả thuyết shortcut:

1. **Ablation 14 frame = 100%, y hệt 64 frame.** Theo đúng logic của template: nếu giảm mạnh thông tin thời gian mà điểm không đổi thì "model có thể đang đoán dựa trên đặc điểm hình ảnh tĩnh, không phải chuyển động thật". Đây chưa phải static-frame test nhưng đi cùng hướng.
2. **Cosine test→train 0.9922.** Mỗi clip test có bản gần-song sinh trong tập train.
3. **Random clip cũng đạt 100%** — cách chọn đoạn clip không ảnh hưởng gì.

> **Đánh giá:** ba dấu hiệu cùng chỉ về một hướng — kết quả 100% nhiều khả năng đến từ **đặc trưng tĩnh của khung hình** (bố cục bàn, vật thể đang có mặt, trạng thái cụm lắp ráp), chứ không phải từ hiểu chuyển động. Chỉ static-frame control mới kết luận được dứt điểm.

---

## Mục 6: Ví dụ lỗi cụ thể

**Model đúng 84/84 — không có ca sai nào để liệt kê.**

Việc không thể hoàn thành mục này **chính là phát hiện quan trọng nhất của báo cáo**. Một benchmark mà model đạt trần tuyệt đối thì không còn khả năng phân biệt model tốt với model kém, và không cung cấp được thông tin gì để cải thiện.

Thay vào đó, dưới đây là **5 thiếu sót của chính thí nghiệm** — đóng vai trò tương đương "danh sách lỗi" cho mục này:

### Thiếu sót 1 — Không có static-frame control
- **Cần làm gì:** lấy 1 frame giữa clip, lặp lại thành 64 frame, chạy lại probe.
- **Kỳ vọng phân biệt:** nếu vẫn ≈100% → xác nhận shortcut tĩnh, kết quả hiện tại vô hiệu. Nếu tụt >20–30% → model thật sự dùng chuyển động.
- **Vì sao quan trọng nhất:** đây là phép kiểm duy nhất tách được hai giả thuyết đang cạnh tranh.

### Thiếu sót 2 — Không có temporal shuffle control
- **Cần làm gì:** xáo trộn/đảo ngược thứ tự frame rồi chạy lại.
- **Kỳ vọng:** nếu điểm không đổi → model bỏ qua hoàn toàn chiều thời gian.

### Thiếu sót 3 — Tương đồng test↔train 0.9922 chưa được xử lý
- **Cần làm gì:** chia tập theo operator / trạm / ngày quay thay vì chỉ theo cycle, hoặc kiểm thử trên dây chuyền khác.
- **Vấn đề:** chia theo cycle không tạo được sự khác biệt phân phối thật.

### Thiếu sót 4 — Không lưu xác suất từng mẫu
- **Cần làm gì:** lưu `predict_proba` cho từng clip.
- **Vấn đề:** không có nó thì không vẽ được reliability diagram, không phân tích được abstention — vốn là yêu cầu của B10 (mà notebook đã đúng khi *không* claim B10).

### Thiếu sót 5 — Không có ablation không-chuyển-động ở mức đặc trưng
- **Cần làm gì:** che vật thể/dụng cụ hoặc thay nền, giữ nguyên chuyển động.
- **Vấn đề:** chưa biết model bám vào vật thể hay bám vào hành động.

---

## Mục 7: Kết luận cuối cùng

**Dataset: HATREC (probe tuyến tính trên 12 cycle giữ lại, n = 84)**

Theo đúng con số:
- ☑ **Model này THẮNG RÕ RÀNG so với baseline đơn giản** — Macro-F1 1.0 vs 0.0357; accuracy 100% vs 14.29%.
- ☐ NGANG BẰNG
- ☐ THUA

**Nhưng kết luận về hiệu lực (đè lên kết luận trên):**

> 🚨 **KẾT QUẢ KHÔNG TIN CẬY ĐƯỢC — không dùng để claim năng lực.**
>
> Ba lý do: (1) thiếu static-frame control, (2) ablation 14 frame cho gap 0.0, (3) cosine test→train 0.9922.
>
> Điều thí nghiệm này chứng minh được: **7 công đoạn HATREC tách tuyến tính được trong không gian biểu diễn của V-JEPA 2**.
> Điều thí nghiệm này **không** chứng minh được: V-JEPA 2 hiểu chuyển động, hiểu trình tự thời gian, hay tổng quát hoá sang dây chuyền khác.

Mọi con số HATREC báo cáo ra ngoài từ nay phải kèm ba điều kiện: tên file đã trung tính hoá, chia tập theo cycle/participant, **và có static-frame control**. Hai điều kiện đầu đã đạt; điều kiện thứ ba chưa.

---

## Mục 8: Lưu ý / giới hạn

**Mẫu test có đủ lớn không?**
**Hơi nhỏ.** 84 clip từ 12 cycle. Wilson CI cận dưới là 95.63% — nghĩa là kể cả với 84/84 đúng, thống kê chỉ đảm bảo được "ít nhất 95.6%". Quan trọng hơn: 12 cycle là **12 đơn vị độc lập thật sự**, không phải 84. Cluster bootstrap đã tính đúng theo cycle và vẫn ra [1.0, 1.0], nhưng nền tảng vẫn chỉ là 12 cụm.

**Có phát hiện leak không? Kết quả Mục 4 có đáng tin?**
Leak **theo quy trình**: không — mọi kiểm soát pass. Leak **theo nội dung**: có vấn đề — cosine 0.9922 cho thấy test và train gần như cùng phân phối thị giác.
**Kết quả Mục 4 KHÔNG đáng tin** như một tuyên bố về năng lực. Nó đáng tin như một tuyên bố hẹp hơn: "với dữ liệu cùng trạm làm việc này, probe tuyến tính trên embedding V-JEPA 2 phân loại được 7 công đoạn".

**Có điều gì chưa kiểm tra mà làm thêm sẽ chắc chắn hơn?**

1. **Static-frame control** — ưu tiên tuyệt đối. Rẻ, nhanh, và quyết định toàn bộ hiệu lực của báo cáo này.
2. **Temporal shuffle control** — ưu tiên hai, cùng lý do.
3. **Che vật thể / can thiệp nền** — tách "nhận ra dụng cụ" khỏi "nhận ra hành động".
4. **Chia tập theo operator/trạm/sản phẩm** — cần metadata, hiện chưa có.
5. **Lưu xác suất từng mẫu** — để làm reliability diagram và phân tích abstention.
6. **Cần một benchmark khó hơn.** HATREC ở trần 100% không còn giá trị phân biệt. Muốn đo năng lực video thật cần dataset có đa dạng trạm/người/sản phẩm, hoặc chuyển sang bài toán chưa cắt sẵn (phát hiện ranh giới công đoạn — chính là B2 mà notebook đã đúng khi không claim).

---

## Nguồn dữ liệu và mã nguồn

| Loại | Đường dẫn |
|---|---|
| Metrics | `research/hatrec_cosmos3/vjepa2_hatrec_results/hatrec_vjepa2_metrics.json` |
| Dự đoán | `.../vjepa2_hatrec_results/hatrec_vjepa2_predictions.csv` |
| Kết luận nghiên cứu | `.../vjepa2_hatrec_results/deep_analysis/VJEPA2_HATREC_RESEARCH_VERDICT.md` |
| Chẩn đoán leak/shortcut | `.../deep_analysis/leakage_shortcut_diagnostics.csv` |
| Báo cáo per-class | `.../deep_analysis/per_class_report.csv` |
| Khoảng tin cậy | `.../deep_analysis/uncertainty_intervals.csv` |
| Notebook chạy | `research/hatrec_cosmos3/VJEPA2_HATREC_LeakageSafe.ipynb` |
| Notebook phân tích | `.../vjepa2_hatrec_results/VJEPA2_HATREC_Deep_Analysis.executed.ipynb` |

**Hình nên gắn kèm khi trình bày:**
`deep_analysis/01_results_and_controls.png`, `deep_analysis/03_similarity_audit.png` (hình quan trọng nhất — thể hiện phân bố cosine 0.99), `hatrec_vjepa2_analysis.png`.
