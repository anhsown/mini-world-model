# Báo cáo Phase 1 — Chỉ mục

**Người test:** Sơn · **Cập nhật:** 2026-08-01 · **Repo:** `https://github.com/anhsown/mini-world-model`

Mỗi cặp (model × dataset) một báo cáo riêng, viết theo cấu trúc 8 mục thống nhất của team.

| # | Báo cáo | Model | Dataset | Độ phủ |
|---|---|---|---|---|
| 01 | [Qwen2-VL-2B × MMAD](01_qwen2vl_2b_mmad.md) | Qwen2-VL-2B-Instruct | MMAD | 39,670/39,670 (**100%**) |
| 02 | [Cosmos 3 Nano × MMAD](02_cosmos3_nano_mmad.md) | Cosmos 3 Nano Reasoner | MMAD | 622/39,670 (**1.57%**) |
| 03 | [Cosmos 3 Nano × HATREC](03_cosmos3_nano_hatrec.md) | Cosmos 3 Nano Reasoner | HATREC | 546/546 (**100%**) |
| 04 | [V-JEPA 2 ViT-L × HATREC](04_vjepa2_hatrec.md) | V-JEPA 2 ViT-L 300M | HATREC | 84 clip test (12 cycle) |

---

## Bảng tổng hợp chung

| Model | Dataset | Metric chính | Baseline cùng dữ liệu | Thắng/Ngang/Thua | Shortcut / Leak? |
|---|---|---|---|---|---|
| **Qwen2-VL-2B** | MMAD (đủ 4 nguồn) | Macro-F1 **0.650** · micro 64.73% | majority-letter 30.38% (F1 0.117) | **THẮNG rõ ràng** | Không leak protocol. **Chưa test blind-image** |
| **Qwen2-VL-2B** | MMAD — *riêng subtask Anomaly Detection* | F1 **0.584** · recall 0.481 | "luôn nói có lỗi" F1 **0.756** | **THUA** | Miss rate 51.9% |
| **Cosmos3-Nano** (BNB8, Kaggle T4×2) | MMAD (380 câu, **chỉ MVTec-AD**) | Macro-F1 **0.746** · micro 73.16% | majority-letter 35.53% | **THẮNG rõ ràng** | Không leak. **Thiên lệch độ phủ nặng** |
| **Cosmos3-Nano** (endpoint chính thức) | MMAD (221 câu, **chỉ MVTec-AD**) | Macro-F1 **0.496** · micro 49.77% | majority-letter **51.13%** | **THUA** | như trên |
| **Cosmos3-Nano** | MMAD — *riêng subtask Anomaly Detection* | F1 **0.250** · recall 0.143 | "luôn nói có lỗi" F1 **0.830** | **THUA nặng** | Miss rate 85.7% |
| **Cosmos3-Nano** | HATREC (classification) | Macro-F1 **0.216** · acc 23.63% | majority-class F1 0.036 | **THẮNG** (nhưng không dùng được) | Không leak. **Chưa test static-frame**; bằng chứng thời gian chỉ 17.6% |
| **V-JEPA 2 ViT-L** (⚠️ có probe, **không zero-shot**) | HATREC | Macro-F1 **1.0** · acc 100% | majority blind F1 0.036 | Thắng về số — **KHÔNG TIN CẬY ĐƯỢC** | Leak quy trình PASS, nhưng **similarity 0.9922** + **thiếu static-frame** |
| GPT-4o *(tham chiếu từ paper MMAD)* | MMAD | ~74.9% | — | Cao hơn nhiều model open-source | Paper có human baseline để đối chiếu |

---

## Ba kết luận cần nhấn mạnh khi trình bày

### 1. So sánh Cosmos vs Qwen trên MMAD phải khớp nguồn dữ liệu

Toàn bộ 601 record parse-valid của Cosmos đều thuộc **MVTec-AD** — đúng nguồn dễ nhất trong 4 nguồn của MMAD.

| So sánh | Cosmos 3 Nano | Qwen2-VL-2B | Kết quả |
|---|---|---|---|
| ❌ Sai — Cosmos (MVTec-AD) vs Qwen (cả 4 nguồn) | 73.16% | 64.73% | "Cosmos hơn 8.4 điểm" |
| ✅ Đúng — cả hai trên MVTec-AD | 73.16% | **75.25%** | **Qwen hơn 2.1 điểm** |

Bảng đặt 73.16% cạnh 64.73% là **so sánh không hợp lệ** và đảo ngược kết luận. Chênh giữa nguồn dễ nhất (MVTec-AD 75.25%) và khó nhất (MVTec-LOCO 55.22%) lên tới 20 điểm.

### 2. Cả hai VLM đều thất bại đúng ở chỗ quan trọng nhất

Trên subtask **Anomaly Detection** — sát nhất với bài toán B3 của công ty — cả hai đều **thua baseline ngu**:

| Model | Recall | F1 | Baseline F1 | Miss rate |
|---|---|---|---|---|
| Qwen2-VL-2B | 0.481 | 0.584 | 0.756 | 51.9% |
| Cosmos 3 Nano | 0.143 | 0.250 | 0.830 | **85.7%** |

Cả hai đều thiên lệch mạnh về phía "không có lỗi". Accuracy tổng che mất điều này — đúng lý do template cấm dùng accuracy làm metric chính cho dữ liệu mất cân bằng.

### 3. Kết quả 100% của V-JEPA 2 chưa được phép công bố như một thành tựu

Ablation 14 frame cũng cho 100% (gap 0.0), cosine test→train 0.9922, và static-frame control **chưa chạy**. Điều chứng minh được chỉ là: 7 công đoạn HATREC **tách tuyến tính được** trong không gian embedding của V-JEPA 2.

---

## Còn thiếu — chưa có báo cáo

| Model × Dataset | Trạng thái | Ghi chú |
|---|---|---|
| **Cosmos3-Nano × PIADE/ALPI (B0 + B5)** | ❌ **0%** | **Đầu việc bắt buộc, ưu tiên cao nhất** theo handoff §12. Chưa có dữ liệu, script hay thư mục nào trong workspace |
| Qwen2-VL-2B × HATREC | ❌ Không có dữ liệu trên máy | Con số 15.02% trong bảng team là của người khác |
| Cosmos3-Nano × MMAD — 3 nguồn còn lại | ❌ Chưa chạy | GoodsAD, VisA, MVTec-LOCO đều n = 0 |

---

## Việc bắt buộc phải bổ sung cho các báo cáo đã có

Xếp theo tỉ lệ giá trị/chi phí:

1. **Static-frame control cho cả hai báo cáo HATREC** (03, 04) — rẻ, nhanh, quyết định hiệu lực của cả hai. Với V-JEPA đây là điều kiện tiên quyết để con số 100% có ý nghĩa.
2. **Chạy 100 câu MMAD giống hệt nhau trên cả hai backend Cosmos** — để giải thích chênh lệch 73.16% vs 49.77%. Chưa hiểu nguyên nhân thì mọi record gộp về sau đều mang lỗi hệ thống.
3. **Blind-image control trên ~500 câu MMAD** — cho cả Qwen và Cosmos, xác nhận model thật sự dùng ảnh.
4. **Mở rộng Cosmos sang MVTec-LOCO** — nguồn khó nhất, cho cận dưới thực tế.
5. **Lưu probability outputs** ở mọi lần chạy sau — hiện chưa report được ECE/abstention cho hai model MMAD.
6. **Temporal shuffle control** cho HATREC.

---

## Ghi chú về quy trình

- Mọi baseline trong các báo cáo này được **tính lại bằng code trên đúng tập record mà model đã trả lời**, không mượn từ nguồn khác và không lấy từ paper.
- Metric chính là **Macro-F1** ở mọi nơi dữ liệu mất cân bằng; accuracy chỉ báo kèm khi đó là protocol công bố của dataset (MMAD).
- Theo lưu ý chung của team: **chờ khung codebase chung từ Quỳnh Anh** trước khi viết code cho các lần chạy tiếp theo, để định dạng kết quả thống nhất và gộp được. Bốn báo cáo này viết từ dữ liệu **đã thu thập xong**, không phát sinh code chạy model mới.
- Chi phí đến thời điểm này: **0 đ** — toàn bộ chạy trên Kaggle free tier và NVIDIA Build free tier, không thuê server.
