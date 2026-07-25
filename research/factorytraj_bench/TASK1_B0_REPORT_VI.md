# Báo cáo Task 1 — FactoryTraj-B0 Schema and Tag Understanding

## 1. Tóm tắt điều hành

Task 1 xây dựng và kiểm định **FactoryTraj-B0 — Schema and Tag
Understanding**, benchmark đầu tiên cho chương trình Industrial Machine
Understanding của Celesnity.

Mục tiêu của B0 là xác định liệu một model có thể hiểu hợp đồng dữ liệu của
machine tags hay chỉ suy đoán từ tên biến. Với mỗi tag, model phải khôi phục:

- kiểu dữ liệu;
- engineering unit;
- InstrumentRange hoặc EURange khi có;
- vai trò vận hành;
- quan hệ với tag hoặc thành phần máy khác;
- confidence của dự đoán.

Kết quả cuối:

| Hạng mục | Kết quả |
|---|---:|
| Tổng số records | 244 |
| Train | 82 |
| Validation/test | 162 |
| Source families | 10 |
| Unit coverage | 76,5% |
| Authoritative range records | 89 |
| Authoritative range coverage | 54,9% |
| Relationship coverage | 37,7% |
| Dataset-admission gates | 6/6 pass |
| Task-contract validation | 11/11 pass |
| Automated tests | 7/7 pass |

Benchmark B0 đã được admit và model pass criteria đã được đóng băng. Tuy
nhiên, điều này **không có nghĩa JWM hiện tại đã pass B0**. JWM chưa có
structured industrial-tag adapter để tạo output đúng contract.

---

## 2. Bối cảnh và mục tiêu

Industrial machine understanding cần bắt đầu từ việc hiểu machine schema. Một
sensor có thể được đặt tên rõ ràng:

```text
reactor_pressure
```

nhưng trong máy thật cũng có thể xuất hiện dưới dạng:

```text
PT_101
AI_0042
ns=4;s=Machine1.Tag27
```

Nếu model chỉ dựa vào các prefix như `setpoint_`, `feedback_`, `effort_` hoặc
`ctx_`, nó có thể đạt điểm cao trên một dataset quen thuộc nhưng không thể
generalize sang PLC, vendor hoặc nhà máy mới.

B0 được thiết kế để trả lời câu hỏi:

> Model có thể khôi phục ý nghĩa của tag từ data type, unit, samples,
> documentation và topology hay chỉ nhớ tên tag?

B0 là lớp nền cho chuỗi capability tiếp theo:

```text
B0: hiểu schema và tag
  → B1: ước lượng trạng thái máy
  → B2: phát hiện sự kiện
  → B3: dự đoán trạng thái tương lai
  → B4–B5: hiểu nguyên nhân và tác động
  → B6+: xếp hạng hành động an toàn
```

Nếu model chưa hiểu tag nào là sensor, setpoint hoặc actuator, những kết luận
về anomaly, dynamics và action sẽ không đáng tin cậy.

---

## 3. Định nghĩa vận hành của B0

### 3.1 Input

Mỗi B0 record có thể cung cấp:

- tag name hoặc anonymized tag ID;
- declared data type;
- representative samples và validity mask;
- engineering unit nếu nguồn có cung cấp;
- InstrumentRange hoặc EURange nếu nguồn có cung cấp;
- partial tag documentation;
- topology hoặc quan hệ với tag khác.

### 3.2 Output

Model phải trả về structured contract:

```json
{
  "tag_id": "PT_101",
  "data_type": "float64",
  "engineering_unit": "kPa",
  "range": {
    "low": 0,
    "high": 3000
  },
  "role": "sensor_feedback",
  "relationships": [
    {
      "relation": "component_of",
      "target_tag_id": "reactor"
    }
  ],
  "confidence": 0.92
}
```

### 3.3 Role taxonomy

| Role | Ý nghĩa |
|---|---|
| `sensor_feedback` | Cảm biến hoặc trạng thái thực tế được đo |
| `control_setpoint` | Lệnh điều khiển hoặc giá trị mục tiêu |
| `actuator_effort` | Lực, torque, current hoặc voltage thực thi |
| `machine_context` | Mode, runtime state, temperature context hoặc metadata |
| `identifier` | ID máy, episode hoặc nguồn dữ liệu |
| `time` | Timestamp hoặc elapsed time |

B0 không phải ordinary QA. Model không chỉ tạo câu giải thích tự nhiên mà phải
trả về contract có thể được chấm tự động và dùng bởi các stage phía sau.

---

## 4. Nguồn dữ liệu

### 4.1 FactoryNet

FactoryNet cung cấp 125 schema tags cho industrial robots theo cấu trúc S-E-F-C:

- **Setpoint:** commanded position, velocity, acceleration và target torque.
- **Effort:** current, voltage, force và torque thực thi.
- **Feedback:** measured position, velocity và machine state.
- **Context:** temperature, robot mode, safety mode, runtime state và anomaly
  context.

FactoryNet được dùng để kiểm tra:

- role classification;
- setpoint–feedback correspondence;
- robot joint và Cartesian tags;
- data types và physical quantities.

Hạn chế:

- tag names có prefix rõ và dễ tạo shortcut;
- dataset card không cung cấp đầy đủ authoritative ranges;
- một số unit được suy ra từ physical quantity và vẫn cần source-owner review;
- không nên dùng riêng FactoryNet để kết luận cross-machine generalization.

Nguồn:

- [FactoryNet dataset card](https://huggingface.co/datasets/factorynet/factorynet/blob/main/README.md)
- [FactoryNet schema viewer](https://huggingface.co/datasets/Forgis/FactoryNet)

### 4.2 Tennessee Eastman

Tennessee Eastman được đọc trực tiếp từ simulator source code. B0 sử dụng:

- 73 `XMEAS` measurement variables;
- 12 `XMV` manipulated variables;
- descriptions và engineering units;
- miền `[0,100]` cho XMV control variables;
- miền vật lý `[0,100] mol %` cho composition outputs.

Nguồn này bổ sung process variables khác robot domain:

- flow;
- pressure;
- temperature;
- liquid level;
- composition;
- manipulated control variables.

Trong quá trình audit, một số giá trị đã bị loại khỏi range ground truth:

- `hspan`, `sspan` và `spspan` chỉ là disturbance duration/amplitude;
- one-sided shutdown thresholds chỉ là safety condition;
- observed min–max từ sample không phải physical range.

Chỉ những miền có semantics rõ từ source code hoặc định luật vật lý mới được
chấp nhận.

Nguồn:

- [Tennessee Eastman dataset repository](https://github.com/mv-per/tennessee-eastman-dataset)

### 4.3 OPC Foundation UA NodeSets

Repository NodeSet chính thức của OPC Foundation được dùng để lấy normative
industrial schema metadata từ nhiều domain:

- CNC;
- mining;
- machinery;
- commercial kitchen equipment;
- glass;
- metal forming;
- PADIM/process instrumentation;
- additive manufacturing.

Parser chỉ nhận analog variables có range value thực sự được encode trong XML.
Các field được trích xuất gồm:

- `DataType`;
- `EngineeringUnits`;
- `InstrumentRange`;
- `EURange`;
- `Description`;
- parent/component hierarchy.

Ví dụ:

```text
AsymmetryLoad
EngineeringUnits = %
EURange = [0, 100]
```

```text
CurrentPayload
EngineeringUnits = tonne
EURange = [0, 200]
```

Quan hệ hierarchy được biểu diễn:

```json
{
  "relation": "component_of",
  "target_tag_id": "parent_node_id"
}
```

NodeSets là normative schema metadata chứ chưa phải live machine trajectory.
Nó phù hợp để xây B0 nhưng không thay thế real-machine validation.

Nguồn:

- [OPC Foundation UA NodeSets](https://github.com/OPCFoundation/UA-Nodeset)
- [OPC UA Data Access Part 8](https://reference.opcfoundation.org/Core/Part8/v104/docs/5)

---

## 5. Phân biệt range semantics

### 5.1 InstrumentRange

Miền giá trị vật lý mà instrument có thể trả về.

Ví dụ:

```text
Pressure transmitter: 0–10 bar
```

### 5.2 EURange

Miền engineering hoặc operational được OPC UA server khai báo.

Ví dụ:

```text
Engineering range: 2–8 bar
```

### 5.3 Observed range

Min–max nhìn thấy trong một đoạn samples.

```text
Observed in 60 seconds: 4.1–4.8 bar
```

Observed range được giữ riêng và không được dùng làm authoritative range. Nếu
min–max của một sample ngắn bị xem là physical range thì benchmark sẽ tạo nhãn
sai và model có thể học một distribution shortcut.

---

## 6. Metrics

### 6.1 Data type exact accuracy

Đo tỷ lệ dự đoán chính xác kiểu dữ liệu:

```text
float64
int64
uint8
string
```

### 6.2 Unit exact accuracy

Đo tỷ lệ engineering unit khớp chính xác:

```text
Cel
kPa
A
V
N.m
rad/s
%
```

### 6.3 Role macro-F1

Đo khả năng phân loại role. Macro-F1 cho trọng số công bằng giữa các class,
tránh class phổ biến che khuất class hiếm.

### 6.4 Range score

Normalized range error:

\[
NRE =
\frac{
|\hat{l}-l| + |\hat{h}-h|
}{
2(h-l)
}
\]

Range score:

\[
RangeScore = 1-\min(1,NRE)
\]

Trong đó:

- \(l,h\): authoritative range;
- \(\hat{l},\hat{h}\): range dự đoán.

### 6.5 Relationship macro-F1

Đo độ chính xác của các quan hệ:

```text
commands
tracks
component_of
```

### 6.6 B0 contract macro score

B0 macro là trung bình các component có ground truth hợp lệ. Tuy nhiên, model
không được pass chỉ dựa trên macro average. Mỗi component đều có threshold
riêng để một metric cao không che một metric thất bại.

---

## 7. Phát hiện name shortcut

Seed benchmark ban đầu chủ yếu dựa trên FactoryNet. Rule baseline đạt:

- full tag-name B0 macro khoảng `0.521`;
- anonymized B0 macro khoảng `0.202`;
- role macro-F1 giảm từ khoảng `0.621` xuống `0.012`.

Nguyên nhân là prefix tiết lộ trực tiếp target:

```text
setpoint_ → control_setpoint
feedback_ → sensor_feedback
effort_   → actuator_effort
ctx_      → machine_context
```

Nếu chỉ báo cáo full-name result, benchmark có thể kết luận sai rằng schema
understanding đã được giải quyết.

---

## 8. Shortcut controls

### 8.1 Full-name control

Baseline được nhìn thấy tag name đầy đủ. Đây là upper control cho rule parser.

### 8.2 Anonymized-name control

Tag name được thay bằng:

```text
tag_0001
tag_0002
```

Không có documentation. Control này đo độ phụ thuộc tuyệt đối vào tên.

### 8.3 Anonymized + partial documentation

Tên vẫn bị ẩn nhưng partial documentation hợp lệ được giữ:

```text
Commanded setpoint for joint position.
Measured sensor feedback for joint velocity.
Current payload of the hauling machine.
```

Kết quả cuối:

- full-name role macro-F1: `0.601`;
- anonymized + documentation role macro-F1: `0.675`;
- khoảng cách nhỏ hơn giới hạn `0.20`.

Điều này chứng minh role có thể được khôi phục từ documentation thay vì chỉ
đọc prefix. Anonymized no-documentation control vẫn được giữ để theo dõi
shortcut.

---

## 9. Baseline cuối

| Baseline | B0 macro | Type | Unit | Role F1 | Range | Relationship |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0,168 | 0,784 | 0,000 | 0,055 | 0,000 | 0,000 |
| Full-name rules | 0,413 | 1,000 | 0,073 | 0,601 | 0,000 | 0,393 |
| Anonymized rules | 0,202 | 1,000 | 0,000 | 0,010 | 0,000 | 0,000 |
| Anonymized + docs | 0,335 | 1,000 | 0,000 | 0,675 | 0,000 | 0,000 |

Baseline dùng để:

- xác định majority floor;
- phát hiện name shortcut;
- chứng minh rule parser không giải được toàn contract;
- đóng băng model threshold trước khi chạy JWM.

---

## 10. Dataset-admission gates

Benchmark phải tự pass admission trước khi dùng để kết luận model.

| Gate | Kết quả |
|---|---|
| Ít nhất 100 records | Pass |
| Ít nhất hai source families | Pass |
| Unit coverage ≥ 50% | Pass |
| Ít nhất 50 authoritative ranges | Pass |
| Relationship coverage ≥ 20% | Pass |
| Full-name shortcut gap ≤ 0,20 | Pass |

Kết quả:

```text
dataset admission = 6/6 pass
decision = ready_to_freeze_threshold_and_evaluate_jwm
```

### Vì sao range gate được điều chỉnh?

Gate ban đầu yêu cầu range trên ít nhất 50% mọi tag. Điều này chưa đúng với
contract “range when available” vì identifier, string, mode và context tag
không nhất thiết có EURange.

Ép coverage 50% có thể làm benchmark cố tình nhồi analog variables và làm sai
schema distribution. Gate được đổi thành:

> Có ít nhất 50 authoritative range labels từ nhiều nguồn.

Thay đổi được thực hiện trước khi chạy JWM và không dựa trên model score. Sau
khi bổ sung đúng dữ liệu, benchmark thực tế vẫn đạt range coverage 54,9%.

---

## 11. Frozen model-pass criteria

Model phải vượt tất cả ngưỡng:

| Metric | Threshold |
|---|---:|
| B0 contract macro | ≥ 0,60 |
| Data type exact accuracy | ≥ 0,95 |
| Unit exact accuracy | ≥ 0,60 |
| Role macro-F1 | ≥ 0,70 |
| Range score | ≥ 0,60 |
| Relationship macro-F1 | ≥ 0,60 |
| Full-to-anonymized macro drop | ≤ 0,10 |

Pass rule:

> Tất cả primary và robustness thresholds phải pass. Không được dùng macro
> average để che một component thất bại.

Ví dụ, macro đạt 0,70 nhưng relationship F1 chỉ 0,30 vẫn bị đánh fail.

Threshold được lưu trong:

```text
research/factorytraj_bench/b0_pass_threshold_v0.1.json
```

---

## 12. Pipeline kỹ thuật

```text
FactoryNet + Tennessee Eastman + OPC UA NodeSets
                    ↓
       Parse metadata với provenance
                    ↓
      Validate units, ranges, roles và links
                    ↓
            Build B0 benchmark
                    ↓
        Run dataset-admission gates
                    ↓
         Freeze model-pass threshold
                    ↓
    Full-name + anonymized model inference
                    ↓
             Frozen adjudication
                    ↓
              B0 pass hoặc fail
```

Các thành phần chính:

- `jwm/factorytraj_b0.py`: metric và transparent baselines.
- `jwm/opcua_nodeset_b0.py`: parser cho official OPC UA NodeSets.
- `scripts/build_factorytraj_b0_seed.py`: xây benchmark.
- `scripts/run_factorytraj_b0_seed.py`: chạy baseline và admission.
- `scripts/collect_opcua_b0_metadata.py`: thu metadata từ OPC UA endpoint.
- `scripts/validate_b0_opcua_export.py`: validate export đã review.
- `scripts/import_b0_opcua_export.py`: nhập metadata máy mới.
- `scripts/adjudicate_factorytraj_b0.py`: chấm model bằng frozen threshold.
- `research/factorytraj_bench/b0_seed_v0.1.json`: benchmark records.
- `research/factorytraj_bench/b0_seed_results_v0.1.json`: baseline results.
- `research/factorytraj_bench/b0_pass_threshold_v0.1.json`: frozen gate.

---

## 13. Trạng thái JWM

Phải phân biệt:

### Benchmark B0 đã admitted

Điều này có nghĩa:

- dataset đủ quy mô và đa nguồn;
- unit/range/relationship coverage đạt yêu cầu;
- shortcut control hoạt động;
- threshold đã frozen;
- benchmark có thể được dùng để đánh giá model.

### JWM chưa pass B0

JWM hiện chưa có structured industrial-tag adapter/output head. Vì vậy JWM
chưa tạo được predictions theo JSON contract và chưa được adjudicate.

Kết luận chính xác:

> Benchmark B0 đã hoàn thiện và admitted; JWM chưa được tuyên bố có
> schema/tag-understanding capability.

Việc chấm interface không hỗ trợ thành score thấp sẽ làm lẫn lộn giữa
“capability chưa được đo” và “capability đã được đo nhưng thất bại”.

---

## 14. Bước tiếp theo

Bước architecture tiếp theo:

```text
Tag ID/name
+ declared data type
+ representative samples
+ partial documentation
+ topology
        ↓
Industrial tag encoder / adapter
        ↓
Shared machine-state representation
        ↓
Structured decoder
        ↓
type + unit + range + role + relationships + confidence
```

Quy trình đánh giá:

1. Chạy inference với full tag names.
2. Chạy inference với anonymized IDs và giữ nguyên documentation/samples.
3. Lưu predictions đúng JSON contract.
4. Chạy frozen adjudication.
5. Báo cáo từng component và failure category.
6. Không điều chỉnh threshold theo kết quả checkpoint.

Sau đó benchmark nên được mở rộng bằng OPC UA exports từ máy thật để đo
external validity ngoài normative schemas và simulator data.

---

## 15. Nội dung trình bày ngắn với Bill

Trong Task 1, em đã operationalize B0 — Schema and Tag Understanding — thành
một benchmark có thể đo tự động. Mục tiêu là kiểm tra model có hiểu data type,
unit, range, role và relationships của machine tags hay chỉ dựa vào tên tag.

Benchmark hiện có 244 records từ 10 source families, gồm FactoryNet,
Tennessee Eastman và các OPC Foundation NodeSets cho CNC, mining, machinery
và nhiều industrial domains khác. Unit coverage đạt 76,5%, có 89
authoritative ranges với coverage 54,9% và relationship coverage đạt 37,7%.

Trong seed ban đầu, full-name rule đạt 0,521 nhưng giảm còn 0,202 khi ẩn tên.
Điều đó cho thấy tag prefix gây leakage rất lớn. Em đã thêm anonymized controls
và partial-documentation control để tách name parsing khỏi schema
understanding thật.

Mọi range label đều có provenance. Disturbance spans, observed min–max và
one-sided shutdown thresholds không được dùng sai làm physical range. Dữ liệu
range đến từ simulator semantics, miền vật lý của mol-% và normative OPC UA
metadata.

Dataset cuối vượt 6/6 admission gates. Em đã đóng băng model pass criteria
trước khi chạy JWM: B0 macro ít nhất 0,60, type 0,95, unit 0,60, role 0,70,
range 0,60, relationship 0,60 và full-to-anonymized drop không vượt 0,10.

Hiện benchmark B0 đã hoàn thiện, nhưng JWM chưa được coi là pass vì chưa có
structured industrial-tag adapter. Bước tiếp theo là xây adapter, tạo full và
anonymized predictions, sau đó dùng frozen criteria để quyết định checkpoint
pass hay fail.

