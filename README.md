# 🤖 J.A.R.V.I.S. — Trợ lý ảo giọng nói

Trợ lý ảo lấy cảm hứng từ Iron Man: **bạn ra lệnh bằng tiếng Việt**, JARVIS trả lời bằng **giọng Anh-Anh điềm tĩnh như trong phim** (gọi bạn là "sir").

## Cài đặt

```powershell
pip install -r requirements.txt
```

## Chạy

```powershell
python jarvis.py
```

## 👁️ Chế độ Visual Reasoner

Bản `Jarvis-Vision` kết hợp trợ lý giọng nói hiện có với Qwen3-VL Reasoner và
camera DirectShow, hoàn toàn trong ứng dụng desktop — không dùng Gradio hay giao
diện web.

Luồng sử dụng:

1. Nói **“Hey Jarvis”** và đợi Jarvis trả lời.
2. Nói **“truy cập camera”**, **“mở camera”** hoặc **“nhìn xung quanh”**.
3. HUD chuyển sang camera mode; logo thu nhỏ vào góc và video chiếm vùng chính.
4. Hỏi, ví dụ: **“Vật tôi đang cầm là gì?”**, **“Có bao nhiêu người?”** hoặc
   **“Thứ màu xanh bên trái là gì?”**.
5. Nói **“tắt camera”** để trở lại HUD thường.

Reasoner được nạp lần đầu khi camera bật, thường cần khoảng 20–25 giây trên GTX
1650. Mỗi lượt sau mất khoảng 9–15 giây. Hãy giữ vật rõ, đủ sáng và gần giữa
khung hình; vật nhỏ, bị cắt hoặc che nhiều vẫn là failure case của model 2B.

Mọi phiên được lưu riêng tại:

```text
data/vision_sessions/<session-id>/events.jsonl
data/vision_sessions/<session-id>/frames/*-raw.jpg
data/vision_sessions/<session-id>/frames/*-overlay.jpg
```

`events.jsonl` chứa câu hỏi, câu trả lời, bbox, confidence, latency, raw model
output và đường dẫn ảnh để phân tích benchmark về sau.

### 🔵 Giao diện HUD hologram

Khi bạn gọi “Hey Jarvis”, HUD bung ra **toàn màn hình** với logo và đồng hồ ở
giữa. Khi camera được bật, logo lùi vào góc trái, luồng camera mở ở vùng chính,
còn bbox và câu trả lời được phủ trực tiếp lên hình. Khi JARVIS nói, các vòng
xoay nhanh và lõi phát sáng theo nhịp. Về chế độ chờ thì HUD tự ẩn.

Tùy chỉnh trong [config.py](config.py): `HUD_FULLSCREEN = True` để giữ toàn màn
hình, `HUD_ENABLED = False` để tắt giao diện.

### 🎙️ Wake word — "Hey Jarvis"

Dùng **openWakeWord** — engine chuyên dụng nhận diện chính xác âm thanh "Hey Jarvis" **chạy offline trên máy** (không phiên âm, không đoán chữ, không cần internet cho việc đánh thức). Chỉnh độ nhạy bằng `WAKE_THRESHOLD` trong [config.py](config.py) (thấp hơn = nhạy hơn).

Jarvis khởi động ở **chế độ chờ** (im lặng, chạy nền):

- Nói **"Hey Jarvis"** → tỉnh dậy: *"Yes, sir?"* → nói lệnh của bạn
- Nói liền **"Hey Jarvis, mấy giờ rồi"** → thực hiện luôn
- Sau khi trả lời, Jarvis tiếp tục nghe; **im lặng ~2 lượt** hoặc nói **"ngủ đi"** → quay về chế độ chờ
- Nói **"goodbye"** → *"Goodbye, sir. Powering down."* — thoát hẳn
- Nói **"goodnight"** / **"chúc ngủ ngon"** → *"Good night, sir. Sleep well."* — thoát hẳn

Không thích wake word? Đặt `WAKE_WORD_ENABLED = False` trong [config.py](config.py) — Jarvis sẽ nghe lệnh liên tục.

### 🎧 Speech-to-text chống tiếng nền

Bản Vision không còn để Whisper tự do chuyển mọi âm thanh thành câu lệnh. Pipeline
mới gồm:

```text
Microphone Array Realtek → signal-quality gate → Silero VAD
→ Whisper large-v3-turbo → language/confidence/script gate → brain
```

- Microphone được khóa vào `Microphone Array (Realtek)` để không lấy nhầm mic V380.
- Chỉ tiếng Việt và các lệnh tiếng Anh rõ ràng như `open camera`, `goodbye` được nhận.
- Transcript Trung/Nhật/Hàn, ngôn ngữ ngoài whitelist, decoder confidence thấp và
  tiếng nền không đủ speech activity sẽ bị loại, không gửi sang model hội thoại.
- Sau một lệnh thông thường Jarvis trở về standby; điều này ngăn TV hoặc loa nền
  kéo trợ lý vào chuỗi hội thoại giả. Vision mode vẫn cho phép hỏi liên tục.
- Các lần ASR bị loại được ghi tại `data/asr_logs/<date>/asr_events.jsonl`; audio
  bị loại được giữ dạng WAV để audit.

### 🚀 Tự khởi động cùng Windows

Jarvis được cài chạy ẩn mỗi khi đăng nhập Windows (file `JARVIS.vbs` trong thư mục Startup) — bật máy lên là "Hey Jarvis" gọi được ngay, không cần mở gì.

- **Tắt Jarvis đang chạy nền**: nói "Hey Jarvis... goodbye" (hoặc "goodnight"), hoặc Task Manager → kết thúc tiến trình `python`.
- **Bỏ tự khởi động**: xóa file `JARVIS.vbs` trong thư mục mở bằng lệnh `Win+R` → gõ `shell:startup`.
- Jarvis có khóa chống chạy trùng — mở tay `python jarvis.py` khi bản nền đang chạy sẽ tự thoát.

- Không micro → tự chuyển sang gõ phím.
- Nói **"chế độ gõ"** / **"chế độ nói"** để chuyển đổi.

## Jarvis làm được gì?

| Nhóm | Ví dụ câu lệnh |
|---|---|
| 🕐 Giờ & ngày | "mấy giờ rồi", "hôm nay thứ mấy" |
| 📱 Mở ứng dụng | "mở chrome", "mở máy tính", "mở cài đặt" |
| 🌐 Mở website | "mở youtube", "vào facebook", "mở shopee" |
| 🔍 Tìm kiếm | "tìm kiếm cách nấu phở bò" |
| 🎵 Phát nhạc | "phát nhạc Sơn Tùng", "mở bài hát Nơi này có anh" |
| ☀️ Thời tiết | "thời tiết", "thời tiết ở Đà Nẵng" |
| 📰 Tin tức | "tin tức mới nhất", "đọc báo" |
| 💵 Tỷ giá | "tỷ giá đô la" |
| 🔊 Âm lượng | "tăng âm lượng", "âm lượng 50", "tắt tiếng" |
| ⏰ Nhắc nhở | "nhắc tôi uống nước sau 30 phút", "nhắc tôi họp lúc 3 giờ chiều" |
| ⏲️ Hẹn giờ | "hẹn giờ 5 phút" |
| 📝 Ghi chú | "ghi chú mua sữa", "đọc ghi chú", "xóa ghi chú" |
| 🔒 Khóa máy | "khóa màn hình" |

## 🧠 Bộ não AI — model TỰ TRAIN, chạy offline, không phụ thuộc ai

Mặc định, bộ não là **model bạn tự huấn luyện** (Qwen2.5-0.5B đã được distill theo phong cách JARVIS), chạy hẳn trên máy bạn qua một server nội bộ — **không API key, không provider, không internet**. Jarvis tự bật server này khi khởi động.

- Đã train sẵn ở `training/jarvis-model/`. Chạy lại / làm thông minh hơn: xem [training/README_TRAIN.md](training/README_TRAIN.md).
- Muốn train từ đầu: `python training/build_dataset.py` → `python training/train_local.py` (~17 phút trên GTX 1650).

### Muốn dùng dịch vụ ngoài thay vì model local?

Bộ não dùng **chuẩn OpenAI-compatible**: đổi nhà cung cấp = sửa 2 dòng `AI_BASE_URL` + `AI_MODEL` trong [config.py](config.py), thêm key vào `.env` (`AI_API_KEY=...`). Không đụng code.

| Dịch vụ | AI_BASE_URL | Ghi chú |
|---|---|---|
| **Model local (mặc định)** | `http://localhost:8080/v1` | Tự train, offline, không key |
| Groq | `https://api.groq.com/openai/v1` | Miễn phí, Llama 70B, cần key `gsk_...` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | Có bậc miễn phí |
| Anthropic Claude | `https://api.anthropic.com/v1` | Thông minh nhất, trả phí |

## Tùy chỉnh

Mở [config.py](config.py) để đổi:
- **Giọng nói**: JARVIS phim (`en-GB-RyanNeural`, `en-GB-ThomasNeural`) hoặc giọng Việt (`vi-VN-NamMinhNeural`, `vi-VN-HoaiMyNeural`) — lưu ý nếu đổi về giọng Việt thì cần dịch lại các câu trả lời trong `core/`
- **Tốc độ & độ trầm**: `SPEECH_RATE`, `PITCH`
- **Thành phố mặc định** cho thời tiết

## Cấu trúc project

```
jarvis.py            # Chương trình chính: nghe → xử lý → nói
config.py            # Cấu hình
core/
├── speech.py        # 🎙️ Realtek mic → VAD → Whisper → confidence/language gate
├── voice.py         # 🔊 Đọc thành tiếng (edge-tts + pygame)
├── brain.py         # 🧠 Điều phối lệnh → kỹ năng → AI
├── hud.py           # 🔵 Giao diện hologram (pywebview)
├── ai.py            # 🔌 Kết nối Claude API (tùy chọn)
└── skills/
    ├── apps.py      # Mở ứng dụng, website
    ├── system.py    # Giờ, ngày, âm lượng, khóa máy
    ├── web.py       # Tìm kiếm, nhạc, thời tiết, tin tức, tỷ giá
    └── reminders.py # Nhắc nhở, hẹn giờ, ghi chú
data/                # Ghi chú & nhắc nhở (tự tạo khi dùng)
```

> Lưu ý: cần **kết nối internet** cho nhận diện giọng nói, tạo giọng đọc, thời tiết, tin tức.
