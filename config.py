# ============================================================
#  Cấu hình Jarvis — chỉnh các giá trị ở đây theo ý bạn
# ============================================================

# --- Giọng nói (Text-to-Speech) ---
# Giọng JARVIS phim (Anh-Anh): "en-GB-RyanNeural" | "en-GB-ThomasNeural"
# Giọng Việt: nam "vi-VN-NamMinhNeural" | nữ "vi-VN-HoaiMyNeural"
# (nếu dùng giọng Việt, hãy đổi cả các câu trả lời trong core/ về tiếng Việt)
VOICE = "en-GB-RyanNeural"
SPEECH_RATE = "-5%"           # nói chậm rãi, điềm tĩnh kiểu JARVIS
PITCH = "-5Hz"                # trầm hơn một chút
TTS_SYNTHESIS_TIMEOUT = 10

# --- Nhận diện giọng nói (Speech-to-Text) cho LỆNH ---
# Whisper: chạy offline trên máy, tự hiểu CẢ tiếng Việt LẪN tiếng Anh (bạn nói
# tiếng nào cũng được). Đây là engine mặc định.
WHISPER_ENABLED = True
# Measured on this 4-core i5: the failing "what am I holding?" sample is
# decoded correctly in ~1.6 s instead of ~25 s with large-v3-turbo.
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"        # dành GPU 4 GB cho Visual Reasoner
WHISPER_COMPUTE = "int8"

# Khóa đúng microphone gần người dùng; tránh lấy nhầm microphone của camera V380.
MICROPHONE_DEVICE_INDEX = None
MICROPHONE_NAME_HINT = "Microphone Array (Realtek"

# Pipeline ASR: signal gate -> Silero VAD -> Whisper -> confidence/language gate.
ASR_ALLOWED_LANGUAGES = ("vi", "en")
ASR_MIN_LANGUAGE_PROBABILITY_VI = 0.40
ASR_MIN_LANGUAGE_PROBABILITY_EN = 0.72
# Open-ended camera questions use a contextual prompt and a less rigid
# language gate while retaining the acoustic/VAD/confidence safety gates.
ASR_VISION_MIN_LANGUAGE_PROBABILITY_VI = 0.32
ASR_VISION_MIN_LANGUAGE_PROBABILITY_EN = 0.55
ASR_MIN_RMS_DBFS = -48.0
ASR_MIN_ACTIVE_RATIO = 0.015
ASR_MIN_AVG_LOGPROB = -0.85
ASR_MIN_WORD_PROBABILITY = 0.42
ASR_MAX_NO_SPEECH_PROBABILITY = 0.68
ASR_ALLOW_ENGLISH_FREEFORM = False
ASR_SAVE_REJECTED_AUDIO = True
ASR_SAVE_ACCEPTED_AUDIO = False
ASR_LOG_ROOT = "data/asr_logs"
ASR_AMBIENT_CALIBRATION_SECONDS = 1.0
ASR_PAUSE_THRESHOLD = 0.65
ASR_PHRASE_THRESHOLD = 0.25
ASR_NON_SPEAKING_DURATION = 0.35
# Long prompt/hotword lists were echoed by the base model when a short command
# was noisy. Keep the first pass neutral; a cloud fallback handles ambiguous
# speech instead of allowing prompt text to become a false command.
ASR_INITIAL_PROMPT = "JARVIS."
ASR_HOTWORDS = None
ASR_VISION_INITIAL_PROMPT = (
    "JARVIS. Tiếng Việt hoặc English. Camera, vật đang cầm, what am I holding?"
)
# A long hotword list caused the small model to echo prompt examples. The short
# contextual prompt above is both faster and more faithful on the captured audio.
ASR_VISION_HOTWORDS = None
ASR_BEAM_SIZE = 1
ASR_LANGUAGE_DETECTION_SEGMENTS = 1
ASR_CLOUD_FALLBACK_ENABLED = True
ASR_CLOUD_FALLBACK_LANGUAGES = ("vi-VN", "en-US")
ASR_CLOUD_TIMEOUT = 5
ASR_FAILURES_TO_STANDBY = 5

LANGUAGE = "vi-VN"            # chỉ dùng cho nhánh Google dự phòng (khi tắt Whisper)
LISTEN_TIMEOUT = 6            # giây chờ bạn bắt đầu nói
PHRASE_LIMIT = 12            # độ dài tối đa của một câu nói (giây)

# --- Giao diện HUD (vòng tròn hologram) ---
HUD_ENABLED = True            # False = chạy Jarvis không có giao diện
HUD_SIZE = 1280               # fallback size; fullscreen is the normal mode
HUD_FULLSCREEN = True         # wake word luôn mở HUD toàn màn hình
HUD_DEBUG = False

# --- Visual Reasoner + camera ---
VISION_MODEL = "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit"
VISION_CAMERA_INDEX = 0
VISION_CAMERA_WIDTH = 1280
VISION_CAMERA_HEIGHT = 720
VISION_CAMERA_WARMUP_FRAMES = 3
VISION_CAMERA_FPS = 30        # yêu cầu camera capture ở 30fps (CAP_PROP_FPS)
VISION_DISPLAY_FPS = 30       # HUD hiển thị 30fps (trước đây 5)
VISION_DISPLAY_WIDTH = 800    # giảm từ 960 để encode+push nằm trong ngân sách 33ms/frame
VISION_DISPLAY_JPEG_QUALITY = 60
VISION_DISPLAY_AUTO_DEGRADE = True   # tự hạ width (800->640->512) nếu fps đo được < 80% mục tiêu
# Camera V380 vật lý max ~21.5fps @720p (đo thực tế). Bật nội suy cross-fade để
# HUD render đủ 30 khung hình RIÊNG BIỆT mỗi giây (frame-rate conversion chuẩn);
# tắt thì HUD hiển thị đúng nhịp camera (~21.5fps).
VISION_DISPLAY_INTERPOLATE = True
VISION_MEMORY_FRAMES = 8
VISION_MEMORY_INTERVAL = 0.75
VISION_REASONING_FRAMES = 1   # một frame cho latency ~9-12 giây trên GTX 1650
VISION_MAX_NEW_TOKENS = 72
VISION_MAX_IMAGE_EDGE = 336
VISION_MAX_GPU_MEMORY = "2300MiB"
VISION_MAX_CPU_MEMORY = "8GiB"
VISION_LOG_ROOT = "data/vision_sessions"
VISION_REASONER_READY_TIMEOUT = 30
VISION_INFERENCE_TIMEOUT = 45
VISION_WORKER_STARTUP_TIMEOUT = 60
VISION_WORKER_RESPONSE_TIMEOUT = 120
VISION_CAMERA_MAX_READ_FAILURES = 12
VISION_CAMERA_RECONNECT_SECONDS = 1.0

# --- World Brain (JWM — bộ não world-model tự train, jwm/checkpoints/) ---
# "off"     = tắt hoàn toàn
# "shadow"  = chạy song song với Qwen3-VL, chỉ ghi log dự đoán để đánh giá (an toàn)
# "primary" = dùng trực tiếp câu trả lời của JWM (chỉ cho cảnh thuộc synthetic domain)
WORLD_BRAIN_MODE = "shadow"
WORLD_BRAIN_TRIAL_LOG = "data/world_brain_trials"

# Unified turn/state telemetry. ASR and vision retain their richer domain logs.
RUNTIME_EVENT_LOG = "data/runtime/events.jsonl"

# Bản clone dùng khóa riêng, không tranh chấp với Jarvis gốc nếu cần đối chiếu.
INSTANCE_PORT = 47822

# --- Wake word: "Hey Jarvis" ---
WAKE_WORD_ENABLED = True      # False = nghe lệnh liên tục, không cần gọi tên

# Engine chính: openWakeWord — model chuyên dụng nhận diện "Hey Jarvis" offline.
WAKE_MODEL = "hey_jarvis"     # model huấn luyện sẵn của openWakeWord
WAKE_THRESHOLD = 0.5          # 0-1: cao hơn = khó đánh thức hơn (ít nhầm), thấp hơn = nhạy hơn

# Dự phòng: nếu máy không cài được openWakeWord, quay về Google STT + đoán chữ.
WAKE_LANGUAGE = "en-US"
WAKE_WORDS = ("jarvis", "javis", "jarvit", "jervis", "travis", "travers",
              "travel", "service", "charvis", "gia vit")
SILENCE_TO_STANDBY = 2        # số lượt im lặng liên tiếp trước khi quay về chế độ chờ
ONE_SHOT_AFTER_WAKE = True    # lệnh thường xong thì về standby; vision mode vẫn hội thoại liên tục

# --- Thông tin mặc định ---
DEFAULT_CITY = "Hà Nội"       # thành phố mặc định khi hỏi thời tiết
USER_NAME = "sir"             # JARVIS gọi bạn là "sir" như trong phim

# --- Nguồn tin tức (RSS) ---
# BBC (tiếng Anh, hợp giọng JARVIS). Muốn tin Việt Nam:
#   NEWS_RSS = "https://vnexpress.net/rss/tin-moi-nhat.rss"; NEWS_SOURCE = "VnExpress"
NEWS_RSS = "https://feeds.bbci.co.uk/news/world/rss.xml"
NEWS_SOURCE = "the BBC"

# --- Bộ não AI (chuẩn OpenAI-compatible — KHÔNG khóa vào provider nào) ---
# MẶC ĐỊNH: bộ não bạn TỰ TRAIN, chạy offline trên máy (không key, không provider).
#   Bật bằng cách mở cửa sổ riêng chạy:  python training/serve_brain.py
#   (xem training/README_TRAIN.md để train lại hoặc làm thông minh hơn)
AI_BASE_URL = "http://localhost:8080/v1"
AI_MODEL = "jarvis"
# Muốn dùng dịch vụ ngoài thay vì model local? Đổi 2 dòng trên, ví dụ Groq (miễn phí):
#   AI_BASE_URL = "https://api.groq.com/openai/v1"; AI_MODEL = "llama-3.3-70b-versatile"
#   rồi thêm AI_API_KEY=gsk_... vào file .env. Xem thêm core/ai.py.
AI_MAX_TOKENS = 500           # trả lời ngắn gọn vì sẽ được đọc thành tiếng
AI_HISTORY_LIMIT = 20         # số lượt hội thoại nhớ được
