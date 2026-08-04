"""Kỹ năng: tìm kiếm, phát nhạc, thời tiết, tin tức, tỷ giá.

Bạn ra lệnh tiếng Việt — JARVIS trả lời tiếng Anh (giọng phim).
Ví dụ: "tìm kiếm cách nấu phở", "phát nhạc Sơn Tùng",
       "thời tiết Đà Nẵng", "tin tức mới nhất", "tỷ giá đô la"
"""

import re
import urllib.parse
import webbrowser
import xml.etree.ElementTree as ET

import requests

import config

# Mô tả thời tiết theo mã WMO của Open-Meteo
WEATHER_CODES = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast skies",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "light showers", 81: "rain showers", 82: "violent showers",
    95: "thunderstorms", 96: "thunderstorms with light hail", 99: "thunderstorms with hail",
}



def handle(text: str, norm: str) -> str | None:
    # --- Tìm kiếm Google ---
    query = _after_prefix(norm, (
        "tim kiem ", "tra cuu ", "google ", "search for ", "search ", "look up ",
    ))
    if query:
        webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
        return f"Here are the search results for {query}, sir."

    # --- Phát nhạc / video trên YouTube ---
    query = _after_prefix(norm, (
        "phat nhac ", "mo nhac ", "bat nhac ", "phat bai ", "mo bai hat ",
        "phat video ", "mo video ", "phat ", "play "))
    if query:
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        webbrowser.open(url)
        return f"Playing {query} on YouTube, sir."
    if norm in ("phat nhac", "mo nhac", "bat nhac", "play music", "play some music"):
        webbrowser.open("https://www.youtube.com/results?search_query=nh%E1%BA%A1c+hay")
        return "Opening YouTube music, sir. Next time, tell me the song name for a precise match."

    # --- Thời tiết (lấy tên thành phố từ giữa câu) ---
    if "thoi tiet" in norm or "weather" in norm:
        m = re.search(r"(?:thoi tiet|weather)(?: (?:o|tai|in|at|for))? (.+)", norm)
        city = m.group(1).strip() if m else None
        return _weather(city or config.DEFAULT_CITY)

    # --- Tin tức ---
    if any(k in norm for k in ("tin tuc", "tin moi", "co gi moi", "doc bao",
                              "the news", "latest news", "headlines", "what's happening",
                              "whats happening")):
        return _news()

    # --- Tỷ giá USD ---
    if any(k in norm for k in ("ty gia", "gia do la", "gia usd", "do la bao nhieu",
                              "exchange rate", "dollar rate", "usd rate")):
        return _usd_rate()

    return None


def _after_prefix(norm: str, prefixes: tuple) -> str | None:
    """Nếu câu bắt đầu bằng một prefix, trả về phần còn lại (đã chuẩn hóa).

    Dùng chuỗi đã chuẩn hóa (không dấu, không dấu câu) cho bền — Google/YouTube
    tìm kiếm không dấu vẫn ra kết quả đúng.
    """
    for prefix in prefixes:
        if norm.startswith(prefix) and len(norm) > len(prefix):
            return norm[len(prefix):].strip()
    return None


def _weather(city: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "vi"},
            timeout=8,
        ).json()
        if not geo.get("results"):
            return f"I could not locate {city}, sir."
        place = geo["results"][0]

        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=8,
        ).json()

        current = data["current"]
        daily = data["daily"]
        description = WEATHER_CODES.get(current["weather_code"], "")
        return (
            f"It is currently {round(current['temperature_2m'])} degrees Celsius "
            f"in {place['name']}"
            f"{' with ' + description if description else ''}, "
            f"humidity {current['relative_humidity_2m']} percent. "
            f"Today's high is {round(daily['temperature_2m_max'][0])}, "
            f"low {round(daily['temperature_2m_min'][0])} degrees."
        )
    except requests.RequestException:
        return "I could not retrieve the weather data, sir. Please check the connection."
    except (KeyError, IndexError, ValueError):
        return "The weather service returned unexpected data, sir. Do try again later."


def _news(limit: int = 5) -> str:
    try:
        response = requests.get(config.NEWS_RSS, timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        titles = [item.findtext("title", "") for item in root.iter("item")][:limit]
        titles = [t.strip() for t in titles if t and t.strip()]
        if not titles:
            return "I could not fetch the news at the moment, sir."
        numbered = ". ".join(f"Headline {i + 1}: {t}" for i, t in enumerate(titles))
        return f"Here are the top {len(titles)} headlines from {config.NEWS_SOURCE}. {numbered}."
    except (requests.RequestException, ET.ParseError):
        return "I could not retrieve the news, sir. Please check the connection."


def _usd_rate() -> str:
    try:
        data = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8).json()
        vnd = data["rates"]["VND"]
        return f"One US dollar is currently worth approximately {round(vnd):,} Vietnamese dong, sir."
    except requests.RequestException:
        return "I could not retrieve the exchange rate, sir. Please check the connection."
    except (KeyError, ValueError):
        return "The exchange rate service returned unexpected data, sir."
