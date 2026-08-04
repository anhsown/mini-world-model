"""Gộp các file part_*.jsonl (do 'thầy' Claude viết) thành dataset huấn luyện.

Chạy:  python training/build_dataset.py

- Kiểm tra từng dòng là JSON hợp lệ, đúng cấu trúc {"messages": [...]}
- Loại các dòng lỗi / trùng lặp
- Xáo trộn (deterministic) và ghi ra dataset.jsonl
"""

import glob
import json
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset.jsonl")

SYSTEM_EXPECTED = (
    "You are JARVIS, a refined British AI assistant. "
    "Address the user as sir. Reply in concise spoken English."
)


def valid(obj) -> bool:
    if not isinstance(obj, dict) or "messages" not in obj:
        return False
    msgs = obj["messages"]
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False
    if msgs[0].get("role") != "system":
        return False
    roles = [m.get("role") for m in msgs[1:]]
    # phải xen kẽ user/assistant, bắt đầu bằng user, kết thúc bằng assistant
    if roles[0] != "user" or roles[-1] != "assistant":
        return False
    expected = ["user", "assistant"] * (len(roles) // 2)
    if roles != expected:
        return False
    for m in msgs:
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False
    return True


def main() -> None:
    rows = []
    seen = set()
    parts = sorted(glob.glob(os.path.join(HERE, "part_*.jsonl")))
    if not parts:
        print("⚠️  Chưa có file part_*.jsonl nào. Chờ 'thầy' viết xong đã.")
        return

    stats = {}
    for path in parts:
        name = os.path.basename(path)
        kept = bad = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not valid(obj):
                    bad += 1
                    continue
                # chuẩn hóa system prompt cho đồng nhất
                obj["messages"][0]["content"] = SYSTEM_EXPECTED
                key = json.dumps(obj["messages"][1:], ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(obj)
                kept += 1
        stats[name] = (kept, bad)

    # xáo trộn deterministic (không phụ thuộc random seed hệ thống)
    rows.sort(key=lambda o: hash(json.dumps(o["messages"], ensure_ascii=False)) & 0xFFFFFFFF)

    with open(OUT, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("📊 Thống kê từng phần (giữ / lỗi):")
    for name, (kept, bad) in stats.items():
        print(f"   {name}: {kept} giữ, {bad} lỗi")
    print(f"\n✅ Tổng cộng {len(rows)} mẫu -> {OUT}")


if __name__ == "__main__":
    main()
