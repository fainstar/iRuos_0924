# send_discord_webhook.py
import os
import json
import sys
import requests
import argparse
from datetime import datetime

# 讀取 webhook（只從 --webhook 參數取得）
parser = argparse.ArgumentParser(description="Send tomorrow trading signal to Discord webhook")
parser.add_argument('--webhook', '-w', required=True, help='Discord webhook URL (required)')
parser.add_argument('--json-path', '-j', help='Path to JSON file (overrides default log/tomorrow_trading_signal.json)')
parser.add_argument('--dry-run', action='store_true', help='Do not POST, just print payload')
args = parser.parse_args()

WEBHOOK_URL = args.webhook

# 預設 JSON 路徑（改為相對於此腳本的 log 資料夾，避免絕對路徑問題）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = args.json_path or os.path.join(BASE_DIR, "log", "tomorrow_trading_signal.json")

if not os.path.isfile(JSON_PATH):
    print(f"找不到 JSON 檔案: {JSON_PATH}\n請確認檔案存在或傳入正確路徑（可用 --json-path 參數）", file=sys.stderr)
    sys.exit(2)

try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"讀取 JSON 檔案失敗（格式錯誤）: {JSON_PATH} - {e}", file=sys.stderr)
    sys.exit(3)

# 格式化與顏色設定
action = data.get("recommended_action", "").lower()
color_map = {"buy": 0x00FF00, "hold": 0xFFFF00, "sell": 0xFF0000}
color = color_map.get(action, 0xCCCCCC)

confidence = data.get("confidence", 0)
confidence_pct = f"{confidence*100:.2f}%"

# 加入 action_probabilities 作為一個欄位（格式化）
probs = data.get("action_probabilities", {})
def prob_bar(pct, width=20):
    """Return a simple text progress bar and percentage for pct in [0,1]."""
    try:
        p = float(pct)
    except Exception:
        p = 0.0
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0
    filled = int(round(p * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {p*100:6.2f}%"

# build prob_text as a code block with bars for readability
order = (("buy", "Buy"), ("hold", "Hold"), ("sell", "Sell"))
lines = []
for key, label in order:
    pct = probs.get(key, 0) or 0
    lines.append(f"{label:<5} {prob_bar(pct)}")
prob_text = "```text\n" + "\n".join(lines) + "\n```"

# 小工具：把 ISO-like 字串格式化為 YYYY-MM-DD HH:MM:SS
def fmt_iso_to_readable(s):
    if not s:
        return "N/A"
    try:
        # 支援尾端帶 Z 的 UTC 表示
        s2 = s.replace("Z", "+00:00") if isinstance(s, str) and s.endswith("Z") else s
        dt = datetime.fromisoformat(s2)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # 如果解析失敗，回傳原始字串的前 19 字元作為 fallback
        return str(s)[:19]

# 格式化時間字串
signal_time = fmt_iso_to_readable(data.get("signal_date"))
for_date = fmt_iso_to_readable(data.get("for_date"))
# 建 embed
action_display = action.upper() if action else "N/A"
action_emoji = "🟩" if action == "buy" else ("🟨" if action == "hold" else ("🟥" if action == "sell" else "❔"))

# 建 embed（使用欄位而非框線表格）
embed = {
    "title": "📈 明日交易建議 / Tomorrow Trading Signal",
    "description": f"**建議動作：{action_display} {action_emoji}  •  信心度：{confidence_pct}**",
    "color": color,
    "fields": [
        {"name": "📅 訊號時間", "value": signal_time, "inline": True},
        {"name": "🔁 對應日期", "value": for_date, "inline": True},
        {"name": "💰 當前價格", "value": f"{data.get('current_price', 0):.2f}", "inline": True},
        {"name": "📝 建議動作", "value": f"{action_display} {action_emoji}", "inline": True},
        {"name": "🔎 信心度", "value": confidence_pct, "inline": True},
        {"name": "⚖️ Action Probabilities", "value": prob_text, "inline": False},
    ],
    "footer": {"text": f"模型來源 • 產生時間: {signal_time}"},
}

payload = {"embeds": [embed]}

# POST
resp = requests.post(WEBHOOK_URL, json=payload)
if resp.status_code // 100 == 2:
    print("已送出到 Discord webhook")
else:
    print(f"發生錯誤: HTTP {resp.status_code} 內容: {resp.text}")