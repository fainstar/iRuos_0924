"""Send trading signal (tomorrow_trading_signal.json) to a Discord webhook.

This script is refactored into functions for readability and testing.
"""

from typing import Any, Dict, Optional

import os
import json
import sys
import time
import requests
import argparse
from datetime import datetime


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Send tomorrow trading signal to Discord webhook")
    parser.add_argument("--webhook", "-w", required=True, help="Discord webhook URL (required)")
    parser.add_argument("--json-path", "-j", help="Path to JSON file (overrides default log/tomorrow_trading_signal.json)")
    parser.add_argument("--dry-run", action="store_true", help="Do not POST, just print payload")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries on failure (default: 1)")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds to wait between retries (default: 10)")
    return parser.parse_args(argv)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON from path, exit with error code on failure."""
    if not os.path.isfile(path):
        print(f"找不到 JSON 檔案: {path}\n請確認檔案存在或傳入正確路徑（可用 --json-path 參數）", file=sys.stderr)
        sys.exit(2)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"讀取 JSON 檔案失敗（格式錯誤）: {path} - {e}", file=sys.stderr)
        sys.exit(3)


def prob_bar(pct: float, width: int = 20) -> str:
    """Return a simple text progress bar and percentage for pct in [0,1]."""
    try:
        p = float(pct)
    except Exception:
        p = 0.0
    p = max(0.0, min(1.0, p))
    filled = int(round(p * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {p*100:6.2f}%"


def fmt_iso_to_readable(s: Optional[str]) -> str:
    """Format an ISO-like string to YYYY-MM-DD (fallback to partial original)."""
    if not s:
        return "N/A"
    try:
        s2 = s.replace("Z", "+00:00") if isinstance(s, str) and s.endswith("Z") else s
        dt = datetime.fromisoformat(s2)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(s)[:19]


def build_embed(data: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the Discord embed payload from data dict."""
    action = data.get("recommended_action", "").lower()
    color_map = {"buy": 0x00FF00, "hold": 0xFFFF00, "sell": 0xFF0000}
    color = color_map.get(action, 0xCCCCCC)

    confidence = data.get("confidence", 0) or 0
    confidence_pct = f"{confidence*100:.2f}%"

    probs = data.get("action_probabilities", {}) or {}
    order = (("buy", "Buy"), ("hold", "Hold"), ("sell", "Sell"))
    lines = []
    for key, label in order:
        pct = probs.get(key, 0) or 0
        lines.append(f"{label:<5} {prob_bar(pct)}")
    prob_text = "```text\n" + "\n".join(lines) + "\n```"

    signal_time = fmt_iso_to_readable(data.get("signal_date"))
    for_date = fmt_iso_to_readable(data.get("for_date"))

    action_display = action.upper() if action else "N/A"
    action_emoji = "🟩" if action == "buy" else ("🟨" if action == "hold" else ("🟥" if action == "sell" else "❔"))

    embed = {
        "title": "📈 明日交易建議",
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
    return embed


def post_webhook(webhook_url: str, payload: Dict[str, Any], dry_run: bool = False, retries: int = 1, retry_delay: int = 10) -> int:
    """POST payload to webhook_url with retries.

    retries: number of retry attempts after the initial try (e.g. 1 means try once, then retry once)
    retry_delay: seconds to wait between retries
    Return HTTP status code (0 for dry-run), or -1 on exception.
    """
    if dry_run:
        print("Dry-run mode: payload would be:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    attempts = 1 + max(0, int(retries))
    last_status = -1
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(webhook_url, json=payload)
        except Exception as e:
            print(f"Error sending webhook on attempt {attempt}/{attempts}: {e}")
            last_status = -1
        else:
            last_status = resp.status_code
            if resp.status_code // 100 == 2:
                print(f"已送出到 Discord webhook (attempt {attempt}/{attempts})")
                return resp.status_code
            else:
                print(f"發生錯誤 on attempt {attempt}/{attempts}: HTTP {resp.status_code} 內容: {resp.text}")

        # 如果還有剩餘嘗試次數，等待後重試
        if attempt < attempts:
            print(f"等待 {retry_delay} 秒後重試...")
            time.sleep(max(0, int(retry_delay)))

    return last_status


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = args.json_path or os.path.join(base_dir, "log", "tomorrow_trading_signal.json")

    data = load_json(json_path)
    embed = build_embed(data)
    payload = {"embeds": [embed]}

    status = post_webhook(args.webhook, payload, dry_run=args.dry_run, retries=args.retries, retry_delay=args.retry_delay)
    # Return non-zero on HTTP failure
    if status == -1:
        return 4
    if status != 0 and status // 100 != 2:
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())