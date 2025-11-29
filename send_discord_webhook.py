"""將明日交易訊號 (tomorrow_trading_signal.json) 推送到 Discord webhook 的工具。"""

from typing import Any, Dict, Optional

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import requests

from logging_config import setup_logging


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """
    解析命令列參數。
    
    Args:
        argv (list, optional): 參數列表。預設為 None (使用 sys.argv)。
        
    Returns:
        argparse.Namespace: 解析後的參數物件。
    """
    parser = argparse.ArgumentParser(description="發送明日交易訊號至 Discord Webhook")
    parser.add_argument("--webhook", "-w", required=True, help="Discord webhook URL (必填)")
    parser.add_argument("--json-path", "-j", help="JSON 檔案路徑 (覆蓋預設值 log/tomorrow_trading_signal.json)")
    parser.add_argument("--dry-run", action="store_true", help="不發送 POST 請求，僅印出 Payload")
    parser.add_argument("--retries", type=int, default=1, help="失敗重試次數 (預設: 1)")
    parser.add_argument("--retry-delay", type=int, default=10, help="重試間隔秒數 (預設: 10)")
    return parser.parse_args(argv)


def load_json(path: str) -> Dict[str, Any]:
    """
    從路徑載入 JSON，若失敗則以錯誤代碼退出。
    
    Args:
        path (str): JSON 檔案路徑。
        
    Returns:
        Dict[str, Any]: 載入的 JSON 數據。
    """
    if not os.path.isfile(path):
        logger.error("JSON file not found: %s", path)
        sys.exit(2)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON file %s: %s", path, e)
        sys.exit(3)


def prob_bar(pct: float, width: int = 20) -> str:
    """
    回傳一個簡單的文字進度條與百分比，pct 範圍為 [0,1]。
    
    Args:
        pct (float): 百分比數值 (0.0 到 1.0)。
        width (int): 進度條寬度 (字元數)。
        
    Returns:
        str: 格式化後的進度條字串。
    """
    try:
        p = float(pct)
    except Exception:
        p = 0.0
    p = max(0.0, min(1.0, p))
    filled = int(round(p * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {p*100:6.2f}%"


def fmt_iso_to_readable(s: Optional[str]) -> str:
    """
    將 ISO 格式字串轉換為 YYYY-MM-DD (若失敗則回傳部分原始字串)。
    
    Args:
        s (str, optional): ISO 日期字串。
        
    Returns:
        str: 格式化後的日期字串。
    """
    if not s:
        return "N/A"
    try:
        s2 = s.replace("Z", "+00:00") if isinstance(s, str) and s.endswith("Z") else s
        dt = datetime.fromisoformat(s2)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(s)[:19]


def build_embed(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    從數據字典建構 Discord embed payload。
    
    Args:
        data (Dict[str, Any]): 訊號數據。
        
    Returns:
        Dict[str, Any]: Discord embed 物件。
    """
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
    """將 payload 以 POST 方式送至指定 webhook，並支援重試機制"""
    if dry_run:
        logger.info("Dry-run mode: payload would be:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    attempts = 1 + max(0, int(retries))
    last_status = -1
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(webhook_url, json=payload)
        except Exception as e:
            logger.exception("Error sending webhook on attempt %s/%s", attempt, attempts)
            last_status = -1
        else:
            last_status = resp.status_code
            if resp.status_code // 100 == 2:
                logger.info("Webhook delivered successfully (attempt %s/%s)", attempt, attempts)
                return resp.status_code
            else:
                logger.error(
                    "Webhook attempt %s/%s failed: HTTP %s %s",
                    attempt,
                    attempts,
                    resp.status_code,
                    resp.text,
                )

        # 如果還有剩餘嘗試次數，等待後重試
        if attempt < attempts:
            logger.warning("Retrying webhook in %s seconds", retry_delay)
            time.sleep(max(0, int(retry_delay)))

    return last_status


def main(argv: Optional[list] = None) -> int:
    """命令列入口點"""
    args = parse_args(argv)

    setup_logging()

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