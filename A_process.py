import json
import subprocess
import sys
import schedule
import time
import os


# 這個檔案會從同目錄下的 `stock.json` 讀取要處理的股票(symbol)與對應的 webhook_url
# 支援的格式範例：
# 單一物件:
# {"symbol": "ETH-USD", "webhook_url": "https://..."}
# 或是陣列:
# [{"symbol": "BTC-USD", "webhook_url": "https://..."}, {"symbol": "ETH-USD", "webhook_url": "https://..."}]
# 或是連續多個 JSON 物件 (多個物件串在一起 / NDJSON)：本程式會嘗試逐一解析


def load_stock_entries(path="stock.json"):
    """載入 stock.json，回傳一個 dict list。若檔案不存在或解析失敗，會回傳預設值清單。"""
    default = [{
        "symbol": "ETH-USD",
        "webhook_url": "https://discord.com/api/webhooks/1426931603870978181/TQPCP9zPF8AbCEZokiZ-rrfpaeprmWWs6X0mvVtvuntCdIaFCmFpEgZ0vokelDjcEPfz"
    }]

    if not os.path.exists(path):
        print(f"{path} not found, using default entries: {default}")
        return default

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print(f"{path} is empty, using default entries")
        return default

    # 1) 嘗試整體解析（支援物件或陣列）
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2) 支援多個連續 JSON 物件（raw_decode 逐一解析）
    decoder = json.JSONDecoder()
    idx = 0
    entries = []
    L = len(text)
    while idx < L:
        try:
            obj, end = decoder.raw_decode(text, idx)
            entries.append(obj)
            idx = end
            # 跳過空白
            while idx < L and text[idx].isspace():
                idx += 1
        except ValueError:
            break

    if entries:
        return entries

    print(f"Failed to parse {path}, using default entries")
    return default


def run_pipeline_for(entry: dict):
    """對單一 entry 執行整個 pipeline：fetch -> features -> ... -> send webhook。"""
    symbol = entry.get("symbol", "ETH-USD")
    webhook = entry.get("webhook_url")
    try:
        subprocess.run([sys.executable, "fetch.py", "-t", symbol, "-y", "10"], check=True)
        subprocess.run([sys.executable, "features.py"], check=True)
        subprocess.run([sys.executable, "rolling.py"], check=True)
        subprocess.run([sys.executable, "pretidy.py"], check=True)
        subprocess.run([sys.executable, "bayesian_unified.py", "--window-weeks", "10", "--bins", "4"], check=True)
        subprocess.run([sys.executable, "backtest_signal_based.py"], check=True)
        if webhook:
            subprocess.run([sys.executable, "send_discord_webhook.py", "-w", webhook], check=True)
        else:
            print(f"No webhook provided for {symbol}, skipping webhook step")
        print(f"Pipeline for {symbol} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the pipeline for {symbol}: {e}")


def run_all(entries):
    for entry in entries:
        run_pipeline_for(entry)


if __name__ == "__main__":
    entries = load_stock_entries("stock.json")
    run_all(entries)
    print("Initial run completed. Setting up scheduler...")
    # 每天20點執行所有 entries
    schedule.every().day.at("20:00").do(run_all, entries)

    print("Scheduler started. Waiting for the next run...")
    while True:
        schedule.run_pending()
        time.sleep(30)
