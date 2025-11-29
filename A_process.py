import json
import os
import schedule
import time

from logging_config import get_logger, setup_logging
from pipeline import PipelineConfig, TradingPipeline


logger = get_logger(__name__)


# 這個檔案會從同目錄下的 `stock.json` 讀取要處理的股票(symbol)與對應的 webhook_url
# 支援的格式範例：
# 單一物件:
# {"symbol": "ETH-USD", "webhook_url": "https://..."}
# 或是陣列:
# [{"symbol": "BTC-USD", "webhook_url": "https://..."}, {"symbol": "ETH-USD", "webhook_url": "https://..."}]
# 或是連續多個 JSON 物件 (多個物件串在一起 / NDJSON)：本程式會嘗試逐一解析


def load_stock_entries(path="stock.json"):
    """
    從 stock.json 載入股票設定。
    
    Args:
        path (str): JSON 檔案路徑。
        
    Returns:
        list: 股票設定字典列表。
    """
    default = [{
        "symbol": "ETH-USD",
        "webhook_url": "https://discord.com/api/webhooks/1426931603870978181/TQPCP9zPF8AbCEZokiZ-rrfpaeprmWWs6X0mvVtvuntCdIaFCmFpEgZ0vokelDjcEPfz"
    }]

    if not os.path.exists(path):
        logger.warning("%s not found, using default entries: %s", path, default)
        return default

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        logger.warning("%s is empty, using default entries", path)
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

    logger.warning("Failed to parse %s, using default entries", path)
    return default


def run_pipeline_for(entry: dict):
    """依照設定執行單一標的的完整流水線"""
    symbol = entry.get("symbol", "ETH-USD")
    config = PipelineConfig(
        symbol=symbol,
        years=entry.get("years", 10),
        window_weeks=entry.get("window_weeks", 12),
        num_bins=entry.get("num_bins", 4),
        lookback_days=entry.get("lookback_days", 5),
        initial_train_cutoff=entry.get("initial_train_cutoff", "2019-12-31"),
        initial_capital=entry.get("initial_capital", 1_000_000.0),
        commission_rate=entry.get("commission_rate", 0.001425),
        webhook_url=entry.get("webhook_url"),
    )

    try:
        pipeline = TradingPipeline(config)
        result = pipeline.run()
        if isinstance(result, dict) and result.get("skipped"):
            logger.info("No new data for %s. Pipeline skipped.", symbol)
        else:
            logger.info("Pipeline for %s completed successfully", symbol)
        return result
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error while running the pipeline for %s", symbol)
        return None


def run_all(entries):
    """依序對所有設定執行流水線"""
    results = []
    for entry in entries:
        results.append(run_pipeline_for(entry))
    return results


if __name__ == "__main__":
    setup_logging()
    entries = load_stock_entries("stock.json")
    run_all(entries)
    logger.info("Initial run completed. Setting up scheduler...")
    # 每天15點執行所有 entries
    schedule.every().day.at("15:00").do(run_all, entries)

    logger.info("Scheduler started. Waiting for the next run...")
    while True:
        schedule.run_pending()
        time.sleep(30)
