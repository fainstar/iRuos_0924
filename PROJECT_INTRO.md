# iRuos_0924 專案介紹

## 專案概要

此專案聚焦於以程式化方式整合數據抓取、技術指標計算、特徵離散化、貝葉斯分類滾動驗證與訊號回測的完整自動化交易流程。透過統一的 `TradingPipeline` 類別，使用者可以對指定標的自動完成資料處理、模型訓練、績效評估與多渠道回報。

## 核心特點

- **SQLite 增量抓取**：`fetch.py` 搭配 `database.py`，只更新最新行情，避免重複下載，確保資料完整性。
- **技術指標與分箱標準化**：`features.py` 計算多種技術指標，`rolling.py` 以 12 週窗口、4 分箱離散化特徵，使模型能處理分類資料。
- **貝葉斯滾動驗證**：`bayesian_unified.py` 針對每日資料進行連續驗證，每逢週三重訓，並輸出詳細的每日預測結果與摘要。
	最新版本採用預編碼快取：首次訓練後儲存已 LabelEncoder 處理的特徵矩陣，後續週三重訓與最終訓練僅需切片同一份 `numpy` 陣列，大幅縮短迭代時間且不影響結果。
- **策略回測與視覺化**：`backtest_signal_based.py` 依訊號進行交易模擬，輸出統計檔、交易紀錄與年度走勢圖，並在圖表與檔名中標註標的資訊。
- **自動通知與後端回報**：`TradingPipeline` 會推送 Discord webhook，並將最新行情與預測結果同步提交至 Google Form，便於後續資訊整合。

## 流程總覽

1. **資料抓取**：`fetch.py` 依指定年數從 Yahoo Finance 下載資料，經正規化後寫入 `data/data.csv`，並保存至 SQLite 以利增量更新。
2. **特徵工程**：`features.py` 對原始資料加入 RSI、MACD、布林通道等技術指標，輸出 `data/feature.csv`。
3. **滾動窗口分箱**：`rolling.py` 利用 `RollingWindowBinner` 將特徵按週期分箱，形成 `data/rolling_window_12weeks_4bins.csv`。
4. **資料整理**：`pretidy.py` 保留必要欄位與分箱結果，去除缺失值，輸出 `data/final_data.csv`。
5. **模型訓練與驗證**：`bayesian_unified.py` 執行滾動驗證與最終模型訓練。過程中會先建立完整資料的預編碼快取，再於每次週期性重訓直接切片重用，產出每日預測與明日訊號於 `log/` 目錄。
6. **策略回測**：`backtest_signal_based.py` 讀取驗證結果 CSV，模擬以隔天開盤價交易，生成統計 JSON、每日資金變化 CSV、交易紀錄與年度圖表。
7. **通知與後端提交**：若設定 Webhook，`send_discord_webhook.py` 會推送嵌入訊息；同時 `TradingPipeline` 會將最新行情與預測機率上傳至 Google Form。

## 主要模組職責

| 模組 | 角色 | 主要輸出 |
| --- | --- | --- |
| `fetch.py` | 下載並正規化行情資料 | `data/data.csv`、SQLite 市場資料 |
| `features.py` | 計算技術指標 | `data/feature.csv` |
| `rolling.py` | 建立滾動窗口並分箱 | `data/rolling_window_12weeks_4bins.csv` |
| `pretidy.py` | 篩選欄位並清洗資料 | `data/final_data.csv` |
| `bayesian_unified.py` | 滾動驗證、輸出訊號 | `log/rolling_validation_daily_details.csv`、`log/tomorrow_trading_signal.json` 等 |
| `backtest_signal_based.py` | 根據訊號進行回測 | 回測統計 JSON、交易紀錄 CSV、年度圖表 |
| `pipeline.py` | 將各階段整合成單一流程 | 同時整合 webhook 與表單提交結果 |
| `A_process.py` | 讀取 `stock.json`，排程執行流水線 | 多標的批次執行與每日排程 |

## 使用方式

1. 於 `stock.json` 列出欲處理的標的、Webhook 及可選參數（如年數、初始資金等）。
2. 執行 `python3 A_process.py`，即可依設定依序完成整套流程，並於每日 15:00 自動重跑。
3. 所有中間產物與結果會存放於 `data/` 與 `log/` 目錄，便於定期檢視或進一步分析。

## 延伸方向

- 增加更多模型或特徵選擇策略，以比較不同策略績效。
- 將回測結果導入視覺化 dashboard，提供互動式分析。
- 加入測試框架（如 pytest）與 CI/CD 自動化流程。
- 強化錯誤處理與重試機制，確保排程任務的穩定性。
