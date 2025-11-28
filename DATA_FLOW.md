# 資料流與訓練週期概覽

本文整理 iRuos_0924 專案在每日排程中從資料抓取到模型滾動驗證、回測的資料流向與週期設定，方便理解流程中各階段使用的資料範圍與重新訓練頻率。

## 整體流程一覽

1. **行情抓取 (`fetch.py`)**
   - 讀取 `stock.json` 指定的股票代號與年數（預設 10 年）。
   - 透過 SQLite 追蹤已下載的最晚日期，僅補抓缺少的最新資料；首次執行會抓取完整歷史。
   - 將乾淨的行情儲存至 `data/data.csv`，並同步寫入 `data/market_data.db`。

2. **技術指標計算 (`features.py`)**
   - 以 `data/data.csv` 為輸入，為全部日期計算 RSI、MACD、布林通道、ADX 等複合指標。
   - 產生完整的特徵表 `data/feature.csv`。

3. **滾動窗口分箱 (`rolling.py`)**
   - 使用 `RollingWindowBinner` 針對特徵資料建構 *12 週* 視窗，並將數值特徵離散化為 *4 分箱*。
   - 每個視窗以週為單位滑動，產出 `data/rolling_window_12weeks_4bins.csv`。

4. **資料整理 (`pretidy.py`)**
   - 保留基本欄位、分箱欄與視窗計算欄位，移除缺漏值。
   - 輸出模型用最終資料 `data/final_data.csv`。

5. **貝葉斯滾動驗證 (`bayesian_unified.py`)**
   - 以 `data/final_data.csv` 為基礎進行滾動式驗證，以下為關鍵設定：
     - **訓練資料期間**：所有日期中 *2019-12-31 以前* 的資料（可透過 `initial_train_cutoff` 調整）。
     - **滾動驗證範圍**：訓練集之後的全部資料，會逐日產生預測。
     - **重新訓練週期**：實務上以「每週三」為條件重新訓練（程式中透過 `weekday()==2` 判斷）。
     - **模型 lookback**：預設以過去五個交易日 (`lookback_days=5`) 的資料計算未來報酬率分類。
       - **預編碼快取**：首次訓練前會先用 LabelEncoder 將所有分箱欄位轉為 `numpy` 陣列，後續週三重訓與最終訓練僅需對快取切片，能維持結果一致同時節省大量轉換時間。
   - 輸出內容：
     - 每日預測表 `log/rolling_validation_daily_details.csv`
     - 驗證摘要 `log/rolling_validation_results.json`
     - 明日訊號 `log/tomorrow_trading_signal.json`
     - 儲存最新模型 `log/bayesian_classifier_model.pkl`

6. **基於訊號的回測 (`backtest_signal_based.py`)**
   - 讀取 `log/rolling_validation_daily_details.csv`，模擬隔日開盤價交易，並套用手續費、資本設定。
   - 輸出：回測統計 JSON、每日資金變化 CSV、交易紀錄 CSV、年度走勢圖。

7. **通知與後端提交 (`pipeline.py` 整合)**
   - 若設定 webhook，呼叫 `send_discord_webhook.py` 推送訊息。
   - 以最新行情與預測結果填寫 Google Form，記錄於管理後端。

## 資料範圍與週期重點

| 階段 | 資料範圍 | 週期/頻率 | 備註 |
| --- | --- | --- | --- |
| 行情抓取 | 首次：完整歷史；之後：自上一筆日期至今 | 每日排程執行一次 | 透過 SQLite 確保增量更新 |
| 特徵計算 | `data/data.csv` 的所有日期 | 每次管線執行 | 指標需完整序列支援滾動計算 |
| 滾動分箱 | 同上，滑動 12 週視窗 | 每次管線執行 | 視窗重疊，結果寫入 CSV |
| 滾動驗證訓練集 | `Date <= 2019-12-31` 的資料 | 每次重新訓練時使用 | `initial_train_cutoff` 可在 `stock.json` 中 per-entry 調整 |
| 滾動驗證重新訓練 | 每逢週三 (`weekday()==2`) | 驗證迴圈內判斷 | 重新訓練後立即用當天資料預測，並透過預編碼快取避免重建特徵 |
| Lookback 天數 | 5 日 | 固定設定，可在 `stock.json` 覆寫 | 影響未來報酬率計算與分類 | 
| 回測 | 驗證結果涵蓋的所有日期 | 每次管線執行 | 以隔日開盤成交，模擬全程策略 |

## 後續調整建議

- 若欲縮短訓練集期間，可在 `stock.json` 的條目中設定 `"initial_train_cutoff": "YYYY-MM-DD"`。
- 若實際作業希望只以最新 12 週資料重算，可延伸以下方向：
  1. 保留前次特徵與分箱結果，只針對增量資料處理。
  2. 針對 `bayesian_unified.py` 增加增量訓練或窗口限制（例如僅保留最近 N 週資料）。
- 若週期需要調整，每週重新訓練的判斷（預設週三）可變更為不同 weekday 或固定日數 (`retrain_frequency`)。

## 流程示意圖

```
stock.json
    ↓ (A_process.py)
TradingPipeline
    ↓ fetch_stock_data → data/data.csv + SQLite 市場資料庫
    ↓ add_technical_indicators → data/feature.csv
    ↓ RollingWindowBinner → data/rolling_window_12weeks_4bins.csv
    ↓ PreTidyProcessor → data/final_data.csv
    ↓ BayesianStateClassifier (40% 閾值, 週三再訓練)
          ↳ log/rolling_validation_daily_details.csv
          ↳ log/rolling_validation_results.json
          ↳ log/tomorrow_trading_signal.json
    ↓ SignalBasedBacktest → 回測報表與圖表
    ↓ Discord Webhook + Google Form 提交
```

以上即為目前資料流與訓練週期的整體說明，方便未來調整設定或延伸開發時參考。
