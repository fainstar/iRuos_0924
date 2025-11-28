# 專案總覽

- `A_process.py` 作為入口，讀取 `stock.json` 逐一執行 `TradingPipeline`，首輪完成後以 `schedule` 安排每日 15:00 重新執行。
- `pipeline.TradingPipeline` 負責整合資料抓取、特徵工程、滾動分箱、資料整理、貝氏分類、回測、Discord 通知與 Google 表單回填，並統一管理輸出至 `data/` 與 `log/`。
- `fetch.py` 搭配 `database.py` 實作多來源的 Yahoo Finance 擷取策略，透過 SQLite (`data/market_data.db`) 做增量更新與快取。
- `features.py`、`rolling.py`、`pretidy.py` 分別產生技術指標、切出週期滾動視窗並分箱，以及選取欄位加上缺值清理，輸出建模用 `final_data.csv`。
- `bayesian_unified.py` 使用 CategoricalNB 搭配 LabelEncoder 進行週三重訓的 walk-forward 驗證，輸出每日預測 CSV、儲存模型與明日訊號。
- `backtest_signal_based.py` 依據明日開盤價模擬交易訊號（40% 閾值），計算報酬/統計並繪製年度圖表；`send_discord_webhook.py` 組裝含 ASCII 機率條的 Discord embed 並推送。

## 優點與亮點

- 流程清楚（抓取 → 特徵 → 分箱 → 整理 → 模型 → 回測 → 通知），且各模組皆保留 CLI 介面便於單步調試。
- SQLite 支援資料增量同步，減少重複下載；輸出紀錄（CSV、JSON、log）完整，方便追蹤。
- 模型與訊號檔案齊全，易於驗證與重複使用；Discord 通知包含信心度與機率資訊，閱讀體驗佳。

## 風險與痛點

- `bayesian_unified.py`、`backtest_signal_based.py` 體積龐大且邏輯與 CLI 混雜，提升維護成本。
- `stock.json` 直接暴露正式 webhook，沒有秘密管理或環境區隔；整體缺乏自動化測試與 CI，也未提供依賴清單。
- `A_process.py` 的排程遇到例外僅輸出錯誤訊息，缺乏重試/告警；策略參數（40% 閾值、週三重訓等）硬編寫彈性不足。
- `data/`、`log/` 缺乏清理政策，長期執行可能造成磁碟壓力。

## 改進建議

1. **設定與秘密管理**：將 webhook、閾值、排程時間移至 `.env` 或設定檔，提供 `stock.json.example` 並透過環境變數注入正式值。
2. **模組化重構**：拆分大型模組為子套件（例如 `models/`、`backtest/`、`notifications/`），萃取共用工具以提升重用性與測試性。
3. **測試與 CI**：為特徵計算、滾動分箱、模型資料前處理等撰寫單元測試，並在 GitHub Actions 上建立自動化流程。
4. **依賴與部署文件**：補齊 `requirements.txt` 或 `pyproject.toml`，於 README 說明安裝與執行步驟，推薦使用虛擬環境或 Docker。
5. **監控與告警**：排程或任一階段失敗時發送警示（Discord/Email）；關鍵步驟增加重試與異常處理。
6. **策略彈性**：將回測閾值、佣金、模型 retrain 頻率等抽為設定項，並考慮支援多模型或策略比較。
7. **資料治理**：為 `data/`、`log/` 制定保留策略（壓縮、定期清理），寫檔時採臨時檔再 rename 以避免半成品。
8. **報表化輸出**：將結果匯整成 HTML 或 Markdown 報表，集中呈現績效、指標與交易明細。
9. **長期部署**：若需長期常駐，建議使用 Docker 搭配 cron/systemd 管理，以確保安裝與重啟穩定。
10. **文件補強**：撰寫 README 或架構圖，描述完整 pipeline、資料流與輸出說明，降低新進或後續維護門檻。
