from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

from backtest_signal_based import SignalBasedBacktest
from bayesian_unified import BayesianStateClassifier
from logging_config import get_logger
from fetch import fetch_stock_data
from features import add_technical_indicators
from pretidy import PreTidyConfig, PreTidyProcessor
from rolling import RollingWindowBinner
from send_discord_webhook import build_embed, post_webhook


class NoNewDataError(RuntimeError):
    """Raised when no new market data is available for processing."""


FORM_SUBMIT_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLScqVfPDwwqP4QsM0SVNBLZSt1-8PXDl03EyQSfx9PmRxvvgMg/formResponse"
)


@dataclass
class PipelinePaths:
    """流水線輸出檔案的標準路徑設定"""

    raw: Path = Path("data/data.csv")
    features: Path = Path("data/feature.csv")
    rolling: Path = Path("data/rolling_window_12weeks_4bins.csv")
    final: Path = Path("data/final_data.csv")
    log_dir: Path = Path("log")
    rolling_daily: Path = Path("log/rolling_validation_daily_details.csv")
    rolling_summary: Path = Path("log/rolling_validation_results.json")
    tomorrow_signal: Path = Path("log/tomorrow_trading_signal.json")
    model: Path = Path("log/bayesian_classifier_model.pkl")


@dataclass
class PipelineConfig:
    """交易流程的主要參數設定"""

    symbol: str
    years: int = 10
    window_weeks: int = 12
    num_bins: int = 4
    lookback_days: int = 5
    initial_train_cutoff: str = "2019-12-31"
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.001425
    webhook_url: Optional[str] = None


class TradingPipeline:
    """將資料抓取、特徵處理、模型驗證與回測串成單一流程"""

    def __init__(self, config: PipelineConfig, paths: Optional[PipelinePaths] = None) -> None:
        self.config = config
        self.paths = paths or PipelinePaths()

        self.raw_data: Optional[pd.DataFrame] = None
        self.feature_data: Optional[pd.DataFrame] = None
        self.rolling_data: Optional[pd.DataFrame] = None
        self.final_data: Optional[pd.DataFrame] = None
        self.classifier: Optional[BayesianStateClassifier] = None
        self.backtest_stats: Optional[Dict[str, Any]] = None
        self.tomorrow_signal: Optional[Dict[str, Any]] = None
        self.latest_market_row: Optional[pd.Series] = None
        self.latest_data_date: Optional[pd.Timestamp] = None
        self.backend_status: Optional[int] = None

        self._prepare_directories()
        self.logger = get_logger(f"TradingPipeline[{self.config.symbol}]")

    def _prepare_directories(self) -> None:
        """建立流程所需的資料夾結構"""
        self.paths.raw.parent.mkdir(parents=True, exist_ok=True)
        self.paths.features.parent.mkdir(parents=True, exist_ok=True)
        self.paths.rolling.parent.mkdir(parents=True, exist_ok=True)
        self.paths.final.parent.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)

    def _find_market_row(self, target_date: pd.Timestamp) -> Optional[pd.Series]:
        """在原始行情資料中尋找最接近指定日期的列"""
        if self.raw_data is None or 'Date' not in self.raw_data.columns:
            return None
        market_df = self.raw_data.sort_values('Date').reset_index(drop=True)
        matches = market_df[market_df['Date'] == target_date]
        if not matches.empty:
            return matches.iloc[-1]
        earlier = market_df[market_df['Date'] < target_date]
        if not earlier.empty:
            return earlier.iloc[-1]
        return None

    def fetch_data(self) -> pd.DataFrame:
        """抓取指定標的的行情資料並儲存為 CSV"""
        self.logger.info("開始抓取 %s 歷史資料", self.config.symbol)
        data = fetch_stock_data(self.config.symbol, years=self.config.years)
        if data is None or data.empty:
            raise ValueError(f"無法取得 {self.config.symbol} 的行情資料")
        has_new_rows = bool(getattr(data, "attrs", {}).get("has_new_rows", True))
        if not has_new_rows:
            self.logger.info("未偵測到新的交易資料，流程將跳過後續步驟")
            raise NoNewDataError(f"No new data for {self.config.symbol}")
        self.raw_data = data.copy()
        if 'Date' in self.raw_data.columns:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        self.raw_data.to_csv(self.paths.raw, index=False)
        self.logger.info("原始資料共 %d 筆，已寫入 %s", len(self.raw_data), self.paths.raw)
        return self.raw_data

    def generate_features(self) -> pd.DataFrame:
        """計算技術指標並輸出特徵表"""
        if self.raw_data is None:
            raise RuntimeError("尚未抓取資料，無法計算特徵")
        self.logger.info("計算技術指標特徵")
        enriched = add_technical_indicators(self.raw_data.copy())
        self.feature_data = enriched
        self.feature_data.to_csv(self.paths.features, index=False)
        self.logger.info("特徵表共 %d 筆，已寫入 %s", len(self.feature_data), self.paths.features)
        return self.feature_data

    def build_rolling_windows(self) -> pd.DataFrame:
        """依設定建立滾動窗口並完成分箱"""
        if self.feature_data is None:
            raise RuntimeError("尚未產生特徵，無法進行滾動窗口分箱")
        self.logger.info("開始 %d 週、%d 分箱的滾動窗口處理", self.config.window_weeks, self.config.num_bins)
        processor = RollingWindowBinner()
        rolling_df = processor.process(
            self.feature_data.copy(),
            window_weeks=self.config.window_weeks,
            num_bins=self.config.num_bins,
        )
        if rolling_df.empty:
            raise ValueError("滾動窗口處理未產生結果")
        self.rolling_data = rolling_df
        self.rolling_data.to_csv(self.paths.rolling, index=False)
        self.logger.info("滾動窗口輸出共 %d 筆，已寫入 %s", len(self.rolling_data), self.paths.rolling)
        return self.rolling_data

    def pretidy_data(self) -> pd.DataFrame:
        """整理分箱後資料，留下模型所需欄位"""
        if self.rolling_data is None:
            raise RuntimeError("尚未有滾動窗口輸出，無法整理資料")
        self.logger.info("整理分箱後資料")
        config = PreTidyConfig(output_path=self.paths.final)
        processor = PreTidyProcessor(config)
        final_df = processor.process(self.rolling_data.copy())
        self.final_data = final_df
        self.final_data.to_csv(self.paths.final, index=False)
        self.logger.info("整理後資料共 %d 筆，已寫入 %s", len(self.final_data), self.paths.final)
        return self.final_data

    def run_model(self) -> Dict[str, Any]:
        """執行滾動驗證並輸出模型與訊號"""
        if self.final_data is None or self.final_data.empty:
            raise RuntimeError("尚未完成資料整理，無法執行模型驗證")
        self.logger.info("啟動貝葉斯分類滾動驗證")
        df = self.final_data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        cutoff = pd.to_datetime(self.config.initial_train_cutoff)
        initial_train_size = len(df[df['Date'] <= cutoff])
        classifier = BayesianStateClassifier(lookback_days=self.config.lookback_days)
        self.classifier = classifier

        rolling_results = classifier.rolling_validation(
            df,
            initial_train_size=initial_train_size,
            retrain_frequency=7,
        )
        with open(self.paths.rolling_summary, 'w', encoding='utf-8') as fh:
            json.dump(rolling_results, fh, ensure_ascii=False, indent=2, default=str)
        self.logger.info("滾動驗證摘要已寫入 %s", self.paths.rolling_summary)

        if not self.paths.rolling_daily.exists():
            raise FileNotFoundError(
                f"找不到每日驗證輸出 {self.paths.rolling_daily}，回測階段無法繼續"
            )

        if len(df) == 0:
            raise ValueError("整理後資料不足以產生訊號")

        final_train_size = max(int(len(df) * 0.8), self.config.lookback_days + 1)
        classifier.train_until(final_train_size)
        classifier.save_model(str(self.paths.model))
        self.logger.info("最新模型已保存至 %s", self.paths.model)

        latest_row = df.iloc[-1]
        latest_date = latest_row['Date'] if 'Date' in latest_row else None
        if isinstance(latest_date, pd.Timestamp):
            self.latest_data_date = latest_date
            self.latest_market_row = self._find_market_row(latest_date)
            if self.latest_market_row is None:
                self.logger.warning("找不到 %s 的對應市場資料，後端回報可能缺少欄位", latest_date.date())
        else:
            self.latest_data_date = None
            self.latest_market_row = None

        signal = classifier.get_next_day_signal(latest_row)
        signal['symbol'] = self.config.symbol
        self.tomorrow_signal = signal
        with open(self.paths.tomorrow_signal, 'w', encoding='utf-8') as fh:
            json.dump(signal, fh, ensure_ascii=False, indent=2, default=str)
        self.logger.info("明日訊號已寫入 %s", self.paths.tomorrow_signal)

        return rolling_results

    def run_backtest(self) -> Dict[str, Any]:
        """根據最新滾動驗證結果執行回測"""
        self.logger.info("開始回測交易策略")
        backtest = SignalBasedBacktest(
            csv_file=str(self.paths.rolling_daily),
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            symbol=self.config.symbol,
        )
        if not backtest.load_data():
            raise RuntimeError("載入回測資料失敗")
        backtest.run_backtest()
        backtest.print_summary()
        stats = backtest.save_results()
        backtest.plot_results()
        self.backtest_stats = stats
        return stats

    def send_webhook(self) -> Optional[int]:
        """若設定 Discord webhook，則送出明日訊號"""
        if not self.config.webhook_url:
            self.logger.info("未設定 webhook，略過通知步驟")
            return None
        if not self.tomorrow_signal:
            raise RuntimeError("尚未產生明日訊號，無法推送 webhook")
        self.logger.info("向 Discord webhook 送出明日訊號")
        payload = {"embeds": [build_embed(self.tomorrow_signal)]}
        status = post_webhook(self.config.webhook_url, payload)
        if status // 100 != 2 and status != 0:
            self.logger.warning("Webhook 回應非 2xx：%s", status)
        return status

    def submit_backend_form(self) -> Optional[int]:
        """將最新行情與模型預測回填到後端表單"""
        if not self.tomorrow_signal:
            self.logger.info("尚未產生交易訊號，略過後端表單填寫")
            return None

        if self.latest_market_row is None:
            self.logger.warning("缺少對應的市場資料，無法提交後端表單")
            return None

        market_row = self.latest_market_row
        probabilities = self.tomorrow_signal.get('action_probabilities', {})

        def _safe_float(value: Any, digits: int = 4) -> str:
            if value is None or pd.isna(value):
                return "0"
            return f"{float(value):.{digits}f}"

        open_price = _safe_float(market_row.get('Open'))
        close_price = _safe_float(market_row.get('Close'))
        high_price = _safe_float(market_row.get('High'))
        low_price = _safe_float(market_row.get('Low'))
        volume_value = market_row.get('Volume')
        volume = "0"
        if volume_value is not None and not pd.isna(volume_value):
            volume = str(int(float(volume_value)))

        buy_prob = _safe_float(probabilities.get('buy', 0.0))
        sell_prob = _safe_float(probabilities.get('sell', 0.0))
        action = self.tomorrow_signal.get('recommended_action', 'hold')

        form_payload = {
            'entry.1837756886': self.config.symbol,
            'entry.1446693099': open_price,
            'entry.2111181123': close_price,
            'entry.1470983622': high_price,
            'entry.1477170485': low_price,
            'entry.624137556': volume,
            'entry.1287453826': buy_prob,
            'entry.1294058507': sell_prob,
            'entry.908197911': action,
        }

        try:
            response = requests.post(
                FORM_SUBMIT_URL,
                data=form_payload,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10,
            )
        except requests.RequestException as exc:
            self.logger.warning("提交 Google Form 失敗：%s", exc)
            return None

        self.backend_status = response.status_code
        if response.status_code // 100 != 2:
            self.logger.warning("Google Form 回應狀態碼 %s", response.status_code)
        else:
            self.logger.info("已提交表單至後端，狀態碼 %s", response.status_code)

        return response.status_code

    def run(self) -> Dict[str, Any]:
        """執行整體流程並回傳各階段輸出"""
        self.logger.info("=== 開始執行 %s 流水線 ===", self.config.symbol)
        try:
            self.fetch_data()
        except NoNewDataError:
            self.logger.info("今日尚無新資料，已跳過 %s 流水線", self.config.symbol)
            return {
                "skipped": True,
                "reason": "no_new_data",
                "symbol": self.config.symbol,
            }
        self.generate_features()
        self.build_rolling_windows()
        self.pretidy_data()
        rolling_results = self.run_model()
        backtest_stats = self.run_backtest()
        webhook_status = self.send_webhook()
        backend_status = self.submit_backend_form()
        self.logger.info("=== %s 流水線完成 ===", self.config.symbol)
        return {
            "rolling_results": rolling_results,
            "tomorrow_signal": self.tomorrow_signal,
            "backtest_stats": backtest_stats,
            "webhook_status": webhook_status,
            "backend_status": backend_status,
        }
