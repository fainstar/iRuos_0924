from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringConfig:
    input_path: Path = Path("data/data.csv")
    output_path: Path = Path("data/feature.csv")


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic_oscillator(
    data: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    low_min = data["Low"].rolling(window=k_period).min()
    high_max = data["High"].rolling(window=k_period).max()
    k_percent = 100 * ((data["Close"] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent


def calculate_macd(
    data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = data["Close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    highest_high = data["High"].rolling(window=period).max()
    lowest_low = data["Low"].rolling(window=period).min()
    williams_r = -100 * ((highest_high - data["Close"]) / (highest_high - lowest_low))
    return williams_r


def calculate_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad_tp = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=False
    )
    cci = (typical_price - sma_tp) / (0.015 * mad_tp)
    return cci


def calculate_obv(data: pd.DataFrame) -> pd.Series:
    obv = pd.Series(index=data.index, dtype=float)
    obv.iloc[0] = 0

    for i in range(1, len(data)):
        if data["Close"].iloc[i] > data["Close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + data["Volume"].iloc[i]
        elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - data["Volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    return obv


def calculate_adx(data: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            np.abs(data["High"] - data["Close"].shift(1)),
            np.abs(data["Low"] - data["Close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)

    dm_plus = np.where(
        (data["High"] - data["High"].shift(1)) > (data["Low"].shift(1) - data["Low"]),
        np.maximum(data["High"] - data["High"].shift(1), 0),
        0,
    )
    dm_minus = np.where(
        (data["Low"].shift(1) - data["Low"]) > (data["High"] - data["High"].shift(1)),
        np.maximum(data["Low"].shift(1) - data["Low"], 0),
        0,
    )

    atr = tr.rolling(window=period).mean()
    di_plus = 100 * (pd.Series(dm_plus).rolling(window=period).mean() / atr)
    di_minus = 100 * (pd.Series(dm_minus).rolling(window=period).mean() / atr)

    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()

    return adx, di_plus, di_minus


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values("Date").copy()

    data["RSI_3"] = calculate_rsi(data, period=3)
    data["RSI_5"] = calculate_rsi(data, period=5)
    data["RSI_7"] = calculate_rsi(data, period=7)
    data["RSI_10"] = calculate_rsi(data, period=10)
    data["RSI_14"] = calculate_rsi(data, period=14)

    data["%K"], data["%D"] = calculate_stochastic_oscillator(data)

    data["MACD"], data["MACD_Signal"], data["MACD_Histogram"] = calculate_macd(data)

    data["SMA_5"] = data["Close"].rolling(window=5).mean()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()

    data["EMA_5"] = data["Close"].ewm(span=5, adjust=False).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    sma_20 = data["Close"].rolling(window=20).mean()
    std_20 = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = sma_20 + (std_20 * 2)
    data["BB_Lower"] = sma_20 - (std_20 * 2)
    data["BB_Middle"] = sma_20

    data["ATR_14"] = calculate_atr(data, period=14)
    data["Williams_R"] = calculate_williams_r(data, period=14)
    data["CCI_20"] = calculate_cci(data, period=20)
    data["OBV"] = calculate_obv(data)
    data["ADX"], data["DI_Plus"], data["DI_Minus"] = calculate_adx(data, period=14)

    data["Price_Change_Pct"] = data["Close"].pct_change() * 100
    data["Volume_SMA_20"] = data["Volume"].rolling(window=20).mean()
    data["Price_SMA_20_Diff"] = data["Close"] - data["SMA_20"]
    data["Price_EMA_12_Diff"] = data["Close"] - data["EMA_12"]

    return data


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到輸入檔案: {path}")
    logger.info("讀取原始資料: %s", path)
    return pd.read_csv(path)


def save_dataset(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    logger.info("已將特徵資料輸出至: %s", path)


def run_feature_engineering(config: FeatureEngineeringConfig) -> pd.DataFrame:
    dataset = load_dataset(config.input_path)
    enriched = add_technical_indicators(dataset)
    save_dataset(enriched, config.output_path)
    return enriched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="對原始股價資料產生技術指標特徵")
    parser.add_argument(
        "--input",
        type=Path,
        default=FeatureEngineeringConfig.input_path,
        help="輸入的原始股價 CSV (預設: data/data.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FeatureEngineeringConfig.output_path,
        help="輸出的特徵 CSV (預設: data/feature.csv)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日誌層級 (預設: INFO)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level))

    config = FeatureEngineeringConfig(input_path=args.input, output_path=args.output)

    try:
        result = run_feature_engineering(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "技術指標已成功添加，共 %d 筆資料 (日期範圍: %s ~ %s)",
        len(result),
        result["Date"].min() if "Date" in result.columns else "未知",
        result["Date"].max() if "Date" in result.columns else "未知",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
