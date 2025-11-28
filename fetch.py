from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from json.decoder import JSONDecodeError
from typing import Callable, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
import database  # Added database import

if sys.platform == "win32":
    import io as _io  # type: ignore

    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")  # type: ignore[attr-defined]


_YF_SESSION: Optional[requests.Session] = None
_YF_CRUMB: Optional[str] = None


@dataclass
class FetchAttempt:
    label: str
    loader: Callable[[], pd.DataFrame]


def get_yf_session() -> requests.Session:
    global _YF_SESSION
    if _YF_SESSION is not None:
        return _YF_SESSION

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        }
    )

    _YF_SESSION = session
    return session


def get_yahoo_crumb(session: requests.Session) -> str:
    global _YF_CRUMB
    if _YF_CRUMB:
        return _YF_CRUMB

    resp = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15)
    resp.raise_for_status()
    crumb = resp.text.strip()
    if not crumb:
        raise ValueError("Yahoo Finance crumb 為空")
    _YF_CRUMB = crumb
    return crumb


def warm_up_session(session: requests.Session, ticker_symbol: str) -> None:
    warm_up_urls = [
        f"https://finance.yahoo.com/quote/{ticker_symbol}",
        f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker_symbol}",
    ]

    for url in warm_up_urls:
        try:
            session.get(url, timeout=10)
        except Exception:
            continue

    try:
        get_yahoo_crumb(session)
    except Exception:
        pass


def _normalize_history_df(raw_df: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        raise ValueError("yfinance 回傳了空的資料集")

    df = raw_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in df.columns]

    if "Date" not in df.columns:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)

    if "Date" not in df.columns:
        if "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
        else:
            first_col = df.columns[0]
            df.rename(columns={first_col: "Date"}, inplace=True)

    if "Date" not in df.columns:
        raise ValueError("資料缺少日期欄位，無法後續處理")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        if col not in df.columns:
            raise ValueError(f"資料缺少必要欄位 {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").round(6)

    df = df.dropna(subset=price_columns)

    if "Volume" not in df.columns:
        df["Volume"] = 0
    else:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    for col in ["Dividends", "Stock Splits"]:
        if col not in df.columns:
            df[col] = 0

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def _direct_download_loader(ticker_symbol: str, years: int) -> pd.DataFrame:
    session = get_yf_session()
    warm_up_session(session, ticker_symbol)
    try:
        crumb = get_yahoo_crumb(session)
    except Exception:
        crumb = None

    end = datetime.utcnow()
    start = end - timedelta(days=int(years * 366))

    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    if crumb:
        params["crumb"] = crumb

    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker_symbol}"
    response = session.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.text[:120]}")

    if not response.text or response.text.strip().startswith("<!DOCTYPE"):
        raise ValueError("回傳內容非 CSV")

    csv_buffer = io.StringIO(response.text)
    df = pd.read_csv(csv_buffer)
    return df


def _chart_api_loader(ticker_symbol: str, years: int) -> pd.DataFrame:
    session = get_yf_session()

    params = {
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
        "corsDomain": "finance.yahoo.com",
    }

    params["range"] = "max" if years > 10 else f"{max(1, years)}y"

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker_symbol}"
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()

    try:
        payload = response.json()
    except JSONDecodeError as exc:
        raise ValueError(f"chart api JSON 解析失敗: {exc}") from exc

    chart = payload.get("chart", {})
    error_info = chart.get("error")
    if error_info:
        raise ValueError(f"chart api error: {error_info}")

    results = chart.get("result")
    if not results:
        raise ValueError("chart api 無返回結果")

    result = results[0]
    timestamps = result.get("timestamp")
    indicators = result.get("indicators", {})
    quote_list = indicators.get("quote", [{}])
    adjclose_list = indicators.get("adjclose", [{}])

    if not timestamps:
        raise ValueError("chart api 無時間戳")

    quote = quote_list[0] if quote_list else {}
    adjclose = adjclose_list[0] if adjclose_list else {}

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Adj Close": adjclose.get("adjclose"),
            "Volume": quote.get("volume"),
        }
    )

    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        df["Close"] = df["Adj Close"].where(df["Adj Close"].notna(), df["Close"])

    df.drop(columns=[col for col in df.columns if col.lower() == "adj close"], inplace=True, errors="ignore")

    return df


def build_fetch_attempts(
    ticker_symbol: str, years: int, session: requests.Session
) -> Tuple[List[FetchAttempt], Optional[str]]:
    attempts: List[FetchAttempt] = []
    initialization_error: Optional[str] = None

    history_periods = [f"{years}y"]
    if years > 5:
        history_periods.append("5y")
    history_periods.append("max")

    try:
        ticker = yf.Ticker(ticker_symbol, session=session)
    except Exception as exc:
        initialization_error = f"初始化 yfinance Ticker 失敗: {exc}"
    else:
        for period in history_periods:
            attempts.append(
                FetchAttempt(
                    label=f"Ticker.history(period={period})",
                    loader=lambda p=period: ticker.history(
                        period=p,
                        interval="1d",
                        auto_adjust=False,
                        actions=True,
                    ),
                )
            )

    download_periods = [f"{years}y", "max"]
    for period in download_periods:
        attempts.append(
            FetchAttempt(
                label=f"yf.download(period={period})",
                loader=lambda p=period: yf.download(
                    tickers=ticker_symbol,
                    period=p,
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    threads=False,
                    session=session,
                ),
            )
        )

    attempts.append(
        FetchAttempt(
            label="direct_csv_download",
            loader=lambda: _direct_download_loader(ticker_symbol, years),
        )
    )

    attempts.append(
        FetchAttempt(
            label="chart_api",
            loader=lambda: _chart_api_loader(ticker_symbol, years),
        )
    )

    return attempts, initialization_error


def run_fetch_attempts(
    ticker_symbol: str,
    attempts: Iterable[FetchAttempt],
    initial_error: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    last_error = initial_error

    for attempt in attempts:
        try:
            raw_data = attempt.loader()
        except (ValueError, JSONDecodeError) as err:
            last_error = f"{attempt.label} 解析失敗: {err}"
            continue
        except Exception as err:
            last_error = f"{attempt.label} 發生例外: {err}"
            continue

        if raw_data is None or (hasattr(raw_data, "empty") and raw_data.empty):
            last_error = f"{attempt.label} 回傳空資料"
            continue

        try:
            normalized = _normalize_history_df(raw_data, ticker_symbol)
        except Exception as err:
            last_error = f"{attempt.label} 正規化資料失敗: {err}"
            continue

        return normalized, None

    return None, last_error or "未知原因"


def fetch_stock_data(ticker_symbol: str, years: int = 5) -> Optional[pd.DataFrame]:
    ticker_symbol = ticker_symbol.upper()
    years = max(1, min(years, 20))

    # 1. Check DB for latest date
    latest_date_str = database.get_latest_date(ticker_symbol)
    new_data = None

    if latest_date_str:
        print(f"Found existing data for {ticker_symbol} up to {latest_date_str}")
        try:
            latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
            start_date = latest_date + timedelta(days=1)
            
            if start_date > datetime.now():
                print("Data is up to date.")
            else:
                print(f"Fetching new data from {start_date.strftime('%Y-%m-%d')}...")
                session = get_yf_session()
                # Use yf.download for incremental update
                new_data = yf.download(
                    tickers=ticker_symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    threads=False,
                    session=session,
                )
                
                if new_data is not None and not new_data.empty:
                    try:
                        new_data = _normalize_history_df(new_data, ticker_symbol)
                    except Exception as e:
                        print(f"Error normalizing incremental data: {e}")
                        new_data = None
        except Exception as e:
            print(f"Error checking/fetching incremental data: {e}")
            # Fallback to full fetch if incremental fails
            latest_date_str = None 

    if latest_date_str is None:
        # Full fetch
        try:
            print(f"Fetching {ticker_symbol} data for {years} years...")
        except Exception:
            print(f"Fetching {ticker_symbol} data for {years} years...")

        session = get_yf_session()
        warm_up_session(session, ticker_symbol)

        attempts, initialization_error = build_fetch_attempts(ticker_symbol, years, session)
        new_data, error_message = run_fetch_attempts(ticker_symbol, attempts, initialization_error)
        
        if new_data is None:
            reason = error_message or "未知原因"
            print(f"Error fetching {ticker_symbol}: {reason}")

    # 2. Save new data to DB
    if new_data is not None and not new_data.empty:
        database.save_market_data(ticker_symbol, new_data)

    # 3. Load full data from DB
    full_data = database.load_market_data(ticker_symbol)

    if not full_data.empty:
        try:
            print(f"成功抓取 {len(full_data)} 條數據 (從 {full_data['Date'].min()} 到 {full_data['Date'].max()})")
        except Exception:
            print(f"Successfully fetched {len(full_data)} records")
        return full_data

    return None


def build_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器"""
    parser = argparse.ArgumentParser(description="抓取股票歷史資料")
    parser.add_argument("-t", "--ticker", type=str, help="股票代號 (例如: AAPL, 2330.TW, ETH-USD)")
    parser.add_argument("-y", "--years", type=int, default=5, help="要抓取的年數 (預設: 5)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/data.csv",
        help="輸出文件路徑 (預設: data/data.csv)",
    )
    parser.add_argument("--interactive", action="store_true", help="使用互動模式輸入參數")
    return parser


def get_user_input() -> Tuple[str, int]:
    """互動式取得使用者輸入的股票代號與年數"""
    print("=" * 50)
    print("📈 股票資料抓取工具")
    print("=" * 50)

    while True:
        ticker = input("請輸入股票代號 (例如: AAPL, 2330.TW, ETH-USD): ").strip().upper()
        if ticker:
            break
        print("股票代號不能為空，請重新輸入")

    while True:
        years_input = input("請輸入要抓取的年數 (預設5年，直接按Enter使用預設值): ").strip()
        if not years_input:
            return ticker, 5
        try:
            years = int(years_input)
        except ValueError:
            print("請輸入有效的數字")
            continue
        if 0 < years <= 20:
            return ticker, years
        print("年數必須在1-20之間")


def main(argv: Optional[Iterable[str]] = None) -> int:
    """命令列入口點"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.interactive or (not args.ticker):
        ticker, years = get_user_input()
        filename = args.output
    else:
        ticker = args.ticker.upper()
        years = args.years
        filename = args.output

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = fetch_stock_data(ticker, years=years)

    if data is not None:
        data.to_csv(filename, index=False)
        print(f"✅ 股票 {ticker} 的 {years} 年原始資料已成功抓取，並保存到 {filename}")
        print(f"📊 數據概覽: {len(data)} 條記錄，日期範圍: {data['Date'].min()} - {data['Date'].max()}")
        return 0

    print(f"❌ 無法獲取股票 {ticker} 的資料，請檢查股票代號是否正確")
    return 1


if __name__ == "__main__":
    sys.exit(main())

