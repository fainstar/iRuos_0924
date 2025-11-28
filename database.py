"""
市場數據的資料庫管理模組。
處理 SQLite 連線與資料持久化。
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = 'data/market_data.db'

def get_db_connection() -> sqlite3.Connection:
    """
    建立與 SQLite 資料庫的連線。
    
    Returns:
        sqlite3.Connection: 資料庫連線物件。
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    初始化資料庫，若資料表不存在則建立之。
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 建立 market_data 表
    # 使用 (symbol, date) 作為複合主鍵，防止重複資料
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data (
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        dividends REAL,
        stock_splits REAL,
        PRIMARY KEY (symbol, date)
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("資料庫初始化成功。")

def get_latest_date(symbol: str) -> str | None:
    """
    取得資料庫中指定股票的最新日期。
    
    Args:
        symbol (str): 股票代號 (例如: 'AAPL')。
        
    Returns:
        str | None: 最新日期字串 (YYYY-MM-DD)，若無資料則回傳 None。
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT MAX(date) FROM market_data WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

def save_market_data(symbol: str, df: pd.DataFrame) -> None:
    """
    將市場數據儲存至資料庫，使用 Upsert (插入或取代) 方式。
    
    Args:
        symbol (str): 股票代號。
        df (pd.DataFrame): 包含市場數據的 DataFrame，需包含欄位:
                           Date, Open, High, Low, Close, Volume。
    """
    if df is None or df.empty:
        logger.warning(f"沒有資料可儲存: {symbol}")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 準備資料記錄
    records = []
    for _, row in df.iterrows():
        records.append((
            symbol,
            row['Date'], # 假設 Date 已經是格式化好的字串 YYYY-MM-DD
            row.get('Open'),
            row.get('High'),
            row.get('Low'),
            row.get('Close'),
            row.get('Volume'),
            row.get('Dividends', 0),
            row.get('Stock Splits', 0)
        ))
    
    # 使用 INSERT OR REPLACE 處理重複資料 (Upsert)
    cursor.executemany('''
    INSERT OR REPLACE INTO market_data 
    (symbol, date, open, high, low, close, volume, dividends, stock_splits)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', records)
    
    conn.commit()
    conn.close()
    logger.info(f"已儲存 {len(records)} 筆記錄到資料庫 ({symbol})")

def load_market_data(symbol: str) -> pd.DataFrame:
    """
    從資料庫讀取指定股票的所有歷史資料。
    
    Args:
        symbol (str): 股票代號。
        
    Returns:
        pd.DataFrame: 包含歷史市場數據的 DataFrame。
    """
    conn = get_db_connection()
    
    query = 'SELECT * FROM market_data WHERE symbol = ? ORDER BY date ASC'
    df = pd.read_sql_query(query, conn, params=(symbol,))
    
    conn.close()
    
    if not df.empty:
        # 重新命名欄位以符合應用程式慣例 (首字大寫)
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'dividends': 'Dividends',
            'stock_splits': 'Stock Splits'
        })
    
    return df

# 模組載入時初始化資料庫
init_db()
