from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List
import sys
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreTidyConfig:
    input_path: Path = Path('data/rolling_window_10weeks_4bins.csv')
    output_path: Path = Path('data/final_data.csv')
    base_columns: List[str] = field(default_factory=lambda: [
        'Date',
        'Open',
        'Close',
    ])
    bin_suffix: str = '_bin'


class PreTidyProcessor:
    def __init__(self, config: PreTidyConfig):
        self.config = config

    def select_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("正在選擇欄位...")
        selected_columns = self.config.base_columns.copy()
        bin_columns = [col for col in data.columns if col.endswith(self.config.bin_suffix)]
        selected_columns.extend(bin_columns)
        selected_columns = [col for col in selected_columns if col in data.columns]
        logger.debug("選擇了 %d 個欄位", len(selected_columns))
        return data[selected_columns]

    def drop_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("刪除缺失值行...")
        before_drop = len(data)
        cleaned = data.dropna()
        logger.info("刪除了 %d 行缺失值", before_drop - len(cleaned))
        return cleaned

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("開始整理數據...")
        df = self.select_columns(data)
        df = self.drop_missing(df)
        logger.info("整理完成，共 %d 條記錄", len(df))
        return df


def load_rolling_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到滾動窗口輸入檔: {path}")
    logger.info("讀取滾動窗口資料: %s", path)
    return pd.read_csv(path)


def save_final_data(data: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    logger.info("整理後資料已輸出至: %s", path)
    return path


def run_pretidy(config: PreTidyConfig) -> pd.DataFrame:
    data = load_rolling_table(config.input_path)
    processor = PreTidyProcessor(config)
    result = processor.process(data)
    save_final_data(result, config.output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="整理滾動窗口輸出數據")
    parser.add_argument('--input', type=Path, default=PreTidyConfig.input_path, help='滾動窗口輸入檔案')
    parser.add_argument('--output', type=Path, default=PreTidyConfig.output_path, help='整理後輸出檔案')
    parser.add_argument(
        '--base-columns',
        nargs='*',
        default=None,
        help='基本欄位列表，預設為 config 中定義欄位'
    )
    parser.add_argument('--bin-suffix', default=PreTidyConfig.bin_suffix, help='分箱欄位後綴字')
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='日誌層級'
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level))

    default_base_columns = PreTidyConfig().base_columns
    base_columns = args.base_columns if args.base_columns else default_base_columns
    config = PreTidyConfig(
        input_path=args.input,
        output_path=args.output,
        base_columns=base_columns,
        bin_suffix=args.bin_suffix,
    )

    try:
        run_pretidy(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("資料整理完成並寫入 %s", config.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())