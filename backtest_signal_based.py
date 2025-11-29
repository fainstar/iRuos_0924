#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基於信號的回測系統
根據CSV檔案中的買賣信號進行回測
條件：
- 如果訊號是sell且機率>40%就賣出
- 如果訊號是buy且機率>40%就買入
- 使用下一筆的開盤價進行操作
"""

import argparse
import json
import logging
import os
from datetime import datetime

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logging_config import get_logger, setup_logging

# 設定中文字體
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Hiragino Sans GB', 'Heiti TC', 'Microsoft JhengHei', 'Arial Unicode MS']

class SignalBasedBacktest:
    """
    基於交易訊號的回測系統。
    根據 'buy'/'sell' 訊號與機率閾值執行交易。
    """
    def __init__(
        self,
        csv_file: str,
        initial_capital: float = 1000000,
        commission_rate: float = 0.001425,
        symbol: str | None = None,
    ):
        """
        初始化回測系統。
        
        Args:
            csv_file (str): 包含訊號的 CSV 檔案路徑。
            initial_capital (float): 回測初始資金。
            commission_rate (float): 每筆交易的手續費率。
            symbol (str | None): 股票代號或標的名稱。
        """
        self.csv_file = csv_file
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.data = None
        self.results = []
        self.trades = []
        self.symbol = symbol
        display_name = symbol or os.path.splitext(os.path.basename(csv_file))[0] or "unknown"
        self.logger = get_logger(f"SignalBacktest[{display_name}]")

    def _derive_symbol_from_data(self) -> str:
        """
        從資料或檔名推導股票代號。
        """
        if self.data is not None:
            candidate_columns = ['symbol', 'Symbol', 'ticker', 'Ticker', '股票代號', '標的', '代號']
            for col in candidate_columns:
                if col in self.data.columns:
                    value = str(self.data[col].iloc[0]).strip()
                    if value:
                        return value

        base_name = os.path.basename(self.csv_file)
        fallback = os.path.splitext(base_name)[0]
        return fallback if fallback else "未知標的"
        
    def load_data(self) -> bool:
        """
        從 CSV 檔案載入數據。
        
        Returns:
            bool: 若成功載入回傳 True，否則回傳 False。
        """
        try:
            self.data = pd.read_csv(self.csv_file)
            self.data['日期'] = pd.to_datetime(self.data['日期'])
            
            # 轉換機率欄位為數值
            prob_columns = ['買入機率', '持有機率', '賣出機率']
            for col in prob_columns:
                if col in self.data.columns:
                    self.data[col] = self.data[col].str.replace('%', '').astype(float) / 100
            
            self.logger.info("成功載入數據，共 %d 筆記錄", len(self.data))
            self.logger.info(
                "數據期間：%s 到 %s",
                self.data['日期'].min(),
                self.data['日期'].max(),
            )

            if not self.symbol:
                self.symbol = self._derive_symbol_from_data()
                self.logger.info("使用標的：%s", self.symbol)
                self.logger = get_logger(f"SignalBacktest[{self.symbol}]")
            
        except Exception:
            self.logger.exception("載入數據失敗")
            return False
        return True
    
    def run_backtest(self) -> None:
        """
        執行回測邏輯。
        """
        if self.data is None:
            self.logger.error("尚未載入數據，無法進行回測")
            return
        
        # 初始化
        capital = self.initial_capital
        position = 0  # 持倉股數
        cash = capital  # 現金
        
        # 回測結果記錄
        daily_records = []
        
        for i in range(len(self.data) - 1):  # 最後一筆無法執行（沒有下一筆開盤價）
            current = self.data.iloc[i]
            next_row = self.data.iloc[i + 1]
            
            date = current['日期']
            action = current['動作']
            open_price = current['開盤']
            close_price = current['收盤']
            next_open = next_row['開盤']  # 下一筆開盤價
            
            buy_prob = current.get('買入機率', 0)
            sell_prob = current.get('賣出機率', 0)
            
            # 判斷交易信號
            trade_signal = None
            trade_price = next_open
            
            if action == 'sell' and sell_prob > 0.4:
                trade_signal = 'sell'
            elif action == 'buy' and buy_prob > 0.4:
                trade_signal = 'buy'
            
            # 除錯信息：記錄前幾筆數據的交易判斷
            if i < 20 or (trade_signal is not None):
                self.logger.debug(
                    "日期: %s, 動作: %s, 買入機率: %.1f%%, 賣出機率: %.1f%%, 交易信號: %s, 現金: %s, 持股: %s",
                    date.strftime('%Y-%m-%d'),
                    action,
                    buy_prob * 100,
                    sell_prob * 100,
                    trade_signal,
                    f"{cash:,.0f}",
                    position,
                )
            
            # 執行交易
            trade_executed = False
            trade_amount = 0
            commission = 0
            
            if trade_signal == 'buy' and cash > 0:
                # 買入：計算考慮手續費後可買的股數
                # 總成本 = 股數 * 價格 * (1 + 手續費率)
                max_shares = int(cash / (trade_price * (1 + self.commission_rate)))
                self.logger.debug(
                    "嘗試買入：現金 %s，價格 %.2f，可買股數 %s",
                    f"{cash:,.0f}",
                    trade_price,
                    max_shares,
                )
                
                if max_shares > 0:
                    trade_amount = max_shares * trade_price
                    commission = trade_amount * self.commission_rate
                    total_cost = trade_amount + commission
                    self.logger.debug(
                        "交易金額 %s，手續費 %s，需要總金額 %s",
                        f"{trade_amount:,.0f}",
                        f"{commission:,.0f}",
                        f"{total_cost:,.0f}",
                    )
                    
                    if cash >= total_cost:
                        position += max_shares
                        cash -= total_cost
                        trade_executed = True
                        self.logger.info(
                            "執行買入：%s 股，價格 %.2f，金額 %s",
                            max_shares,
                            trade_price,
                            f"{trade_amount:,.0f}",
                        )
                        
                        # 記錄交易
                        trade_record = {
                            '日期': date,
                            '動作': 'BUY',
                            '價格': trade_price,
                            '股數': max_shares,
                            '金額': trade_amount,
                            '手續費': commission,
                            '買入機率': buy_prob,
                            '賣出機率': sell_prob
                        }
                        self.trades.append(trade_record)
                    else:
                        self.logger.warning(
                            "資金不足，無法買入（需要 %s，僅有 %s）",
                            f"{total_cost:,.0f}",
                            f"{cash:,.0f}",
                        )
            
            elif trade_signal == 'sell' and position > 0:
                # 賣出：賣出所有持股
                trade_amount = position * trade_price
                commission = trade_amount * self.commission_rate
                
                cash += (trade_amount - commission)
                shares_sold = position
                position = 0
                trade_executed = True
                self.logger.info(
                    "執行賣出：%s 股，價格 %.2f，金額 %s",
                    shares_sold,
                    trade_price,
                    f"{trade_amount:,.0f}",
                )
                
                # 記錄交易
                trade_record = {
                    '日期': date,
                    '動作': 'SELL',
                    '價格': trade_price,
                    '股數': shares_sold,
                    '金額': trade_amount,
                    '手續費': commission,
                    '買入機率': buy_prob,
                    '賣出機率': sell_prob
                }
                self.trades.append(trade_record)
            
            # 計算當日總價值
            total_value = cash + position * close_price
            
            # 記錄每日數據
            daily_record = {
                '日期': date,
                '開盤': open_price,
                '收盤': close_price,
                '現金': cash,
                '持股': position,
                '持股價值': position * close_price,
                '總價值': total_value,
                '報酬率': (total_value - self.initial_capital) / self.initial_capital,
                '動作': action,
                '交易信號': trade_signal if trade_executed else None,
                '買入機率': buy_prob,
                '賣出機率': sell_prob
            }
            daily_records.append(daily_record)
        
        # 轉換為DataFrame
        self.results = pd.DataFrame(daily_records)
        
        self.logger.info("回測完成")
        self.logger.info("總交易次數：%d", len(self.trades))
        self.logger.info("最終總價值：%s", f"{self.results['總價值'].iloc[-1]:,.0f}")
        self.logger.info("總報酬率：%.2f%%", self.results['報酬率'].iloc[-1] * 100)
    
    def calculate_statistics(self):
        """計算回測統計數據"""
        if self.results is None or len(self.results) == 0:
            return {}
        
        final_value = self.results['總價值'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 計算日報酬率
        daily_returns = self.results['總價值'].pct_change().dropna()
        
        # 計算年化報酬率和波動率
        trading_days_per_year = 252
        total_days = len(self.results)
        years = total_days / trading_days_per_year
        
        annual_return = (final_value / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        annual_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
        
        # 計算最大回撤
        cumulative = self.results['總價值']
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # 計算夏普比率（假設無風險利率為2%）
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 計算勝率
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            if len(trades_df) >= 2:
                # 配對買賣交易來計算損益
                buy_trades = trades_df[trades_df['動作'] == 'BUY']
                sell_trades = trades_df[trades_df['動作'] == 'SELL']
                
                profitable_trades = 0
                total_trade_pairs = min(len(buy_trades), len(sell_trades))
                
                for i in range(total_trade_pairs):
                    buy_price = buy_trades.iloc[i]['價格']
                    sell_price = sell_trades.iloc[i]['價格']
                    if sell_price > buy_price:
                        profitable_trades += 1
                
                win_rate = profitable_trades / total_trade_pairs if total_trade_pairs > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        stats = {
            '初始資金': self.initial_capital,
            '最終價值': final_value,
            '總報酬率': total_return,
            '年化報酬率': annual_return,
            '年化波動率': annual_volatility,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe_ratio,
            '交易次數': len(self.trades),
            '勝率': win_rate,
            '回測天數': len(self.results)
        }
        
        return stats
    
    def plot_results(self):
        """繪製簡潔的年度股價走勢與交易策略圖表"""
        if self.results is None or len(self.results) == 0:
            self.logger.warning("沒有回測結果可以繪製")
            return

        symbol_display = self.symbol or "未命名標的"
        symbol_title = symbol_display.upper() if symbol_display else "未命名標的"
        
        # 獲取所有年份
        self.results['年份'] = self.results['日期'].dt.year
        years = sorted(self.results['年份'].unique())
        num_years = len(years)
        
        # 計算圖表佈局
        cols = min(3, num_years)  # 最多3列
        rows = (num_years + cols - 1) // cols  # 計算需要的行數
        
        # 創建圖表
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if num_years == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # 主標題
        fig.suptitle(f'📈 {symbol_title} 年度股價走勢與交易策略分析', fontsize=20, fontweight='bold', y=0.98)
        
        # 轉換交易記錄為DataFrame便於處理
        trades_df = pd.DataFrame(self.trades) if len(self.trades) > 0 else pd.DataFrame()
        
        # 為每一年繪製子圖
        for i, year in enumerate(years):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 獲取該年度數據
            year_data = self.results[self.results['年份'] == year].copy()
            
            if len(year_data) == 0:
                ax.text(0.5, 0.5, f'{year}年\n無數據', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{symbol_title} {year}年', fontsize=14, fontweight='bold')
                continue
            
            # 創建雙軸 - 左軸股價，右軸累積收益率
            ax2 = ax.twinx()
            
            # 繪製股價走勢（左軸）
            line1 = ax.plot(year_data['日期'], year_data['收盤'], 
                           color='#1f77b4', linewidth=2, label='收盤價')
            
            # 計算該年度累積收益率（右軸）
            year_start_value = year_data['總價值'].iloc[0]
            cumulative_return = ((year_data['總價值'] - year_start_value) / year_start_value * 100)
            
            # 繪製累積收益率線（右軸）
            line2 = ax2.plot(year_data['日期'], cumulative_return, 
                            color='#ff7f0e', linewidth=2, linestyle='--', 
                            label='累積收益率', alpha=0.8)
            
            # 標記持倉期間
            is_holding = year_data['持股'] > 0
            holding_periods = []
            start_date = None
            
            for j, (date, holding) in enumerate(zip(year_data['日期'], is_holding)):
                if holding and start_date is None:
                    start_date = date
                elif not holding and start_date is not None:
                    holding_periods.append((start_date, date))
                    start_date = None
            
            # 處理年末仍持倉的情況
            if start_date is not None:
                holding_periods.append((start_date, year_data['日期'].iloc[-1]))
            
            # 繪製持倉期間背景
            for start, end in holding_periods:
                ax.axvspan(start, end, alpha=0.15, color='green', label='持倉期間' if start == holding_periods[0][0] else "")
            
            # 標記買入賣出點
            if not trades_df.empty:
                year_trades = trades_df[trades_df['日期'].dt.year == year]
                
                # 買入點
                buy_trades = year_trades[year_trades['動作'] == 'BUY']
                if len(buy_trades) > 0:
                    ax.scatter(buy_trades['日期'], buy_trades['價格'], 
                             color='red', marker='^', s=100, zorder=10,
                             label=f'買入 ({len(buy_trades)}次)')
                
                # 賣出點
                sell_trades = year_trades[year_trades['動作'] == 'SELL']
                if len(sell_trades) > 0:
                    ax.scatter(sell_trades['日期'], sell_trades['價格'], 
                             color='green', marker='v', s=100, zorder=10,
                             label=f'賣出 ({len(sell_trades)}次)')
            
            # 計算該年度報酬率
            start_value = year_data['總價值'].iloc[0]
            end_value = year_data['總價值'].iloc[-1]
            year_return = (end_value - start_value) / start_value * 100
            
            # 計算該年度最大回撤
            year_values = year_data['總價值'].values
            peak = year_values[0]
            max_drawdown = 0
            for value in year_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # 設置軸標籤和標題
            ax.set_ylabel('股價 (TWD)', fontsize=11, color='#1f77b4')
            ax2.set_ylabel('累積收益率 (%)', fontsize=11, color='#ff7f0e')
            ax.tick_params(axis='y', labelcolor='#1f77b4')
            ax2.tick_params(axis='y', labelcolor='#ff7f0e')
            
            # 設定子圖標題和格式
            ax.set_title(
                f'{symbol_title} {year}年 (年度報酬: {year_return:+.1f}%, 最大回撤: -{max_drawdown:.1f}%)', 
                fontsize=12,
                fontweight='bold',
                pad=15,
            )
            
            # 合併圖例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if len(lines1) > 0 or len(lines2) > 0:
                ax.legend(lines1 + lines2, labels1 + labels2, 
                         loc='upper left', fontsize=9, framealpha=0.9)
            
            ax.grid(True, alpha=0.3)
            
            # 格式化x軸日期
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            
            # 調整y軸範圍使圖表更美觀
            price_min = year_data['收盤'].min()
            price_max = year_data['收盤'].max()
            price_range = price_max - price_min
            ax.set_ylim(price_min - price_range*0.05, price_max + price_range*0.1)
        
        # 隱藏多餘的子圖
        for i in range(num_years, len(axes)):
            axes[i].set_visible(False)
        
        # 調整佈局
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_symbol = symbol_title.replace('/', '_').replace(' ', '_')
        filename = f"log/{safe_symbol}_yearly_price_strategy_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        self.logger.info("年度分析圖表已保存至：%s", filename)
        
        # plt.show()
    
    def save_results(self):
        """保存回測結果"""
        
        # 保存統計數據
        stats = self.calculate_statistics()
        stats_file = f"log/signal_backtest_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存每日詳細數據
        daily_file = f"log/signal_backtest_daily_values.csv"
        self.results.to_csv(daily_file, index=False, encoding='utf-8-sig')
        
        # 保存交易記錄
        if len(self.trades) > 0:
            trades_file = f"log/signal_backtest_trades.csv"
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')
        
        self.logger.info("回測結果已保存")
        self.logger.info("統計數據：%s", stats_file)
        self.logger.info("每日數據：%s", daily_file)
        if len(self.trades) > 0:
            self.logger.info("交易記錄：%s", trades_file)
        
        return stats
    
    def print_summary(self):
        """印出回測摘要"""
        stats = self.calculate_statistics()
        
        self.logger.info("=" * 60)
        self.logger.info("基於信號的回測結果摘要")
        self.logger.info("=" * 60)
        self.logger.info("初始資金：        $%s", f"{stats['初始資金']:,.0f}")
        self.logger.info("最終價值：        $%s", f"{stats['最終價值']:,.0f}")
        self.logger.info("總報酬率：        %.2f%%", stats['總報酬率'] * 100)
        self.logger.info("年化報酬率：      %.2f%%", stats['年化報酬率'] * 100)
        self.logger.info("年化波動率：      %.2f%%", stats['年化波動率'] * 100)
        self.logger.info("最大回撤：        %.2f%%", stats['最大回撤'] * 100)
        self.logger.info("夏普比率：        %.3f", stats['夏普比率'])
        self.logger.info("交易次數：        %d", stats['交易次數'])
        self.logger.info("勝率：            %.2f%%", stats['勝率'] * 100)
        self.logger.info("回測天數：        %d", stats['回測天數'])
        self.logger.info("=" * 60)
        
        # 顯示最近幾筆交易
        if len(self.trades) > 0:
            self.logger.info("最近的交易記錄：")
            trades_df = pd.DataFrame(self.trades)
            self.logger.info("\n%s", trades_df.tail(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    """
    解析命令列參數。
    """
    parser = argparse.ArgumentParser(description="基於交易訊號的回測系統")
    parser.add_argument(
        "--csv",
        default="log/rolling_validation_daily_details.csv",
        help="輸入的交易訊號 CSV 檔案",
    )
    parser.add_argument("--symbol", default=None, help="股票代號或標的名稱")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000000,
        help="初始資金 (預設: 1000000)",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.001425,
        help="手續費率 (預設: 0.001425)",
    )
    return parser.parse_args()


def main():
    """主函數"""
    setup_logging()
    cli_logger = get_logger("SignalBacktestCLI")
    args = parse_args()
    
    # 確保log目錄存在
    os.makedirs("log", exist_ok=True)
    
    # 創建回測實例
    backtest = SignalBasedBacktest(
        csv_file=args.csv,
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
        symbol=args.symbol,
    )
    
    # 執行回測
    if backtest.load_data():
        backtest.run_backtest()
        backtest.print_summary()
        backtest.save_results()
        backtest.plot_results()
    else:
        cli_logger.error("回測失敗")


if __name__ == "__main__":
    main()