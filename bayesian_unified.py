"""
貝葉斯分類器滾動驗證系統
使用樸素貝葉斯來分類狀態，預測5天後的報酬率分類
專注於滾動驗證（Walk-Forward Validation）
回測使用隔天的開盤價進行交易
"""

import pandas as pd
import numpy as np
import json
import logging
import joblib
import warnings
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

class BayesianStateClassifier:
    """
    貝葉斯狀態分類器。
    使用樸素貝葉斯 (Naive Bayes) 根據離散化特徵預測未來報酬率的分類。
    """
    
    def __init__(self, lookback_days: int = 5):
        self.lookback_days = lookback_days
        self.model = CategoricalNB()
        self.feature_columns = []
        self.label_encoders = {}
        self.logger = self._setup_logger()
        
        # 分類閾值
        self.return_threshold_buy = 0.03   # 3%
        self.return_threshold_sell = -0.03  # -3%
        
        # 分類標籤
        self.class_labels = {'buy': 0, 'hold': 1, 'sell': 2}
        self.inverse_labels = {0: 'buy', 1: 'hold', 2: 'sell'}

        # 預先編碼資料快取
        self._precomputed_X: np.ndarray | None = None
        self._precomputed_y: np.ndarray | None = None
        self._precomputed_indices: np.ndarray | None = None
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌配置。"""
        logger = logging.getLogger('BayesianClassifier')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 創建文件處理器，設置UTF-8編碼
            handler = logging.FileHandler('log/bayesian_classifier.log', encoding='utf-8')
            
            # 設置更詳細的日誌格式
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-5s | %(funcName)-25s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 控制台處理器只顯示重要信息
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-5s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.ERROR)  # 控制台只顯示錯誤
            logger.addHandler(console_handler)
            
        return logger
    
    def _calculate_future_return(self, data: pd.DataFrame, current_idx: int) -> float:
        """計算用於標記的未來報酬率。"""
        if current_idx + self.lookback_days >= len(data):
            return None
        
        current_price = data.iloc[current_idx]['Close']
        future_price = data.iloc[current_idx + self.lookback_days]['Close']
        
        return (future_price - current_price) / current_price
    
    def _classify_return(self, return_value: float) -> str:
        """將報酬率分類為買入/賣出/持有。"""
        if return_value > self.return_threshold_buy:
            return 'buy'
        elif return_value < self.return_threshold_sell:
            return 'sell'
        else:
            return 'hold'
    
    def _encode_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """使用 LabelEncoder 對類別特徵進行編碼。"""
        X_encoded = X.copy()
        
        for col in X.columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X_encoded[col] = self.label_encoders[col].transform(X[col])
                
        
        return X_encoded.values

    def _precompute_full_dataset(self, data: pd.DataFrame) -> None:
        """對完整資料集進行一次性編碼供後續重複使用"""
        X, y, indices = self.prepare_data(data)
        X_encoded = self._encode_features(X, fit=True)
        self._precomputed_X = X_encoded
        self._precomputed_y = y
        self._precomputed_indices = np.array(indices)

    def _ensure_precomputed(self) -> None:
        if self._precomputed_X is None or self._precomputed_y is None or self._precomputed_indices is None:
            raise ValueError("尚未建立預先編碼資料，無法重用快取")

    def _build_training_mask(self, end_row: int) -> np.ndarray:
        """根據資料列數建立訓練子集遮罩"""
        self._ensure_precomputed()
        limit = end_row - self.lookback_days
        if limit <= 0:
            return np.zeros_like(self._precomputed_indices, dtype=bool)
        return self._precomputed_indices < limit

    def train_until(self, end_row: int) -> dict:
        """使用預編碼資料訓練至指定列數"""
        mask = self._build_training_mask(end_row)
        if mask.sum() == 0:
            raise ValueError("訓練資料不足，無法進行模型訓練")
        return self.train(None, mask=mask, reuse_encoded=True)
    
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """準備訓練數據"""
        self.logger.info("準備訓練數據...")
        
        # 識別bin特徵
        self.feature_columns = [col for col in data.columns if col.endswith('_bin')]
        self.logger.info(f"使用 {len(self.feature_columns)} 個bin特徵: {self.feature_columns}")
        
        # 準備特徵和標籤
        X_list = []
        y_list = []
        valid_indices = []
        
        for i in range(len(data) - self.lookback_days):
            future_return = self._calculate_future_return(data, i)
            if future_return is not None:
                X_list.append(data.iloc[i][self.feature_columns])
                y_list.append(self.class_labels[self._classify_return(future_return)])
                valid_indices.append(i)
                
        
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        
        self.logger.info(f"準備了 {len(X)} 個有效樣本")
        
        # 統計標籤分布
        unique, counts = np.unique(y, return_counts=True)
        for label_idx, count in zip(unique, counts):
            action = self.inverse_labels[label_idx]
            percentage = count / len(y) * 100
            self.logger.info(f"{action}: {count} 樣本 ({percentage:.1f}%)")
        
        return X, y, valid_indices
    
    def train(
        self,
        data: pd.DataFrame | None = None,
        *,
        mask: np.ndarray | None = None,
        reuse_encoded: bool = False,
    ):
        """訓練貝葉斯分類器"""
        self.logger.info("=" * 50)
        self.logger.info("🚀 開始訓練貝葉斯分類器...")
        self.logger.info("=" * 50)
        
        if reuse_encoded:
            if mask is None:
                raise ValueError("缺少訓練遮罩，無法重用預編碼資料")
            self._ensure_precomputed()
            X_encoded = self._precomputed_X[mask]
            y = self._precomputed_y[mask]
        else:
            if data is None:
                raise ValueError("未提供訓練資料")
            # 準備數據
            self.logger.info("📊 準備訓練數據...")
            X, y, _ = self.prepare_data(data)
            
            # 編碼特徵
            self.logger.info("🔧 編碼特徵數據...")
            X_encoded = self._encode_features(X, fit=True)

        if len(y) == 0:
            raise ValueError("訓練資料不足，無法訓練模型")
        
        # 分割數據
        self.logger.info("✂️ 分割訓練和測試數據...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"訓練集: {len(X_train)} 樣本")
        self.logger.info(f"測試集: {len(X_test)} 樣本")
        
        # 訓練模型
        self.model.fit(X_train, y_train)
        
        # 評估模型
        self.logger.info("📈 評估模型性能...")
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.logger.info(f"✅ 訓練準確率: {train_score:.4f}")
        self.logger.info(f"🎯 測試準確率: {test_score:.4f}")
        
        # 詳細評估
        y_pred = self.model.predict(X_test)
        
        # 分類報告
        target_names = ['buy', 'hold', 'sell']
        class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        self.logger.info("📊 各動作分類性能:")
        for action in target_names:
            metrics = class_report[action]
            self.logger.info(f"   {action:>4s}: 精確率={metrics['precision']:.3f}, 召回率={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info(f"🔍 混淆矩陣:\n{cm}")
        
        # 交叉驗證
        self.logger.info("🔄 執行5折交叉驗證...")
        cv_scores = cross_val_score(self.model, X_encoded, y, cv=5)
        self.logger.info(f"📊 5折交叉驗證準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 保存評估結果
        self.evaluation_results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        self.logger.info("✅ 訓練完成")
        self.logger.info("=" * 50)
        
        return self.evaluation_results
    
    def predict(self, features: pd.Series) -> dict:
        """預測單個樣本"""
        # 準備特徵
        X = pd.DataFrame([features[self.feature_columns]])
        X_encoded = self._encode_features(X, fit=False)
        
        # 預測
        prediction = self.model.predict(X_encoded)[0]
        probabilities = self.model.predict_proba(X_encoded)[0]
        
        # 轉換結果
        predicted_action = self.inverse_labels[prediction]
        action_probabilities = {
            action: float(probabilities[idx]) 
            for idx, action in self.inverse_labels.items()
        }
        
        # 計算信心度
        confidence = max(probabilities)
        
        return {
            'action': predicted_action,
            'confidence': confidence,
            'probabilities': action_probabilities,
            'prediction_index': int(prediction)
        }
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """批量預測"""
        predictions = []
        
        for i in range(len(data)):
            try:
                result = self.predict(data.iloc[i])
                predictions.append(result)
            except Exception as e:
                self.logger.warning(f"預測失敗 (第{i}行): {e}")
                predictions.append({'action': 'hold', 'confidence': 0.0, 'probabilities': {}})
                
        
        return pd.DataFrame(predictions)
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """計算最大回撤"""
        if len(portfolio_values) == 0:
            return 0.0
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def get_feature_importance(self) -> dict:
        """獲取特徵重要性（基於條件概率）"""
        if not hasattr(self.model, 'feature_log_prob_'):
            return {}
        
        try:
            # 計算每個特徵對每個類別的貢獻
            feature_importance = {}
            
            # 對於CategoricalNB，feature_log_prob_是一個列表
            for i, feature in enumerate(self.feature_columns):
                importance = 0
                for class_idx in range(len(self.class_labels)):
                    if i < len(self.model.feature_log_prob_[class_idx]):
                        # 計算該特徵在該類別中的平均重要性
                        class_importance = np.exp(self.model.feature_log_prob_[class_idx][i]).var()
                        importance += class_importance
                feature_importance[feature] = importance
            
            # 排序
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
        except Exception as e:
            self.logger.warning(f"無法計算特徵重要性: {e}")
            # 返回平均重要性
            return {feature: 1.0 for feature in self.feature_columns}
    
    def get_next_day_signal(self, current_data: pd.Series) -> dict:
        """根據今天的數據預測明天的交易訊號"""
        prediction = self.predict(current_data)
        
        signal_info = {
            'signal_date': datetime.now().isoformat(),
            'for_date': (datetime.now() + pd.Timedelta(days=1)).isoformat(),
            'current_price': float(current_data['Close']),
            'recommended_action': prediction['action'],
            'confidence': prediction['confidence'],
            'action_probabilities': prediction['probabilities'],
            'reasoning': self._get_signal_reasoning(prediction)
        }
        
        return signal_info
    
    def _get_signal_reasoning(self, prediction: dict) -> str:
        """生成訊號推理說明"""
        action = prediction['action']
        confidence = prediction['confidence']
        
        if action == 'buy':
            if confidence > 0.7:
                return f"高信心度({confidence:.1%})建議買入，預期未來5天報酬率可能超過3%"
            elif confidence > 0.5:
                return f"中等信心度({confidence:.1%})建議買入，預期未來5天可能有正報酬"
            else:
                return f"低信心度({confidence:.1%})建議買入，但需謹慎考慮"
        elif action == 'sell':
            if confidence > 0.7:
                return f"高信心度({confidence:.1%})建議賣出，預期未來5天報酬率可能低於-3%"
            elif confidence > 0.5:
                return f"中等信心度({confidence:.1%})建議賣出，預期未來5天可能有負報酬"
            else:
                return f"低信心度({confidence:.1%})建議賣出，但需謹慎考慮"
        else:  # hold
            return f"建議持有({confidence:.1%}信心度)，預期報酬率在-3%到3%之間"

    def rolling_validation(self, data: pd.DataFrame, 
                          initial_train_size: int = None,
                          retrain_frequency: int = 5) -> dict:  # 5天
        """
        連續滾動驗證（不分輪次）
        
        Args:
            data: 完整數據集
            initial_train_size: 初始訓練集大小（預設為總數據的60%）
            retrain_frequency: 重新訓練的頻率（5天）
        """
        self.logger.info("🔄 開始連續滾動驗證...")
        self.logger.info("=" * 60)
        
        if initial_train_size is None:
            # 預設使用2019年12月31日作為訓練集截止日期
            cutoff_date = pd.to_datetime('2019-12-31')
            initial_train_size = len(data[data['Date'] <= cutoff_date])
        
        # 確保有足夠的數據
        min_required = initial_train_size + self.lookback_days
        if len(data) < min_required:
            raise ValueError(f"數據不足，至少需要 {min_required} 行數據")
        
        # 準備初始訓練數據
        train_data = data[:initial_train_size]
        
        # 驗證數據是剩餘的所有數據
        validation_start = initial_train_size
        validation_end = len(data) - self.lookback_days
        validation_data = data[validation_start:validation_end]
        
        self.logger.info(f"📅 訓練期間: {train_data['Date'].min()} - {train_data['Date'].max()}")
        self.logger.info(f"📅 驗證期間: {validation_data['Date'].min()} - {validation_data['Date'].max()}")
        self.logger.info(f"📊 訓練樣本數: {len(train_data)}, 驗證樣本數: {len(validation_data)}")
        
        # 建立預先編碼資料
        self.logger.info("🧮 建立預編碼特徵快取...")
        self._precompute_full_dataset(data)
        
        # 初始訓練模型
        self.logger.info("🚀 初始訓練模型...")
        model_perf = self.train_until(initial_train_size)
        
        # 進行連續滾動驗證，每週三重新訓練
        print(f"開始連續滾動驗證，共 {len(validation_data)} 天...")
        validation_results = self._validate_with_frequent_retrain(
            validation_data, 
            initial_train_size, 
            retrain_frequency, 
            data
        )
        
        # 準備結果結構
        validation_results.update({
            'train_start': train_data['Date'].min().isoformat(),
            'train_end': train_data['Date'].max().isoformat(),
            'validation_start': validation_data['Date'].min().isoformat(),
            'validation_end': validation_data['Date'].max().isoformat(),
            'train_size': len(train_data),
            'validation_size': len(validation_data)
        })
        
        # 保存每日詳細狀態 CSV
        if 'daily_results' in validation_results:
            daily_df = validation_results['daily_results'].copy()
            
            # 設置您要求的欄位順序
            columns_order = [
                'date',           # 日期
                'open',           # 開盤
                'close',          # 收盤
                'predicted_action', # 動作
                'prob_buy',       # 買入機率
                'prob_hold',      # 持有機率
                'prob_sell'       # 賣出機率
            ]
            
            # 確保所有欄位都存在
            for col in columns_order:
                if col not in daily_df.columns:
                    daily_df[col] = None
                    
            daily_df = daily_df[columns_order]
            
            # 在重新命名前計算統計資訊
            if 'is_wednesday' in validation_results['daily_results'].columns:
                original_df = validation_results['daily_results']
                wednesday_retrains = original_df[original_df['is_wednesday'] == True]
                wednesday_count = len(wednesday_retrains)
            else:
                wednesday_count = validation_results.get('retrain_count', 0)
            
            # 重新命名欄位為中文
            column_rename_map = {
                'date': '日期',
                'open': '開盤',
                'close': '收盤',
                'predicted_action': '動作',
                'prob_buy': '買入機率',
                'prob_hold': '持有機率',
                'prob_sell': '賣出機率'
            }
            daily_df = daily_df.rename(columns=column_rename_map)
            
            # 保存每日詳細狀態 CSV
            csv_filename = 'log/rolling_validation_daily_details.csv'
            daily_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"每日詳細狀態已保存到: {csv_filename}")
            print(f"\n每日詳細狀態已保存到: {csv_filename}")
            print(f"總共 {len(daily_df)} 天的詳細記錄")
            
            # 顯示一些統計資訊
            print(f"週三重新訓練次數: {wednesday_count} 次")
        
        self.logger.info("🎉 連續滾動驗證完成")
        self.logger.info("=" * 60)
        
        # 準備純預測驗證摘要
        summary = {
            'prediction_accuracy': validation_results.get('prediction_accuracy', 0),
            'action_accuracy': validation_results.get('action_accuracy', {}),
            'avg_confidence': validation_results.get('avg_confidence', 0),
            'total_predictions': validation_results.get('total_predictions', 0),
            'correct_predictions': validation_results.get('correct_predictions', 0),
            'retrain_count': validation_results.get('retrain_count', 0)
        }
        
        return {
            'summary': summary,
            'detailed_results': [validation_results],  # 包裝成列表以保持一致性
            'validation_config': {
                'initial_train_size': initial_train_size,
                'retrain_frequency': '每週三',
                'total_validation_days': len(validation_data)
            },
            'daily_results_csv': csv_filename if 'daily_results' in validation_results else None
        }
    
    def _validate_with_frequent_retrain(self, validation_data: pd.DataFrame, 
                                       train_end_position: int, 
                                       retrain_frequency: int, 
                                       full_data: pd.DataFrame) -> dict:
        """在驗證期間內進行預測驗證，並在每週三重新訓練模型"""
        results = []
        retrain_count = 0
        
        # 在驗證期間內逐日預測
        days_to_process = len(validation_data) - self.lookback_days
        with tqdm(total=days_to_process, desc="每日預測驗證", unit="天", leave=False) as day_pbar:
            for i in range(days_to_process):
                # 當前驗證日的數據
                row = validation_data.iloc[i]
                current_date = pd.to_datetime(row['Date'])
                current_close = row['Close']
                
                day_pbar.set_description(f"驗證日: {current_date.strftime('%Y-%m-%d')}")
                
                # 檢查是否為週三（weekday() == 2，週一=0，週二=1，週三=2）
                if current_date.weekday() == 2 and i > 0:  # 週三且不是第一天
                    retrain_count += 1
                    # 使用到當前時間點的所有數據重新訓練
                    retrain_end = train_end_position + i
                    day_pbar.set_description(f"週三重訓: {current_date.strftime('%Y-%m-%d')}")
                    self.logger.info(f"🔄 週三重新訓練 (第 {retrain_count} 次): {current_date.strftime('%Y-%m-%d')}, 數據範圍: 第1-{retrain_end}行")
                    try:
                        self.train_until(retrain_end)
                        self.logger.info(f"✅ 第 {retrain_count} 次重訓完成")
                    except Exception as e:
                        self.logger.error(f"❌ 重新訓練失敗: {e}")
                
                # 生成當天的預測訊號
                try:
                    prediction = self.predict(row)
                    action = prediction['action']
                    confidence = prediction['confidence']
                except Exception as e:
                    self.logger.warning(f"⚠️ 預測失敗: {e}")
                    action = 'hold'
                    confidence = 0.0
                    prediction = {'probabilities': {'buy': 0.33, 'hold': 0.34, 'sell': 0.33}}
                
                # 計算實際未來報酬率（用於評估預測準確性）
                actual_return = self._calculate_future_return(validation_data, i)
                if actual_return is None:
                    day_pbar.update(1)
                    continue
                
                actual_action = self._classify_return(actual_return)
                
                results.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'open': row.get('Open', current_close),  # 開盤價，如果沒有則用收盤價
                    'close': current_close,  # 收盤價
                    'predicted_action': action,
                    'prob_buy': f"{prediction['probabilities'].get('buy', 0) * 100:.2f}%",
                    'prob_hold': f"{prediction['probabilities'].get('hold', 0) * 100:.2f}%",
                    'prob_sell': f"{prediction['probabilities'].get('sell', 0) * 100:.2f}%",
                    # 保留其他欄位以備不時之需
                    'weekday': current_date.strftime('%A'),
                    'is_wednesday': current_date.weekday() == 2,
                    'actual_action': actual_action,
                    'confidence': confidence,
                    'actual_return': actual_return,
                    'retrain_count': retrain_count
                })
                
                # 更新進度條
                day_pbar.update(1)
                day_pbar.set_postfix({
                    '預測': action,
                    '實際': actual_action,
                    '信心': f"{confidence:.1%}"
                })
        
        # 計算純預測驗證指標
        df_results = pd.DataFrame(results)
        if len(df_results) == 0:
            return self._empty_validation_result()
        
        # 只計算預測準確率相關指標
        correct_predictions = (df_results['predicted_action'] == df_results['actual_action']).sum()
        prediction_accuracy = correct_predictions / len(df_results) if len(df_results) > 0 else 0
        
        # 按動作分類準確率
        action_accuracy = {}
        for action in ['buy', 'hold', 'sell']:
            action_mask = df_results['actual_action'] == action
            if action_mask.sum() > 0:
                action_correct = ((df_results['predicted_action'] == action) & action_mask).sum()
                action_accuracy[action] = action_correct / action_mask.sum()
            else:
                action_accuracy[action] = 0.0
        
        return {
            'prediction_accuracy': prediction_accuracy,
            'action_accuracy': action_accuracy,
            'avg_confidence': df_results['confidence'].mean(),
            'total_predictions': len(df_results),
            'correct_predictions': correct_predictions,
            'retrain_count': retrain_count,
            'daily_results': df_results
        }
    
    def _empty_validation_result(self) -> dict:
        """返回空的驗證結果"""
        return {
            'prediction_accuracy': 0.0,
            'action_accuracy': {'buy': 0.0, 'hold': 0.0, 'sell': 0.0},
            'avg_confidence': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0,
            'retrain_count': 0,
            'daily_results': pd.DataFrame()
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'class_labels': self.class_labels,
            'inverse_labels': self.inverse_labels,
            'lookback_days': self.lookback_days,
            'return_threshold_buy': self.return_threshold_buy,
            'return_threshold_sell': self.return_threshold_sell,
            'evaluation_results': getattr(self, 'evaluation_results', {})
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """載入模型"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"模型檔案不存在: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.label_encoders = model_data['label_encoders']
        self.class_labels = model_data['class_labels']
        self.inverse_labels = model_data['inverse_labels']
        self.lookback_days = model_data['lookback_days']
        self.return_threshold_buy = model_data['return_threshold_buy']
        self.return_threshold_sell = model_data['return_threshold_sell']
        self.evaluation_results = model_data.get('evaluation_results', {})
        
        self.logger.info(f"模型已從 {filepath} 載入")

def run_rolling_validation():
    """執行滾動驗證的主函數"""
    # 載入數據
    print("載入數據...")
    data = pd.read_csv('data/final_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    print(f"數據形狀: {data.shape}")
    print(f"日期範圍: {data['Date'].min()} 到 {data['Date'].max()}")
    
    # 創建分類器
    classifier = BayesianStateClassifier(lookback_days=5)
    
    # 執行滾動驗證
    print("\n=== 執行連續預測驗證（週三重新訓練）===")
    
    # 配置滾動驗證參數
    # 設定初始訓練集到2019年12月31日
    cutoff_date = pd.to_datetime('2019-12-31')
    initial_train_size = len(data[data['Date'] <= cutoff_date])
    retrain_frequency = 'Wednesday'  # 每週三重新訓練

    print(f"初始訓練集大小: {initial_train_size} 天")
    print(f"重新訓練頻率: 每週三")
    
    # 執行連續滾動驗證
    rolling_results = classifier.rolling_validation(
        data, 
        initial_train_size=initial_train_size,
        retrain_frequency=7  # 傳入7天頻率（實際上會被週三邏輯覆蓋）
    )
    
    # 顯示預測驗證結果
    summary = rolling_results['summary']
    print("\n=== 連續預測驗證結果摘要 ===")
    print(f"預測準確率: {summary['prediction_accuracy']:.2%}")
    print(f"平均信心度: {summary['avg_confidence']:.2%}")
    print(f"總預測次數: {summary['total_predictions']}")
    print(f"正確預測次數: {summary['correct_predictions']}")
    print(f"重新訓練次數: {summary['retrain_count']}")
    print(f"總驗證天數: {rolling_results['validation_config']['total_validation_days']}")
    
    # 顯示各動作預測準確率
    action_acc = summary['action_accuracy']
    print(f"\n各動作預測準確率:")
    print(f"  買入 (buy): {action_acc.get('buy', 0):.2%}")
    print(f"  持有 (hold): {action_acc.get('hold', 0):.2%}")
    print(f"  賣出 (sell): {action_acc.get('sell', 0):.2%}")
    
    # 保存滾動驗證結果
    with open('log/rolling_validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(rolling_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n預測驗證結果已保存到: log/rolling_validation_results.json")
    
    # 顯示驗證結果詳細資訊
    print(f"\n=== 驗證詳細結果 ===")
    result = rolling_results['detailed_results'][0]  # 只有一個結果
    print(f"驗證期間: {result['validation_start'][:10]} - {result['validation_end'][:10]}")
    print(f"預測準確率: {result['prediction_accuracy']:.2%}")
    print(f"平均信心度: {result['avg_confidence']:.2%}")
    print(f"總預測次數: {result['total_predictions']}")
    print(f"重新訓練次數: {result['retrain_count']}")
    
    # 用最新數據重新訓練最終模型
    print(f"\n=== 訓練最終模型 ===")
    final_train_size = int(len(data) * 0.8)
    evaluation_results = classifier.train_until(final_train_size)
    
    # 保存最終模型
    classifier.save_model('log/bayesian_classifier_model.pkl')
    
    # 生成明天的交易訊號
    print("\n=== 明天的交易訊號 ===")
    latest_data = data.iloc[-1]
    tomorrow_signal = classifier.get_next_day_signal(latest_data)
    
    print(f"訊號產生日期: {latest_data['Date'].strftime('%Y-%m-%d')}")
    print(f"執行交易日期: {(latest_data['Date'] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")
    print(f"當前收盤價: ${latest_data['Close']:.2f}")
    if 'Open' in data.columns:
        print(f"當前開盤價: ${latest_data['Open']:.2f}")
    print(f"明天建議動作: {tomorrow_signal['recommended_action']}")
    print(f"信心度: {tomorrow_signal['confidence']:.2%}")
    print(f"推理說明: {tomorrow_signal['reasoning']}")
    print("動作概率:")
    for action, prob in tomorrow_signal['action_probabilities'].items():
        print(f"  {action}: {prob:.2%}")
    
    # 保存明天的訊號
    with open('log/tomorrow_trading_signal.json', 'w', encoding='utf-8') as f:
        json.dump(tomorrow_signal, f, ensure_ascii=False, indent=2)
    
    print(f"\n明天的交易訊號已保存到: log/tomorrow_trading_signal.json")


def main():
    """主函數 - 只執行滾動驗證"""
    run_rolling_validation()


if __name__ == "__main__":
    main()