"""
Модели для детекции мошенничества
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import joblib


class FraudDetectionModel:
    """Модель детекции мошенничества на основе CatBoost"""
    
    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.03,
        depth: int = 6,
        random_seed: int = 42
    ):
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=random_seed,
            verbose=100,
            early_stopping_rounds=50
        )
        self.feature_cols = None
        self.categorical_features = None
        self.threshold = 0.5
        
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'is_fraud',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка признаков для обучения
        
        Args:
            df: исходный DataFrame
            target_col: название колонки с таргетом
            exclude_cols: колонки, которые не должны попасть в признаки
            
        Returns:
            X, y
        """
        if exclude_cols is None:
            exclude_cols = [
                'transaction_id', 'transaction_datetime', 'transaction_date',
                'client_id', 'destination_id', target_col
            ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Определяем категориальные признаки
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Приводим категориальные к строкам
        for col in categorical_features:
            X[col] = X[col].astype(str)
        
        self.feature_cols = feature_cols
        self.categorical_features = categorical_features
        
        print(f"Признаков для обучения: {len(feature_cols)}")
        print(f"Категориальных признаков: {len(categorical_features)}")
        
        return X, y
    
    def temporal_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        valid_size: float = 0.15,
        date_col: str = 'transaction_datetime'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разбиение по времени (избегаем утечки будущего)
        
        Args:
            df: DataFrame с данными
            train_size: доля train
            valid_size: доля valid
            date_col: колонка с датой
            
        Returns:
            train_df, valid_df, test_df
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_size)
        valid_end = int(n * (train_size + valid_size))
        
        train_df = df.iloc[:train_end]
        valid_df = df.iloc[train_end:valid_end]
        test_df = df.iloc[valid_end:]
        
        print(f"Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"Valid: {len(valid_df)} ({len(valid_df)/n*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        
        print(f"\nTrain fraud rate: {train_df['is_fraud'].mean():.4f}")
        print(f"Valid fraud rate: {valid_df['is_fraud'].mean():.4f}")
        print(f"Test fraud rate: {test_df['is_fraud'].mean():.4f}")
        
        return train_df, valid_df, test_df
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ):
        """
        Обучение модели
        """
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=self.categorical_features
        )
        
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = Pool(
                X_valid,
                y_valid,
                cat_features=self.categorical_features
            )
        
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True if eval_set else False
        )
        
        print("\n✓ Модель обучена")
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Предсказание вероятностей"""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Предсказание классов с порогом"""
        if threshold is None:
            threshold = self.threshold
        
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None,
        dataset_name: str = "Test"
    ) -> Dict:
        """
        Оценка качества модели
        
        Returns:
            Словарь с метриками
        """
        if threshold is None:
            threshold = self.threshold
        
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y, y_proba),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'f2': fbeta_score(y, y_pred, beta=2),
            'threshold': threshold
        }
        
        print(f"\n{'='*50}")
        print(f"Метрики на {dataset_name} (threshold={threshold:.3f}):")
        print(f"{'='*50}")
        for metric_name, value in metrics.items():
            if metric_name != 'threshold':
                print(f"{metric_name:15s}: {value:.4f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0         1")
        print(f"Actual 0     {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"Actual 1     {cm[1,0]:6d}    {cm[1,1]:6d}")
        
        # Бизнес-метрики
        total_fraud = y.sum()
        caught_fraud = cm[1, 1]
        false_blocks = cm[0, 1]
        
        print(f"\nБизнес-метрики:")
        print(f"Всего мошеннических транзакций: {total_fraud}")
        print(f"Корректно заблокировано: {caught_fraud} ({caught_fraud/total_fraud*100:.1f}%)")
        print(f"Ложных блокировок: {false_blocks}")
        
        return metrics
    
    def find_optimal_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'f2',
        plot: bool = True
    ) -> float:
        """
        Поиск оптимального порога классификации
        
        Args:
            X: признаки
            y: таргет
            metric: метрика для оптимизации ('f1', 'f2', 'precision', 'recall')
            plot: показать график
            
        Returns:
            Оптимальный порог
        """
        y_proba = self.predict_proba(X)
        
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
        
        # Вычисляем метрику для каждого порога
        if metric == 'f1':
            f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        elif metric == 'f2':
            beta = 2
            f_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        elif metric == 'precision':
            f_scores = precisions
        elif metric == 'recall':
            f_scores = recalls
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Находим оптимальный индекс (исключаем последний элемент, т.к. thresholds короче на 1)
        optimal_idx = np.argmax(f_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\nОптимальный порог ({metric}): {optimal_threshold:.4f}")
        print(f"Precision: {precisions[optimal_idx]:.4f}")
        print(f"Recall: {recalls[optimal_idx]:.4f}")
        print(f"{metric}: {f_scores[optimal_idx]:.4f}")
        
        if plot:
            plt.figure(figsize=(12, 5))
            
            # График Precision-Recall
            plt.subplot(1, 2, 1)
            plt.plot(thresholds, precisions[:-1], label='Precision')
            plt.plot(thresholds, recalls[:-1], label='Recall')
            plt.plot(thresholds, f_scores[:-1], label=metric.upper(), linewidth=2)
            plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal={optimal_threshold:.3f}')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F-score vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Precision-Recall кривая
            plt.subplot(1, 2, 2)
            plt.plot(recalls, precisions, linewidth=2)
            plt.scatter(recalls[optimal_idx], precisions[optimal_idx], 
                       color='red', s=100, zorder=5, label=f'Optimal (th={optimal_threshold:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/usr/src/forte/reports/threshold_optimization.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("\n✓ График сохранён: reports/threshold_optimization.png")
        
        self.threshold = optimal_threshold
        return optimal_threshold
    
    def get_feature_importance(self, top_n: int = 20, plot: bool = True) -> pd.DataFrame:
        """
        Важность признаков
        """
        feature_importance = self.model.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('/usr/src/forte/reports/feature_importance.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("\n✓ График сохранён: reports/feature_importance.png")
        
        return importance_df
    
    def save(self, filepath: str):
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'categorical_features': self.categorical_features,
            'threshold': self.threshold
        }
        joblib.dump(model_data, filepath)
        print(f"\n✓ Модель сохранена: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Загрузка модели"""
        model_data = joblib.load(filepath)
        
        instance = cls()
        instance.model = model_data['model']
        instance.feature_cols = model_data['feature_cols']
        instance.categorical_features = model_data['categorical_features']
        instance.threshold = model_data['threshold']
        
        print(f"\n✓ Модель загружена: {filepath}")
        return instance
