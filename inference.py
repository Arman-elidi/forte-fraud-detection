import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from models.fraud_model import FraudDetectionModel
from models.explainer import ModelExplainer


class FraudPredictor:
    """Класс для предсказания мошенничества на новых транзакциях"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: путь к сохранённой модели
        """
        self.model = FraudDetectionModel.load(model_path)
        self.explainer = ModelExplainer(self.model)
        
    def predict_single_transaction(
        self,
        transaction_data: Dict,
        explain: bool = False
    ) -> Dict:
        """
        Предсказание для одной транзакции
        
        Args:
            transaction_data: словарь с данными транзакции
            explain: вычислять ли объяснение (SHAP)
            
        Returns:
            Словарь с результатом
        """
        # Создаём DataFrame из одной строки
        df = pd.DataFrame([transaction_data])
        
        # Убеждаемся, что все нужные признаки есть
        for col in self.model.feature_cols:
            if col not in df.columns:
                df[col] = -9999  # sentinel value for missing
        
        # Приводим к нужному порядку
        X = df[self.model.feature_cols]
        
        # Fill NaN values
        X = X.fillna(-9999)
        
        # Convert object columns to string (for categorical features)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype(str)
        
        # Предсказание
        fraud_proba = self.model.predict_proba(X)[0]
        is_fraud = self.model.predict(X, threshold=self.model.threshold)[0]
        
        # Рекомендация
        if fraud_proba >= 0.8:
            recommendation = "БЛОКИРОВАТЬ"
        elif fraud_proba >= 0.3:
            recommendation = "ПРОВЕРИТЬ"
        else:
            recommendation = "OK"
        
        result = {
            'fraud_probability': float(fraud_proba),
            'is_fraud': bool(is_fraud),
            'recommendation': recommendation,
            'threshold': float(self.model.threshold)
        }
        
        # Объяснение
        if explain:
            top_features = self.explainer.get_top_features_for_instance(X, 0, top_n=5)
            
            explanations = []
            for _, row in top_features.iterrows():
                explanations.append({
                    'feature': row['feature'],
                    'value': float(row['value']) if pd.api.types.is_numeric_dtype(type(row['value'])) else str(row['value']),
                    'contribution': float(row['shap_value']),
                    'impact': 'увеличивает риск' if row['shap_value'] > 0 else 'снижает риск'
                })
            
            result['top_factors'] = explanations
        
        return result
    
    def predict_batch(
        self,
        transactions_df: pd.DataFrame,
        return_probabilities: bool = True
    ) -> pd.DataFrame:
        """
        Пакетное предсказание для множества транзакций
        
        Args:
            transactions_df: DataFrame с транзакциями
            return_probabilities: возвращать вероятности
            
        Returns:
            DataFrame с предсказаниями
        """
        # Убеждаемся, что все нужные признаки есть
        for col in self.model.feature_cols:
            if col not in transactions_df.columns:
                transactions_df[col] = -9999  # sentinel value for missing
        
        X = transactions_df[self.model.feature_cols].copy()

        # Remove rows that might be headers (where value equals column name)
        # This fixes the "Cannot convert 'last_os_categorical' to float" error if header is in data
        cols_to_check = [c for c in X.columns if X[c].dtype == 'object']
        if cols_to_check:
            for col in cols_to_check:
                # If we find the column name as a value in that column, drop those rows
                is_header_row = X[col].astype(str) == col
                if is_header_row.any():
                    X = X[~is_header_row]

        
        # Convert categorical columns to string first (before fillna)
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = X[col].astype(str)
        
        # Fill NaN values - numeric columns with -9999, object columns with 'missing'
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('missing').astype(str) # Ensure string type
            else:
                X[col] = X[col].fillna(-9999)
        
        # Convert remaining object columns to string (for categorical features)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype(str)
        
        predictions = transactions_df.copy()
        predictions['fraud_probability'] = self.model.predict_proba(X)
        predictions['is_fraud'] = self.model.predict(X, threshold=self.model.threshold)
        
        # Рекомендация
        def get_recommendation(prob):
            if prob >= 0.8:
                return "БЛОКИРОВАТЬ"
            elif prob >= 0.3:
                return "ПРОВЕРИТЬ"
            else:
                return "OK"
        
        predictions['recommendation'] = predictions['fraud_probability'].apply(get_recommendation)
        
        return predictions


def main():
    """Пример использования"""
    
    model_path = '/usr/src/forte/models/fraud_detection_model.pkl'
    
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        print("Пожалуйста, сначала запустите train.py для обучения модели")
        return
    
    # Загрузка модели
    predictor = FraudPredictor(model_path)
    
    print("="*70)
    print("FRAUD DETECTION - INFERENCE")
    print("="*70)
    
    # Пример транзакции
    example_transaction = {
        'amount': 50000,
        'hour': 23,
        'day_of_week': 6,
        'is_night': 1,
        'is_weekend': 1,
        'log_amount': np.log1p(50000),
        'is_new_destination': 1,
        'client_tx_count': 5,
        'client_avg_amount': 5000,
        'amount_vs_median': 10.0,
        # ... остальные признаки
    }
    
    print("\nПример транзакции:")
    for key, value in list(example_transaction.items())[:5]:
        print(f"  {key}: {value}")
    print("  ...")
    
    # Предсказание
    print("\nПредсказание...")
    result = predictor.predict_single_transaction(example_transaction, explain=True)
    
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТ:")
    print(f"{'='*70}")
    print(f"Вероятность мошенничества: {result['fraud_probability']:.4f}")
    print(f"Классификация: {'МОШЕННИЧЕСТВО' if result['is_fraud'] else 'ЧИСТАЯ'}")
    print(f"Рекомендация: {result['recommendation']}")
    print(f"Порог: {result['threshold']:.4f}")
    
    if 'top_factors' in result:
        print(f"\nТоп-5 факторов:")
        for i, factor in enumerate(result['top_factors'], 1):
            print(f"{i}. {factor['feature']}")
            print(f"   Значение: {factor['value']}")
            print(f"   Вклад: {factor['contribution']:.4f} ({factor['impact']})")


if __name__ == '__main__':
    main()
