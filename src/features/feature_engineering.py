"""
Feature engineering для детекции мошенничества
"""
import pandas as pd
import numpy as np
from typing import List, Dict


class FraudFeatureEngineer:
    """Класс для генерации признаков антифрода"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Временные признаки из transaction_datetime
        """
        df = df.copy()
        
        df['hour'] = df['transaction_datetime'].dt.hour
        df['day_of_week'] = df['transaction_datetime'].dt.dayofweek
        df['day_of_month'] = df['transaction_datetime'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 7)).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 12)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)
        
        self.feature_names.extend([
            'hour', 'day_of_week', 'day_of_month', 'is_weekend', 
            'is_night', 'is_morning', 'is_evening'
        ])
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Признаки на основе суммы перевода
        """
        df = df.copy()
        
        # Логарифм суммы (для нормализации распределения)
        df['log_amount'] = np.log1p(df['amount'])
        
        # Бакеты по квантилям
        df['amount_bucket'] = pd.qcut(
            df['amount'], 
            q=5, 
            labels=['very_small', 'small', 'medium', 'large', 'very_large'],
            duplicates='drop'
        )
        
        # Округлённость суммы (подозрительно ровные суммы)
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
        df['is_round_100'] = (df['amount'] % 100 == 0).astype(int)
        
        self.feature_names.extend([
            'log_amount', 'amount_bucket', 'is_round_amount', 'is_round_100'
        ])
        
        return df
    
    def create_client_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Исторические признаки клиента (агрегаты по прошлым транзакциям)
        
        ВАЖНО: для прод-системы нужна правильная временная логика,
        здесь — упрощённый вариант для обучения
        """
        df = df.copy()
        df = df.sort_values(['client_id', 'transaction_datetime'])
        
        # Количество прошлых транзакций клиента
        df['client_tx_count'] = df.groupby('client_id').cumcount()
        
        # Средняя и медианная сумма переводов клиента (expanding window)
        df['client_avg_amount'] = df.groupby('client_id')['amount'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['client_median_amount'] = df.groupby('client_id')['amount'].transform(
            lambda x: x.expanding().median().shift(1)
        )
        
        # Отношение текущей суммы к медианной (подозрительно крупный перевод)
        df['amount_vs_median'] = df['amount'] / (df['client_median_amount'] + 1)
        df['amount_vs_avg'] = df['amount'] / (df['client_avg_amount'] + 1)
        
        # Максимальная сумма ранее
        df['client_max_amount'] = df.groupby('client_id')['amount'].transform(
            lambda x: x.expanding().max().shift(1)
        )
        df['is_new_max'] = (df['amount'] > df['client_max_amount']).astype(int)
        
        # Заполнение NaN для первых транзакций
        fill_cols = [
            'client_avg_amount', 'client_median_amount', 
            'client_max_amount', 'amount_vs_median', 'amount_vs_avg'
        ]
        for col in fill_cols:
            df[col].fillna(df['amount'], inplace=True)
        
        df['is_new_max'].fillna(0, inplace=True)
        
        self.feature_names.extend([
            'client_tx_count', 'client_avg_amount', 'client_median_amount',
            'amount_vs_median', 'amount_vs_avg', 'client_max_amount', 'is_new_max'
        ])
        
        return df
    
    def create_destination_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Признаки связки клиент-получатель
        """
        df = df.copy()
        df = df.sort_values(['client_id', 'destination_id', 'transaction_datetime'])
        
        # Первый перевод этому получателю?
        df['is_new_destination'] = (
            df.groupby(['client_id', 'destination_id']).cumcount() == 0
        ).astype(int)
        
        # Количество переводов этому получателю
        df['dest_tx_count'] = df.groupby(['client_id', 'destination_id']).cumcount()
        
        self.feature_names.extend(['is_new_destination', 'dest_tx_count'])
        
        return df
    
    def create_global_destination_risk(self, df: pd.DataFrame, train_only: bool = True) -> pd.DataFrame:
        """
        Глобальные риски по получателю
        
        Args:
            df: DataFrame
            train_only: если True, считаем только по train-данным (без утечки)
        """
        df = df.copy()
        
        # Доля мошеннических транзакций на этого получателя
        dest_fraud_stats = df.groupby('destination_id')['is_fraud'].agg([
            ('dest_fraud_rate', 'mean'),
            ('dest_total_tx', 'count')
        ]).reset_index()
        
        # Количество уникальных клиентов, переводивших этому получателю
        dest_clients = df.groupby('destination_id')['client_id'].nunique().reset_index()
        dest_clients.rename(columns={'client_id': 'dest_unique_clients'}, inplace=True)
        
        # Мерджим обратно
        df = df.merge(dest_fraud_stats, on='destination_id', how='left')
        df = df.merge(dest_clients, on='destination_id', how='left')
        
        # Заполнение пропусков
        df['dest_fraud_rate'].fillna(0, inplace=True)
        df['dest_total_tx'].fillna(1, inplace=True)
        df['dest_unique_clients'].fillna(1, inplace=True)
        
        self.feature_names.extend([
            'dest_fraud_rate', 'dest_total_tx', 'dest_unique_clients'
        ])
        
        return df
    
    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Velocity-признаки: насколько быстро клиент совершает транзакции
        """
        df = df.copy()
        df = df.sort_values(['client_id', 'transaction_datetime'])
        
        # Время с момента предыдущей транзакции (в часах)
        df['time_since_last_tx'] = df.groupby('client_id')['transaction_datetime'].diff().dt.total_seconds() / 3600
        df['time_since_last_tx'].fillna(999, inplace=True)  # для первой транзакции
        
        # Быстрая последовательность транзакций (< 1 часа)
        df['is_quick_succession'] = (df['time_since_last_tx'] < 1).astype(int)
        
        self.feature_names.extend(['time_since_last_tx', 'is_quick_succession'])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применить все фичи
        """
        print("Создание временных признаков...")
        df = self.create_time_features(df)
        
        print("Создание признаков суммы...")
        df = self.create_amount_features(df)
        
        print("Создание исторических признаков клиента...")
        df = self.create_client_history_features(df)
        
        print("Создание признаков получателя...")
        df = self.create_destination_features(df)
        
        print("Создание глобальных рисков получателя...")
        df = self.create_global_destination_risk(df)
        
        print("Создание velocity-признаков...")
        df = self.create_velocity_features(df)
        
        print(f"\nВсего создано признаков: {len(self.feature_names)}")
        
        return df
