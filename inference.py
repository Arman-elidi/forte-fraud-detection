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
        
        # Генерация недостающих признаков
        df = self.generate_missing_features(df)
        
        # Убеждаемся, что все нужные признаки есть
        for col in self.model.feature_cols:
            if col not in df.columns:
                df[col] = -9999  # sentinel value for missing
        
        # Приводим к нужному порядку
        X = df[self.model.feature_cols]
        
        # ВАЖНО: Обрабатываем категориальные и числовые признаки отдельно
        categorical_cols = self.model.categorical_features if self.model.categorical_features else []
        
        # Для категориальных признаков: конвертируем в строки
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna('missing').astype(str)
        
        # Для числовых признаков: заполняем NaN числовым значением
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
        for col in numeric_cols:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    pass
            X[col] = X[col].fillna(-9999)
        
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
    
    @staticmethod
    def generate_missing_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация недостающих признаков из доступных данных
        
        Args:
            df: DataFrame с базовыми данными
            
        Returns:
            DataFrame с добавленными признаками
        """
        df = df.copy()
        
        # 1. ВРЕМЕННЫЕ ПРИЗНАКИ из transaction_datetime
        if 'transaction_datetime' in df.columns:
            df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
            df['hour'] = df['transaction_datetime'].dt.hour
            df['day_of_week'] = df['transaction_datetime'].dt.dayofweek
            df['day_of_month'] = df['transaction_datetime'].dt.day
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 7)).astype(int)
            df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 12)).astype(int)
            df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)
        
        # transaction_date_key для категориального признака
        if 'transaction_date' in df.columns:
            df['transaction_date_key'] = pd.to_datetime(df['transaction_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # 2. ПРИЗНАКИ СУММЫ из amount
        if 'amount' in df.columns:
            # Конвертируем amount в числовой тип
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            
            df['log_amount'] = np.log1p(df['amount'])
            df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
            df['is_round_100'] = (df['amount'] % 100 == 0).astype(int)
            
            # amount_bucket - категориальный признак
            df['amount_bucket'] = pd.cut(
                df['amount'], 
                bins=[0, 1000, 5000, 10000, 50000, np.inf],
                labels=['0-1k', '1k-5k', '5k-10k', '10k-50k', '50k+']
            ).astype(str)
        
        # 3. АГРЕГИРОВАННЫЕ ПРИЗНАКИ - используем разумные значения по умолчанию
        # Признаки клиента
        df['client_tx_count'] = 1  # Минимум 1 транзакция
        if 'amount' in df.columns:
            median_amount = df['amount'].median()
            if pd.isna(median_amount) or median_amount == 0:
                median_amount = 5000
            df['client_avg_amount'] = median_amount
            df['client_median_amount'] = median_amount
            df['amount_vs_median'] = df['amount'] / (median_amount + 1)
            df['amount_vs_avg'] = df['amount'] / (median_amount + 1)
            df['client_max_amount'] = df['amount']
        else:
            df['client_avg_amount'] = 5000
            df['client_median_amount'] = 5000
            df['amount_vs_median'] = 1.0
            df['amount_vs_avg'] = 1.0
            df['client_max_amount'] = 5000
        
        df['is_new_max'] = 0  # Не новый максимум
        
        # Признаки получателя
        df['is_new_destination'] = 0  # Предполагаем известного получателя
        df['dest_tx_count'] = 1
        df['dest_fraud_rate'] = 0.0  # Низкий риск по умолчанию
        df['dest_total_tx'] = 1
        df['dest_unique_clients'] = 1
        
        # Временные интервалы
        df['time_since_last_tx'] = 86400  # 1 день в секундах
        df['is_quick_succession'] = 0
        
        # Признаки устройств
        df['unique_devices_30d'] = 1
        df['device_changed_24h'] = 0
        df['device_limit_exceeded'] = 0
        df['suspicious_device_activity'] = 0
        
        # Дополнительные логин-признаки (если базовые уже есть)
        if 'avg_login_interval_30d' in df.columns:
            df['avg_login_interval_7d'] = pd.to_numeric(df['avg_login_interval_30d'], errors='coerce').fillna(3600)
        else:
            df['avg_login_interval_7d'] = 3600  # 1 час
            df['avg_login_interval_30d'] = 3600
        
        if 'login_frequency_7d' in df.columns and 'login_frequency_30d' in df.columns:
            # Конвертируем в числовой тип перед вычислениями
            freq_7d = pd.to_numeric(df['login_frequency_7d'], errors='coerce').fillna(0)
            freq_30d = pd.to_numeric(df['login_frequency_30d'], errors='coerce').fillna(0)
            df['freq_change_7d_vs_30d'] = (freq_7d - freq_30d) / (freq_30d + 0.001)
        else:
            df['freq_change_7d_vs_30d'] = 0.0
        
        df['login_frequency_spike'] = 0
        
        if 'burstiness_login_interval' in df.columns:
            burstiness = pd.to_numeric(df['burstiness_login_interval'], errors='coerce').fillna(0)
            df['high_burstiness_flag'] = (burstiness > 0.5).astype(int)
        else:
            df['high_burstiness_flag'] = 0
        
        # Dropper-признаки
        df['incoming_transfers_24h'] = 0
        df['time_since_last_incoming'] = 86400
        df['incoming_transfers_24h_tmp'] = 0
        df['time_since_last_incoming_tmp'] = 86400
        df['dropper_activity_flag'] = 0
        
        return df
    
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
        # ШАГИ ОБРАБОТКИ:
        # 1. Генерация недостающих признаков
        # 2. Маппинг английских названий в русские
        # 3. Предсказание
        # 4. Переименование обратно в английские для UI
        
        # ШАГ 1: Генерация недостающих признаков
        transactions_df = self.generate_missing_features(transactions_df)
        
        # ШАГ 2: Маппинг английских названий в русские (для совместимости со старой моделью)
        reverse_mapping = {
            'monthly_os_changes': 'Количество разных версий ОС (os_ver) за последние 30 дней до transdate — сколько разных ОС/версий использовал клиент',
            'monthly_phone_model_changes': 'Количество разных моделей телефона (phone_model) за последние 30 дней — насколько часто клиент "менял устройство" по логам',
            'last_phone_model': 'Модель телефона из самой последней сессии (по времени) перед transdate',
            'last_phone_model_categorical': 'Модель телефона из самой последней сессии (по времени) перед transdate',
            'last_os_version': 'Версия ОС из самой последней сессии перед transdate',
            'last_os_categorical': 'Версия ОС из самой последней сессии перед transdate',
            'logins_last_7_days': 'Количество уникальных логин-сессий (минутных тайм-слотов) за последние 7 дней до transdate',
            'logins_last_30_days': 'Количество уникальных логин-сессий за последние 30 дней до transdate',
            'login_frequency_7d': 'Среднее число логинов в день за последние 7 дней: logins_last_7_days / 7',
            'login_frequency_30d': 'Среднее число логинов в день за последние 30 дней: logins_last_30_days / 30',
            'freq_change_7d_vs_mean': 'Относительное изменение частоты логинов за 7 дней к средней частоте за 30 дней:\n(freq7d?freq30d)/freq30d(freq_{7d} - freq_{30d}) / freq_{30d}(freq7d?freq30d)/freq30d — показывает, стал клиент заходить чаще или реже недавно',
            'logins_7d_over_30d_ratio': 'Доля логинов за 7 дней от логинов за 30 дней',
            'avg_login_interval_30d': 'Средний интервал (в секундах) между соседними сессиями за последние 30 дней',
            'std_login_interval_30d': 'Стандартное отклонение интервалов между логинами за 30 дней (в секундах), измеряет разброс интервалов',
            'var_login_interval_30d': 'Дисперсия интервалов между логинами за 30 дней (в секундах²), ещё одна мера разброса',
            'ewm_login_interval_7d': 'Экспоненциально взвешенное среднее интервалов между логинами за 7 дней, где более свежие сессии имеют больший вес (коэффициент затухания 0.3)',
            'burstiness_login_interval': 'Показатель "взрывности" логинов: (std−mean)/(std+mean)(std - mean)/(std + mean)(std−mean)/(std+mean) для интервалов',
            'fano_factor_login_interval': 'Fano-factor интервалов: variance / mean',
            'zscore_avg_login_interval_7d': 'Z-скор среднего интервала за последние 7 дней относительно среднего за 30 дней: насколько сильно недавние интервалы отличаются от типичных, в единицах стандартного отклонения'
        }
        
        # Применяем reverse mapping (английские → русские) если есть английские колонки
        for eng_col, rus_col in reverse_mapping.items():
            if eng_col in transactions_df.columns and rus_col in self.model.feature_cols:
                # Переименовываем в русское название для совместимости
                transactions_df.rename(columns={eng_col: rus_col}, inplace=True)
        
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

        # ВАЖНО: Обрабатываем категориальные и числовые признаки отдельно
        # Получаем список категориальных признаков из модели
        categorical_cols = self.model.categorical_features if self.model.categorical_features else []
        
        # Для категориальных признаков: конвертируем в строки
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna('missing').astype(str)
        
        # Для числовых признаков: заполняем NaN числовым значением
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
        for col in numeric_cols:
            if X[col].dtype == 'object':
                # Если это object, но не категориальный признак, пытаемся конвертировать в число
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    pass
            # Заполняем NaN
            X[col] = X[col].fillna(-9999)
        
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
        
        # ФИНАЛЬНОЕ ПЕРЕИМЕНОВАНИЕ: русские названия → английские короткие названия для UI
        final_rename_mapping = {
            'Количество разных версий ОС (os_ver) за последние 30 дней до transdate — сколько разных ОС/версий использовал клиент': 'monthly_os_changes',
            'Количество разных моделей телефона (phone_model) за последние 30 дней — насколько часто клиент "менял устройство" по логам': 'monthly_phone_model_changes',
            'Модель телефона из самой последней сессии (по времени) перед transdate': 'last_phone_model_categorical',
            'Версия ОС из самой последней сессии перед transdate': 'last_os_categorical',
            'Количество уникальных логин-сессий (минутных тайм-слотов) за последние 7 дней до transdate': 'logins_last_7_days',
            'Количество уникальных логин-сессий за последние 30 дней до transdate': 'logins_last_30_days',
            'Среднее число логинов в день за последние 7 дней: logins_last_7_days / 7': 'login_frequency_7d',
            'Среднее число логинов в день за последние 30 дней: logins_last_30_days / 30': 'login_frequency_30d',
            'Относительное изменение частоты логинов за 7 дней к средней частоте за 30 дней:\n(freq7d?freq30d)/freq30d(freq_{7d} - freq_{30d}) / freq_{30d}(freq7d?freq30d)/freq30d — показывает, стал клиент заходить чаще или реже недавно': 'freq_change_7d_vs_mean',
            'Доля логинов за 7 дней от логинов за 30 дней': 'logins_7d_over_30d_ratio',
            'Средний интервал (в секундах) между соседними сессиями за последние 30 дней': 'avg_login_interval_30d',
            'Стандартное отклонение интервалов между логинами за 30 дней (в секундах), измеряет разброс интервалов': 'std_login_interval_30d',
            'Дисперсия интервалов между логинами за 30 дней (в секундах²), ещё одна мера разброса': 'var_login_interval_30d',
            'Дисперсия интервалов между логинами за 30 дней (в секундах?), ещё одна мера разброса': 'var_login_interval_30d',
            'Экспоненциально взвешенное среднее интервалов между логинами за 7 дней, где более свежие сессии имеют больший вес (коэффициент затухания 0.3)': 'ewm_login_interval_7d',
            'Показатель "взрывности" логинов: (std−mean)/(std+mean)(std - mean)/(std + mean)(std−mean)/(std+mean) для интервалов': 'burstiness_login_interval',
            'Показатель "взрывности" логинов: (std?mean)/(std+mean)(std - mean)/(std + mean)(std?mean)/(std+mean) для интервалов': 'burstiness_login_interval',
            'Fano-factor интервалов: variance / mean': 'fano_factor_login_interval',
            'Z-скор среднего интервала за последние 7 дней относительно среднего за 30 дней: насколько сильно недавние интервалы отличаются от типичных, в единицах стандартного отклонения': 'zscore_avg_login_interval_7d'
        }
        predictions.rename(columns=final_rename_mapping, inplace=True)
        
        # Удаляем дубликаты колонок (оставляем первое вхождение)
        predictions = predictions.loc[:, ~predictions.columns.duplicated()]
        
        # Оставляем только нужные колонки (включая результаты предсказания)
        required_columns = [
            'client_id', 'transaction_date', 'transaction_datetime', 'amount', 
            'transaction_id', 'destination_id', 'is_fraud',
            'monthly_os_changes', 'monthly_phone_model_changes', 
            'last_phone_model_categorical', 'last_os_categorical',
            'logins_last_7_days', 'logins_last_30_days', 
            'login_frequency_7d', 'login_frequency_30d',
            'freq_change_7d_vs_mean', 'logins_7d_over_30d_ratio',
            'avg_login_interval_30d', 'std_login_interval_30d', 
            'var_login_interval_30d', 'ewm_login_interval_7d',
            'burstiness_login_interval', 'fano_factor_login_interval', 
            'zscore_avg_login_interval_7d',
            # Добавляем колонки с результатами предсказания
            'fraud_probability', 'recommendation'
        ]
        
        # Фильтруем только те колонки, которые есть в predictions
        available_columns = [col for col in required_columns if col in predictions.columns]
        predictions = predictions[available_columns]
        
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
