"""
Утилиты для загрузки и предобработки данных
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def load_transactions(filepath: str, encoding: str = 'cp1251', sep: str = ';') -> pd.DataFrame:
    """
    Загрузка таблицы транзакций
    
    Args:
        filepath: путь к CSV файлу
        encoding: кодировка файла
        sep: разделитель
        
    Returns:
        DataFrame с транзакциями
    """
    # Загрузка с пропуском первой строки (заголовок внутри данных)
    df = pd.read_csv(filepath, encoding=encoding, sep=sep, low_memory=False)
    
    # Приведение колонок к стандартным именам
    column_mapping = {
        'Уникальный идентификатор клиента': 'client_id',
        'Дата совершенной транзакции': 'transaction_date',
        'Дата и время совершенной транзакции': 'transaction_datetime',
        'Сумма совершенного перевода': 'amount',
        'Уникальный идентификатор транзакции': 'transaction_id',
        'Зашифрованный идентификатор получателя/destination транзакции': 'destination_id',
        'Размеченные транзакции(переводы), где 1 - мошенническая операция , 0 - чистая': 'is_fraud'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Удаляем строки-заголовки внутри данных
    df = df[df['client_id'] != 'cst_dim_id'].copy()
    df = df[df['transaction_date'] != 'transdate'].copy()
    
    # Очистка дат от кавычек и преобразование
    if df['transaction_datetime'].dtype == 'object':
        df['transaction_datetime'] = df['transaction_datetime'].astype(str).str.replace("'", "", regex=False)
    if df['transaction_date'].dtype == 'object':
        df['transaction_date'] = df['transaction_date'].astype(str).str.replace("'", "", regex=False)
    
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    
    # Очистка суммы от кавычек и преобразование
    if df['amount'].dtype == 'object':
        df['amount'] = df['amount'].astype(str).str.replace("'", "", regex=False)
        df['amount'] = df['amount'].str.replace(',', '.').astype(float)
    
    # Приведение ID к строковым типам
    df['client_id'] = df['client_id'].astype(str)
    df['destination_id'] = df['destination_id'].astype(str).str.replace("'", "", regex=False)
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Таргет к числовому типу
    if 'is_fraud' in df.columns:
        df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce').astype('Int64')
    
    # Удаление строк с невалидными данными
    df = df.dropna(subset=['transaction_datetime', 'amount', 'is_fraud'])
    
    print(f"Загружено транзакций: {len(df)}")
    print(f"Период: {df['transaction_datetime'].min()} - {df['transaction_datetime'].max()}")
    print(f"Уникальных клиентов: {df['client_id'].nunique()}")
    
    return df


def load_behavioral_patterns(filepath: str, encoding: str = 'cp1251', sep: str = ';') -> pd.DataFrame:
    """
    Загрузка таблицы поведенческих паттернов
    
    Args:
        filepath: путь к CSV файлу
        encoding: кодировка файла
        sep: разделитель
        
    Returns:
        DataFrame с поведенческими признаками
    """
    df = pd.read_csv(filepath, encoding=encoding, sep=sep, low_memory=False)
    
    # Стандартизация ключевых колонок
    base_mapping = {
        'Уникальный идентификатор клиента': 'client_id',
        'Дата совершенной транзакции': 'transaction_date',
        'UniqueCustomerID': 'client_id',
        'date': 'transaction_date'
    }
    
    # Переименовываем
    cols_to_rename = {k: v for k, v in base_mapping.items() if k in df.columns}
    df.rename(columns=cols_to_rename, inplace=True)
    
    # Удаляем строки-заголовки внутри данных
    if 'client_id' in df.columns:
        df = df[df['client_id'] != 'UniqueCustomerID'].copy()
        df = df[df['client_id'] != 'cst_dim_id'].copy()
    
    # Очистка client_id от кавычек
    df['client_id'] = df['client_id'].astype(str).str.replace("'", "", regex=False)
    
    # Очистка даты от кавычек и преобразование
    if df['transaction_date'].dtype == 'object':
        df['transaction_date'] = df['transaction_date'].astype(str).str.replace("'", "", regex=False)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    
    # Приведение всех числовых колонок
    for col in df.columns:
        if col not in ['client_id', 'transaction_date']:
            # Пропускаем явно категориальные колонки (модель телефона, ОС)
            if 'модель' in col.lower() or 'os' in col.lower() or 'model' in col.lower():
                df[col] = df[col].astype(str).str.replace("'", "", regex=False)
                continue
            
            # Пытаемся конвертировать в числовой тип
            try:
                # Очистка от кавычек
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace("'", "", regex=False)
                    df[col] = df[col].str.replace(',', '.')
                
                # Конвертация в float
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                # Если не получилось - оставляем как есть
                pass
    
    # Удаление строк с невалидной датой
    df = df.dropna(subset=['transaction_date'])
    
    print(f"Загружено поведенческих записей: {len(df)}")
    print(f"Уникальных клиентов: {df['client_id'].nunique()}")
    print(f"Признаков: {len(df.columns)}")
    
    return df


def merge_datasets(transactions: pd.DataFrame, behavioral: pd.DataFrame) -> pd.DataFrame:
    """
    Объединение транзакций с поведенческими признаками
    
    Args:
        transactions: таблица транзакций
        behavioral: таблица поведенческих паттернов
        
    Returns:
        Объединённый DataFrame
    """
    # Приводим transaction_date к одинаковому формату для обеих таблиц
    transactions['transaction_date_key'] = pd.to_datetime(transactions['transaction_date']).dt.date
    behavioral['transaction_date_key'] = pd.to_datetime(behavioral['transaction_date']).dt.date
    
    # Merge
    merged = transactions.merge(
        behavioral,
        left_on=['client_id', 'transaction_date_key'],
        right_on=['client_id', 'transaction_date_key'],
        how='left',
        suffixes=('', '_beh')
    )
    
    # Удаляем дубликаты колонок после merge
    duplicate_cols = [col for col in merged.columns if col.endswith('_beh')]
    merged = merged.drop(columns=duplicate_cols)
    
    print(f"\nРезультаты объединения:")
    print(f"Транзакций до merge: {len(transactions)}")
    print(f"Поведенческих записей: {len(behavioral)}")
    print(f"Транзакций после merge: {len(merged)}")
    
    # Статистика по поведенческим данным
    behavioral_cols = [col for col in behavioral.columns if col not in ['client_id', 'transaction_date', 'transaction_date_key']]
    if behavioral_cols:
        has_behavioral = merged[behavioral_cols[0]].notna().sum()
        print(f"Транзакций с поведенческими данными: {has_behavioral} ({has_behavioral/len(merged)*100:.1f}%)")
    
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка данных: обработка пропусков, приведение типов
    
    Args:
        df: исходный DataFrame
        
    Returns:
        Очищенный DataFrame
    """
    df = df.copy()
    
    # Числовые колонки: заполнение медианой
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'is_fraud':  # таргет не трогаем
            df[col].fillna(df[col].median(), inplace=True)
    
    # Категориальные колонки: заполнение значением "unknown"
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna('unknown', inplace=True)
    
    # Удаление строк без таргета (если есть)
    if 'is_fraud' in df.columns:
        df = df[df['is_fraud'].notna()]
        df['is_fraud'] = df['is_fraud'].astype(int)
    
    return df


def load_and_prepare_data(
    transactions_path: str,
    behavioral_path: str,
    encoding: str = 'cp1251',
    sep: str = ';'
) -> pd.DataFrame:
    """
    Полный пайплайн загрузки и подготовки данных
    
    Args:
        transactions_path: путь к файлу с транзакциями
        behavioral_path: путь к файлу с поведенческими паттернами
        encoding: кодировка
        sep: разделитель
        
    Returns:
        Готовый к обучению DataFrame
    """
    print("Загрузка транзакций...")
    transactions = load_transactions(transactions_path, encoding, sep)
    
    print("Загрузка поведенческих паттернов...")
    behavioral = load_behavioral_patterns(behavioral_path, encoding, sep)
    
    print("Объединение датасетов...")
    merged = merge_datasets(transactions, behavioral)
    
    print("Очистка данных...")
    cleaned = clean_data(merged)
    
    print(f"\nИтоговый размер датасета: {cleaned.shape}")
    print(f"Количество признаков: {cleaned.shape[1]}")
    print(f"Распределение таргета:\n{cleaned['is_fraud'].value_counts()}")
    print(f"Доля мошеннических транзакций: {cleaned['is_fraud'].mean():.4f}")
    
    return cleaned
