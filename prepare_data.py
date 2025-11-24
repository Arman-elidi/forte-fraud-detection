"""
Утилита для подготовки данных из исходных CSV файлов
Автоматически загружает, объединяет и применяет feature engineering
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from utils.data_loader import load_and_prepare_data
from features.feature_engineering import FraudFeatureEngineer


def prepare_data_for_inference(
    transactions_path: str = 'data/транзакции в Мобильном интернет Банкинге.csv',
    behavioral_path: str = 'data/поведенческие паттерны клиентов.csv',
    output_path: str = None
) -> pd.DataFrame:
    """
    Загружает исходные данные, применяет feature engineering и возвращает готовый датасет
    
    Args:
        transactions_path: путь к файлу с транзакциями
        behavioral_path: путь к файлу с поведенческими паттернами
        output_path: путь для сохранения результата (опционально)
        
    Returns:
        DataFrame с полным набором признаков
    """
    print("=" * 70)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ПРОВЕРКИ МОШЕННИЧЕСТВА")
    print("=" * 70)
    
    # Шаг 1: Загрузка и объединение данных
    print("\n📊 Шаг 1: Загрузка исходных данных...")
    print(f"  - Транзакции: {transactions_path}")
    print(f"  - Поведенческие паттерны: {behavioral_path}")
    
    df = load_and_prepare_data(transactions_path, behavioral_path)
    
    print(f"✓ Загружено {len(df)} транзакций")
    print(f"  - Легитимных: {(df['is_fraud']==0).sum()}")
    print(f"  - Мошеннических: {(df['is_fraud']==1).sum()}")
    
    # Шаг 2: Feature Engineering
    print("\n🔧 Шаг 2: Применение feature engineering...")
    fe = FraudFeatureEngineer()
    df_features = fe.fit_transform(df)
    
    print(f"✓ Создано {len(df_features.columns)} признаков")
    
    # Шаг 3: Сохранение (опционально)
    if output_path:
        print(f"\n💾 Шаг 3: Сохранение результата...")
        df_features.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Сохранено в: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ ДАННЫЕ ГОТОВЫ К ИСПОЛЬЗОВАНИЮ")
    print("=" * 70)
    
    return df_features


def create_demo_samples(
    df: pd.DataFrame,
    n_clean: int = 5,
    n_fraud: int = 5,
    output_path: str = 'demo_batch_ready.csv'
) -> pd.DataFrame:
    """
    Создает демо-выборку из подготовленных данных
    
    Args:
        df: DataFrame с полным набором признаков
        n_clean: количество легитимных транзакций
        n_fraud: количество мошеннических транзакций
        output_path: путь для сохранения
        
    Returns:
        DataFrame с демо-выборкой
    """
    print("\n📋 Создание демо-выборки...")
    
    # Выбираем примеры
    clean = df[df['is_fraud'] == 0].sample(min(n_clean, len(df[df['is_fraud'] == 0])), random_state=42)
    fraud = df[df['is_fraud'] == 1].sample(min(n_fraud, len(df[df['is_fraud'] == 1])), random_state=42)
    
    demo_df = pd.concat([clean, fraud])
    
    # Сохраняем
    demo_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Создана демо-выборка:")
    print(f"  - Легитимных: {len(clean)}")
    print(f"  - Мошеннических: {len(fraud)}")
    print(f"  - Сохранено в: {output_path}")
    
    return demo_df


def validate_data_format(df: pd.DataFrame) -> bool:
    """
    Проверяет, что DataFrame содержит все необходимые признаки
    
    Args:
        df: DataFrame для проверки
        
    Returns:
        True если формат корректный, False иначе
    """
    required_columns = [
        'client_id', 'amount', 'transaction_datetime',
        'hour', 'day_of_week', 'is_weekend',
        'is_new_destination', 'client_avg_amount'
    ]
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        print(f"❌ Отсутствуют обязательные колонки: {missing}")
        return False
    
    print(f"✓ Формат данных корректный ({len(df.columns)} колонок)")
    return True


if __name__ == "__main__":
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(description='Подготовка данных для проверки мошенничества')
    parser.add_argument('--transactions', default='data/транзакции в Мобильном интернет Банкинге.csv',
                        help='Путь к файлу с транзакциями')
    parser.add_argument('--behavioral', default='data/поведенческие паттерны клиентов.csv',
                        help='Путь к файлу с поведенческими паттернами')
    parser.add_argument('--output', default='processed_data.csv',
                        help='Путь для сохранения обработанных данных')
    parser.add_argument('--demo', action='store_true',
                        help='Создать демо-выборку')
    parser.add_argument('--demo-clean', type=int, default=5,
                        help='Количество легитимных транзакций в демо')
    parser.add_argument('--demo-fraud', type=int, default=5,
                        help='Количество мошеннических транзакций в демо')
    parser.add_argument('--streamlit', action='store_true',
                        help='Запустить Streamlit после подготовки данных')
    
    args = parser.parse_args()
    
    # Подготовка данных
    df = prepare_data_for_inference(
        transactions_path=args.transactions,
        behavioral_path=args.behavioral,
        output_path=args.output
    )
    
    # Валидация
    validate_data_format(df)
    
    # Создание демо-выборки (опционально)
    if args.demo:
        create_demo_samples(
            df,
            n_clean=args.demo_clean,
            n_fraud=args.demo_fraud,
            output_path='demo_batch_ready.csv'
        )
    
    print("\n✅ Готово!")
    print(f"\nИспользование:")
    print(f"  1. Полный датасет: {args.output}")
    if args.demo:
        print(f"  2. Демо-выборка: demo_batch_ready.csv")
    
    # Запуск Streamlit (опционально)
    if args.streamlit:
        print("\n🚀 Запуск Streamlit...")
        print("=" * 70)
        try:
            subprocess.run(['streamlit', 'run', 'app.py'], check=True)
        except KeyboardInterrupt:
            print("\n\n✓ Streamlit остановлен")
        except Exception as e:
            print(f"\n❌ Ошибка запуска Streamlit: {e}")
            print("Запустите вручную: streamlit run app.py")

