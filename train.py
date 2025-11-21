"""
Основной скрипт обучения модели детекции мошенничества
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path

from utils.data_loader import load_and_prepare_data
from features.feature_engineering import FraudFeatureEngineer
from models.fraud_model import FraudDetectionModel
from models.explainer import ModelExplainer


def main():
    """Основной пайплайн обучения"""
    
    print("="*70)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*70)
    
    # Пути к данным
    DATA_DIR = Path('/usr/src/forte/data')
    MODELS_DIR = Path('/usr/src/forte/models')
    REPORTS_DIR = Path('/usr/src/forte/reports')
    
    # Создаём директории если их нет
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    
    transactions_path = DATA_DIR / 'транзакции в Мобильном интернет Банкинге.csv'
    behavioral_path = DATA_DIR / 'поведенческие паттерны клиентов.csv'
    
    # Проверка наличия файлов
    if not transactions_path.exists():
        print(f"\n❌ Файл не найден: {transactions_path}")
        print("Пожалуйста, поместите файлы в директорию /usr/src/forte/data/")
        return
    
    if not behavioral_path.exists():
        print(f"\n❌ Файл не найден: {behavioral_path}")
        print("Пожалуйста, поместите файлы в директорию /usr/src/forte/data/")
        return
    
    # ========================================================================
    # ШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("="*70)
    
    df = load_and_prepare_data(
        str(transactions_path),
        str(behavioral_path),
        encoding='cp1251',
        sep=';'
    )
    
    # ========================================================================
    # ШАГ 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 2: FEATURE ENGINEERING")
    print("="*70)
    
    fe = FraudFeatureEngineer()
    df = fe.fit_transform(df)
    
    print(f"\nИтоговая размерность: {df.shape}")
    print(f"Всего признаков после FE: {df.shape[1]}")
    
    # ========================================================================
    # ШАГ 3: РАЗБИЕНИЕ НА TRAIN/VALID/TEST
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 3: РАЗБИЕНИЕ НА TRAIN/VALID/TEST")
    print("="*70)
    
    model = FraudDetectionModel(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42
    )
    
    train_df, valid_df, test_df = model.temporal_split(
        df,
        train_size=0.7,
        valid_size=0.15
    )
    
    # Подготовка признаков
    X_train, y_train = model.prepare_features(train_df)
    X_valid, y_valid = model.prepare_features(valid_df)
    X_test, y_test = model.prepare_features(test_df)
    
    # ========================================================================
    # ШАГ 4: ОБУЧЕНИЕ МОДЕЛИ
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 4: ОБУЧЕНИЕ МОДЕЛИ")
    print("="*70)
    
    model.train(X_train, y_train, X_valid, y_valid)
    
    # ========================================================================
    # ШАГ 5: ОЦЕНКА НА TRAIN/VALID/TEST (с дефолтным порогом 0.5)
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 5: ПЕРВИЧНАЯ ОЦЕНКА (threshold=0.5)")
    print("="*70)
    
    model.evaluate(X_train, y_train, threshold=0.5, dataset_name="Train")
    model.evaluate(X_valid, y_valid, threshold=0.5, dataset_name="Valid")
    model.evaluate(X_test, y_test, threshold=0.5, dataset_name="Test")
    
    # ========================================================================
    # ШАГ 6: ПОДБОР ОПТИМАЛЬНОГО ПОРОГА
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 6: ПОДБОР ОПТИМАЛЬНОГО ПОРОГА")
    print("="*70)
    
    optimal_threshold = model.find_optimal_threshold(
        X_valid,
        y_valid,
        metric='f2',  # F2 больше штрафует пропуск мошенничества
        plot=True
    )
    
    # Оценка с оптимальным порогом
    print("\n" + "="*70)
    print(f"ОЦЕНКА С ОПТИМАЛЬНЫМ ПОРОГОМ ({optimal_threshold:.4f})")
    print("="*70)
    
    model.evaluate(X_valid, y_valid, threshold=optimal_threshold, dataset_name="Valid")
    test_metrics = model.evaluate(X_test, y_test, threshold=optimal_threshold, dataset_name="Test")
    
    # ========================================================================
    # ШАГ 7: FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 7: FEATURE IMPORTANCE")
    print("="*70)
    
    importance_df = model.get_feature_importance(top_n=20, plot=True)
    
    # Сохранение в CSV
    importance_df.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)
    print(f"\n✓ Feature importance сохранён: {REPORTS_DIR / 'feature_importance.csv'}")
    
    print("\nТоп-10 важнейших признаков:")
    print(importance_df.head(10).to_string(index=False))
    
    # ========================================================================
    # ШАГ 8: SHAP ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 8: SHAP ANALYSIS")
    print("="*70)
    
    explainer = ModelExplainer(model, X_train.sample(min(100, len(X_train)), random_state=42))
    
    # Глобальное объяснение
    print("\nГлобальное объяснение (на тестовой выборке)...")
    explainer.explain_global(
        X_test,
        max_display=20,
        save_path=str(REPORTS_DIR / 'shap_summary.png')
    )
    
    # Примеры объяснений конкретных транзакций
    print("\nОбъяснение конкретных транзакций...")
    explainer.explain_fraud_vs_clean(
        X_test,
        y_test,
        n_fraud=2,
        n_clean=2,
        save_dir=str(REPORTS_DIR)
    )
    
    # ========================================================================
    # ШАГ 9: СОХРАНЕНИЕ МОДЕЛИ
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 9: СОХРАНЕНИЕ МОДЕЛИ")
    print("="*70)
    
    model_path = MODELS_DIR / 'fraud_detection_model.pkl'
    model.save(str(model_path))
    
    # Сохранение метрик
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(REPORTS_DIR / 'test_metrics.csv', index=False)
    print(f"✓ Метрики сохранены: {REPORTS_DIR / 'test_metrics.csv'}")
    
    # ========================================================================
    # ФИНАЛЬНЫЙ ОТЧЁТ
    # ========================================================================
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*70)
    print(f"\nМодель сохранена: {model_path}")
    print(f"Отчёты сохранены в: {REPORTS_DIR}")
    print(f"\nОсновные метрики на TEST:")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-score:  {test_metrics['f1']:.4f}")
    print(f"  F2-score:  {test_metrics['f2']:.4f}")
    print(f"  Threshold: {test_metrics['threshold']:.4f}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
