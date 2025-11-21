"""
Интерпретация модели через SHAP
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Optional, List


class ModelExplainer:
    """Класс для объяснения предсказаний модели"""
    
    def __init__(self, model, X_background: Optional[pd.DataFrame] = None):
        """
        Args:
            model: обученная модель FraudDetectionModel
            X_background: фоновые данные для SHAP (можно взять sample из train)
        """
        self.model = model.model  # CatBoost model
        self.feature_cols = model.feature_cols
        self.X_background = X_background
        self.explainer = None
        
    def initialize_explainer(self, X_background: Optional[pd.DataFrame] = None):
        """
        Инициализация SHAP explainer
        
        Для CatBoost используем TreeExplainer
        """
        if X_background is not None:
            self.X_background = X_background
        
        if self.X_background is not None:
            # Используем sample для ускорения
            if len(self.X_background) > 100:
                bg_sample = self.X_background.sample(100, random_state=42)
            else:
                bg_sample = self.X_background
        else:
            bg_sample = None
        
        self.explainer = shap.TreeExplainer(self.model)
        print("✓ SHAP explainer инициализирован")
    
    def explain_global(self, X: pd.DataFrame, max_display: int = 20, save_path: Optional[str] = None):
        """
        Глобальное объяснение: какие признаки важнее в среднем
        
        Args:
            X: данные для объяснения
            max_display: сколько признаков показать
            save_path: путь для сохранения графика
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        # Берём sample для ускорения
        if len(X) > 1000:
            X_sample = X.sample(1000, random_state=42)
        else:
            X_sample = X
        
        print("Вычисление SHAP values...")
        shap_values = self.explainer.shap_values(X_sample)
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ График сохранён: {save_path}")
        
        plt.show()
        
        # Bar plot (средняя абсолютная важность)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=max_display, show=False)
        
        if save_path:
            bar_path = save_path.replace('.png', '_bar.png')
            plt.savefig(bar_path, dpi=150, bbox_inches='tight')
            print(f"✓ График сохранён: {bar_path}")
        
        plt.show()
        
        return shap_values
    
    def explain_instance(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0,
        show_data: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Объяснение конкретной транзакции
        
        Args:
            X: данные
            instance_idx: индекс транзакции
            show_data: показать значения признаков
            save_path: путь для сохранения
            
        Returns:
            SHAP values для этой транзакции
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        # Use .loc instead of .iloc for proper index handling
        instance = X.loc[[instance_idx]]
        shap_values = self.explainer.shap_values(instance)
        
        # Waterfall plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=instance.values[0],
                feature_names=X.columns.tolist()
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ График сохранён: {save_path}")
        
        plt.show()
        
        if show_data:
            print("\nЗначения признаков:")
            for col in X.columns:
                print(f"  {col:30s}: {instance[col].values[0]}")
        
        return shap_values
    
    def explain_multiple_instances(
        self,
        X: pd.DataFrame,
        indices: List[int],
        save_prefix: Optional[str] = None
    ):
        """
        Объяснение нескольких транзакций
        
        Args:
            X: данные
            indices: список индексов
            save_prefix: префикс для сохранения файлов
        """
        for i, idx in enumerate(indices):
            print(f"\n{'='*60}")
            print(f"Транзакция #{idx}")
            print(f"{'='*60}")
            
            save_path = None
            if save_prefix:
                save_path = f"{save_prefix}_tx_{idx}.png"
            
            try:
                self.explain_instance(X, idx, show_data=True, save_path=save_path)
            except Exception as e:
                print(f"⚠️ Ошибка при объяснении транзакции {idx}: {e}")
                continue
    
    def get_top_features_for_instance(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Получить топ-N признаков для конкретной транзакции
        
        Returns:
            DataFrame с признаками, их значениями и SHAP values
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        instance = X.iloc[[instance_idx]]
        shap_values = self.explainer.shap_values(instance)[0]
        
        feature_contributions = pd.DataFrame({
            'feature': X.columns,
            'value': instance.values[0],
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        return feature_contributions.head(top_n)
    
    def explain_fraud_vs_clean(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_fraud: int = 3,
        n_clean: int = 3,
        save_dir: Optional[str] = None
    ):
        """
        Сравнение объяснений для мошеннических и чистых транзакций
        
        Args:
            X: признаки
            y: таргет
            n_fraud: сколько мошеннических показать
            n_clean: сколько чистых показать
            save_dir: директория для сохранения
        """
        fraud_indices = y[y == 1].sample(min(n_fraud, (y == 1).sum()), random_state=42).index.tolist()
        clean_indices = y[y == 0].sample(min(n_clean, (y == 0).sum()), random_state=42).index.tolist()
        
        print(f"\n{'='*60}")
        print("МОШЕННИЧЕСКИЕ ТРАНЗАКЦИИ")
        print(f"{'='*60}")
        
        save_prefix_fraud = f"{save_dir}/fraud" if save_dir else None
        # Pass full X, not subset
        self.explain_multiple_instances(X, fraud_indices, save_prefix_fraud)
        
        print(f"\n\n{'='*60}")
        print("ЧИСТЫЕ ТРАНЗАКЦИИ")
        print(f"{'='*60}")
        
        save_prefix_clean = f"{save_dir}/clean" if save_dir else None
        # Pass full X, not subset
        self.explain_multiple_instances(X, clean_indices, save_prefix_clean)
