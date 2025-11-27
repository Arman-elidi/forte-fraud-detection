# Fraud Detection Model Wrapper

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import joblib
from pathlib import Path

class FraudDetectionModel:
    """Wrapper around CatBoost for fraud detection.

    Provides methods for training, inference, threshold optimisation and model persistence.
    """

    def __init__(self, iterations: int = 1000, learning_rate: float = 0.03, depth: int = 6, random_seed: int = 42):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=self.random_seed,
            verbose=False,
        )
        # Default threshold – will be tuned later
        self.threshold = 0.5
        self.feature_cols = []

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'is_fraud', exclude_cols=None) -> tuple:
        """Separate features and target.
        Returns X (DataFrame) and y (Series).
        """
        if exclude_cols is None:
            exclude_cols = []
        cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
        self.feature_cols = cols
        X = df[cols]
        y = df[target_col]
        return X, y

    def temporal_split(self, df: pd.DataFrame, train_size: float = 0.7, valid_size: float = 0.15, date_col: str = 'transaction_datetime') -> tuple:
        """Split data chronologically to avoid leakage.
        Returns train_df, valid_df, test_df.
        """
        df = df.sort_values(date_col)
        n = len(df)
        train_end = int(n * train_size)
        valid_end = train_end + int(n * valid_size)
        train_df = df.iloc[:train_end]
        valid_df = df.iloc[train_end:valid_end]
        test_df = df.iloc[valid_end:]
        return train_df, valid_df, test_df

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame = None, y_valid: pd.Series = None):
        """Fit the CatBoost model.
        If validation data is provided, it will be used for early stopping.
        """
        train_pool = Pool(X_train, y_train)
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = Pool(X_valid, y_valid)
        self.model.fit(train_pool, eval_set=eval_set)
        return self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of class 1 (fraud)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Predict class using a threshold (default self.threshold)."""
        if threshold is None:
            threshold = self.threshold
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = None) -> dict:
        """Calculate precision, recall, f1, fbeta (beta=2) and ROC‑AUC.
        Returns a dictionary with metrics.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
        if threshold is None:
            threshold = self.threshold
        preds = self.predict(X, threshold)
        probs = self.predict_proba(X)
        metrics = {
            'precision': precision_score(y, preds, zero_division=0),
            'recall': recall_score(y, preds, zero_division=0),
            'f1': f1_score(y, preds, zero_division=0),
            'f2': fbeta_score(y, preds, beta=2, zero_division=0),
            'roc_auc': roc_auc_score(y, probs),
        }
        return metrics

    def find_optimal_threshold(self, X: pd.DataFrame, y: pd.Series, metric: str = 'f2', plot: bool = False):
        """Search threshold that maximises the chosen metric (precision, recall, f1, f2).
        Updates self.threshold and optionally plots the metric curve.
        """
        probs = self.predict_proba(X)
        thresholds = np.linspace(0.0, 1.0, 101)
        best_thr = 0.5
        best_score = -1
        scores = []
        from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            if metric == 'precision':
                score = precision_score(y, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y, preds, zero_division=0)
            elif metric == 'f1':
                score = f1_score(y, preds, zero_division=0)
            elif metric == 'f2':
                score = fbeta_score(y, preds, beta=2, zero_division=0)
            else:
                raise ValueError('Unsupported metric')
            scores.append(score)
            if score > best_score:
                best_score = score
                best_thr = thr
        self.threshold = best_thr
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(thresholds, scores, marker='o')
            plt.title(f'Optimal {metric.upper()} threshold')
            plt.xlabel('Threshold')
            plt.ylabel(metric.upper())
            plt.grid(True)
            plt.show()
        return best_thr, best_score

    def get_feature_importance(self, top_n: int = 20, plot: bool = True):
        """Return feature importance from CatBoost.
        If plot=True, displays a bar chart.
        """
        importances = self.model.get_feature_importance(type='PredictionValuesChange')
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        top = importance_df.head(top_n)
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.barh(top['feature'][::-1], top['importance'][::-1])
            plt.title('Top Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
        return top

    def save(self, filepath: str):
        """Serialize model to disk using joblib."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load a previously saved model.
        Returns an instance of FraudDetectionModel.
        """
        model = joblib.load(filepath)
        
        # Handle case where model was saved as a dictionary (legacy or different version)
        if isinstance(model, dict):
            instance = cls()
            if 'model' in model:
                instance.model = model['model']
            if 'feature_cols' in model:
                instance.feature_cols = model['feature_cols']
            if 'categorical_features' in model:
                instance.categorical_features = model['categorical_features']
            if 'threshold' in model:
                instance.threshold = model['threshold']
            return instance
            
        if not isinstance(model, cls):
            raise TypeError('Loaded object is not a FraudDetectionModel')
        return model
