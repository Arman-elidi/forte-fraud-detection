# Model Explainer using SHAP

import shap
import pandas as pd
import numpy as np
from pathlib import Path

class ModelExplainer:
    """Provides SHAP explanations for the FraudDetectionModel.

    - Global explanation (summary plot)
    - Local explanation for a single transaction
    - Top‑N feature contributions
    """

    def __init__(self, model):
        self.model = model
        # Use TreeExplainer for CatBoost
        self.explainer = shap.TreeExplainer(self.model.model)

    def explain_global(self, X: pd.DataFrame, max_display: int = 20):
        """Generate global SHAP summary plot.
        Returns the SHAP values matrix and optionally displays the plot.
        """
        shap_values = self.explainer.shap_values(X)
        if max_display:
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        return shap_values

    def explain_instance(self, instance: pd.Series):
        """Explain a single transaction.
        Returns a dictionary with feature, value and SHAP contribution.
        """
        # Ensure instance is a DataFrame with one row
        df = pd.DataFrame([instance])
        shap_vals = self.explainer.shap_values(df)
        contributions = shap_vals[0]
        result = []
        for feature, value, contrib in zip(df.columns, instance, contributions):
            result.append({
                'feature': feature,
                'value': value,
                'impact': float(contrib)
            })
        # Sort by absolute impact descending
        result.sort(key=lambda x: abs(x['impact']), reverse=True)
        return result

    def get_top_features(self, X: pd.DataFrame, top_n: int = 10):
        """Return top‑n features with highest mean absolute SHAP value.
        Useful for model documentation.
        """
        shap_vals = self.explainer.shap_values(X)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_abs
        }).sort_values('importance', ascending=False).head(top_n)
        return feature_importance
