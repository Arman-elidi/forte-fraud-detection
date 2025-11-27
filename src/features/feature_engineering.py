# Feature Engineering Pipeline

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features required for fraud detection.

    The pipeline covers:
    - Temporal features (hour, day_of_week, is_weekend, is_night)
    - Amount features (log_amount, amount_bucket, is_round_amount)
    - Client historical aggregates (client_tx_count, client_avg_amount, client_median_amount, amount_vs_median)
    - Behavioral patterns (login frequency, OS changes, phone model changes, burstiness, fano factor, z‑score intervals)
    - Velocity features (time_since_last_tx, is_quick_succession)

    The function returns a new DataFrame with original columns plus engineered ones.
    """
    df = df.copy()

    # --- Temporal features ---
    if 'transaction_datetime' in df.columns:
        df['hour'] = df['transaction_datetime'].dt.hour
        df['day_of_week'] = df['transaction_datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 7)).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 12)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)

    # --- Amount features ---
    if 'amount' in df.columns:
        df['log_amount'] = np.log1p(df['amount'])
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
        df['is_round_100'] = (df['amount'] % 100 == 0).astype(int)
        # Quantile bucket (5 buckets)
        df['amount_bucket'] = pd.qcut(df['amount'], q=5, labels=False, duplicates='drop')

    # --- Client historical aggregates ---
    if 'client_id' in df.columns:
        client_agg = df.groupby('client_id')['amount'].agg(['count', 'mean', 'median', 'max']).rename(columns={
            'count': 'client_tx_count',
            'mean': 'client_avg_amount',
            'median': 'client_median_amount',
            'max': 'client_max_amount'
        })
        df = df.merge(client_agg, left_on='client_id', right_index=True, how='left')
        df['amount_vs_median'] = df['amount'] / (df['client_median_amount'] + 1)
        df['amount_vs_avg'] = df['amount'] / (df['client_avg_amount'] + 1)
        df['is_new_max'] = (df['amount'] > df['client_max_amount']).astype(int)

    # --- Behavioral features (placeholders) ---
    # These columns are expected to be present after merging with behavioral data.
    # If they exist, we keep them; otherwise we fill with zeros.
    behavioral_cols = [
        'monthly_os_changes', 'monthly_phone_model_changes', 'last_phone_model_categorical',
        'last_os_categorical', 'logins_last_7_days', 'logins_last_30_days',
        'login_frequency_7d', 'login_frequency_30d', 'freq_change_7d_vs_mean',
        'logins_7d_over_30d_ratio', 'avg_login_interval_30d', 'std_login_interval_30d',
        'var_login_interval_30d', 'ewm_login_interval_7d', 'burstiness_login_interval',
        'fano_factor_login_interval', 'zscore_avg_login_interval_7d'
    ]
    for col in behavioral_cols:
        if col not in df.columns:
            df[col] = 0

    # --- Velocity features ---
    if 'transaction_datetime' in df.columns:
        df = df.sort_values(['client_id', 'transaction_datetime'])
        df['time_since_last_tx'] = df.groupby('client_id')['transaction_datetime'].diff().dt.total_seconds().fillna(0)
        df['is_quick_succession'] = (df['time_since_last_tx'] < 3600).astype(int)  # <1 hour

    return df


class FraudFeatureEngineer:
    """Wrapper class for feature engineering used in training pipeline."""
    def __init__(self):
        pass
    def fit_transform(self, df):
        """Apply engineered features to the DataFrame."""
        return engineer_features(df)
