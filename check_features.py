
import pandas as pd
import sys
import os
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from models.fraud_model import FraudDetectionModel

def check_features():
    print("Loading model...")
    try:
        model_data = joblib.load('models/fraud_detection_model.pkl')
        # Handle both dict and object formats if necessary, but based on fraud_model.py it's a dict
        if isinstance(model_data, dict):
            feature_cols = model_data['feature_cols']
        else:
            feature_cols = model_data.feature_cols
            
        print(f"Model expects {len(feature_cols)} features.")
        print("Expected features:", feature_cols)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nLoading CSV...")
    try:
        df = pd.read_csv('data/merged_data.csv', low_memory=False, nrows=5)
        csv_cols = list(df.columns)
        print(f"CSV has {len(csv_cols)} columns.")
        print("CSV columns:", csv_cols)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("\n" + "="*30)
    print("MISSING FEATURES")
    print("="*30)
    missing = [col for col in feature_cols if col not in csv_cols]
    if missing:
        for col in missing:
            print(f"- {col}")
    else:
        print("All features are present!")

if __name__ == "__main__":
    check_features()
