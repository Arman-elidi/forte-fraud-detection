
import pandas as pd
import sys
import os
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from features.feature_engineering import FraudFeatureEngineer
from inference import FraudPredictor

def process_and_check():
    input_path = 'data/merged_data.csv'
    output_path = 'data/merged_data_processed.csv'
    
    print(f"Loading {input_path}...")
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Pre-processing types
    print("Converting types...")
    try:
        df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        # Ensure client_id is string
        df['client_id'] = df['client_id'].astype(str)
    except Exception as e:
        print(f"Error converting types: {e}")
        return

    # Feature Engineering
    print("Applying Feature Engineering...")
    try:
        fe = FraudFeatureEngineer()
        df_features = fe.fit_transform(df)
        print(f"Features generated: {df_features.shape[1]} columns.")
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save processed data
    print(f"Saving to {output_path}...")
    df_features.to_csv(output_path, index=False)
    
    # Run Prediction Check
    print("\nRunning Prediction on Processed Data...")
    try:
        predictor = FraudPredictor(model_path='models/fraud_detection_model.pkl')
        predictions = predictor.predict_batch(df_features)
        
        total = len(predictions)
        fraud = predictions['is_fraud'].sum()
        blocked = (predictions['recommendation'] == 'БЛОКИРОВАТЬ').sum()
        check = (predictions['recommendation'] == 'ПРОВЕРИТЬ').sum()
        
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        print(f"Total transactions: {total}")
        print(f"Fraud detected: {fraud} ({fraud/total*100:.2f}%)")
        print(f"Recommended to BLOCK: {blocked}")
        print(f"Recommended to CHECK: {check}")
        print("="*30)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_and_check()
