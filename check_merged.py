
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from inference import FraudPredictor

def check_file(filepath):
    print(f"Checking file: {filepath}")
    
    # Load data
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Initialize predictor
    try:
        predictor = FraudPredictor(model_path='models/fraud_detection_model.pkl')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run prediction
    print("Running prediction...")
    try:
        predictions = predictor.predict_batch(df)
        
        # Stats
        total = len(predictions)
        fraud = predictions['is_fraud'].sum()
        blocked = (predictions['recommendation'] == 'БЛОКИРОВАТЬ').sum()
        check = (predictions['recommendation'] == 'ПРОВЕРИТЬ').sum()
        
        print("\n" + "="*30)
        print("RESULTS")
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
    check_file('data/merged_data.csv')
