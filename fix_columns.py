
import pandas as pd

filepath = 'data/merged_data.csv'
print(f"Processing {filepath}...")

try:
    df = pd.read_csv(filepath, low_memory=False)
    
    # Mapping based on inspection
    mapping = {
        'cst_dim_id': 'client_id',
        'transdatetime': 'transaction_datetime',
        'transdate': 'transaction_date',
        'target': 'is_fraud',
        'docno': 'transaction_id',
        'direction': 'destination_id' # Assuming direction/destination mapping
    }
    
    # Rename columns
    df.rename(columns=mapping, inplace=True)
    
    # Save back
    df.to_csv(filepath, index=False)
    print("✓ Columns renamed successfully.")
    print(f"New columns: {list(df.columns)}")
    
except Exception as e:
    print(f"Error: {e}")
