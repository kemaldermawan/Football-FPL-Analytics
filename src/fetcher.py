import pandas as pd
import os

def get_fpl_players():
    file_path = os.path.join('data', 'fpl_static.parquet')
    
    if os.path.exists(file_path):
        return pd.read_parquet(file_path, engine='pyarrow')
    else:
        print("Local data not found. Please run update_engine.py first.")
        return pd.DataFrame()