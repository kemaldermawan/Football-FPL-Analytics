import pandas as pd
import numpy as np

def calculate_price_momentum(player_data):
    """
    Kalkulasi Indeks Momentum Harga (Im) untuk memprediksi fluktuasi nilai aset FPL.
    """
    df = player_data.copy()
    
    epsilon = 0.1
    
    t_in = pd.to_numeric(df["transfers_in_event"], errors="coerce").fillna(0)
    t_out = pd.to_numeric(df["transfers_out_event"], errors="coerce").fillna(0)
    net_transfers = t_in - t_out
    
    cost = df["Cost"]
    ownership = df["Ownership_Pct"]
    
    df["Market_Momentum"] = (net_transfers / (cost * (ownership + epsilon))) * 100
    
    df["Market_Momentum"] = df["Market_Momentum"].round(2)
    
    def classify_trend(momentum):
        if momentum > 500:
            return "Imminent Rise"
        elif momentum > 150:
            return "Rising Trend"
        elif momentum < -500:
            return "Imminent Fall"
        elif momentum < -150:
            return "Falling Trend"
        else:
            return "Stable"
            
    df["Price_Forecast"] = df["Market_Momentum"].apply(classify_trend)
    
    return df