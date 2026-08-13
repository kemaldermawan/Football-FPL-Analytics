import pandas as pd
import numpy as np

def calculate_squad_variance(squad_df):
    """
    Mengalkulasi risiko korelasi (variansi) berdasarkan konsentrasi pemain pada klub yang sama.
    """
    eval_df = squad_df.copy()
    
    if "Team" not in eval_df.columns:
        return {"Total_Variance_Risk": 0.0, "Risk_Status": "Unknown"}
        
    # Menghitung konsentrasi aset per klub
    team_counts = eval_df["Team"].value_counts()
    
    # Penalti variansi meningkat secara eksponensial untuk kepemilikan ganda
    variance_penalty = 0.0
    for team, count in team_counts.items():
        if count > 1:
            variance_penalty += (count ** 2) * 1.5 
            
    risk_status = "Optimal (Diversified)"
    if variance_penalty > 15.0:
        risk_status = "High Risk (Over-concentrated)"
    elif variance_penalty > 7.0:
        risk_status = "Moderate Risk"
        
    return {
        "Total_Variance_Risk": round(variance_penalty, 2),
        "Risk_Status": risk_status,
        "Team_Concentration": team_counts.to_dict()
    }