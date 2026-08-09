import pandas as pd
import numpy as np

def calculate_rotation_risk(player_data, european_teams, density_map=None, current_gw=1):
    """
    Kalkulasi Indeks Risiko Rotasi (Ri) mengintegrasikan volume beban fisik,
    pajak Eropa, pengganda kepadatan jadwal, dan ancaman skorsing disiplin.
    """
    df = player_data.copy()
    
    # Ekstraksi metrik kartu (jatuh kembali ke 0 jika kosong)
    df["yellow_cards"] = pd.to_numeric(df.get("yellow_cards", 0), errors="coerce").fillna(0)
    
    def assess_risk(row):
        risk = 0.0
        
        # 1. Parameter Beban Fisik (Akumulasi Volume Menit)
        if row["Minutes"] > 2200:
            risk += 25.0
        elif row["Minutes"] > 1200:
            risk += 15.0
            
        # 2. Pengganda Kepadatan Jadwal Dinamis
        density_multiplier = 1.0
        if density_map is not None and current_gw in density_map.columns:
            team_matches = density_map.at[row["Team"], current_gw] if row["Team"] in density_map.index else 1
            if team_matches > 1:
                density_multiplier = 1.5 # Pengganda eksponensial untuk jadwal padat (DGW)
        
        # 3. Parameter Pajak Eropa
        if row["Team"] in european_teams:
            risk += (35.0 * density_multiplier)
        elif density_multiplier > 1.0:
            risk += 20.0 # Penalti jadwal padat untuk tim domestik murni
            
        # 4. Parameter Kerentanan Posisi
        if row["Position"] in ["MID", "FWD"]:
            risk += 10.0
            
        # 5. Parameter Disiplin (Ancaman Skorsing Proksimal)
        yc = row["yellow_cards"]
        if yc == 4 or yc == 9:
            risk += 45.0
            
        # 6. Parameter Imunitas (Pemain Premium)
        if row["Cost"] >= 9.0:
            risk -= 30.0
        elif row["Cost"] >= 7.5:
            risk -= 15.0
            
        return max(0.0, min(100.0, risk))
        
    df["Rotation_Risk_Pct"] = df.apply(assess_risk, axis=1)
    
    def classify_risk(row):
        if row["yellow_cards"] in [4, 9]: return "Suspension Risk"
        if row["Rotation_Risk_Pct"] >= 60: return "High Risk"
        if row["Rotation_Risk_Pct"] >= 30: return "Moderate"
        return "Nailed/Safe"
        
    df["Risk_Category"] = df.apply(classify_risk, axis=1)
    
    return df