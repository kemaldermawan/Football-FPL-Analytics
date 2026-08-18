import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

def train_and_predict_points(df):
    """
    Melatih model Gradient Boosting yang dioptimalkan untuk memori rendah
    guna memprediksi poin pemain berbasis metrik historis.
    """
    ml_df = df.copy()
    
    # 1. Seleksi Fitur (Variabel Independen)
    features = ["Cost", "Minutes", "Form", "xG", "xA", "Ownership_Pct"]
    target = "Total Points"
    
    # Injeksi nilai aman untuk mencegah galat NaN
    for col in features + [target]:
        if col not in ml_df.columns:
            ml_df[col] = 0.0
        ml_df[col] = pd.to_numeric(ml_df[col], errors="coerce").fillna(0.0)
        
    # Memfilter pemain yang memiliki menit bermain memadai untuk data latih (mencegah bias)
    train_data = ml_df[ml_df["Minutes"] > 200].copy()
    
    if len(train_data) < 50:
        ml_df["ML_Proj_Pts"] = 0.0
        return ml_df
        
    X_train = train_data[features]
    y_train = train_data[target]
    
    # 2. Standarisasi Skala Fitur (Mempercepat konvergensi algoritma)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. Inisialisasi HistGradientBoosting (Sangat efisien untuk RAM terbatas)
    model = HistGradientBoostingRegressor(
        max_iter=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    
    # 4. Pelatihan Model (*Fitting*)
    model.fit(X_train_scaled, y_train)
    
    # 5. Eksekusi Prediksi pada Seluruh Basis Data
    X_all_scaled = scaler.transform(ml_df[features])
    ml_df["ML_Proj_Pts"] = model.predict(X_all_scaled)
    
    # Normalisasi hasil prediksi (Tidak ada poin minus ekstrem)
    ml_df["ML_Proj_Pts"] = np.maximum(0.0, ml_df["ML_Proj_Pts"].round(2))
    
    return ml_df