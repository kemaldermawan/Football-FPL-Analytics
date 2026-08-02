import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def run_kmeans_clustering(df: pd.DataFrame, min_90s: float = 3.0, n_clusters: int = 12) -> tuple:
    df_clean = df.copy()
    
    if 'Matches_90s' not in df_clean.columns:
        return pd.DataFrame(), None, [], None
        
    df_clean['Matches_90s'] = pd.to_numeric(df_clean['Matches_90s'], errors='coerce').fillna(0)
    
    # Eliminasi anomali statistik dengan menyaring pemain yang menit bermainnya terlalu rendah
    df_filtered = df_clean[df_clean['Matches_90s'] >= min_90s].copy().reset_index(drop=True)
    
    if df_filtered.empty:
        return pd.DataFrame(), None, [], None
        
    # Isolasi kolom metrik numerik murni untuk proses pembelajaran mesin
    exclude_cols = ['Player', 'Team', 'Position', 'league', 'season', 'nation', 'age']
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col not in exclude_cols]
    
    df_filtered[features] = df_filtered[features].fillna(0)
    
    # Normalisasi data (Standardization) agar metrik berskala besar tidak mendominasi model
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[features])
    
    # Eksekusi K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_filtered['Cluster'] = kmeans.fit_predict(scaled_data)
    
    return df_filtered, scaled_data, features, scaler

def find_similar_players(player_name: str, df: pd.DataFrame, scaled_data: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    if player_name not in df['Player'].values:
        return pd.DataFrame()
        
    player_idx = df.index.get_loc(df[df['Player'] == player_name].index[0])
    player_vector = scaled_data[player_idx].reshape(1, -1)
    
    # Kalkulasi Jarak Euclidean antara vektor pemain target dengan seluruh pemain di liga
    distances = euclidean_distances(player_vector, scaled_data)[0]
    
    df_result = df.copy()
    df_result['Similarity_Distance'] = distances
    
    # Mengurutkan pemain dengan jarak matematis terdekat (selain pemain itu sendiri)
    similar_players = df_result[df_result['Player'] != player_name].sort_values('Similarity_Distance').head(top_n)
    
    display_cols = ['Player', 'Team', 'Position', 'Matches_90s', 'Cluster', 'Similarity_Distance']
    return similar_players[display_cols]