import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generate_score_matrix(home_xg: float, away_xg: float, max_goals: int = 5) -> pd.DataFrame:
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    
    matrix = np.outer(home_probs, away_probs)
    
    cols = [f"Away {i}" for i in range(max_goals + 1)]
    idxs = [f"Home {i}" for i in range(max_goals + 1)]
    df_matrix = pd.DataFrame(matrix, columns=cols, index=idxs)
    
    return df_matrix

def calculate_match_odds(score_matrix: pd.DataFrame) -> tuple:
    matrix_vals = score_matrix.values
    home_win = np.tril(matrix_vals, -1).sum()
    draw = np.trace(matrix_vals)
    away_win = np.triu(matrix_vals, 1).sum()
    
    return home_win, draw, away_win

def find_similar_players(df: pd.DataFrame, target_name: str, n_clusters: int = 8) -> pd.DataFrame:
    features = ['Cost', 'Total Points', 'Value (Pts/Cost)']
    ml_data = df.dropna(subset=features).copy()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(ml_data[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    ml_data['Cluster'] = kmeans.fit_predict(scaled_features)
    
    if target_name not in ml_data['Last Name'].values:
        return pd.DataFrame()
        
    target_cluster = ml_data[ml_data['Last Name'] == target_name]['Cluster'].values[0]
    similar_players = ml_data[ml_data['Cluster'] == target_cluster]
    
    return similar_players[similar_players['Last Name'] != target_name].nlargest(5, 'Total Points')

def run_monte_carlo(home_xg: float, away_xg: float, iterations: int = 10000) -> tuple:
    np.random.seed(42)
    
    home_sim = np.random.poisson(home_xg, iterations)
    away_sim = np.random.poisson(away_xg, iterations)
    
    home_wins = np.sum(home_sim > away_sim)
    draws = np.sum(home_sim == away_sim)
    away_wins = np.sum(home_sim < away_sim)
    
    return home_wins / iterations, draws / iterations, away_wins / iterations