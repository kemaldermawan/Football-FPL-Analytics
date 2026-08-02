"""
K-Means statistical scouting: groups players by multi-dimensional per-90
statistical profile, and surfaces the nearest neighbours (by Euclidean
distance in the standardized feature space) to a target player — the
"find a cheaper alternative to X" workflow.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

EXCLUDE_COLS = {"Player", "Team", "Position", "league", "season", "nation", "age", "Cluster"}


def _select_features(df: pd.DataFrame) -> list:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_COLS]


def run_kmeans_clustering(df: pd.DataFrame, min_90s: float = 3.0,
                           n_clusters: int = 12) -> tuple:
    """Returns (clustered_df, scaled_data, feature_list, scaler). All four
    are returned together because downstream similarity search needs the
    exact same scaled feature space the clusters were fit on."""
    if "Matches_90s" not in df.columns or df.empty:
        return pd.DataFrame(), None, [], None

    df_clean = df.copy()
    df_clean["Matches_90s"] = pd.to_numeric(df_clean["Matches_90s"], errors="coerce").fillna(0)

    df_filtered = df_clean[df_clean["Matches_90s"] >= min_90s].copy().reset_index(drop=True)
    if df_filtered.empty:
        return pd.DataFrame(), None, [], None

    features = _select_features(df_filtered)
    if not features:
        return pd.DataFrame(), None, [], None

    df_filtered[features] = df_filtered[features].fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[features])

    n_clusters = max(2, min(n_clusters, len(df_filtered) - 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_filtered["Cluster"] = kmeans.fit_predict(scaled_data)

    return df_filtered, scaled_data, features, scaler


def find_similar_players(player_name: str, df: pd.DataFrame, scaled_data: np.ndarray,
                          top_n: int = 5) -> pd.DataFrame:
    """Nearest neighbours by Euclidean distance in the standardized
    statistical feature space — the mathematical basis for 'this player
    plays like that player, for a fraction of the price'."""
    if df is None or df.empty or player_name not in df["Player"].values:
        return pd.DataFrame()

    player_idx = df.index.get_loc(df[df["Player"] == player_name].index[0])
    player_vector = scaled_data[player_idx].reshape(1, -1)

    distances = euclidean_distances(player_vector, scaled_data)[0]

    df_result = df.copy()
    df_result["Similarity_Distance"] = distances

    similar_players = (
        df_result[df_result["Player"] != player_name]
        .sort_values("Similarity_Distance")
        .head(top_n)
    )

    display_cols = ["Player", "Team", "Position", "Matches_90s", "Cluster", "Similarity_Distance"]
    display_cols = [c for c in display_cols if c in similar_players.columns]
    return similar_players[display_cols]


def project_clusters_2d(scaled_data: np.ndarray) -> np.ndarray:
    """PCA projection of the cluster feature space down to 2 dimensions,
    purely for visualization (the clustering itself is unaffected)."""
    if scaled_data is None or len(scaled_data) < 2:
        return np.zeros((0, 2))
    n_components = min(2, scaled_data.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(scaled_data)
    if n_components == 1:
        projected = np.column_stack([projected, np.zeros(len(projected))])
    return projected
