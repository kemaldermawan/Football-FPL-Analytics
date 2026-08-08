import pandas as pd

def get_optimal_captaincy(player_data, target_metric="Proj_xPts", top_n=5):
    """
    Mengevaluasi rekomendasi kapten dengan hierarki pemecah seri absolut:
    Metrik Utama -> Posisi (FWD/MID > DEF/GKP) -> Form (Tertinggi) -> Kepemilikan (Terendah)
    """
    if target_metric not in player_data.columns:
        target_metric = "ep_next"

    available_players = player_data[player_data["Availability"] == "Available"].copy()
    
    # Injeksi hierarki nilai taktis posisi
    pos_weight = {"FWD": 4, "MID": 3, "DEF": 2, "GKP": 1}
    available_players["Pos_Weight"] = available_players["Position"].map(pos_weight)
    
    # Pengurutan multi-variabel
    ranked_players = available_players.sort_values(
        by=[target_metric, "Pos_Weight", "Form", "Ownership_Pct"], 
        ascending=[False, False, False, True]
    )
    
    return ranked_players.head(top_n)

def find_differentials(player_data, max_ownership=10.0, min_minutes=450, target_metric="Proj_xPts"):
    if target_metric not in player_data.columns:
        target_metric = "ep_next"

    differentials = player_data[
	    (player_data["Availability"] == "Available") &
	    (player_data["Minutes"] >= min_minutes) &
	    (player_data["Ownership_Pct"] <= max_ownership) &
	    (player_data["Position"].isin(["MID", "FWD"])) # Filter posisi eksplisit
	].copy()

    top_differentials = differentials.sort_values(by=target_metric, ascending=False)
    
    return top_differentials

def detect_fixture_anomalies(fixtures_df, team_id_map, current_gw=1, horizon=10):
    """
    Memetakan kepadatan jadwal untuk mendeteksi Blank Gameweeks (BGW) 
    dan Double Gameweeks (DGW) secara algoritmik.
    """
    upcoming = fixtures_df[(fixtures_df["event"] >= current_gw) & (fixtures_df["event"] < current_gw + horizon)].copy()
    
    if upcoming.empty:
        return None

    home_counts = upcoming.groupby(["event", "team_h"]).size().reset_index(name="matches")
    home_counts.rename(columns={"team_h": "team_id"}, inplace=True)
    
    away_counts = upcoming.groupby(["event", "team_a"]).size().reset_index(name="matches")
    away_counts.rename(columns={"team_a": "team_id"}, inplace=True)
    
    combined = pd.concat([home_counts, away_counts])
    density_matrix = combined.groupby(["event", "team_id"])["matches"].sum().reset_index()
    
    density_matrix["Team"] = density_matrix["team_id"].map(team_id_map)
    
    pivot_density = density_matrix.pivot(index="Team", columns="event", values="matches").fillna(0).astype(int)
    
    return pivot_density