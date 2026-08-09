"""
Local data access layer. Reads pre-serialized Parquet files produced by
update_engine.py. Uses Streamlit's cache when available so repeated tab
switches inside the dashboard don't re-hit disk unnecessarily.
"""
import os
import json
import requests
import pandas as pd
import streamlit as st

from src.config import PATH_FPL_STATIC, PATH_TEAM_FORM, PATH_FPL_TEAMS, PATH_NEXT_OPPONENT, PATH_FIXTURE_RUN, PATH_SEASON_STATUS, PATH_FPL_FIXTURES

try:
    import streamlit as st
    _cache = st.cache_data(ttl=600)
except Exception:  # pragma: no cover - allows module use outside Streamlit
    def _cache(func):
        return func

@st.cache_data(ttl=3600)
def get_fpl_players():
    """
    Mengekstrak matriks pemain dari titik akhir statis API FPL
    dan memetakan identitas klub serta posisi.
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 1. Ekstraksi pangkalan data mentah
        df = pd.DataFrame(data["elements"])
        teams_df = pd.DataFrame(data["teams"])
        positions_df = pd.DataFrame(data["element_types"])
        
        # 2. Konstruksi kamus pemetaan (mapping dictionaries)
        team_map = dict(zip(teams_df["id"], teams_df["name"]))
        pos_map = dict(zip(positions_df["id"], positions_df["singular_name_short"]))
        
        # 3. Transformasi tipe data integer ke dalam teks (string)
        df["team_name"] = df["team"].map(team_map)
        df["position_name"] = df["element_type"].map(pos_map)
        
        # 4. Seleksi kolom target secara komprehensif
        target_columns = [
            "id", "first_name", "second_name", "team_name", "position_name",
            "now_cost", "total_points", "ep_next", "minutes", "status",
            "chance_of_playing_next_round", "cost_change_event", "cost_change_start",
            "selected_by_percent", "transfers_in_event", "transfers_out_event",
            "form", "expected_goals", "expected_assists", "goals_scored", "assists",
            "yellow_cards", "news" 
        ]
        
        existing_columns = [col for col in target_columns if col in df.columns]
        df = df[existing_columns].copy()
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Kegagalan telemetri pada peladen FPL. Detail galat: {e}")
        return pd.DataFrame()

@_cache
def get_team_rolling_form() -> pd.DataFrame:
    """Load rolling xG/xGA per team (last-5-match window), used by the
    Dixon-Coles predictor and the Custom xPts model."""
    if os.path.exists(PATH_TEAM_FORM):
        return pd.read_parquet(PATH_TEAM_FORM, engine="pyarrow")
    return pd.DataFrame()

@_cache
def get_fpl_team_strengths() -> pd.DataFrame:
    """Load FPL's own official team strength ratings (attack/defense,
    home/away). This is the basis for the Custom xPts Fixture Difficulty
    Rating multiplier."""
    if os.path.exists(PATH_FPL_TEAMS):
        return pd.read_parquet(PATH_FPL_TEAMS, engine="pyarrow")
    return pd.DataFrame()

@_cache
def get_next_opponent() -> pd.DataFrame:
    """Load each team's next unplayed fixture (opponent + home/away),
    derived from the fixtures endpoint by update_engine.py."""
    if os.path.exists(PATH_NEXT_OPPONENT):
        return pd.read_parquet(PATH_NEXT_OPPONENT, engine="pyarrow")
    return pd.DataFrame()

@_cache
def get_fixture_run() -> pd.DataFrame:
    """Load each team's average official FDR (1=easiest, 5=hardest) over
    their next 5 unplayed fixtures, plus a readable list of opponents."""
    if os.path.exists(PATH_FIXTURE_RUN):
        return pd.read_parquet(PATH_FIXTURE_RUN, engine="pyarrow")
    return pd.DataFrame()

@_cache
def get_season_status() -> dict:
    """Load the season-status flag written by update_engine.py
    (PRE_SEASON / IN_PROGRESS / SEASON_COMPLETE), used to warn the user
    when displayed stats are still carried over from the previous season."""
    if os.path.exists(PATH_SEASON_STATUS):
        with open(PATH_SEASON_STATUS) as f:
            return json.load(f)
    return {"season_status": "UNKNOWN", "message": "Run update_engine.py to determine season status."}

@_cache
def get_all_fixtures() -> pd.DataFrame:
    """Load the complete FPL fixture list for the entire season."""
    if os.path.exists(PATH_FPL_FIXTURES):
        return pd.read_parquet(PATH_FPL_FIXTURES, engine="pyarrow")
    return pd.DataFrame()