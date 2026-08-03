"""
Local data access layer. Reads pre-serialized Parquet files produced by
update_engine.py. Uses Streamlit's cache when available so repeated tab
switches inside the dashboard don't re-hit disk unnecessarily.
"""
import os
import json
import pandas as pd

from src.config import PATH_FPL_STATIC, PATH_TEAM_FORM, PATH_FPL_TEAMS, PATH_NEXT_OPPONENT, PATH_FIXTURE_RUN, PATH_SEASON_STATUS

try:
    import streamlit as st
    _cache = st.cache_data(ttl=600)
except Exception:  # pragma: no cover - allows module use outside Streamlit
    def _cache(func):
        return func


@_cache
def get_fpl_players() -> pd.DataFrame:
    """Load the static FPL player valuation/points table."""
    if os.path.exists(PATH_FPL_STATIC):
        return pd.read_parquet(PATH_FPL_STATIC, engine="pyarrow")
    print("Local data not found. Please run update_engine.py first.")
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