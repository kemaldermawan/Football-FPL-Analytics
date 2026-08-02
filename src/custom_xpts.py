"""
Custom xPts Projection Model — replaces the FPL API's own `ep_next`
estimate with an in-house projection built from four independent factors:

  1. Tactical Fixture Difficulty Rating (FDR): how hard the next opponent
     is, derived from the opponent's Defense Vulnerability rating.
  2. Rotation risk: a discount applied when a club has a Champions/Europa
     League fixture within 3 days of the gameweek, since rotation becomes
     statistically more likely.
  3. Conversion ratio: actual goals scored per unit of xG, smoothing out
     players running hot/cold relative to their underlying numbers.
  4. Projected minutes (xMins): recent minutes trend, discounting players
     who are nursing knocks or out of favour.

Each factor is a multiplier in [0, ~1.3] applied to a baseline points rate,
so the model stays interpretable — you can see exactly why a projection
moved.
"""
import pandas as pd
import numpy as np

FPL_POINTS_PER_GOAL = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
FPL_POINTS_PER_ASSIST = 3
FPL_APPEARANCE_POINTS = 2  # for playing 60+ minutes


def fixture_difficulty_multiplier(opponent_defense_vulnerability: float) -> float:
    """Higher opponent Defense Vulnerability (they concede more than
    average) -> easier fixture -> multiplier above 1.0."""
    if pd.isna(opponent_defense_vulnerability):
        return 1.0
    return float(np.clip(opponent_defense_vulnerability, 0.5, 1.8))


def rotation_risk_multiplier(has_european_fixture_midweek: bool,
                              squad_depth_flag: str = "normal") -> float:
    """Discounts projected minutes when a club juggles a midweek European
    fixture. `squad_depth_flag` can be 'thin', 'normal', or 'deep' to
    reflect how much a manager is likely to rotate."""
    if not has_european_fixture_midweek:
        return 1.0
    depth_discount = {"thin": 0.90, "normal": 0.80, "deep": 0.65}
    return depth_discount.get(squad_depth_flag, 0.80)


def conversion_ratio(actual_goals: float, expected_goals: float, min_xg: float = 0.5) -> float:
    """Goals scored per unit xG, capped to avoid wild extrapolation from
    tiny sample sizes early in a season."""
    if expected_goals is None or expected_goals < min_xg:
        return 1.0
    ratio = actual_goals / expected_goals
    return float(np.clip(ratio, 0.5, 1.8))


def projected_minutes_factor(rolling_minutes: float, full_match_minutes: float = 90.0) -> float:
    return float(np.clip(rolling_minutes / full_match_minutes, 0.0, 1.0))

def build_opponent_defense_map(team_strengths_df: pd.DataFrame) -> dict:
    """Converts FPL's own official team strength ratings into the
    {team_name: defense_vulnerability} map expected by
    `fixture_difficulty_multiplier`. Uses the average of home/away
    defensive strength; a team with below-average defensive strength gets
    a vulnerability > 1.0 (easier fixture for their opponents)."""
    if team_strengths_df.empty:
        return {}

    df = team_strengths_df.copy()
    home_col = next((c for c in df.columns if c == "strength_defence_home"), None)
    away_col = next((c for c in df.columns if c == "strength_defence_away"), None)

    if not home_col or not away_col:
        return {}

    df["avg_defence_strength"] = df[[home_col, away_col]].mean(axis=1)
    league_avg = df["avg_defence_strength"].mean()
    if not league_avg:
        return {}

    df["Defense_Vulnerability"] = league_avg / df["avg_defence_strength"]
    return dict(zip(df["name"], df["Defense_Vulnerability"]))

def compute_custom_xpts(players_df: pd.DataFrame,
                         opponent_defense_map: dict,
                         european_fixture_teams: set,
                         position_col: str = "Position",
                         team_col: str = "Team",
                         opponent_col: str = "Opponent",
                         xg_col: str = "xG",
                         goals_col: str = "Goals",
                         xa_col: str = "xA",
                         assists_col: str = "Assists",
                         rolling_minutes_col: str = "Rolling_Minutes") -> pd.DataFrame:
    """Produces a `Custom_xPts` column per player using the four-factor
    model described in the module docstring."""
    df = players_df.copy()

    for c in (xg_col, goals_col, xa_col, assists_col, rolling_minutes_col):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0.0

    df["FDR_Multiplier"] = df[opponent_col].map(
        lambda opp: fixture_difficulty_multiplier(opponent_defense_map.get(opp, 1.0))
    ) if opponent_col in df.columns else 1.0

    df["Rotation_Multiplier"] = df[team_col].map(
        lambda t: rotation_risk_multiplier(t in european_fixture_teams)
    ) if team_col in df.columns else 1.0

    df["Conversion_Ratio"] = df.apply(
        lambda r: conversion_ratio(r[goals_col], r[xg_col]), axis=1
    )
    df["xMins_Factor"] = df[rolling_minutes_col].apply(projected_minutes_factor)

    df["Goal_Points_Weight"] = df[position_col].map(FPL_POINTS_PER_GOAL).fillna(4)

    # Baseline expected attacking-return points from underlying xG/xA,
    # then scaled by fixture difficulty, rotation risk, and minutes.
    base_attacking_pts = (
        df[xg_col] * df["Conversion_Ratio"] * df["Goal_Points_Weight"]
        + df[xa_col] * FPL_POINTS_PER_ASSIST
    )

    df["Custom_xPts"] = (
        (base_attacking_pts + FPL_APPEARANCE_POINTS)
        * df["FDR_Multiplier"]
        * df["Rotation_Multiplier"]
        * df["xMins_Factor"]
    ).round(2)

    return df
