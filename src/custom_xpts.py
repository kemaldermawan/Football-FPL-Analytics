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


def build_custom_fdr_matrix(fixtures_df: pd.DataFrame, strengths_df: pd.DataFrame, team_map: dict, european_teams: list = None, key_absences: dict = None) -> tuple:
    """
    Builds a continuous 1.0 - 6.0 Custom FDR matrix leveraging Dixon-Coles parameters,
    European fatigue penalty, Rest-day differentials, and Key player absences.
    Returns (pivot_labels, pivot_values).
    """
    if european_teams is None: european_teams = []
    if key_absences is None: key_absences = {}

    upcoming = fixtures_df[fixtures_df["finished"] == False].copy()
    if upcoming.empty or strengths_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    upcoming["Home_Team"] = upcoming["team_h"].map(team_map)
    upcoming["Away_Team"] = upcoming["team_a"].map(team_map)
    upcoming = upcoming.dropna(subset=["Home_Team", "Away_Team"])
    
    # 1. Konversi zona waktu (UTC) untuk komputasi Diferensial Waktu Pemulihan
    upcoming["kickoff_time"] = pd.to_datetime(upcoming["kickoff_time"], errors="coerce", utc=True)
    upcoming = upcoming.sort_values("kickoff_time")

    str_dict = strengths_df.set_index("Team").to_dict(orient="index")
    default_str = {"Attack_Strength": 1.0, "Defense_Vulnerability": 1.0}

    gamma = 1.25
    home_adv = -0.25
    away_adv = 0.25

    rows = []
    last_match_tracker = {}

    for _, fx in upcoming.iterrows():
        gw = fx.get("event")
        if pd.isna(gw): continue

        home_team = fx["Home_Team"]
        away_team = fx["Away_Team"]
        match_time = fx["kickoff_time"]

        # Kalkulasi Diferensial Waktu Istirahat (Rest Days) per Klub
        home_rest = 7.0
        away_rest = 7.0
        if home_team in last_match_tracker and pd.notna(match_time):
            home_rest = (match_time - last_match_tracker[home_team]).total_seconds() / 86400
        if away_team in last_match_tracker and pd.notna(match_time):
            away_rest = (match_time - last_match_tracker[away_team]).total_seconds() / 86400

        # Simpan waktu bermain untuk kalkulasi pertandingan selanjutnya
        if pd.notna(match_time):
            last_match_tracker[home_team] = match_time
            last_match_tracker[away_team] = match_time

        home_str = str_dict.get(home_team, default_str).copy()
        away_str = str_dict.get(away_team, default_str).copy()

        # 2. Injeksi European Fatigue Penalty (Reduksi kekuatan statis 5%)
        if home_team in european_teams:
            home_str["Attack_Strength"] *= 0.95
            home_str["Defense_Vulnerability"] *= 1.05
        if away_team in european_teams:
            away_str["Attack_Strength"] *= 0.95
            away_str["Defense_Vulnerability"] *= 1.05

        # 3. Injeksi Penalti Absennya Pemain Kunci (Kalkulasi dinamis dari FPL endpoint)
        home_penalty = key_absences.get(home_team, 0.0)
        home_str["Attack_Strength"] *= (1.0 - home_penalty)
        home_str["Defense_Vulnerability"] *= (1.0 + home_penalty)

        away_penalty = key_absences.get(away_team, 0.0)
        away_str["Attack_Strength"] *= (1.0 - away_penalty)
        away_str["Defense_Vulnerability"] *= (1.0 + away_penalty)

        # 4. Modifikasi Dinamis Rest Differential (< 4 hari istirahat & selisih >= 2 hari)
        home_rest_mod = 1.10 if (home_rest < 4.0 and (away_rest - home_rest) >= 2.0) else 1.0
        away_rest_mod = 1.10 if (away_rest < 4.0 and (home_rest - away_rest) >= 2.0) else 1.0

        home_lambda = home_str["Attack_Strength"] / home_rest_mod
        home_nu = home_str["Defense_Vulnerability"] * home_rest_mod
        
        away_lambda = away_str["Attack_Strength"] / away_rest_mod
        away_nu = away_str["Defense_Vulnerability"] * away_rest_mod

        # Komputasi probabilitas Dixon-Coles untuk Tim Kandang
        home_ratio = (away_lambda * home_nu) / max((home_lambda * away_nu), 0.01)
        home_fdr = np.clip(3.5 + gamma * np.log(home_ratio) + home_adv, 1.0, 6.99)

        # Komputasi probabilitas Dixon-Coles untuk Tim Tandang
        away_ratio = (home_lambda * away_nu) / max((away_lambda * home_nu), 0.01)
        away_fdr = np.clip(3.5 + gamma * np.log(away_ratio) + away_adv, 1.0, 6.99)

        home_code = home_team[:3].upper()
        away_code = away_team[:3].upper()

        rows.append({"Team": home_team, "GW": f"GW{int(gw)}", "Label": f"{away_code} (H)", "FDR": home_fdr})
        rows.append({"Team": away_team, "GW": f"GW{int(gw)}", "Label": f"{home_code} (A)", "FDR": away_fdr})

    df_melt = pd.DataFrame(rows)
    if df_melt.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Normalisasi agregat untuk menangani kalender DGW (Double Gameweek) dan BGW (Blank)
    df_labels = df_melt.groupby(["Team", "GW"])["Label"].apply(lambda x: "\n".join(x)).reset_index()
    df_values = df_melt.groupby(["Team", "GW"])["FDR"].mean().reset_index()

    pivot_labels = df_labels.pivot(index="Team", columns="GW", values="Label").fillna("-")
    pivot_values = df_values.pivot(index="Team", columns="GW", values="FDR").fillna(3.5)

    gw_cols = sorted(pivot_labels.columns, key=lambda x: int(x.replace("GW", "")))
    return pivot_labels[gw_cols], pivot_values[gw_cols]