"""
Team-level tactical analytics: projected lineups, playmaker indexing,
the attack/possession quadrant matrix, and the Defensive Flank
Vulnerability matrix (which side of an opponent's defense leaks the most
expected goals against).
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PATH_ADVANCED_STATS, COLOR_PITCH, COLOR_TEXT, COLOR_ACCENT, COLOR_LINE


def load_advanced_metrics() -> pd.DataFrame:
    if not os.path.exists(PATH_ADVANCED_STATS):
        return pd.DataFrame()

    df = pd.read_parquet(PATH_ADVANCED_STATS, engine="pyarrow")
    df.columns = [str(col).rstrip("_") for col in df.columns]

    min_col = next((c for c in df.columns if "90s" in c.lower() or "minutes" in c.lower()), None)

    rename_map = {"player": "Player", "team": "Team", "pos": "Position"}
    if min_col:
        rename_map[min_col] = "Matches_90s"

    df = df.rename(columns=rename_map)
    if "Matches_90s" not in df.columns:
        df["Matches_90s"] = 0.0

    return df


def generate_predicted_lineup(df: pd.DataFrame, target_team: str) -> pd.DataFrame:
    team_data = df[df["Team"] == target_team].copy()
    if team_data.empty:
        return pd.DataFrame()

    team_data["Matches_90s"] = pd.to_numeric(team_data["Matches_90s"], errors="coerce").fillna(0)
    projected_squad = team_data.nlargest(11, "Matches_90s")

    tactical_keywords = ["goals", "xg", "passes_completed", "progressive", "tackles"]
    available_cols = ["Player", "Position", "Matches_90s"]

    for col in projected_squad.columns:
        if any(k in col.lower() for k in tactical_keywords) and col not in available_cols:
            available_cols.append(col)

    return projected_squad[available_cols[:8]]


def plot_tactical_quadrant(adv_df: pd.DataFrame) -> plt.Figure:
    xg_col = next((c for c in adv_df.columns if "xg" in c.lower()), None)
    pass_col = next((c for c in adv_df.columns if "passes_completed" in c.lower()), None)

    if not xg_col or not pass_col:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, "Tactical data requirements not met", ha="center", va="center", color=COLOR_TEXT)
        fig.patch.set_facecolor(COLOR_PITCH)
        ax.set_facecolor(COLOR_PITCH)
        return fig

    adv_df = adv_df.copy()
    adv_df[xg_col] = pd.to_numeric(adv_df[xg_col], errors="coerce").fillna(0)
    adv_df[pass_col] = pd.to_numeric(adv_df[pass_col], errors="coerce").fillna(0)

    team_stats = adv_df.groupby("Team").agg(
        Attack_xG=(xg_col, "sum"),
        Possession_Control=(pass_col, "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(COLOR_PITCH)
    ax.set_facecolor(COLOR_PITCH)
    
    sns.scatterplot(data=team_stats, x="Possession_Control", y="Attack_xG",
                     color=COLOR_ACCENT, s=150, edgecolor=COLOR_TEXT, ax=ax)

    for i in range(team_stats.shape[0]):
        ax.text(team_stats["Possession_Control"].iloc[i], team_stats["Attack_xG"].iloc[i] + 0.5,
                team_stats["Team"].iloc[i], color=COLOR_TEXT, fontsize=9, ha="center", weight="bold")

    ax.axhline(team_stats["Attack_xG"].mean(), color=COLOR_LINE, linestyle="--", alpha=0.5)
    ax.axvline(team_stats["Possession_Control"].mean(), color=COLOR_LINE, linestyle="--", alpha=0.5)

    ax.set_title("Tactical Matrix — Attacking Threat (xG) vs Possession Control",
                  color=COLOR_TEXT, fontsize=15, pad=20)
    ax.set_xlabel("Average Completed Passes per Match", color=COLOR_TEXT, fontsize=11)
    ax.set_ylabel("Total Expected Goals (xG)", color=COLOR_TEXT, fontsize=11)
    ax.tick_params(colors=COLOR_TEXT)
    ax.grid(False)

    return fig


def identify_key_playmakers(df: pd.DataFrame, target_team: str) -> pd.DataFrame:
    team_data = df[df["Team"] == target_team].copy()
    if team_data.empty:
        return pd.DataFrame()

    prog_col = next((c for c in team_data.columns if "progressive_passes" in c.lower()), None)
    xa_col = next((c for c in team_data.columns if "xa" in c.lower()), None)

    if not prog_col or not xa_col:
        return pd.DataFrame(columns=["Player", "Error Missing Playmaker Columns"])

    team_data[prog_col] = pd.to_numeric(team_data[prog_col], errors="coerce").fillna(0)
    team_data[xa_col] = pd.to_numeric(team_data[xa_col], errors="coerce").fillna(0)

    team_data["Playmaker_Index"] = (team_data[prog_col] * 0.6) + (team_data[xa_col] * 0.4)
    top_creators = team_data.nlargest(5, "Playmaker_Index")

    return top_creators[["Player", "Position", prog_col, xa_col, "Playmaker_Index"]]


def compute_defensive_flank_vulnerability(shots_df: pd.DataFrame, team_col: str = "team",
                                           y_col: str = "Y") -> pd.DataFrame:
    """Splits conceded shots into Left / Central / Right thirds of the
    pitch (by the Y coordinate of the shot, from the defending team's
    perspective) to flag which flank a team leaks the most xG against.
    Requires event-level shot data with normalized Y in [0, 1]."""
    if shots_df.empty or team_col not in shots_df.columns or y_col not in shots_df.columns:
        return pd.DataFrame()

    df = shots_df.copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[y_col])

    def zone(y):
        if y < 1 / 3:
            return "Right Flank (conceded)"
        if y > 2 / 3:
            return "Left Flank (conceded)"
        return "Central (conceded)"

    df["Flank_Zone"] = df[y_col].apply(zone)

    xg_col = next((c for c in df.columns if c.lower() in ("xg", "xg_shot")), None)

    if xg_col:
        df[xg_col] = pd.to_numeric(df[xg_col], errors="coerce").fillna(0)
        summary = df.groupby([team_col, "Flank_Zone"]).agg(
            Shots_Conceded=(xg_col, "count"),
            xGA=(xg_col, "sum"),
        ).reset_index()
    else:
        summary = df.groupby([team_col, "Flank_Zone"]).size().reset_index(name="Shots_Conceded")
        summary["xGA"] = np.nan

    pivot = summary.pivot_table(index=team_col, columns="Flank_Zone",
                                 values="xGA" if xg_col else "Shots_Conceded",
                                 aggfunc="sum", fill_value=0)

    if not pivot.empty:
        pivot["Most_Vulnerable_Flank"] = pivot.idxmax(axis=1)

    return pivot.reset_index()
