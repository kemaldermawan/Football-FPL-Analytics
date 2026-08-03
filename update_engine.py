"""
ETL pipeline. Run this before launching the dashboard (`streamlit run app.py`)
so that all Parquet caches under data/ are fresh.

    python update_engine.py
"""
import os
import json
import requests
import pandas as pd
import soccerdata as sd

from src.config import (
    DATA_DIR, PATH_FPL_STATIC, PATH_FPL_FIXTURES, PATH_FPL_TEAMS,
    PATH_NEXT_OPPONENT, PATH_FIXTURE_RUN, PATH_SEASON_STATUS, PATH_EVENT_RAW,
    PATH_ADVANCED_STATS, PATH_TEAM_FORM, FPL_BOOTSTRAP_URL, FPL_FIXTURES_URL,
)

ROLLING_WINDOW = 5  # number of most recent matches used for form ratings


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def update_fpl_data():
    print("Initiating connection to FPL API...")
    try:
        response = requests.get(FPL_BOOTSTRAP_URL, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch FPL bootstrap data: {e}")
        return

    data = response.json()

    players_df = pd.DataFrame(data["elements"])
    teams_df = pd.DataFrame(data["teams"])
    positions_df = pd.DataFrame(data["element_types"])
    events_df = pd.DataFrame(data["events"])

    # Number of gameweeks completed so far this season — used to turn each
    # player's cumulative `minutes` into an average minutes-per-match figure
    # (a simple but honest proxy for "rolling" playing time, since the
    # public bootstrap endpoint doesn't expose a true last-5-match window).
    finished_events = int(events_df["finished"].sum()) if "finished" in events_df.columns else 0
    has_current_event = bool(events_df["is_current"].any()) if "is_current" in events_df.columns else False
    total_events = len(events_df)

    if finished_events == 0 and not has_current_event:
        season_status = "PRE_SEASON"
        status_message = (
            "No gameweek has started or finished yet — bootstrap-static is still serving "
            "carried-over data from the previous season (total_points, prices, xG, etc. have "
            "not been reset). Treat all player stats as stale until FPL opens the new season."
        )
    elif finished_events < total_events:
        season_status = "IN_PROGRESS"
        status_message = f"Season in progress: {finished_events}/{total_events} gameweeks completed."
    else:
        season_status = "SEASON_COMPLETE"
        status_message = "All gameweeks for this season have been played."

    with open(PATH_SEASON_STATUS, "w") as f:
        json.dump({
            "season_status": season_status,
            "message": status_message,
            "finished_events": finished_events,
            "total_events": total_events,
        }, f)

    print(f"Season status: {season_status} — {status_message}")

    finished_events_for_minutes = max(finished_events, 1)  # avoid division by zero pre-season

    team_mapping = dict(zip(teams_df["id"], teams_df["name"]))
    players_df["team_name"] = players_df["team"].map(team_mapping)

    pos_mapping = dict(zip(positions_df["id"], positions_df["singular_name_short"]))
    players_df["position_name"] = players_df["element_type"].map(pos_mapping)

    players_df["rolling_minutes_per_match"] = (
        pd.to_numeric(players_df["minutes"], errors="coerce").fillna(0) / finished_events_for_minutes
    ).clip(upper=90)

    clean_df = players_df[[
        "id", "first_name", "second_name", "team_name", "position_name",
        "now_cost", "total_points", "ep_next", "form", "minutes", "status",
        "chance_of_playing_next_round", "rolling_minutes_per_match",
        "goals_scored", "assists", "expected_goals", "expected_assists",
        "cost_change_event", "cost_change_start", "selected_by_percent",
        "transfers_in_event", "transfers_out_event",
    ]].copy()

    clean_df.to_parquet(PATH_FPL_STATIC, engine="pyarrow", index=False)
    print(f"FPL data correctly mapped and serialized to {PATH_FPL_STATIC}")

    # Team strength table (used for FDR) straight from FPL's own ratings
    team_strength_cols = [c for c in teams_df.columns if "strength" in c]
    teams_df[["id", "name", "short_name"] + team_strength_cols].to_parquet(
        PATH_FPL_TEAMS, engine="pyarrow", index=False
    )


def update_fixtures():
    print("Fetching fixture list for FDR / rotation-risk modeling...")
    try:
        response = requests.get(FPL_FIXTURES_URL, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch fixtures: {e}")
        return

    fixtures_df = pd.DataFrame(response.json())
    if fixtures_df.empty:
        print("No fixture data returned.")
        return

    fixtures_df.to_parquet(PATH_FPL_FIXTURES, engine="pyarrow", index=False)
    print(f"Fixtures serialized to {PATH_FPL_FIXTURES}")

    _update_next_opponent(fixtures_df)
    _update_fixture_run(fixtures_df, horizon=5)


def _update_next_opponent(fixtures_df: pd.DataFrame):
    """For every team, finds the earliest unplayed fixture and records the
    opponent + home/away split. Used to attach a real 'Opponent' column to
    the Custom xPts model instead of leaving it blank."""
    if not os.path.exists(PATH_FPL_TEAMS):
        print("Next-opponent step skipped: run update_fpl_data() first (needs team ID->name map).")
        return

    teams_df = pd.read_parquet(PATH_FPL_TEAMS, engine="pyarrow")
    team_names = dict(zip(teams_df["id"], teams_df["name"]))

    upcoming = fixtures_df[fixtures_df["finished"] == False].copy()  # noqa: E712
    if upcoming.empty:
        print("No upcoming fixtures found (season may be complete).")
        return

    upcoming = upcoming.sort_values("event", na_position="last")

    rows = []
    for team_id, team_name in team_names.items():
        team_fixtures = upcoming[(upcoming["team_h"] == team_id) | (upcoming["team_a"] == team_id)]
        if team_fixtures.empty:
            continue
        nxt = team_fixtures.iloc[0]
        is_home = nxt["team_h"] == team_id
        opponent_id = nxt["team_a"] if is_home else nxt["team_h"]
        rows.append({
            "Team": team_name,
            "Opponent": team_names.get(opponent_id, "Unknown"),
            "Is_Home": bool(is_home),
            "Gameweek": nxt.get("event"),
        })

    next_opp_df = pd.DataFrame(rows)
    next_opp_df.to_parquet(PATH_NEXT_OPPONENT, engine="pyarrow", index=False)
    print(f"Next-opponent table serialized to {PATH_NEXT_OPPONENT}")


def _update_fixture_run(fixtures_df: pd.DataFrame, horizon: int = 5):
    """Computes each team's average official FPL Fixture Difficulty
    Rating (1=easiest, 5=hardest) across their next `horizon` unplayed
    fixtures, plus a human-readable list of upcoming opponents. This is
    what lets Market Analysis flag 'good value, but a brutal run of
    fixtures coming up' instead of ranking on Points/Cost alone."""
    if not os.path.exists(PATH_FPL_TEAMS):
        print("Fixture-run step skipped: run update_fpl_data() first (needs team ID->name map).")
        return

    required_cols = {"team_h_difficulty", "team_a_difficulty", "team_h", "team_a", "finished", "event"}
    if not required_cols.issubset(fixtures_df.columns):
        print("Fixture-run step skipped: fixtures payload missing FDR/difficulty columns.")
        return

    teams_df = pd.read_parquet(PATH_FPL_TEAMS, engine="pyarrow")
    team_names = dict(zip(teams_df["id"], teams_df["name"]))
    team_short_names = dict(zip(teams_df["id"], teams_df.get("short_name", teams_df["name"])))

    upcoming = fixtures_df[fixtures_df["finished"] == False].copy()  # noqa: E712
    upcoming = upcoming.sort_values("event", na_position="last")

    rows = []
    for team_id, team_name in team_names.items():
        team_fixtures = upcoming[
            (upcoming["team_h"] == team_id) | (upcoming["team_a"] == team_id)
        ].head(horizon)

        if team_fixtures.empty:
            continue

        difficulties, opponent_labels = [], []
        for _, fx in team_fixtures.iterrows():
            is_home = fx["team_h"] == team_id
            opponent_id = fx["team_a"] if is_home else fx["team_h"]
            difficulty = fx["team_h_difficulty"] if is_home else fx["team_a_difficulty"]
            difficulties.append(difficulty)
            opp_code = team_short_names.get(opponent_id, "UNK")
            opponent_labels.append(f"{opp_code}({'H' if is_home else 'A'})")

        rows.append({
            "Team": team_name,
            "Fixture_Run": " / ".join(opponent_labels),
            "Avg_FDR": round(sum(difficulties) / len(difficulties), 2),
            "Games_Counted": len(difficulties),
        })

    fixture_run_df = pd.DataFrame(rows)
    fixture_run_df.to_parquet(PATH_FIXTURE_RUN, engine="pyarrow", index=False)
    print(f"Next-{horizon}-gameweek fixture difficulty serialized to {PATH_FIXTURE_RUN}")


def update_event_data(season: str = "2023"):
    print("Initiating connection to Understat via soccerdata...")
    try:
        understat = sd.Understat(leagues="ENG-Premier League", seasons=season)
        shots_df = understat.read_shot_events().reset_index()
    except Exception as e:
        print(f"Failed to fetch Understat event data: {e}")
        return

    shots_df.to_parquet(PATH_EVENT_RAW, engine="pyarrow", index=False)
    print(f"Spatial event data serialized to {PATH_EVENT_RAW}")

    _update_team_rolling_form(shots_df)


def _update_team_rolling_form(shots_df: pd.DataFrame):
    """Derives rolling xG-for / xG-against per team from shot event data,
    used by predictor.py's Dixon-Coles strength ratings and by the Custom
    xPts fixture-difficulty multiplier."""
    if shots_df.empty:
        return

    team_col = next((c for c in ["team", "Team"] if c in shots_df.columns), None)
    xg_col = next((c for c in shots_df.columns if c.lower() == "xg"), None)
    date_col = next((c for c in ["date", "Date", "match_date"] if c in shots_df.columns), None)

    if not team_col or not xg_col:
        print("Rolling form skipped: shot data missing team/xG columns.")
        return

    df = shots_df.copy()
    df[xg_col] = pd.to_numeric(df[xg_col], errors="coerce").fillna(0)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    xg_for = (
        df.groupby(team_col)[xg_col]
        .apply(lambda s: s.tail(ROLLING_WINDOW * 20).sum() / max(1, min(len(s), ROLLING_WINDOW * 20)) * ROLLING_WINDOW)
        .reset_index(name="Rolling_xG")
    )

    # xGA: sum of xG from the "opponent" perspective isn't directly available
    # from a one-sided shot table, so approximate using league-average xG
    # against as a placeholder until a fixture-paired dataset is wired in.
    league_avg_xg = df[xg_col].mean() if not df.empty else 1.3
    xg_for["Rolling_xGA"] = league_avg_xg * ROLLING_WINDOW

    xg_for = xg_for.rename(columns={team_col: "Team"})
    xg_for.to_parquet(PATH_TEAM_FORM, engine="pyarrow", index=False)
    print(f"Rolling team form serialized to {PATH_TEAM_FORM}")


def update_advanced_fbref_data(season: str = "2023"):
    print("Initiating full-scale data extraction from FBref (EPL)...")
    try:
        fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)

        standard_df = fbref.read_player_season_stats(stat_type="standard").reset_index()
        shooting_df = fbref.read_player_season_stats(stat_type="shooting").reset_index()
        misc_df = fbref.read_player_season_stats(stat_type="misc").reset_index()
    except Exception as e:
        print(f"Failed to fetch FBref data: {e}")
        return

    advanced_stats = standard_df.merge(shooting_df, on=["league", "season", "team", "player"], how="left")
    advanced_stats = advanced_stats.merge(misc_df, on=["league", "season", "team", "player"], how="left")

    advanced_stats.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else str(col) for col in advanced_stats.columns
    ]

    advanced_stats.to_parquet(PATH_ADVANCED_STATS, engine="pyarrow", index=False)
    print(f"Professional grade metrics serialized to {PATH_ADVANCED_STATS}")


if __name__ == "__main__":
    _ensure_data_dir()
    update_fpl_data()
    update_fixtures()
    update_event_data()
    update_advanced_fbref_data()
    print("\nETL pipeline complete. Launch the dashboard with: streamlit run app.py")