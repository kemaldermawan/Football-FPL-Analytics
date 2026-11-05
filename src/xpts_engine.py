"""
Per-gameweek expected-points (xPts) engine.

Builds a player x gameweek matrix of expected FPL points from first
principles instead of one flat number multiplied by the horizon:

  * Fixture layer  - for every upcoming fixture, each team's expected goals
    for/against from its own Dixon-Coles style ratings (attack, defence,
    venue, rest, European rotation, key absences).
  * Minutes layer  - expected minutes from minutes per team match played,
    scaled by the availability flag.
  * Player layer   - per-90 xG / xA shrunk toward the position average so a
    cameo does not look like a superstar rate.
  * Points layer   - appearance, goals, assists, clean sheet and
    goals-conceded points, summed over every fixture in the gameweek:
    a blank gameweek is 0 and a double gameweek is two fixtures.

Not modelled yet (needs extra columns from the FPL API): bonus points,
goalkeeper saves, cards. Those are the next step.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.custom_xpts import iter_fixture_ratings

HOME_GOAL_FACTOR = 1.12        # home side's goals x1.12, away side's / 1.12 (ratio ~1.25)
DEFAULT_LEAGUE_GOALS = 1.40    # goals per team per match when too few results exist
MIN_MATCHES_FOR_LEAGUE_AVG = 10

GOAL_POINTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
GC_PENALTY_WEIGHT = {"GKP": 1.0, "DEF": 1.0, "MID": 0.0, "FWD": 0.0}  # -1 per 2 goals conceded
ASSIST_POINTS = 3.0

STARTER_MINUTES = 75.0   # average minutes at which a player is a near-certain 60+ minute starter
CAMEO_MINUTES = 25.0     # average minutes at which a player almost always gets on the pitch


def league_goals_per_team_match(fixtures_df: pd.DataFrame, default: float = DEFAULT_LEAGUE_GOALS) -> float:
    """Average goals scored by one team in one match, from finished fixtures."""
    needed = {"team_h_score", "team_a_score", "finished"}
    if fixtures_df is None or fixtures_df.empty or not needed.issubset(fixtures_df.columns):
        return default
    played = fixtures_df[fixtures_df["finished"].fillna(False).astype(bool)]
    h = pd.to_numeric(played["team_h_score"], errors="coerce")
    a = pd.to_numeric(played["team_a_score"], errors="coerce")
    ok = h.notna() & a.notna()
    if ok.sum() < MIN_MATCHES_FOR_LEAGUE_AVG:
        return default
    avg = float((h[ok] + a[ok]).sum() / (2 * ok.sum()))
    return avg if avg > 0 else default


def expected_goals_conceded_penalty(xga, max_goals: int = 12) -> np.ndarray:
    """E[floor(GC / 2)] for GC ~ Poisson(xga): the expected points a
    defender/goalkeeper loses from goals conceded (-1 per two goals)."""
    mu = np.atleast_1d(np.asarray(xga, dtype=float))
    k = np.arange(0, max_goals + 1)
    pmf = poisson.pmf(k[None, :], mu[:, None])
    return (np.floor(k / 2.0)[None, :] * pmf).sum(axis=1)


def build_fixture_expectations(fixtures_df: pd.DataFrame, strengths_df: pd.DataFrame, team_map: dict,
                               european_teams: list = None, key_absences: dict = None,
                               league_goals: float = None,
                               home_goal_factor: float = HOME_GOAL_FACTOR) -> pd.DataFrame:
    """One row per (team, upcoming fixture):

      GW, Team, Opponent, Venue, xGF, xGA, P_CS, E_GC_Pen, Att_Mult

    Att_Mult is how much the team's usual scoring output is scaled in this
    fixture (opponent weakness x venue x its own rotation/absence effects),
    relative to a neutral game against an average opponent.
    """
    cols = ["GW", "Team", "Opponent", "Venue", "xGF", "xGA", "P_CS", "E_GC_Pen", "Att_Mult"]
    lg = league_goals if league_goals is not None else league_goals_per_team_match(fixtures_df)
    eps = 0.01
    rows = []

    for r in iter_fixture_ratings(fixtures_df, strengths_df, team_map, european_teams, key_absences):
        hgf = home_goal_factor
        xgf_home = lg * r["home_lambda"] * r["away_nu"] * hgf
        xgf_away = lg * r["away_lambda"] * r["home_nu"] / hgf
        rows.append({
            "GW": r["gw"], "Team": r["home"], "Opponent": r["away"], "Venue": "H",
            "xGF": xgf_home, "xGA": xgf_away,
            "Att_Mult": (r["home_lambda"] / max(r["home_lambda_base"], eps)) * r["away_nu"] * hgf,
        })
        rows.append({
            "GW": r["gw"], "Team": r["away"], "Opponent": r["home"], "Venue": "A",
            "xGF": xgf_away, "xGA": xgf_home,
            "Att_Mult": (r["away_lambda"] / max(r["away_lambda_base"], eps)) * r["home_nu"] / hgf,
        })

    if not rows:
        return pd.DataFrame(columns=cols)

    table = pd.DataFrame(rows)
    table["P_CS"] = np.exp(-table["xGA"])
    table["E_GC_Pen"] = expected_goals_conceded_penalty(table["xGA"].to_numpy())
    return table[cols]


def _shrunk_per90(players_df: pd.DataFrame, total_col: str, exposure_90s: pd.Series,
                  prior_90s: float) -> pd.Series:
    """Per-90 rate shrunk toward the position average. `prior_90s` is how many
    full matches of position-average evidence are mixed in, so 30 minutes of
    data cannot masquerade as a season-long rate."""
    totals = pd.to_numeric(players_df[total_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = pd.Series(0.0, index=players_df.index)
    for _, idx in players_df.groupby("Position").groups.items():
        tot_exposure = exposure_90s.loc[idx].sum()
        prior_rate = totals.loc[idx].sum() / tot_exposure if tot_exposure > 0 else 0.0
        out.loc[idx] = (totals.loc[idx] + prior_rate * prior_90s) / (exposure_90s.loc[idx] + prior_90s)
    return out


def compute_xpts_matrix(players_df: pd.DataFrame, fixture_table: pd.DataFrame, team_matches: dict,
                        n_gws: int = 8, prior_90s: float = 3.0):
    """Player x gameweek expected points.

    players_df needs: Team, Position (GKP/DEF/MID/FWD), Minutes, xG, xA and
    optionally Chance_of_Playing (0-100) and status ('u'/'n' = unavailable).
    team_matches maps team name -> matches played so far.

    Returns (matrix, gw_numbers): `matrix` has columns xPts_GW1..xPts_GW{n_gws}
    (GW1 = the next gameweek that has fixtures) and the same index as
    players_df; `gw_numbers` lists the real gameweek behind each column.
    """
    out_cols = [f"xPts_GW{k}" for k in range(1, n_gws + 1)]
    result = pd.DataFrame(0.0, index=players_df.index, columns=out_cols)
    if players_df.empty or fixture_table is None or fixture_table.empty:
        return result, []

    gws = sorted(fixture_table["GW"].unique())[:n_gws]
    agg = fixture_table.groupby(["Team", "GW"]).agg(
        n_fix=("GW", "size"), att=("Att_Mult", "sum"),
        cs=("P_CS", "sum"), gc=("E_GC_Pen", "sum"))

    team = players_df["Team"]
    pos = players_df["Position"]
    minutes = pd.to_numeric(players_df["Minutes"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # --- Minutes layer ------------------------------------------------------
    tm = pd.Series(team_matches, dtype=float)
    fallback_matches = float(tm.median()) if len(tm) else 1.0
    matches = team.map(tm).fillna(fallback_matches).clip(lower=1.0)
    m = (minutes / matches).clip(0.0, 90.0)          # average minutes per team match

    if "Chance_of_Playing" in players_df.columns:
        chance = pd.to_numeric(players_df["Chance_of_Playing"], errors="coerce")
    else:
        chance = pd.Series(np.nan, index=players_df.index)
    avail = (chance / 100.0).clip(0.0, 1.0).fillna(1.0)
    if "status" in players_df.columns:
        avail = avail.mask(players_df["status"].isin(["u", "n"]), 0.0)

    p60 = avail * (m / STARTER_MINUTES).clip(0.0, 1.0)     # P(plays 60+ minutes)
    p_any = avail * (m / CAMEO_MINUTES).clip(0.0, 1.0)     # P(plays at all)
    e_min90 = avail * m / 90.0                             # expected share of a full match

    # --- Player layer -------------------------------------------------------
    exposure = minutes / 90.0
    xg90 = _shrunk_per90(players_df, "xG", exposure, prior_90s)
    xa90 = _shrunk_per90(players_df, "xA", exposure, prior_90s)

    goal_w = pos.map(GOAL_POINTS).fillna(0.0)
    cs_w = pos.map(CS_POINTS).fillna(0.0)
    gc_w = pos.map(GC_PENALTY_WEIGHT).fillna(0.0)
    attack_value = e_min90 * (xg90 * goal_w + xa90 * ASSIST_POINTS)  # points per unit of Att_Mult

    # --- Points layer, one gameweek at a time -------------------------------
    for k, gw in enumerate(gws, start=1):
        sub = agg.xs(gw, level="GW")                 # indexed by Team
        n_fix = team.map(sub["n_fix"]).fillna(0.0)
        att = team.map(sub["att"]).fillna(0.0)
        cs = team.map(sub["cs"]).fillna(0.0)
        gc = team.map(sub["gc"]).fillna(0.0)

        result[f"xPts_GW{k}"] = (
            n_fix * (p_any + p60)                    # 1 pt for playing + 1 more for 60+ minutes
            + att * attack_value                     # goals + assists, scaled by fixture
            + p60 * (cs_w * cs - gc_w * gc)          # clean sheet (60+ min) minus goals conceded
        )

    return result, [int(g) for g in gws]