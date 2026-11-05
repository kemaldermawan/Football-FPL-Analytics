"""
Match outcome prediction: dynamic Poisson strength ratings (Dixon & Coles,
1997 style) plus a low-score dependency correction, and a Monte Carlo
simulation layer for variance-aware probabilities.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

RHO_DEFAULT = -0.13  # empirical low-score correlation dampener (Dixon-Coles)


def compute_team_strengths(team_form: pd.DataFrame,
                            team_col: str = "Team",
                            xg_col: str = "Rolling_xG",
                            xga_col: str = "Rolling_xGA") -> pd.DataFrame:
    """Convert rolling xG/xGA into relative Attack Strength / Defense
    Vulnerability ratings, normalized against the league average — the
    core Dixon-Coles parameterization."""
    df = team_form.copy()
    league_avg_xg = df[xg_col].mean()
    league_avg_xga = df[xga_col].mean()

    df["Attack_Strength"] = df[xg_col] / league_avg_xg
    df["Defense_Vulnerability"] = df[xga_col] / league_avg_xga
    return df[[team_col, "Attack_Strength", "Defense_Vulnerability"]]

FPL_ATTACK_RATING_COLS = ("strength_attack_home", "strength_attack_away")
FPL_DEFENCE_RATING_COLS = ("strength_defence_home", "strength_defence_away")


def _valid_rating(teams: pd.DataFrame, cols: list) -> pd.Series:
    """Row-average of the given FPL rating columns, using only values that
    are numeric, finite and > 0. Missing/zero ratings (e.g. before FPL has
    published new-season values) are filled with the league median, and if
    nothing is usable every team gets a flat 1.0 prior - never NaN/inf."""
    block = teams[cols].apply(pd.to_numeric, errors="coerce")
    block = block.where(np.isfinite(block) & (block > 0))
    rating = block.mean(axis=1)
    if rating.notna().sum() == 0:
        return pd.Series(1.0, index=teams.index)
    return rating.fillna(rating.median())


def compute_team_strengths_from_results(fixtures_df: pd.DataFrame,
                                         teams_df: pd.DataFrame,
                                         prior_matches: float = 8.0,
                                         team_col: str = "Team") -> pd.DataFrame:
    """Attack Strength / Defense Vulnerability (league average = 1.0) built
    from the goals actually scored and conceded in finished FPL fixtures.

    Every team's observed rate is shrunk toward FPL's own official team
    ratings, as if the prior were worth `prior_matches` games of evidence:
    early in the season (few matches) the ratings dominate, and as results
    accumulate the observed goals take over. Teams are keyed by FPL team id
    internally, so names always match the rest of the app (no Understat /
    FPL name mismatch, and promoted clubs get a sensible prior).

    Returns columns: Team, Attack_Strength, Defense_Vulnerability,
    Matches_Played. Higher Defense_Vulnerability = leakier defence.
    """
    cols = [team_col, "Attack_Strength", "Defense_Vulnerability", "Matches_Played"]
    if teams_df is None or teams_df.empty or "id" not in teams_df.columns:
        return pd.DataFrame(columns=cols)

    teams = teams_df.drop_duplicates("id").set_index("id")

    # --- Prior: FPL's official ratings (higher rating = better) -----------
    atk_cols = [c for c in FPL_ATTACK_RATING_COLS if c in teams.columns]
    def_cols = [c for c in FPL_DEFENCE_RATING_COLS if c in teams.columns]
    if atk_cols and def_cols:
        prior_atk_raw = _valid_rating(teams, atk_cols)
        prior_def_raw = _valid_rating(teams, def_cols)
    elif "strength" in teams.columns:
        prior_atk_raw = _valid_rating(teams, ["strength"])
        prior_def_raw = prior_atk_raw.copy()
    else:
        prior_atk_raw = pd.Series(1.0, index=teams.index)
        prior_def_raw = prior_atk_raw.copy()

    prior_atk = prior_atk_raw / prior_atk_raw.mean()
    prior_def_vuln = prior_def_raw.mean() / prior_def_raw  # strong defence -> low vulnerability

    # --- Evidence: goals from finished fixtures ----------------------------
    zeros = pd.Series(0.0, index=teams.index)
    n, gf, ga = zeros.copy(), zeros.copy(), zeros.copy()

    needed = {"team_h", "team_a", "team_h_score", "team_a_score", "finished"}
    if fixtures_df is not None and not fixtures_df.empty and needed.issubset(fixtures_df.columns):
        played = fixtures_df[fixtures_df["finished"].fillna(False).astype(bool)].copy()
        played["team_h_score"] = pd.to_numeric(played["team_h_score"], errors="coerce")
        played["team_a_score"] = pd.to_numeric(played["team_a_score"], errors="coerce")
        played = played.dropna(subset=["team_h_score", "team_a_score"])
        played = played[played["team_h"].isin(teams.index) & played["team_a"].isin(teams.index)]

        if not played.empty:
            home = played.groupby("team_h").agg(
                n=("team_h", "size"), gf=("team_h_score", "sum"), ga=("team_a_score", "sum"))
            away = played.groupby("team_a").agg(
                n=("team_a", "size"), gf=("team_a_score", "sum"), ga=("team_h_score", "sum"))
            total = home.add(away, fill_value=0.0).reindex(teams.index).fillna(0.0)
            n, gf, ga = total["n"], total["gf"], total["ga"]

    league_goals_per_team_match = gf.sum() / n.sum() if n.sum() > 0 else 0.0

    if league_goals_per_team_match > 0:
        safe_n = n.where(n > 0)  # NaN where a team has no matches yet
        obs_atk = ((gf / safe_n) / league_goals_per_team_match).fillna(0.0)
        obs_def = ((ga / safe_n) / league_goals_per_team_match).fillna(0.0)
        w = n / (n + max(float(prior_matches), 1e-9))  # weight on evidence
        atk = w * obs_atk + (1.0 - w) * prior_atk
        dfn = w * obs_def + (1.0 - w) * prior_def_vuln
    else:
        atk, dfn = prior_atk, prior_def_vuln

    # Floor avoids zero/negative ratings downstream (log, division),
    # then re-center so the league average is exactly 1.0.
    atk = atk.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.2)
    dfn = dfn.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.2)
    atk, dfn = atk / atk.mean(), dfn / dfn.mean()

    return pd.DataFrame({
        team_col: teams["name"].values,
        "Attack_Strength": atk.values,
        "Defense_Vulnerability": dfn.values,
        "Matches_Played": n.astype(int).values,
    }).reset_index(drop=True)


def expected_goals(home_team: str, away_team: str, strengths: pd.DataFrame,
                    league_avg_home_goals: float = 1.55,
                    league_avg_away_goals: float = 1.20,
                    home_advantage: float = 1.10,
                    team_col: str = "Team") -> tuple:
    """Derive fixture-specific home/away expected goals from team ratings."""
    home = strengths[strengths[team_col] == home_team]
    away = strengths[strengths[team_col] == away_team]

    if home.empty or away.empty:
        # Fall back to league-average expectation if a team is missing
        return league_avg_home_goals, league_avg_away_goals

    home_xg = (
        league_avg_home_goals
        * home["Attack_Strength"].values[0]
        * away["Defense_Vulnerability"].values[0]
        * home_advantage
    )
    away_xg = (
        league_avg_away_goals
        * away["Attack_Strength"].values[0]
        * home["Defense_Vulnerability"].values[0]
    )
    return float(home_xg), float(away_xg)


def _dixon_coles_adjustment(home_goals: int, away_goals: int, home_xg: float,
                             away_xg: float, rho: float) -> float:
    """Low-score correlation correction (tau function from Dixon & Coles,
    1997), applied to scorelines of 0-0, 1-0, 0-1, and 1-1."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (home_xg * away_xg * rho)
    if home_goals == 0 and away_goals == 1:
        return 1 + (home_xg * rho)
    if home_goals == 1 and away_goals == 0:
        return 1 + (away_xg * rho)
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def generate_score_matrix(home_xg: float, away_xg: float, max_goals: int = 6,
                           rho: float = RHO_DEFAULT) -> pd.DataFrame:
    """Full scoreline probability matrix with the Dixon-Coles correlation
    adjustment applied to low-scoring outcomes."""
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]

    matrix = np.outer(home_probs, away_probs)

    for h in range(min(2, max_goals) + 1):
        for a in range(min(2, max_goals) + 1):
            matrix[h, a] *= _dixon_coles_adjustment(h, a, home_xg, away_xg, rho)

    matrix = matrix / matrix.sum()  # renormalize after adjustment

    cols = [f"Away {i}" for i in range(max_goals + 1)]
    idxs = [f"Home {i}" for i in range(max_goals + 1)]
    return pd.DataFrame(matrix, columns=cols, index=idxs)


def calculate_match_odds(score_matrix: pd.DataFrame) -> tuple:
    matrix_vals = score_matrix.values
    home_win = np.tril(matrix_vals, -1).sum()
    draw = np.trace(matrix_vals)
    away_win = np.triu(matrix_vals, 1).sum()
    return float(home_win), float(draw), float(away_win)


def calculate_clean_sheet_probability(score_matrix: pd.DataFrame) -> tuple:
    """P(away scores 0) = home clean sheet; P(home scores 0) = away clean sheet."""
    home_clean_sheet = score_matrix.iloc[:, 0].sum()
    away_clean_sheet = score_matrix.iloc[0, :].sum()
    return float(home_clean_sheet), float(away_clean_sheet)


def run_monte_carlo(home_xg: float, away_xg: float, iterations: int = 10000,
                     rho: float = RHO_DEFAULT, seed: int = 42) -> tuple:
    """Stochastic simulation layer — draws correlated-ish scorelines by
    sampling independent Poisson variates and re-weighting a small fraction
    of 0-0/1-1 draws to reflect the Dixon-Coles low-score dependency,
    rather than assuming pure independence."""
    rng = np.random.default_rng(seed)

    home_sim = rng.poisson(home_xg, iterations)
    away_sim = rng.poisson(away_xg, iterations)

    # Nudge a slice of simulations toward draws when rho indicates negative
    # correlation at low scores (mirrors the DC tau correction in expectation)
    if rho != 0:
        low_score_mask = (home_sim <= 1) & (away_sim <= 1)
        flip_fraction = abs(rho) * 0.5
        flip_n = int(low_score_mask.sum() * flip_fraction)
        if flip_n > 0:
            idxs = rng.choice(np.where(low_score_mask)[0], size=flip_n, replace=False)
            away_sim[idxs] = home_sim[idxs]

    home_wins = np.sum(home_sim > away_sim)
    draws = np.sum(home_sim == away_sim)
    away_wins = np.sum(home_sim < away_sim)

    return home_wins / iterations, draws / iterations, away_wins / iterations