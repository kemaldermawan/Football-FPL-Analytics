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
