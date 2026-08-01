import numpy as np
import pandas as pd
from scipy.stats import poisson

def generate_score_matrix(home_xg: float, away_xg: float, max_goals: int = 5) -> pd.DataFrame:
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    
    matrix = np.outer(home_probs, away_probs)
    
    cols = [f"Away {i}" for i in range(max_goals + 1)]
    idxs = [f"Home {i}" for i in range(max_goals + 1)]
    df_matrix = pd.DataFrame(matrix, columns=cols, index=idxs)
    
    return df_matrix

def calculate_match_odds(score_matrix: pd.DataFrame) -> tuple:
    matrix_vals = score_matrix.values
    home_win = np.tril(matrix_vals, -1).sum()
    draw = np.trace(matrix_vals)
    away_win = np.triu(matrix_vals, 1).sum()
    
    return home_win, draw, away_win