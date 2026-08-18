"""
Expected Threat (xT) — grid-based possession value model.

This is a from-scratch, simplified implementation of the Karun Singh-style
xT concept: the pitch is divided into a 16x12 grid, each zone is assigned a
"threat" value (probability-weighted proxy for how dangerous it is to have
the ball there), and a player's xT contribution from an action is the
difference in zone value between where the move started and where it ended.

Rather than importing a third-party published grid verbatim, the zone
values here are generated analytically from two intuitive components:
  1. Proximity to the opponent's goal (distance decay).
  2. Centrality (actions through the middle of the pitch are more
     dangerous than actions hugging the touchline).
This keeps the model self-contained and easy to recalibrate once real
shot-conversion data is available (see `calibrate_grid_from_shots`).
"""
import numpy as np
import pandas as pd

GRID_COLS = 16  # divisions along the length of the pitch (attacking direction)
GRID_ROWS = 12  # divisions across the width of the pitch


def build_xt_grid(cols: int = GRID_COLS, rows: int = GRID_ROWS) -> np.ndarray:
    """Generate a synthetic-but-principled xT value grid, normalized 0-1.
    Pitch coordinates are assumed to run 0-1 in both x (own goal -> opponent
    goal) and y (touchline -> touchline)."""
    x = (np.arange(cols) + 0.5) / cols
    y = (np.arange(rows) + 0.5) / rows

    xx, yy = np.meshgrid(x, y)

    # Distance-to-goal component: goal sits at (x=1, y=0.5)
    goal_dist = np.sqrt((1 - xx) ** 2 + (yy - 0.5) ** 2)
    proximity = np.exp(-3.2 * goal_dist)

    # Central-lane bonus: zones near y=0.5 get an additional multiplier
    centrality = 1.0 - 0.6 * np.abs(yy - 0.5) * 2

    grid = proximity * centrality
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    return grid


def calibrate_grid_from_shots(shots_df: pd.DataFrame, x_col: str, y_col: str,
                               cols: int = GRID_COLS, rows: int = GRID_ROWS) -> np.ndarray:
    """Optional: reweight the synthetic grid using actual shot density from
    the Understat event data, so zones that generate more real shots in this
    dataset carry proportionally more threat."""
    grid = build_xt_grid(cols, rows)
    if shots_df.empty or x_col not in shots_df.columns or y_col not in shots_df.columns:
        return grid

    xs = pd.to_numeric(shots_df[x_col], errors="coerce").clip(0, 1).fillna(0)
    ys = pd.to_numeric(shots_df[y_col], errors="coerce").clip(0, 1).fillna(0)

    density, _, _ = np.histogram2d(
        xs, ys, bins=[cols, rows], range=[[0, 1], [0, 1]]
    )
    density = density.T  # align to (rows, cols) like the grid
    if density.sum() > 0:
        density_norm = density / density.max()
        grid = 0.7 * grid + 0.3 * density_norm
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)
    return grid


def _zone_index(value: float, n_bins: int) -> int:
    idx = int(np.clip(value, 0, 0.999) * n_bins)
    return min(idx, n_bins - 1)


def compute_action_xt(start_x: float, start_y: float, end_x: float, end_y: float,
                       grid: np.ndarray) -> float:
    """xT added by a single pass/carry, given normalized 0-1 coordinates."""
    rows, cols = grid.shape
    sx, sy = _zone_index(start_x, cols), _zone_index(start_y, rows)
    ex, ey = _zone_index(end_x, cols), _zone_index(end_y, rows)
    return float(grid[ey, ex] - grid[sy, sx])


def compute_player_xt(actions_df: pd.DataFrame, grid: np.ndarray,
                       start_x_col: str, start_y_col: str,
                       end_x_col: str, end_y_col: str,
                       player_col: str = "Player") -> pd.DataFrame:
    """Aggregate xT added per player across a dataframe of progressive
    actions (passes/carries) with start and end coordinates."""
    df = actions_df.copy()
    for c in (start_x_col, start_y_col, end_x_col, end_y_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[start_x_col, start_y_col, end_x_col, end_y_col, player_col])

    df["xT_added"] = df.apply(
        lambda r: compute_action_xt(
            r[start_x_col], r[start_y_col], r[end_x_col], r[end_y_col], grid
        ),
        axis=1,
    )
    # Only count positive (progressive) contributions toward the threat index
    df["xT_added_positive"] = df["xT_added"].clip(lower=0)

    summary = (
        df.groupby(player_col)
        .agg(
            Actions=("xT_added", "count"),
            Total_xT=("xT_added_positive", "sum"),
            Avg_xT_per_Action=("xT_added_positive", "mean"),
        )
        .reset_index()
        .sort_values("Total_xT", ascending=False)
    )
    return summary
    
def simulate_player_xt_proxy(df):
    """
    Menghasilkan skor proxy xT untuk mendeteksi Deep-Lying Playmakers
    karena absensi telemetri kordinat umpan dari API publik FPL.
    """
    xt_df = df.copy()
    
    # Injeksi tipe data aman
    target_cols = ["expected_assists", "expected_goals", "minutes", "Cost"]
    for col in target_cols:
        if col not in xt_df.columns:
            xt_df[col] = 0.0
        xt_df[col] = pd.to_numeric(xt_df[col], errors="coerce").fillna(0.0)
        
    valid_players = xt_df[xt_df["minutes"] > 200].copy()
    
    def compute_xt_index(row):
        pos = row.get("Position", "MID")
        xa_per_90 = (row["expected_assists"] / row["minutes"]) * 90 if row["minutes"] > 0 else 0
        xg_per_90 = (row["expected_goals"] / row["minutes"]) * 90 if row["minutes"] > 0 else 0
        
        # Gelandang menerima bobot spasial lebih tinggi karena mereka menginisiasi progresi sentral
        pos_multiplier = 1.4 if pos == "MID" else (1.1 if pos == "DEF" else 0.8)
        
        # Formula heuristik untuk estimasi ancaman spasial
        xt_index = ((xa_per_90 * 2.5) + (xg_per_90 * 0.5)) * pos_multiplier
        
        # Penyesuaian nilai investasi (Value Metric)
        xt_value = xt_index / (row["Cost"] - 4.0 + 0.1) if row["Cost"] > 4.0 else xt_index
        
        return round(xt_index, 3), round(xt_value, 3)

    valid_players[["xT_Index_per90", "xT_Value_Ratio"]] = valid_players.apply(
        lambda row: pd.Series(compute_xt_index(row)), axis=1
    )
    
    return valid_players.sort_values("xT_Index_per90", ascending=False)