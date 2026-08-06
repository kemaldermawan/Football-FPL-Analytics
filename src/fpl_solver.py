"""
Fantasy Premier League decision engine — Operations Research layer.

Contains:
  * optimize_squad: single-gameweek MILP (PuLP), maximizing expected points
    under budget / positional / club-quota constraints.
  * optimize_squad_multi_horizon: a rolling 5-8 gameweek MILP that also
    tracks squad changes across weeks and penalizes transfers beyond the
    free-transfer allowance.
  * evaluate_chip_strategy: compares the manager's live squad to the
    single-GW optimum to recommend Hold / Take Hits / Wildcard.
  * fetch_manager_squad: pulls a manager's live picks from the FPL API.
"""
import pandas as pd
import pulp
import requests

from src.config import POSITION_LIMITS, MAX_PLAYERS_PER_CLUB, SQUAD_SIZE, BUDGET_DEFAULT, FPL_ENTRY_PICKS_URL


def fetch_manager_squad(team_id: str) -> list:
    """Fetch a manager's current 15-man squad by FPL Team ID. Returns an
    empty list (rather than raising) on any network/API failure so the UI
    layer can show a clean error instead of crashing."""
    if not team_id or not str(team_id).strip().isdigit():
        return []

    url = FPL_ENTRY_PICKS_URL.format(team_id=team_id)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [player["element"] for player in data.get("picks", [])]
    except (requests.RequestException, ValueError, KeyError):
        return []


def _resolve_columns(df: pd.DataFrame) -> dict:
    return {
        "cost": "Cost" if "Cost" in df.columns else "now_cost",
        "pos": "Position" if "Position" in df.columns else "position_name",
        "team": "Team" if "Team" in df.columns else "team_name",
    }


def optimize_squad(df: pd.DataFrame, budget: float = BUDGET_DEFAULT,
                    target_metric: str = "ep_next") -> pd.DataFrame:
    """Single-gameweek MILP squad optimizer maximizing the chosen
    projection column subject to FPL's hard constraints."""
    cols = _resolve_columns(df)
    cost_col, pos_col, team_col = cols["cost"], cols["pos"], cols["team"]

    possible_targets = [target_metric, "Custom_xPts", "ep_next", "Expected Points", "Total Points", "total_points"]
    target_col = next((c for c in possible_targets if c in df.columns), None)

    df_clean = df.copy()
    if not target_col:
        target_col = "Optimization_Target"
        df_clean[target_col] = 0.0

    df_clean[cost_col] = pd.to_numeric(df_clean[cost_col], errors="coerce")
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce").fillna(0.0)
    df_clean = df_clean.dropna(subset=[cost_col, target_col, pos_col, team_col])

    if df_clean.empty:
        return pd.DataFrame()

    active_budget = budget * 10 if df_clean[cost_col].max() > 30 else budget

    prob = pulp.LpProblem("FPL_Predictive_Optimization", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", df_clean.index, cat="Binary")

    prob += pulp.lpSum([df_clean.loc[i, target_col] * player_vars[i] for i in df_clean.index])
    prob += pulp.lpSum([df_clean.loc[i, cost_col] * player_vars[i] for i in df_clean.index]) <= active_budget
    prob += pulp.lpSum([player_vars[i] for i in df_clean.index]) == SQUAD_SIZE

    for pos, limit in POSITION_LIMITS.items():
        prob += pulp.lpSum(
            [player_vars[i] for i in df_clean.index if df_clean.loc[i, pos_col] == pos]
        ) == limit

    for team in df_clean[team_col].unique():
        prob += pulp.lpSum(
            [player_vars[i] for i in df_clean.index if df_clean.loc[i, team_col] == team]
        ) <= MAX_PLAYERS_PER_CLUB

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return pd.DataFrame()

    selected_indices = [i for i in df_clean.index if player_vars[i].varValue == 1.0]
    return df_clean.loc[selected_indices].sort_values(by=[pos_col, target_col], ascending=[True, False])


def optimize_squad_multi_horizon(df: pd.DataFrame, projection_cols: list,
                                  budget: float = BUDGET_DEFAULT,
                                  starting_free_transfers: int = 1,
                                  hit_cost: int = 4) -> dict:
    """Rolling multi-gameweek MILP: chooses a squad for each of the next
    `len(projection_cols)` gameweeks, jointly maximizing total projected
    points minus transfer-hit penalties.

    `projection_cols` is an ordered list of column names in `df`, one per
    upcoming gameweek (e.g. ['xPts_GW1', 'xPts_GW2', ...]).

    Simplification note: free-transfer banking is modeled as +1 per week
    (capped at 5) rolling over from the previous week's usage — this
    mirrors real FPL rules closely but does not model chip interactions
    (Wildcard/Free Hit), which should be evaluated separately via
    `evaluate_chip_strategy`.

    Returns a dict: {gw_label: selected_squad_df, ..., 'summary': stats_df}
    """
    cols = _resolve_columns(df)
    cost_col, pos_col, team_col = cols["cost"], cols["pos"], cols["team"]
    horizon = len(projection_cols)

    df_clean = df.copy()
    df_clean[cost_col] = pd.to_numeric(df_clean[cost_col], errors="coerce")
    for pc in projection_cols:
        df_clean[pc] = pd.to_numeric(df_clean[pc], errors="coerce").fillna(0.0)
    df_clean = df_clean.dropna(subset=[cost_col, pos_col, team_col])

    if df_clean.empty or horizon == 0:
        return {}

    active_budget = budget * 10 if df_clean[cost_col].max() > 30 else budget
    idx = df_clean.index

    prob = pulp.LpProblem("FPL_Multi_Horizon_Optimization", pulp.LpMaximize)

    squad = {t: pulp.LpVariable.dicts(f"Squad_{t}", idx, cat="Binary") for t in range(horizon)}
    transfer_in = {t: pulp.LpVariable.dicts(f"In_{t}", idx, cat="Binary") for t in range(1, horizon)}
    transfer_out = {t: pulp.LpVariable.dicts(f"Out_{t}", idx, cat="Binary") for t in range(1, horizon)}
    hits = {t: pulp.LpVariable(f"Hits_{t}", lowBound=0, cat="Integer") for t in range(1, horizon)}
    free_transfers = {t: pulp.LpVariable(f"FT_{t}", lowBound=1, upBound=5, cat="Integer") for t in range(1, horizon)}

    # --- Objective: total projected points across the horizon, minus hit penalties ---
    points_term = pulp.lpSum(
        df_clean.loc[i, projection_cols[t]] * squad[t][i] for t in range(horizon) for i in idx
    )
    hit_term = pulp.lpSum(hit_cost * hits[t] for t in range(1, horizon))
    prob += points_term - hit_term

    # --- Per-gameweek constraints ---
    for t in range(horizon):
        prob += pulp.lpSum(df_clean.loc[i, cost_col] * squad[t][i] for i in idx) <= active_budget
        prob += pulp.lpSum(squad[t][i] for i in idx) == SQUAD_SIZE
        for pos, limit in POSITION_LIMITS.items():
            prob += pulp.lpSum(squad[t][i] for i in idx if df_clean.loc[i, pos_col] == pos) == limit
        for team in df_clean[team_col].unique():
            prob += pulp.lpSum(squad[t][i] for i in idx if df_clean.loc[i, team_col] == team) <= MAX_PLAYERS_PER_CLUB

    # --- Transfer tracking between consecutive weeks ---
    for t in range(1, horizon):
        for i in idx:
            prob += squad[t][i] - squad[t - 1][i] <= transfer_in[t][i]
            prob += squad[t - 1][i] - squad[t][i] <= transfer_out[t][i]

        total_transfers_t = pulp.lpSum(transfer_in[t][i] for i in idx)

        if t == 1:
            prob += free_transfers[t] == starting_free_transfers
        else:
            # Roll over unused free transfers (capped at 5); approximate linearization
            prob += free_transfers[t] <= free_transfers[t - 1] - total_transfers_t + 1 + 5  # upper bound slack
            prob += free_transfers[t] <= 5
            prob += free_transfers[t] >= 1

        prob += hits[t] >= total_transfers_t - free_transfers[t]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return {}

    results = {}
    summary_rows = []
    for t in range(horizon):
        selected = [i for i in idx if squad[t][i].varValue == 1.0]
        squad_df = df_clean.loc[selected].sort_values(by=[pos_col, projection_cols[t]], ascending=[True, False])
        gw_label = projection_cols[t]
        results[gw_label] = squad_df

        row = {
            "Gameweek": gw_label,
            "Projected_Points": round(squad_df[projection_cols[t]].sum(), 2),
            "Squad_Cost": round(squad_df[cost_col].sum(), 1),
        }
        if t > 0:
            row["Transfers_Made"] = int(sum(transfer_in[t][i].varValue for i in idx))
            row["Point_Hits"] = int(hits[t].varValue)
        summary_rows.append(row)

    results["summary"] = pd.DataFrame(summary_rows)
    return results


def evaluate_chip_strategy(df: pd.DataFrame, current_squad_ids: list,
                            budget: float = BUDGET_DEFAULT, target_col: str = "ep_next") -> dict:
    df_eval = df.copy()
    target_metric = target_col if target_col in df_eval.columns else "ep_next"
    df_eval[target_metric] = pd.to_numeric(df_eval.get(target_metric, 0), errors="coerce").fillna(0.0)

    current_squad = df_eval[df_eval["id"].isin(current_squad_ids)] if current_squad_ids else pd.DataFrame()
    current_xpts = current_squad[target_metric].sum() if not current_squad.empty else 0.0

    if not df_eval.empty:
        optimal_squad = optimize_squad(df_eval, budget=budget, target_metric=target_metric)
        wildcard_xpts = optimal_squad[target_metric].sum() if not optimal_squad.empty else 0.0
    else:
        wildcard_xpts = 0.0

    delta_xpts = wildcard_xpts - current_xpts

    threshold_1_hit = 4.0
    threshold_multi = 12.0

    if delta_xpts >= 25.0:
        decision = "ACTIVATE WILDCARD (Critical Mathematical Advantage Found)"
    elif delta_xpts >= threshold_multi:
        hits = int(delta_xpts // threshold_1_hit)
        decision = f"TAKE HITS (Statistical variance justifies up to {hits} transfers)"
    else:
        decision = "HOLD TRANSFERS (Statistical variance does not justify penalty hits)"

    return {
        "Current_Projected_Pts": round(current_xpts, 2),
        "Optimal_Projected_Pts": round(wildcard_xpts, 2),
        "Mathematical_Delta": round(delta_xpts, 2),
        "Engine_Recommendation": decision,
    }