import pandas as pd
import pulp
import requests

def fetch_manager_squad(team_id: str) -> list:
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/picks/"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return [player['element'] for player in data['picks']]
    return []

def optimize_squad(df: pd.DataFrame, budget: float = 100.0, target_metric: str = 'ep_next') -> pd.DataFrame:
    cost_col = 'Cost' if 'Cost' in df.columns else 'now_cost'
    pos_col = 'Position' if 'Position' in df.columns else 'position_name'
    team_col = 'Team' if 'Team' in df.columns else 'team_name'
    
    possible_targets = [target_metric, 'ep_next', 'Expected Points', 'Total Points', 'total_points']
    target_col = next((c for c in possible_targets if c in df.columns), None)
    
    df_clean = df.copy()
    
    if not target_col:
        target_col = 'Optimization_Target'
        df_clean[target_col] = 0.0
        
    df_clean[cost_col] = pd.to_numeric(df_clean[cost_col], errors='coerce')
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce').fillna(0.0)
    df_clean = df_clean.dropna(subset=[cost_col, target_col, pos_col, team_col])
    
    active_budget = budget * 10 if df_clean[cost_col].max() > 30 else budget

    prob = pulp.LpProblem("FPL_Predictive_Optimization", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", df_clean.index, cat='Binary')
    
    prob += pulp.lpSum([df_clean.loc[i, target_col] * player_vars[i] for i in df_clean.index])
    
    prob += pulp.lpSum([df_clean.loc[i, cost_col] * player_vars[i] for i in df_clean.index]) <= active_budget
    
    prob += pulp.lpSum([player_vars[i] for i in df_clean.index]) == 15
    
    pos_limits = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, limit in pos_limits.items():
        prob += pulp.lpSum([player_vars[i] for i in df_clean.index if df_clean.loc[i, pos_col] == pos]) == limit
        
    for team in df_clean[team_col].unique():
        prob += pulp.lpSum([player_vars[i] for i in df_clean.index if df_clean.loc[i, team_col] == team]) <= 3
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    selected_indices = [i for i in df_clean.index if player_vars[i].varValue == 1.0]
    return df_clean.loc[selected_indices].sort_values(by=[pos_col, target_col], ascending=[True, False])

def evaluate_chip_strategy(df: pd.DataFrame, current_squad_ids: list, budget: float = 100.0) -> dict:
    df_eval = df.copy()
    df_eval['ep_next'] = pd.to_numeric(df_eval['ep_next'], errors='coerce').fillna(0.0)
    
    current_squad = df_eval[df_eval['id'].isin(current_squad_ids)] if current_squad_ids else pd.DataFrame()
    current_xpts = current_squad['ep_next'].sum() if not current_squad.empty else 0.0
       
    if not df_eval.empty:
        optimal_squad = optimize_squad(df_eval, budget=budget, target_metric='ep_next')
        wildcard_xpts = optimal_squad['ep_next'].sum() if not optimal_squad.empty else 0.0
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
        'Current_Projected_Pts': round(current_xpts, 2),
        'Optimal_Projected_Pts': round(wildcard_xpts, 2),
        'Mathematical_Delta': round(delta_xpts, 2),
        'Engine_Recommendation': decision
    }