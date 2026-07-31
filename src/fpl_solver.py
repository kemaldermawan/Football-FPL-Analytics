import pandas as pd
import pulp

def optimize_squad(df: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
    df_clean = df.dropna(subset=['Cost', 'Total Points']).copy()
    
    prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
    
    player_vars = pulp.LpVariable.dicts("Player", df_clean.index, cat='Binary')
    
    # Fungsi Objektif Memaksimalkan Total Poin
    prob += pulp.lpSum([df_clean.loc[i, 'Total Points'] * player_vars[i] for i in df_clean.index])
    
    # Konstrain Anggaran Maksimal
    prob += pulp.lpSum([df_clean.loc[i, 'Cost'] * player_vars[i] for i in df_clean.index]) <= budget
    
    # Konstrain Total Pemain Skuad
    prob += pulp.lpSum([player_vars[i] for i in df_clean.index]) == 15
    
    # Konstrain Kuota Posisi Absolut
    pos_limits = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, limit in pos_limits.items():
        prob += pulp.lpSum([player_vars[i] for i in df_clean.index if df_clean.loc[i, 'Position'] == pos]) == limit
        
    # Konstrain Maksimal 3 Pemain Per Tim
    for team in df_clean['Team'].unique():
        prob += pulp.lpSum([player_vars[i] for i in df_clean.index if df_clean.loc[i, 'Team'] == team]) <= 3
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    selected_indices = [i for i in df_clean.index if player_vars[i].varValue == 1.0]
    return df_clean.loc[selected_indices].sort_values(by=['Position', 'Total Points'], ascending=[True, False])