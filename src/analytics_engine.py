import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_advanced_metrics() -> pd.DataFrame:
    file_path = os.path.join('data', 'advanced_fbref_stats.parquet')
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    df.columns = [str(col).rstrip('_') for col in df.columns]
    
    min_col = next((c for c in df.columns if '90s' in c.lower() or 'minutes' in c.lower()), None)
    
    rename_map = {
        'player': 'Player',
        'team': 'Team',
        'pos': 'Position'
    }
    
    if min_col:
        rename_map[min_col] = 'Matches_90s'
        
    df = df.rename(columns=rename_map)
    
    if 'Matches_90s' not in df.columns:
        df['Matches_90s'] = 0.0
        
    return df

def generate_predicted_lineup(df: pd.DataFrame, target_team: str) -> pd.DataFrame:
    team_data = df[df['Team'] == target_team].copy()
    
    if team_data.empty:
        return pd.DataFrame()
        
    team_data['Matches_90s'] = pd.to_numeric(team_data['Matches_90s'], errors='coerce').fillna(0)
    projected_squad = team_data.nlargest(11, 'Matches_90s')
    
    tactical_keywords = ['goals', 'xg', 'passes_completed', 'progressive', 'tackles']
    available_cols = ['Player', 'Position', 'Matches_90s']
    
    for col in projected_squad.columns:
        if any(keyword in col.lower() for keyword in tactical_keywords) and col not in available_cols:
            available_cols.append(col)
            
    return projected_squad[available_cols[:8]]

def plot_tactical_quadrant(adv_df: pd.DataFrame) -> plt.Figure:
    xg_col = next((c for c in adv_df.columns if 'xg' in c.lower()), None)
    pass_col = next((c for c in adv_df.columns if 'passes_completed' in c.lower()), None)
    
    if not xg_col or not pass_col:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, 'Tactical data requirements not met', ha='center', va='center', color='red')
        fig.patch.set_facecolor('#22312b')
        return fig

    adv_df[xg_col] = pd.to_numeric(adv_df[xg_col], errors='coerce').fillna(0)
    adv_df[pass_col] = pd.to_numeric(adv_df[pass_col], errors='coerce').fillna(0)
    
    team_stats = adv_df.groupby('Team').agg(
        Attack_xG=(xg_col, 'sum'),
        Possession_Control=(pass_col, 'mean')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#22312b')
    ax.set_facecolor('#22312b')
    
    sns.scatterplot(data=team_stats, x='Possession_Control', y='Attack_xG', color='#69ea82', s=150, edgecolor='white', ax=ax)
    
    for i in range(team_stats.shape[0]):
        ax.text(team_stats['Possession_Control'][i], team_stats['Attack_xG'][i] + 0.5, 
                team_stats['Team'][i], color='white', fontsize=9, ha='center', weight='bold')
        
    ax.axhline(team_stats['Attack_xG'].mean(), color='#c7d5cc', linestyle='--', alpha=0.5)
    ax.axvline(team_stats['Possession_Control'].mean(), color='#c7d5cc', linestyle='--', alpha=0.5)
    
    ax.set_title("Tactical Matrix Attacking Threat (xG) vs Possession Control", color='white', fontsize=15, pad=20)
    ax.set_xlabel("Average Completed Passes per Match", color='white', fontsize=11)
    ax.set_ylabel("Total Expected Goals (xG)", color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.grid(False)
    
    return fig

def identify_key_playmakers(df: pd.DataFrame, target_team: str) -> pd.DataFrame:
    team_data = df[df['Team'] == target_team].copy()
    if team_data.empty:
        return pd.DataFrame()

    prog_col = next((c for c in team_data.columns if 'progressive_passes' in c.lower()), None)
    xa_col = next((c for c in team_data.columns if 'xa' in c.lower()), None)

    if not prog_col or not xa_col:
        return pd.DataFrame(columns=['Player', 'Error Missing Playmaker Columns'])

    team_data[prog_col] = pd.to_numeric(team_data[prog_col], errors='coerce').fillna(0)
    team_data[xa_col] = pd.to_numeric(team_data[xa_col], errors='coerce').fillna(0)

    team_data['Playmaker_Index'] = (team_data[prog_col] * 0.6) + (team_data[xa_col] * 0.4)
    top_creators = team_data.nlargest(5, 'Playmaker_Index')

    return top_creators[['Player', 'Position', prog_col, xa_col, 'Playmaker_Index']]