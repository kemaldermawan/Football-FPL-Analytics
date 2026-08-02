import requests
import pandas as pd
import os
import soccerdata as sd

def update_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    print("Initiating connection to FPL API...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        players_df = pd.DataFrame(data['elements'])
        teams_df = pd.DataFrame(data['teams'])
        positions_df = pd.DataFrame(data['element_types'])
        
        team_mapping = dict(zip(teams_df['id'], teams_df['name']))
        players_df['team_name'] = players_df['team'].map(team_mapping)
        
        pos_mapping = dict(zip(positions_df['id'], positions_df['singular_name_short']))
        players_df['position_name'] = players_df['element_type'].map(pos_mapping)
        
        clean_df = players_df[['id', 'first_name', 'second_name', 'team_name', 'position_name', 'now_cost', 'total_points', 'ep_next', 'form']].copy()

        output_path = os.path.join('data', 'fpl_static.parquet')
        clean_df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"FPL data correctly mapped and serialized to {output_path}")
    else:
        print(f"Failed to fetch FPL data. HTTP Error {response.status_code}")

def update_event_data():
    print("Initiating connection to Understat via soccerdata...")
    understat = sd.Understat(leagues="ENG-Premier League", seasons="2023")
    
    shots_df = understat.read_shot_events().reset_index()
    arsenal_shots = shots_df[shots_df['team'] == 'Arsenal']
    
    output_path = os.path.join('data', 'epl_raw.parquet')
    arsenal_shots.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Spatial event data serialized to {output_path}")

def update_advanced_fbref_data():
    print("Initiating full-scale data extraction from FBref (EPL)...")
    
    fbref = sd.FBref(leagues="ENG-Premier League", seasons="2023")
    
    standard_df = fbref.read_player_season_stats(stat_type="standard").reset_index()
    shooting_df = fbref.read_player_season_stats(stat_type="shooting").reset_index()
    misc_df = fbref.read_player_season_stats(stat_type="misc").reset_index()
    
    advanced_stats = standard_df.merge(shooting_df, on=['league', 'season', 'team', 'player'], how='left')
    advanced_stats = advanced_stats.merge(misc_df, on=['league', 'season', 'team', 'player'], how='left')
    
    advanced_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in advanced_stats.columns]
    
    output_path = os.path.join('data', 'advanced_fbref_stats.parquet')
    advanced_stats.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Professional grade metrics serialized to {output_path}")

if __name__ == "__main__":
    update_fpl_data()
    update_event_data()
    update_advanced_fbref_data()