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
        
        clean_df = players_df[['first_name', 'second_name', 'team', 'element_type', 'now_cost', 'total_points']].copy()
        output_path = os.path.join('data', 'fpl_static.parquet')
        clean_df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"FPL data serialized to {output_path}")
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

if __name__ == "__main__":
    update_fpl_data()
    update_event_data()