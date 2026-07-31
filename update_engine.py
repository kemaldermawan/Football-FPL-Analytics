import requests
import pandas as pd
import os

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
        
        clean_df = players_df[['first_name', 'second_name', 'team_name', 'position_name', 'now_cost', 'total_points']].copy()
        
        output_path = os.path.join('data', 'fpl_static.parquet')
        clean_df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Data successfully serialized and saved to {output_path}")
    else:
        print(f"Failed to fetch data. HTTP Error: {response.status_code}")

if __name__ == "__main__":
    update_fpl_data()