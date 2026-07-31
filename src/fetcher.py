import requests
import pandas as pd

def get_fpl_players():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
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
        
        return players_df
    else:
        print(f"HTTP Error {response.status_code}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = get_fpl_players()
    print("Data extraction and mapping successful.")
    print(df[['first_name', 'second_name', 'team_name', 'position_name', 'now_cost']].head())