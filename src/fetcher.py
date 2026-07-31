import requests
import pandas as pd

def get_fpl_players():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        players_df = pd.DataFrame(data['elements'])
        return players_df
    else:
        print(f"HTTP Error {response.status_code}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = get_fpl_players()
    print("Data extraction successful.")
    print(f"Total players retrieved: {len(df)}")
    print(df[['first_name', 'second_name', 'now_cost', 'total_points']].head())