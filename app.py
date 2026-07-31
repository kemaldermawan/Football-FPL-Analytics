import streamlit as st
import altair as alt
from src.fetcher import get_fpl_players

st.set_page_config(
    page_title="FPL Analytics",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football - Fantasy Premier League Analytics")
st.markdown("Welcome to the FPL analytics system. Displaying live data from the official API.")

raw_data = get_fpl_players()

if not raw_data.empty:
    display_data = raw_data[['first_name', 'second_name', 'team_name', 'position_name', 'now_cost', 'total_points']].copy()
    display_data['now_cost'] = display_data['now_cost'] / 10.0
    
    display_data['value'] = (display_data['total_points'] / display_data['now_cost']).round(2)
    
    display_data.rename(columns={
        'first_name': 'First Name',
        'second_name': 'Last Name',
        'team_name': 'Team',
        'position_name': 'Position',
        'now_cost': 'Cost',
        'total_points': 'Total Points',
        'value': 'Value (Pts/Cost)'
    }, inplace=True)
    
    st.sidebar.header("Control Parameters")
    
    teams_list = sorted(display_data['Team'].unique())
    positions_list = display_data['Position'].unique()
    
    selected_teams = st.sidebar.multiselect("Select Teams", teams_list, default=teams_list)
    selected_positions = st.sidebar.multiselect("Select Positions", positions_list, default=positions_list)
    
    filtered_data = display_data[
        (display_data['Team'].isin(selected_teams)) &
        (display_data['Position'].isin(selected_positions))
    ]
    
    st.subheader("Top 5 Efficient Players (Value for Money)")
    top_value_players = filtered_data.nlargest(5, 'Value (Pts/Cost)')
    
    metric_cols = st.columns(5)
    for col, (_, player) in zip(metric_cols, top_value_players.iterrows()):
        player_name = f"{player['First Name'][0]}. {player['Last Name']}"
        col.metric(
            label=player_name,
            value=f"{player['Value (Pts/Cost)']}",
            delta=f"{player['Total Points']} Pts | £{player['Cost']}M",
            delta_color="off"
        )
    
    st.subheader(f"Live Player Metrics ({len(filtered_data)} Players)")
    st.dataframe(filtered_data, use_container_width=True)
    
    st.subheader("Cost vs Total Points Analysis")
    
    scatter_chart = alt.Chart(filtered_data).mark_circle(size=60).encode(
        x=alt.X('Cost', scale=alt.Scale(zero=False)),
        y=alt.Y('Total Points', scale=alt.Scale(zero=False)),
        color='Position',
        tooltip=['First Name', 'Last Name', 'Team', 'Cost', 'Total Points', 'Value (Pts/Cost)']
    ).interactive()
    
    st.altair_chart(scatter_chart, use_container_width=True)

else:
    st.error("System failed to retrieve data from the FPL API.")