import streamlit as st
from src.fetcher import get_fpl_players

st.set_page_config(
    page_title="FPL Analytics",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football - Fantasy Premier League Analytics")
st.markdown("Welcome to the FPL analytics system. Displaying live data from the official API.")

st.sidebar.header("Control Parameters")
st.sidebar.info("Data extraction module successfully integrated.")

st.subheader("Live Player Metrics")

raw_data = get_fpl_players()

if not raw_data.empty:
    display_data = raw_data[['first_name', 'second_name', 'team', 'element_type', 'now_cost', 'total_points']].copy()
    display_data['now_cost'] = display_data['now_cost'] / 10.0
    
    display_data.rename(columns={
        'first_name': 'First Name',
        'second_name': 'Last Name',
        'team': 'Team ID',
        'element_type': 'Position ID',
        'now_cost': 'Cost',
        'total_points': 'Total Points'
    }, inplace=True)
    
    st.dataframe(display_data, use_container_width=True)
else:
    st.error("System failed to retrieve data from the FPL API.")