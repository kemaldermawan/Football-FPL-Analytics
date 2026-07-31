import streamlit as st
from src.fetcher import get_fpl_players
from src.visuals import create_scatter_plot, create_team_bar_chart
from src.fpl_solver import optimize_squad

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
    
    csv_export = filtered_data.to_csv(index=False).encode('utf-8')
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Export Data to CSV",
        data=csv_export,
        file_name='fpl_filtered_data.csv',
        mime='text/csv',
    )
    
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
    
    st.subheader("Data Visualizations")
    tab1, tab2 = st.tabs(["Cost vs Points (Scatter)", "Team Performance (Bar)"])
    
    with tab1:
        scatter_chart = create_scatter_plot(filtered_data)
        st.altair_chart(scatter_chart, use_container_width=True)
    
    with tab2:
        bar_chart = create_team_bar_chart(filtered_data)
        st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Operations Research: MILP Squad Optimizer")
    st.info("Execute the deterministic PuLP solver to generate the mathematically optimal 15-man squad under strict FPL budget and quota constraints.")
    
    if st.button("Generate Optimal Squad"):
        with st.spinner("Calculating optimal integer combinations..."):
            optimal_squad = optimize_squad(display_data, budget=100.0)
            
            st.success("Mathematical optimization complete.")
            st.dataframe(optimal_squad, use_container_width=True)
            
            total_cost = optimal_squad['Cost'].sum()
            total_pts = optimal_squad['Total Points'].sum()
            
            col1, col2 = st.columns(2)
            col1.metric("Squad Expected Total Points", f"{total_pts} Pts")
            col2.metric("Total Budget Utilized", f"£{total_cost:.1f}M")
else:
    st.error("System failed to retrieve data from the FPL API.")