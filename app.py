import streamlit as st
from src.fetcher import get_fpl_players
from src.visuals import create_scatter_plot, create_team_bar_chart, create_pizza_chart
from src.fpl_solver import optimize_squad, fetch_manager_squad
from src.tactical import draw_pass_network
from src.predictor import generate_score_matrix, calculate_match_odds, find_similar_players, run_monte_carlo

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
    
    st.subheader("Data Visualizations & Analytical Models")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cost vs Points", "Team", "Tactical Pitch", "Poisson Predictor", "ML Scouting"])
    
    with tab1:
        scatter_chart = create_scatter_plot(filtered_data)
        st.altair_chart(scatter_chart, use_container_width=True)
        
    with tab2:
        bar_chart = create_team_bar_chart(filtered_data)
        st.altair_chart(bar_chart, use_container_width=True)
        
    with tab3:
        st.info("Displaying spatial event data mapping using mplsoccer.")
        tactical_fig = draw_pass_network()
        st.pyplot(tactical_fig)
        
    with tab4:
        st.markdown("### Match Simulator (Poisson & Monte Carlo)")
        col1, col2 = st.columns(2)
        home_xg = col1.slider("Home Team Expected Goals (xG)", 0.1, 4.0, 1.5, 0.1)
        away_xg = col2.slider("Away Team Expected Goals (xG)", 0.1, 4.0, 1.2, 0.1)
        
        score_matrix = generate_score_matrix(home_xg, away_xg)
        mc_home, mc_draw, mc_away = run_monte_carlo(home_xg, away_xg, 10000)
        
        st.markdown("#### 10,000x Monte Carlo Stable Probabilities")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Home Win", f"{mc_home * 100:.1f}%")
        m_col2.metric("Draw", f"{mc_draw * 100:.1f}%")
        m_col3.metric("Away Win", f"{mc_away * 100:.1f}%")
        
        st.markdown("#### Exact Score Probability Matrix")
        st.dataframe(score_matrix.style.background_gradient(cmap='YlGn', axis=None).format("{:.2%}"), use_container_width=True)

    with tab5:
        st.markdown("### Pro Scouting: K-Means & Percentile Radar")
        player_list = sorted(filtered_data['Last Name'].dropna().unique())
        target_player = st.selectbox("Select Target Player", player_list)
        
        target_series = filtered_data[filtered_data['Last Name'] == target_player].iloc[0]
        pos_population = filtered_data[filtered_data['Position'] == target_series['Position']]
        
        col_radar, col_similar = st.columns([1, 1])
        
        with col_radar:
            radar_fig = create_pizza_chart(target_series, pos_population)
            st.pyplot(radar_fig)
            
        with col_similar:
            st.markdown(f"**Similar Statistical Profiles (K-Means)**")
            similar_df = find_similar_players(filtered_data, target_player)
            if not similar_df.empty:
                st.dataframe(similar_df[['First Name', 'Last Name', 'Cost', 'Total Points']], use_container_width=True)
            else:
                st.warning("Insufficient data for clustering.")

    st.markdown("---")
    st.subheader("Operations Research: MILP Squad Optimizer & Sync")
    st.info("Execute deterministic PuLP solver and synchronize live manager squad via FPL API.")
    
    col_opt, col_sync = st.columns(2)
    
    with col_opt:
        if st.button("Generate Mathematically Optimal Squad"):
            with st.spinner("Calculating optimal integer combinations..."):
                optimal_squad = optimize_squad(display_data, budget=100.0)
                st.success("Mathematical optimization complete.")
                st.dataframe(optimal_squad, use_container_width=True)
                
                total_cost = optimal_squad['Cost'].sum()
                total_pts = optimal_squad['Total Points'].sum()
                st.metric("Expected Points & Budget", f"{total_pts} Pts", f"£{total_cost:.1f}M Utilized")

    with col_sync:
        fpl_team_id = st.text_input("Enter FPL Team ID (e.g., 123456)")
        if st.button("Sync Current Squad"):
            if fpl_team_id:
                with st.spinner("Fetching live data from FPL servers..."):
                    squad_ids = fetch_manager_squad(fpl_team_id)
                    if squad_ids:
                        my_squad = raw_data[raw_data['id'].isin(squad_ids)].copy()
                        my_squad = my_squad[['first_name', 'second_name', 'team_name', 'position_name', 'now_cost', 'total_points']]
                        st.success("Squad synchronized successfully.")
                        st.dataframe(my_squad, use_container_width=True)
                    else:
                        st.error("Invalid Team ID or API rate limit exceeded.")
            else:
                st.warning("Team ID is strictly required.")
else:
    st.error("System failed to retrieve data from the FPL API.")