import streamlit as st
import pandas as pd
from src.fetcher import get_fpl_players
from src.visuals import create_scatter_plot, create_team_bar_chart, create_pizza_chart
from src.tactical import draw_pass_network
from src.predictor import generate_score_matrix, calculate_match_odds, run_monte_carlo
from src.analytics_engine import load_advanced_metrics, generate_predicted_lineup, plot_tactical_quadrant, identify_key_playmakers
from src.fpl_solver import optimize_squad, fetch_manager_squad, evaluate_chip_strategy
from src.scouting_engine import run_kmeans_clustering, find_similar_players

st.set_page_config(
    page_title="Football Intelligence Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Platform Navigation")
app_module = st.sidebar.radio("Select Analytical Module", ["FPL Decision Engine", "Tactical Football Analyst"])
st.sidebar.markdown("---")

if app_module == "FPL Decision Engine":
    st.title("⚽ Fantasy Premier League Decision Engine")
    st.markdown("Deterministic Operations Research and Stochastic Squad Optimization.")

    raw_data = get_fpl_players()

    if not raw_data.empty:
        display_data = raw_data[['id', 'first_name', 'second_name', 'team_name', 'position_name', 'now_cost', 'total_points', 'ep_next']].copy()
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
        
        st.sidebar.header("FPL Control Parameters")
        teams_list = sorted(display_data['Team'].unique())
        positions_list = display_data['Position'].unique()
        
        selected_teams = st.sidebar.multiselect("Select Teams", teams_list, default=teams_list)
        selected_positions = st.sidebar.multiselect("Select Positions", positions_list, default=positions_list)
        
        filtered_data = display_data[
            (display_data['Team'].isin(selected_teams)) &
            (display_data['Position'].isin(selected_positions))
        ]
        
        csv_export = filtered_data.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Export FPL Data",
            data=csv_export,
            file_name='fpl_filtered_data.csv',
            mime='text/csv',
        )
        
        tab_market, tab_milp, tab_chip = st.tabs(["Market Analysis", "MILP Squad Optimizer", "Stochastic Chip Evaluator"])
        
        with tab_market:
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
            
            col_scatter, col_bar = st.columns(2)
            with col_scatter:
                scatter_chart = create_scatter_plot(filtered_data)
                st.altair_chart(scatter_chart, use_container_width=True)
            with col_bar:
                bar_chart = create_team_bar_chart(filtered_data)
                st.altair_chart(bar_chart, use_container_width=True)

        with tab_milp:
            st.subheader("Expected Points Maximization")
            st.info("Execute deterministic PuLP solver to generate mathematically optimal constraints.")
            if st.button("Generate Mathematically Optimal Squad"):
                with st.spinner("Calculating optimal integer combinations..."):
                    optimal_squad = optimize_squad(display_data, budget=100.0, target_metric='ep_next')
                    st.success("Mathematical optimization complete.")
                    st.dataframe(optimal_squad, use_container_width=True)
                    
                    total_cost = optimal_squad['Cost'].sum()
                    total_pts = optimal_squad['Total Points'].sum()
                    st.metric("Expected Points & Budget", f"{total_pts} Pts", f"£{total_cost:.1f}M Utilized")

        with tab_chip:
            st.subheader("Live Squad Sync & Strategy Evaluation")
            fpl_team_id = st.text_input("Enter FPL Team ID (e.g., 123456)")
            if st.button("Evaluate Current Strategy"):
                if fpl_team_id:
                    with st.spinner("Fetching live data and running stochastic evaluation..."):
                        squad_ids = fetch_manager_squad(fpl_team_id)
                        if squad_ids:
                            my_squad = display_data[display_data['id'].isin(squad_ids)].copy()
                            st.success("Squad synchronized successfully.")
                            st.dataframe(my_squad[['First Name', 'Last Name', 'Team', 'Cost', 'ep_next']], use_container_width=True)
                            
                            st.markdown("#### Stochastic Chip Evaluator")
                            eval_results = evaluate_chip_strategy(raw_data, squad_ids)
                            
                            e_col1, e_col2, e_col3 = st.columns(3)
                            e_col1.metric("Current xPts", eval_results['Current_Projected_Pts'])
                            e_col2.metric("Optimal xPts", eval_results['Optimal_Projected_Pts'])
                            e_col3.metric("Mathematical Delta", eval_results['Mathematical_Delta'])
                            
                            st.info(f"**Decision Engine Recommendation:** {eval_results['Engine_Recommendation']}")
                        else:
                            st.error("Invalid Team ID or API rate limit exceeded.")
                else:
                    st.warning("Team ID is strictly required.")
    else:
        st.error("System failed to retrieve data from the FPL API.")

elif app_module == "Tactical Football Analyst":
    st.title("🛡️ Professional Tactical Analyst Hub")
    st.markdown("Spatial deconstruction, machine learning scouting, and tactical quadrant matrix.")
    
    tab_tactical, tab_scouting, tab_spatial, tab_sim = st.tabs([
        "Pro Analytics & Lineups", 
        "ML Scouting", 
        "Spatial Pass Network", 
        "Match Simulator"
    ])
    
    with tab_tactical:
        st.markdown("### Advanced FBref Tactical Engine")
        st.info("Comprehensive multi-league metrics for Premier League and UEFA Champions League.")
        
        adv_data = load_advanced_metrics()
        
        if not adv_data.empty:
            team_list = sorted(adv_data['Team'].dropna().unique())
            selected_team = st.selectbox("Select Club for Tactical Deconstruction", team_list)
            
            col_lineup, col_playmaker = st.columns(2)
            
            with col_lineup:
                st.markdown(f"#### Projected Starting XI")
                lineup_df = generate_predicted_lineup(adv_data, selected_team)
                st.dataframe(lineup_df, use_container_width=True)
                
            with col_playmaker:
                st.markdown(f"#### Key Playmakers (xT & xA Index)")
                playmakers_df = identify_key_playmakers(adv_data, selected_team)
                st.dataframe(playmakers_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Tactical Style Matrix")
            st.info("Top Right quadrant identifies High Possession & High Threat teams.")
            tactical_matrix_fig = plot_tactical_quadrant(adv_data)
            st.pyplot(tactical_matrix_fig)
        else:
            st.warning("Advanced metrics database not found. Please run the ETL pipeline.")

    with tab_scouting:
        st.markdown("### K-Means Statistical Scouting")
        st.info("Unsupervised machine learning identifying hidden gems via Euclidean distance across 20+ tactical metrics.")
        
        adv_data = load_advanced_metrics()
        if not adv_data.empty:
            clustered_df, scaled_data, features, scaler = run_kmeans_clustering(adv_data)
            
            if not clustered_df.empty:
                player_list = sorted(clustered_df['Player'].unique())
                target_player = st.selectbox("Select Premium Player to Find Tactical Alternatives", player_list)
                
                st.markdown(f"#### Most Similar Profiles to {target_player}")
                similar_df = find_similar_players(target_player, clustered_df, scaled_data)
                
                if not similar_df.empty:
                    st.dataframe(similar_df, use_container_width=True)
                else:
                    st.warning("Player data isolated or invalid.")
            else:
                st.warning("Insufficient data for clustering processing. Adjust minutes played filter.")
        else:
            st.warning("Advanced metrics database not found. Please run the ETL pipeline.")

    with tab_spatial:
        st.markdown("### Spatial Event Deconstruction")
        st.info("Displaying spatial event data mapping using mplsoccer.")
        tactical_fig = draw_pass_network()
        st.pyplot(tactical_fig)

    with tab_sim:
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