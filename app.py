import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.fetcher import get_fpl_players, get_team_rolling_form, get_fpl_team_strengths, get_next_opponent, get_fixture_run, get_season_status, get_all_fixtures
from src.visuals import create_scatter_plot, create_team_bar_chart, create_pizza_chart, create_xpts_vs_cost_chart
from src.club_colors import style_table_by_club
from src.tactical import draw_pass_network
from src.predictor import (
    compute_team_strengths, expected_goals, generate_score_matrix,
    calculate_match_odds, calculate_clean_sheet_probability, run_monte_carlo,
)
from src.analytics_engine import (
    load_advanced_metrics, generate_predicted_lineup, plot_tactical_quadrant,
    identify_key_playmakers, compute_defensive_flank_vulnerability,
)
from src.fpl_solver import optimize_squad, optimize_squad_multi_horizon, fetch_manager_squad, evaluate_chip_strategy
from src.scouting_engine import run_kmeans_clustering, find_similar_players
from src.xt_model import build_xt_grid
from src.custom_xpts import compute_custom_xpts, build_opponent_defense_map, build_custom_fdr_matrix
from src.config import COLOR_BG, COLOR_PANEL, COLOR_ACCENT, COLOR_TEXT, COLOR_MUTED, POSITION_ORDER

st.set_page_config(
    page_title="Football Intelligence Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theming: a dark "pitch at night" aesthetic applied consistently across the
# Streamlit chrome, so native widgets don't clash with the matplotlib/altair
# dark-themed charts already used throughout the modules.
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {COLOR_PANEL}; }}
    h1, h2, h3, h4 {{ color: {COLOR_TEXT} !important; }}
    p, li, span, label {{ color: {COLOR_TEXT}; }}
    div[data-testid="stMetric"] {{
        background-color: {COLOR_PANEL};
        border: 1px solid #2a2f3a;
        border-radius: 10px;
        padding: 12px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {COLOR_MUTED} !important; }}
    .stTabs [data-baseweb="tab"] {{ color: {COLOR_MUTED}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: {COLOR_ACCENT} !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {COLOR_ACCENT} !important; }}
    .block-container {{ padding-top: 2rem; }}
    div[data-testid="stButton"] button {{
        background-color: {COLOR_ACCENT};
        color: #ffffff;
        font-weight: 600;
        border: none;
    }}
    div[data-testid="stButton"] button:hover {{
        background-color: #3d78dd;
        color: #ffffff;
    }}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("⚽ Platform Navigation")
app_module = st.sidebar.radio(
    "Select Analytical Module",
    ["FPL Decision Engine", "Tactical Football Analyst"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data refreshed via `python update_engine.py`. Re-run before each gameweek for live prices, form, and fixtures.")

season_info = get_season_status()
if season_info.get("season_status") == "PRE_SEASON":
    st.warning(
        "⚠️ **Pre-season data notice.** " + season_info.get("message", "") +
        " Total Points, prices, and xG figures shown across this dashboard are last "
        "season's carried-over values, not a live reset. Re-run `update_engine.py` closer "
        "to kickoff once FPL opens the new season."
    )
elif season_info.get("season_status") == "SEASON_COMPLETE":
    st.info("ℹ️ " + season_info.get("message", "Season complete."))

# ===========================================================================
# MODULE 1 — FPL DECISION ENGINE
# ===========================================================================
if app_module == "FPL Decision Engine":
    st.title("⚽ Fantasy Premier League Decision Engine")
    st.markdown("Deterministic Operations Research and stochastic squad optimization — no gut feel, only constraints and objective functions.")

    raw_data = get_fpl_players()

    if not raw_data.empty:
        display_data = raw_data[[
            "id", "first_name", "second_name", "team_name", "position_name",
            "now_cost", "total_points", "ep_next", "minutes", "status",
            "chance_of_playing_next_round", "cost_change_event", "cost_change_start",
            "selected_by_percent", "transfers_in_event", "transfers_out_event",
        ]].copy()
        
        display_data["now_cost"] = display_data["now_cost"] / 10.0

        display_data.rename(columns={
            "first_name": "First Name", "second_name": "Last Name",
            "team_name": "Team", "position_name": "Position",
            "now_cost": "Cost", "total_points": "Total Points",
            "minutes": "Minutes", "chance_of_playing_next_round": "Chance_of_Playing",
            "cost_change_event": "Price_Change_GW", "cost_change_start": "Price_Change_Season",
            "selected_by_percent": "Ownership_Pct",
        }, inplace=True)

        # 1. Komputasi VORP (Value Over Replacement Player)
        base_costs = {"GKP": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}
        display_data["Base_Cost"] = display_data["Position"].map(base_costs)
        cost_diff = (display_data["Cost"] - display_data["Base_Cost"]).clip(lower=0.1)
        display_data["VORP"] = (display_data["Total Points"] / cost_diff).round(2)
        
        # 2. Injeksi Form dan Regresi xG/xA
        display_data["Form"] = pd.to_numeric(raw_data["form"], errors="coerce").fillna(0.0)
        display_data["xG"] = pd.to_numeric(raw_data["expected_goals"], errors="coerce").fillna(0.0)
        display_data["xA"] = pd.to_numeric(raw_data["expected_assists"], errors="coerce").fillna(0.0)
        
        goals = pd.to_numeric(raw_data["goals_scored"], errors="coerce").fillna(0.0)
        assists = pd.to_numeric(raw_data["assists"], errors="coerce").fillna(0.0)
        
        # Negatif = Underperforming (Akan segera mencetak gol). Positif = Overperforming (Berisiko seret gol)
        display_data["xG_xA_Delta"] = ((goals + assists) - (display_data["xG"] + display_data["xA"])).round(2)

        STATUS_LABELS = {
            "a": "Available", "d": "Doubtful", "i": "Injured",
            "s": "Suspended", "u": "Unavailable", "n": "Not in Squad",
        }
        display_data["Availability"] = display_data["status"].map(STATUS_LABELS).fillna("Unknown")
        display_data["Chance_of_Playing"] = pd.to_numeric(display_data["Chance_of_Playing"], errors="coerce")
        display_data["Ownership_Pct"] = pd.to_numeric(display_data["Ownership_Pct"], errors="coerce")
        display_data["Price_Change_GW"] = display_data["Price_Change_GW"] / 10.0
        display_data["Price_Change_Season"] = display_data["Price_Change_Season"] / 10.0
        display_data["Net_Transfers_GW"] = (
            pd.to_numeric(display_data["transfers_in_event"], errors="coerce").fillna(0)
            - pd.to_numeric(display_data["transfers_out_event"], errors="coerce").fillna(0)
        ).astype(int)

        st.sidebar.header("FPL Control Parameters")
        teams_list = sorted(display_data["Team"].dropna().unique())
        positions_list = display_data["Position"].dropna().unique()

        selected_teams = st.sidebar.multiselect("Select Teams", teams_list, default=teams_list)
        selected_positions = st.sidebar.multiselect("Select Positions", positions_list, default=positions_list)

        filtered_data = display_data[
            display_data["Team"].isin(selected_teams) & display_data["Position"].isin(selected_positions)
        ]

        fixture_run_df = get_fixture_run()
        if not fixture_run_df.empty:
            filtered_data = filtered_data.merge(fixture_run_df, on="Team", how="left")

        csv_export = filtered_data.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "📥 Export FPL Data", data=csv_export,
            file_name="fpl_filtered_data.csv", mime="text/csv",
        )

        tab_market, tab_matrix, tab_xpts, tab_milp, tab_horizon, tab_chip = st.tabs([
            "Market Analysis", "Advanced Fixture Matrix", "Custom xPts Model", "MILP Squad Optimizer",
            "Multi-Horizon Planner", "Stochastic Chip Evaluator",
        ])

        # --- Market Analysis ---
        with tab_market:
            st.subheader("Top Efficient Players (Value Over Replacement)")
            st.caption(
                "Ranked per position using VORP (Points per £M above the absolute positional base cost). "
                "Restricted to available players clearing a minutes threshold to avoid small-sample flukes."
            )

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1, 1])
            with col_ctrl1:
                min_minutes = st.slider(
                    "Minimum minutes played", min_value=0, max_value=2500, value=450, step=90,
                    help="450 minutes ≈ 5 full matches. Filters out small-sample outliers.",
                )
            with col_ctrl2:
                only_available = st.checkbox("Only show available players", value=True,
                                              help="Excludes players flagged Injured / Suspended / Unavailable.")
            with col_ctrl3:
                rank_basis = st.radio(
                    "Rank by", ["Historical (Total Points)", "Projection (ep_next)"],
                    help="Historical uses season-to-date points. Projection uses the FPL API's "
                         "forward-looking ep_next estimate — better for 'who to buy now'.",
                )

            if "Avg_FDR" in filtered_data.columns:
                max_fdr = st.slider(
                    "Max acceptable average fixture difficulty (next 5 GWs)",
                    min_value=1.0, max_value=5.0, value=5.0, step=0.5,
                    help="FPL's own official FDR: 1 = easiest run of fixtures, 5 = hardest.",
                )
            else:
                max_fdr = None
                st.caption("Fixture-run data not found — run `python update_engine.py` to enable the difficulty filter.")

            eligible = filtered_data[filtered_data["Minutes"] >= min_minutes].copy()
            if only_available:
                eligible = eligible[eligible["Availability"] == "Available"]
            if max_fdr is not None:
                eligible = eligible[eligible["Avg_FDR"].fillna(max_fdr) <= max_fdr]

            # Re-kalkulasi VORP berdasarkan filter ranking pilihan (Historis atau Proyeksi)
            if rank_basis.startswith("Historical"):
                eligible["VORP"] = (eligible["Total Points"] / (eligible["Cost"] - eligible["Base_Cost"]).clip(lower=0.1)).round(2)
                rank_metric = "Total Points"
            else:
                eligible["ep_next"] = pd.to_numeric(eligible["ep_next"], errors="coerce").fillna(0.0)
                eligible["VORP"] = (eligible["ep_next"] / (eligible["Cost"] - eligible["Base_Cost"]).clip(lower=0.1)).round(2)
                rank_metric = "ep_next"

            if eligible.empty:
                st.warning("No players meet the current filters — try lowering the minutes threshold.")
            else:
                POSITION_SQUAD_SLOTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

                col_n, col_note = st.columns([1, 2])
                with col_n:
                    picks_per_position = st.slider(
                        "Value picks to show per position", min_value=3, max_value=15, value=8,
                    )
                with col_note:
                    st.caption(
                        f"Squad quota for reference: {POSITION_SQUAD_SLOTS['GKP']} GKP · "
                        f"{POSITION_SQUAD_SLOTS['DEF']} DEF · {POSITION_SQUAD_SLOTS['MID']} MID · "
                        f"{POSITION_SQUAD_SLOTS['FWD']} FWD (15 total)."
                    )

                st.markdown(f"#### Top {picks_per_position} VORP Picks per Position")
                position_tabs = st.tabs(["GKP", "DEF", "MID", "FWD"])
                for pos, pos_tab in zip(["GKP", "DEF", "MID", "FWD"], position_tabs):
                    with pos_tab:
                        # Sorting menggunakan VORP
                        pos_players = eligible[eligible["Position"] == pos].nlargest(
                            picks_per_position, "VORP"
                        )
                        if pos_players.empty:
                            st.info("No qualifying players at this position under the current filters.")
                            continue

                        st.caption(f"{len(pos_players)} shown · squad quota: {POSITION_SQUAD_SLOTS[pos]}")

                        players_list = list(pos_players.iterrows())
                        for row_start in range(0, len(players_list), 4):
                            row_players = players_list[row_start:row_start + 4]
                            metric_cols = st.columns(len(row_players))
                            for col, (_, player) in zip(metric_cols, row_players):
                                player_name = f"{player['First Name'][0]}. {player['Last Name']}"
                                chance = player.get("Chance_of_Playing")
                                chance_note = (
                                    f" | {int(chance)}% chance to play" if pd.notna(chance) else ""
                                )
                                fdr_val = player.get("Avg_FDR")
                                fdr_note = f" | Next-5 FDR {fdr_val:.1f}" if pd.notna(fdr_val) else ""
                                col.metric(
                                    label=f"{player_name} ({player['Team']})",
                                    value=f"{player['VORP']} VORP",
                                    delta=f"{player[rank_metric]:.1f} {rank_metric} | £{player['Cost']}M | {int(player['Minutes'])} min{chance_note}{fdr_note}",
                                    delta_color="off",
                                )

                st.markdown("#### VORP Comparison — Top 8 Overall")
                top_overall = eligible.nlargest(8, "VORP")
                value_bar = alt.Chart(top_overall).mark_bar(color=COLOR_ACCENT).encode(
                    x=alt.X("VORP:Q", title="Value Over Replacement"),
                    y=alt.Y("Last Name:N", sort="-x", title=None),
                    color=alt.Color("Position:N", scale=alt.Scale(range=["#e0b04f", "#4f8cf0", "#3fb27f", "#e2685f"])),
                    tooltip=["First Name", "Last Name", "Team", "Position", "Cost", rank_metric, "VORP", "Form"],
                ).properties(height=320)
                st.altair_chart(value_bar, use_container_width=True)

            st.markdown("---")
            sync_filters = st.checkbox(
                "Apply the minutes / availability filters above to the table and charts below",
                value=True,
                help="Off = table and charts show every player matching only the sidebar Team/Position filters.",
            )
            table_data = eligible if sync_filters and not eligible.empty else filtered_data

            st.subheader(f"Live Player Metrics ({len(table_data)} Players)")

            show_advanced = st.checkbox(
                "Show advanced columns (price trend, net transfers, upcoming fixtures list)",
                value=False,
                help="Keeps the table focused on the columns most people scan first; tick this for the full picture.",
            )

            sort_choice = st.radio(
                "Sort by", ["VORP", "Club", "Position", "Form"], horizontal=True,
            )

            if sort_choice == "VORP":
                table_sorted = table_data.sort_values("VORP", ascending=False).copy()
            elif sort_choice == "Club":
                table_sorted = table_data.sort_values(["Team", "VORP"], ascending=[True, False]).copy()
            elif sort_choice == "Form":
                table_sorted = table_data.sort_values("Form", ascending=False).copy()
            else:  # Position order
                table_sorted = table_data.copy()
                table_sorted["_pos_order"] = table_sorted["Position"].apply(
                    lambda p: POSITION_ORDER.index(p) if p in POSITION_ORDER else len(POSITION_ORDER)
                )
                table_sorted = table_sorted.sort_values(
                    ["_pos_order", "VORP"], ascending=[True, False]
                ).drop(columns="_pos_order")

            # Mengganti Value (Pts/Cost) dengan VORP, Form, dan xG_xA_Delta pada kolom utama
            core_cols = [
                "First Name", "Last Name", "Team", "Position", "Cost",
                "Total Points", "Form", "VORP", "xG_xA_Delta", "Ownership_Pct",
            ]
            if "Avg_FDR" in table_sorted.columns:
                core_cols.append("Avg_FDR")

            advanced_cols = [
                "Minutes", "Availability", "Chance_of_Playing",
                "Price_Change_GW", "Price_Change_Season", "Net_Transfers_GW",
            ]
            if "Fixture_Run" in table_sorted.columns:
                advanced_cols.append("Fixture_Run")

            display_cols = core_cols + advanced_cols if show_advanced else core_cols

            st.caption("🎨 Team column is colored by club identity — same color makes it easy to spot players from the same club at a glance.")

            st.dataframe(
                style_table_by_club(table_sorted[display_cols], team_col="Team"),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "First Name": st.column_config.TextColumn("First Name", width="small"),
                    "Last Name": st.column_config.TextColumn("Last Name", width="small"),
                    "Cost": st.column_config.NumberColumn("Cost", format="£%.1fM"),
                    "Total Points": st.column_config.NumberColumn("Total Pts"),
                    "Form": st.column_config.NumberColumn("Form", format="%.1f", help="Average points per match over the last 30 days."),
                    "VORP": st.column_config.NumberColumn(
                        "VORP", format="%.2f",
                        help="Value Over Replacement. Efficiency relative to the absolute minimum cost for the position.",
                    ),
                    "xG_xA_Delta": st.column_config.NumberColumn(
                        "Perf. Delta", format="%+.2f",
                        help="Negative = Underperforming (Due for a goal). Positive = Overperforming (Scoring from low-probability chances).",
                    ),
                    "Ownership_Pct": st.column_config.ProgressColumn(
                        "Ownership %", min_value=0, max_value=100, format="%.1f%%",
                    ),
                    "Avg_FDR": st.column_config.ProgressColumn(
                        "Next-5 FDR", min_value=1, max_value=5, format="%.1f",
                    ),
                    "Minutes": st.column_config.NumberColumn("Minutes"),
                    "Chance_of_Playing": st.column_config.NumberColumn("Chance to Play %", format="%.0f%%"),
                    "Price_Change_GW": st.column_config.NumberColumn("Price Δ (GW)", format="%+.1f"),
                    "Price_Change_Season": st.column_config.NumberColumn("Price Δ (Season)", format="%+.1f"),
                    "Net_Transfers_GW": st.column_config.NumberColumn("Net Transfers (GW)", format="%+d"),
                    "Fixture_Run": st.column_config.TextColumn("Upcoming Fixtures", width="large"),
                },
            )
            st.markdown("---")
            st.markdown("#### Player Value Distribution (Cost vs Points)")
            
            scatter_charts = create_scatter_plot(table_data)
            for chart in scatter_charts:
                st.altair_chart(chart, use_container_width=True)
            
            st.markdown("#### Accumulated Points per Team")
            st.altair_chart(create_team_bar_chart(table_data), use_container_width=True)

        # --- Advanced Fixture Matrix ---
        with tab_matrix:
            st.subheader("Advanced Fixture Matrix (Custom FDR)")
            st.info("Logarithmic Dixon-Coles difficulty matrix mapping opponent strength vs team vulnerability (1.0 = Easiest, 6.0+ = Brutal). Split into Attack and Defense modules.")

            fixtures_full_df = get_all_fixtures()
            team_form_data = get_team_rolling_form()

            if fixtures_full_df.empty:
                st.warning("Fixture data is empty. Run `update_engine.py` to populate full fixtures.")
            else:
                # Modulator Parameter Dinamis di Sidebar khusus tab ini
                with st.sidebar.expander("FDR Algorithm Parameters", expanded=True):
                    fdr_gamma = st.slider("Gamma (Scaling Factor)", 0.5, 2.5, 1.25, 0.05)
                    fdr_home_adv = st.slider("Home Advantage (Delta)", -1.0, 0.0, -0.25, 0.05)
                    fdr_away_adv = st.slider("Away Disadvantage (Delta)", 0.0, 1.0, 0.25, 0.05)

                fpl_teams = get_fpl_team_strengths()
                team_id_map = dict(zip(fpl_teams["id"], fpl_teams["name"]))

                dc_strengths = compute_team_strengths(team_form_data)
                
                if dc_strengths.empty or len(dc_strengths) < 20:
                    dc_strengths = pd.DataFrame({
                        "Team": fpl_teams["name"],
                        "Attack_Strength": fpl_teams["strength"] / 3.0,
                        "Defense_Vulnerability": 3.0 / fpl_teams["strength"]
                    })

                # 3. Definisi Array European Clubs Musim 2026/2027
                uefa_clubs = [
                    "Arsenal", "Man City", "Man Utd", "Aston Villa", 
                    "Liverpool", "Bournemouth", "Sunderland", 
                    "Crystal Palace", "Brighton"
                ]

                key_absences_map = {}
                for t_id, t_name in team_id_map.items():
                    team_roster = raw_data[raw_data["team_name"] == t_name].copy()
                    absent_players = team_roster[team_roster["status"].isin(["i", "s", "d", "u"])]
                    key_missing_count = sum(pd.to_numeric(absent_players["selected_by_percent"], errors="coerce") > 5.0)
                    key_absences_map[t_name] = min(key_missing_count * 0.05, 0.20)

                matrix_results = build_custom_fdr_matrix(
                    fixtures_full_df, dc_strengths, team_id_map, 
                    european_teams=uefa_clubs, key_absences=key_absences_map,
                    gamma=fdr_gamma, home_adv=fdr_home_adv, away_adv=fdr_away_adv
                )

                if matrix_results:
                    horizon_weeks = st.slider("Projection Horizon (Gameweeks)", min_value=5, max_value=38, value=12, step=1)
                    
                    def apply_fdr_colors(data_df, val_df):
                        css_df = pd.DataFrame("", index=data_df.index, columns=data_df.columns)
                        for col in data_df.columns:
                            if col in val_df.columns:
                                for idx in data_df.index:
                                    v = val_df.at[idx, col]
                                    label = str(data_df.at[idx, col])
                                    
                                    # Logika BGW (Blank Gameweek)
                                    if v >= 99.0 or label == "BLANK":
                                        css = "background-color: #000000; color: #555555; font-weight: 600;"
                                    elif v < 2.0: css = "background-color: #1a522a; color: #ffffff; font-weight: 600;"
                                    elif v < 3.0: css = "background-color: #27ae60; color: #ffffff; font-weight: 600;"
                                    elif v < 4.0: css = "background-color: #f1c40f; color: #000000; font-weight: 600;"
                                    elif v < 5.0: css = "background-color: #e67e22; color: #000000; font-weight: 600;"
                                    elif v < 6.0: css = "background-color: #e74c3c; color: #ffffff; font-weight: 600;"
                                    else: css = "background-color: #641e16; color: #ffffff; font-weight: 600;"
                                    
                                    # Logika DGW (Double Gameweek)
                                    if "\n" in label:
                                        css += " border: 3px solid #e0b04f;"
                                        
                                    css_df.at[idx, col] = css
                        return css_df

                    atk_lbl, atk_val = matrix_results["attack"]
                    def_lbl, def_val = matrix_results["defense"]
                    cols_to_show = list(atk_lbl.columns)[:horizon_weeks]

                    st.markdown("#### Attack FDR Matrix (Targeting Opponent Defense Vulnerability)")
                    styler_atk = atk_lbl[cols_to_show].style.apply(lambda df: apply_fdr_colors(df, atk_val[cols_to_show]), axis=None)
                    st.dataframe(styler_atk, use_container_width=True, height=500)

                    st.markdown("#### Defense FDR Matrix (Targeting Opponent Attack Strength)")
                    styler_def = def_lbl[cols_to_show].style.apply(lambda df: apply_fdr_colors(df, def_val[cols_to_show]), axis=None)
                    st.dataframe(styler_def, use_container_width=True, height=500)
                    
        # --- Custom xPts Model ---
        with tab_xpts:
            st.subheader("Custom xPts Projection Model")
            st.info(
                "Replaces the FPL API's own `ep_next` figure with an in-house projection combining "
                "Fixture Difficulty (FDR, from FPL's own team strength ratings), rotation risk from "
                "European fixtures, actual-vs-expected conversion ratio, and projected minutes (xMins)."
            )

            if season_info.get("season_status") == "PRE_SEASON":
                st.error(
                    "🚧 **This model is not meaningful right now.** The season hasn't started, so "
                    "the xG, goals, assists, and minutes feeding this projection are all leftover "
                    "values from last season carried over by the FPL API. Custom_xPts below is "
                    "arithmetically correct but reflects last season's form, not this one. Re-run "
                    "`update_engine.py` once gameweek 1 has kicked off for a meaningful projection."
                )

            team_strengths_df = get_fpl_team_strengths()
            next_opponent_df = get_next_opponent()
            opponent_defense_map = build_opponent_defense_map(team_strengths_df)

            with st.expander("Select teams with a midweek European fixture (rotation risk)", expanded=False):
                st.caption(
                    "The public FPL API doesn't expose Champions/Europa League fixtures, so this "
                    "list is manual — tick any club playing in Europe this week to apply the "
                    "rotation-risk discount to their squad's projection."
                )
                european_teams = set(st.multiselect("Teams in European competition this week", teams_list))

            xpts_data = filtered_data.rename(columns={"Last Name": "Player"}).copy()

            # Merge in xG/xA/goals/assists/rolling-minutes from the FPL bootstrap payload.
            # IMPORTANT: expected_goals/goals_scored/etc from the API are SEASON-CUMULATIVE
            # totals, but Custom_xPts projects a single upcoming gameweek — so these are
            # converted to a per-90-minute rate first. Feeding season totals directly here
            # was the earlier bug that made Custom_xPts blow up into triple digits.
            matches_played_est = (pd.to_numeric(raw_data["minutes"], errors="coerce") / 90).replace(0, np.nan)
            raw_metrics = pd.DataFrame({
                "id": raw_data["id"],
                "xG": pd.to_numeric(raw_data["expected_goals"], errors="coerce") / matches_played_est,
                "xA": pd.to_numeric(raw_data["expected_assists"], errors="coerce") / matches_played_est,
                "Goals": pd.to_numeric(raw_data["goals_scored"], errors="coerce") / matches_played_est,
                "Assists": pd.to_numeric(raw_data["assists"], errors="coerce") / matches_played_est,
                "rolling_minutes_per_match": raw_data["rolling_minutes_per_match"],
            }).fillna(0.0)

            xpts_data = xpts_data.merge(raw_metrics, on="id", how="left")
            xpts_data = xpts_data.rename(columns={"rolling_minutes_per_match": "Rolling_Minutes"})

            # Merge in each player's next opponent (by team)
            if not next_opponent_df.empty:
                xpts_data = xpts_data.merge(next_opponent_df[["Team", "Opponent"]], on="Team", how="left")
            else:
                xpts_data["Opponent"] = np.nan

            missing_opponent = xpts_data["Opponent"].isna().sum()
            missing_fdr = len(opponent_defense_map) == 0

            if missing_fdr or next_opponent_df.empty:
                st.warning(
                    "Fixture data not found — run `python update_engine.py` to populate team strength "
                    "ratings and next-opponent fixtures. Showing projections without the FDR/opponent "
                    "adjustment until then."
                )
            elif missing_opponent > 0:
                st.caption(f"{missing_opponent} player(s) have no upcoming fixture on file (team may be on a bye or season data incomplete).")

            custom_result = compute_custom_xpts(
                xpts_data, opponent_defense_map=opponent_defense_map,
                european_fixture_teams=european_teams,
            )

            top_n = st.slider("How many players to show", min_value=5, max_value=50, value=15, step=5)
            top_custom = custom_result.nlargest(top_n, "Custom_xPts")[
                ["Player", "Team", "Position", "Cost", "Opponent", "Custom_xPts", "ep_next",
                 "FDR_Multiplier", "Rotation_Multiplier", "xMins_Factor"]
            ]
            st.dataframe(top_custom, use_container_width=True)
            st.altair_chart(create_xpts_vs_cost_chart(custom_result, "Custom_xPts"), use_container_width=True)

        # --- MILP Squad Optimizer ---
        with tab_milp:
            st.subheader("Expected Points Maximization")
            st.info("Executes a deterministic PuLP solver to generate the mathematically optimal 15-man squad under budget, positional, and per-club constraints.")
            if st.button("Generate Mathematically Optimal Squad"):
                with st.spinner("Calculating optimal integer combinations..."):
                    optimal_squad = optimize_squad(display_data, budget=100.0, target_metric="ep_next")
                    if optimal_squad.empty:
                        st.error("No feasible solution found under current constraints.")
                    else:
                        st.success("Mathematical optimization complete.")
                        st.dataframe(optimal_squad, use_container_width=True)
                        total_cost = optimal_squad["Cost"].sum()
                        total_pts = optimal_squad["Total Points"].sum()
                        st.metric("Expected Points & Budget", f"{total_pts} Pts", f"£{total_cost:.1f}M Utilized")

        # --- Multi-Horizon Planner ---
        with tab_horizon:
            st.subheader("Multi-Horizon MILP Optimization (5–8 Gameweeks)")
            st.info(
                "Extends the single-week solver into a rolling dynamic program: it selects a squad "
                "for every gameweek in the horizon simultaneously, tracks transfers made between "
                "consecutive weeks, and only takes point hits when the projected gain outweighs the "
                "-4 penalty."
            )
            horizon_weeks = st.slider("Planning horizon (gameweeks)", 5, 8, 5)
            starting_ft = st.number_input("Free transfers currently banked", min_value=1, max_value=5, value=1)

            if st.button("Run Multi-Horizon Optimization"):
                with st.spinner(f"Solving joint MILP across {horizon_weeks} gameweeks..."):
                    horizon_df = display_data.copy()
                    # Without live per-gameweek fixture projections wired in yet,
                    # each week reuses ep_next as its projection column — replace
                    # these with per-week Custom_xPts outputs once fixtures are loaded.
                    projection_cols = []
                    for w in range(1, horizon_weeks + 1):
                        col_name = f"xPts_GW{w}"
                        horizon_df[col_name] = pd.to_numeric(horizon_df["ep_next"], errors="coerce").fillna(0.0)
                        projection_cols.append(col_name)

                    results = optimize_squad_multi_horizon(
                        horizon_df, projection_cols, budget=100.0,
                        starting_free_transfers=int(starting_ft),
                    )

                    if not results:
                        st.error("No feasible multi-horizon solution found.")
                    else:
                        st.success("Multi-horizon optimization complete.")
                        st.markdown("#### Gameweek-by-Gameweek Summary")
                        st.dataframe(results["summary"], use_container_width=True)

                        gw_choice = st.selectbox("Inspect squad for gameweek", projection_cols)
                        st.dataframe(results[gw_choice], use_container_width=True)

        # --- Stochastic Chip Evaluator ---
        with tab_chip:
            st.subheader("Live Squad Sync & Strategy Evaluation")
            fpl_team_id = st.text_input("Enter FPL Team ID (e.g., 123456)")
            if st.button("Evaluate Current Strategy"):
                if fpl_team_id:
                    with st.spinner("Fetching live data and running stochastic evaluation..."):
                        squad_ids = fetch_manager_squad(fpl_team_id)
                        if squad_ids:
                            my_squad = display_data[display_data["id"].isin(squad_ids)].copy()
                            st.success("Squad synchronized successfully.")
                            st.dataframe(my_squad[["First Name", "Last Name", "Team", "Cost"]], use_container_width=True)

                            st.markdown("#### Stochastic Chip Evaluator")
                            eval_results = evaluate_chip_strategy(raw_data, squad_ids)

                            e_col1, e_col2, e_col3 = st.columns(3)
                            e_col1.metric("Current xPts", eval_results["Current_Projected_Pts"])
                            e_col2.metric("Optimal xPts", eval_results["Optimal_Projected_Pts"])
                            e_col3.metric("Mathematical Delta", eval_results["Mathematical_Delta"])

                            st.info(f"**Decision Engine Recommendation:** {eval_results['Engine_Recommendation']}")
                        else:
                            st.error("Invalid Team ID, private squad, or FPL API rate limit exceeded.")
                else:
                    st.warning("Team ID is strictly required.")
    else:
        st.error("System failed to retrieve local FPL data. Run `python update_engine.py` first.")

# ===========================================================================
# MODULE 2 — TACTICAL FOOTBALL ANALYST
# ===========================================================================
elif app_module == "Tactical Football Analyst":
    st.title("🛡️ Professional Tactical Analyst Hub")
    st.markdown("Spatial deconstruction, machine learning scouting, and tactical quadrant matrix.")

    tab_tactical, tab_scouting, tab_spatial, tab_flank, tab_sim = st.tabs([
        "Pro Analytics & Lineups", "ML Scouting", "Spatial Pass Network",
        "Defensive Flank Matrix", "Match Simulator",
    ])

    with tab_tactical:
        st.markdown("### Advanced FBref Tactical Engine")
        st.info("Comprehensive multi-league metrics for Premier League and UEFA Champions League.")

        adv_data = load_advanced_metrics()

        if not adv_data.empty:
            team_list = sorted(adv_data["Team"].dropna().unique())
            selected_team = st.selectbox("Select Club for Tactical Deconstruction", team_list)

            col_lineup, col_playmaker = st.columns(2)
            with col_lineup:
                st.markdown("#### Projected Starting XI")
                st.dataframe(generate_predicted_lineup(adv_data, selected_team), use_container_width=True)
            with col_playmaker:
                st.markdown("#### Key Playmakers (Progressive Passes & xA Index)")
                st.dataframe(identify_key_playmakers(adv_data, selected_team), use_container_width=True)

            st.markdown("---")
            st.markdown("#### Tactical Style Matrix")
            st.info("Top-right quadrant identifies High Possession & High Threat teams.")
            st.pyplot(plot_tactical_quadrant(adv_data))
        else:
            st.warning("Advanced metrics database not found. Please run the ETL pipeline.")

    with tab_scouting:
        st.markdown("### K-Means Statistical Scouting")
        st.info("Unsupervised machine learning identifying hidden gems via Euclidean distance across 20+ tactical metrics.")

        adv_data = load_advanced_metrics()
        if not adv_data.empty:
            clustered_df, scaled_data, features, scaler = run_kmeans_clustering(adv_data)

            if not clustered_df.empty:
                player_list = sorted(clustered_df["Player"].unique())
                target_player = st.selectbox("Select Premium Player to Find Tactical Alternatives", player_list)

                st.markdown(f"#### Most Similar Profiles to {target_player}")
                similar_df = find_similar_players(target_player, clustered_df, scaled_data)

                if not similar_df.empty:
                    st.dataframe(similar_df, use_container_width=True)
                else:
                    st.warning("Player data isolated or invalid.")
            else:
                st.warning("Insufficient data for clustering. Try lowering the minimum-minutes filter.")
        else:
            st.warning("Advanced metrics database not found. Please run the ETL pipeline.")

    with tab_spatial:
        st.markdown("### Spatial Event Deconstruction")
        st.info("Displaying spatial shot event data using mplsoccer.")
        st.pyplot(draw_pass_network())

        with st.expander("About the Expected Threat (xT) grid model"):
            st.caption(
                "The platform includes a 16x12 zonal xT grid (`src/xt_model.py`) that values pitch "
                "zones by proximity-to-goal and centrality, then measures a player's threat "
                "contribution as the change in zone value between the start and end of a pass or "
                "carry. Wire in event data with start/end X-Y coordinates to activate per-player xT "
                "leaderboards."
            )
            xt_grid = build_xt_grid()
            st.dataframe(pd.DataFrame(xt_grid).round(2), use_container_width=True)

    with tab_flank:
        st.markdown("### Defensive Flank Vulnerability Matrix")
        st.info(
            "Splits conceded shots into Left / Central / Right thirds of the pitch to flag which "
            "flank a defense leaks the most expected goals against — a direct input for exploiting "
            "an opponent's weak side."
        )
        import os
        from src.config import PATH_EVENT_RAW
        if os.path.exists(PATH_EVENT_RAW):
            shots_df = pd.read_parquet(PATH_EVENT_RAW, engine="pyarrow")
            team_col = next((c for c in ["team", "Team"] if c in shots_df.columns), None)
            y_col = next((c for c in ["Y", "y"] if c in shots_df.columns), None)
            if team_col and y_col:
                flank_matrix = compute_defensive_flank_vulnerability(shots_df, team_col=team_col, y_col=y_col)
                st.dataframe(flank_matrix, use_container_width=True)
            else:
                st.warning("Shot event data is missing the team or Y-coordinate column needed for this view.")
        else:
            st.warning("Event data not found. Please run the ETL pipeline.")

    with tab_sim:
        st.markdown("### Match Simulator (Dixon-Coles Poisson & Monte Carlo)")

        team_form = get_team_rolling_form()
        use_dynamic = not team_form.empty

        if use_dynamic:
            strengths = compute_team_strengths(team_form)
            team_options = sorted(strengths["Team"].unique())
            col_h, col_a = st.columns(2)
            home_team = col_h.selectbox("Home Team", team_options, index=0)
            away_team = col_a.selectbox("Away Team", team_options, index=min(1, len(team_options) - 1))
            home_xg, away_xg = expected_goals(home_team, away_team, strengths)
            st.caption(f"Derived from rolling form: Home xG={home_xg:.2f}, Away xG={away_xg:.2f}")
        else:
            st.info("Rolling team-form table not found — falling back to manual xG sliders. Run `update_engine.py` to enable dynamic team-strength ratings.")
            col1, col2 = st.columns(2)
            home_xg = col1.slider("Home Team Expected Goals (xG)", 0.1, 4.0, 1.5, 0.1)
            away_xg = col2.slider("Away Team Expected Goals (xG)", 0.1, 4.0, 1.2, 0.1)

        score_matrix = generate_score_matrix(home_xg, away_xg)
        mc_home, mc_draw, mc_away = run_monte_carlo(home_xg, away_xg, 10000)
        home_cs, away_cs = calculate_clean_sheet_probability(score_matrix)

        st.markdown("#### 10,000x Monte Carlo Stable Probabilities")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Home Win", f"{mc_home * 100:.1f}%")
        m_col2.metric("Draw", f"{mc_draw * 100:.1f}%")
        m_col3.metric("Away Win", f"{mc_away * 100:.1f}%")

        st.markdown("#### Clean Sheet Probability")
        cs_col1, cs_col2 = st.columns(2)
        cs_col1.metric("Home Clean Sheet", f"{home_cs * 100:.1f}%")
        cs_col2.metric("Away Clean Sheet", f"{away_cs * 100:.1f}%")

        st.markdown("#### Exact Score Probability Matrix (Dixon-Coles adjusted)")
        st.dataframe(
            score_matrix.style.background_gradient(cmap="YlGn", axis=None).format("{:.2%}"),
            use_container_width=True,
        )