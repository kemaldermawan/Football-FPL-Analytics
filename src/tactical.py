"""
Spatial event deconstruction: shot maps and passing networks. The passing
network uses graph theory (via networkx) to compute eigenvector centrality
— identifying which player sits at the structural heart of a team's
build-up, not just who touches the ball most.
"""
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import os
import streamlit as st

from src.config import PATH_EVENT_RAW, COLOR_PITCH, COLOR_LINE, COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TEXT


def draw_pass_network():
    if not os.path.exists(PATH_EVENT_RAW):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Data not found", ha="center", va="center")
        return fig

    shots_df = pd.read_parquet(PATH_EVENT_RAW, engine="pyarrow")

    possible_x = ["X", "x", "location_x", "shot_x", "start_x"]
    possible_y = ["Y", "y", "location_y", "shot_y", "start_y"]

    x_col = next((c for c in possible_x if c in shots_df.columns), None)
    y_col = next((c for c in possible_y if c in shots_df.columns), None)

    if not x_col or not y_col:
        st.error(f"Coordinate columns not found. System detected the following schema: {shots_df.columns.tolist()}")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Coordinate resolution error", ha="center", va="center", color="red")
        return fig

    shots_df = shots_df.copy()
    shots_df[x_col] = pd.to_numeric(shots_df[x_col], errors="coerce") * 120
    shots_df[y_col] = pd.to_numeric(shots_df[y_col], errors="coerce") * 80

    pitch = Pitch(pitch_type="statsbomb", pitch_color=COLOR_PITCH, line_color=COLOR_LINE)
    fig, ax = pitch.draw(figsize=(8, 5))

    res_col = next((c for c in ["result", "Result", "outcome"] if c in shots_df.columns), None)

    if res_col:
        goals = shots_df[shots_df[res_col] == "Goal"]
        misses = shots_df[shots_df[res_col] != "Goal"]
    else:
        goals = pd.DataFrame(columns=shots_df.columns)
        misses = shots_df

    pitch.scatter(misses[x_col], misses[y_col], ax=ax, s=100, color=COLOR_NEGATIVE,
                  edgecolors="black", alpha=0.6, label="Miss")
    pitch.scatter(goals[x_col], goals[y_col], ax=ax, s=200, color=COLOR_POSITIVE,
                  edgecolors="black", zorder=2, label="Goal")

    ax.set_title("Tactical Spatial Analysis (Shot Map)", color=COLOR_TEXT, fontsize=14)
    ax.legend(loc="lower left")
    fig.patch.set_facecolor(COLOR_PITCH)

    return fig


def build_passing_network(passes_df: pd.DataFrame, passer_col: str = "player",
                           recipient_col: str = "pass_recipient",
                           start_x_col: str = "start_x", start_y_col: str = "start_y") -> tuple:
    """Builds a directed weighted graph of completed passes and computes
    eigenvector centrality per player — a measure of how structurally
    important a player is to ball progression (well-connected to other
    well-connected players), not just raw pass volume.

    Returns (node_positions_df, edges_df) ready for plotting, where
    node_positions_df includes an `Eigenvector_Centrality` column.
    """
    import networkx as nx

    df = passes_df.dropna(subset=[passer_col, recipient_col]).copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    edge_weights = df.groupby([passer_col, recipient_col]).size().reset_index(name="weight")

    G = nx.DiGraph()
    for _, row in edge_weights.iterrows():
        G.add_edge(row[passer_col], row[recipient_col], weight=row["weight"])

    try:
        centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        centrality = {node: 0.0 for node in G.nodes}

    avg_positions = df.groupby(passer_col).agg(
        avg_x=(start_x_col, "mean"),
        avg_y=(start_y_col, "mean"),
        Pass_Volume=(passer_col, "count"),
    ).reset_index()

    avg_positions["Eigenvector_Centrality"] = avg_positions[passer_col].map(centrality).fillna(0)

    return avg_positions, edge_weights
