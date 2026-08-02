"""
Chart rendering for the FPL Market Analysis tab: Altair interactive charts
(themed to match the dashboard's dark, neutral aesthetic) and a Matplotlib
percentile radar ("pizza chart") for individual player profiling.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import altair as alt

from src.config import COLOR_PITCH, COLOR_ACCENT, COLOR_TEXT, COLOR_LINE

# Distinct, non-monochrome palette for the four playing positions —
# avoids the "everything is one shade of green" look of a single
# sequential color scheme.
POSITION_COLOR_RANGE = ["#e0b04f", "#4f8cf0", "#3fb27f", "#e2685f"]  # GKP, DEF, MID, FWD


def _apply_dark_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure(background=COLOR_PITCH).configure_axis(
        labelColor=COLOR_TEXT, titleColor=COLOR_TEXT, gridColor="#2a2f3a"
    ).configure_legend(labelColor=COLOR_TEXT, titleColor=COLOR_TEXT)


def create_scatter_plot(data: pd.DataFrame) -> alt.Chart:
    scatter_chart = alt.Chart(data).mark_circle(size=70, opacity=0.85).encode(
        x=alt.X("Cost", scale=alt.Scale(zero=False), title="Cost (£M)"),
        y=alt.Y("Total Points", scale=alt.Scale(zero=False)),
        color=alt.Color("Position", scale=alt.Scale(range=POSITION_COLOR_RANGE)),
        tooltip=["First Name", "Last Name", "Team", "Cost", "Total Points", "Value (Pts/Cost)"],
    ).interactive().properties(height=380)
    return _apply_dark_theme(scatter_chart)


def create_team_bar_chart(data: pd.DataFrame) -> alt.Chart:
    bar_chart = alt.Chart(data).mark_bar(color=COLOR_ACCENT).encode(
        x=alt.X("sum(Total Points):Q", title="Accumulated Points"),
        y=alt.Y("Team:N", sort="-x", title="Team"),
        tooltip=["Team", "sum(Total Points)"],
    ).interactive().properties(height=380)
    return _apply_dark_theme(bar_chart)


def create_xpts_vs_cost_chart(data: pd.DataFrame, xpts_col: str = "Custom_xPts") -> alt.Chart:
    """Market-anomaly scatter: Custom xPts projection vs Cost, to spot
    undervalued players the crowd-sourced ep_next figure might miss."""
    chart = alt.Chart(data).mark_circle(size=70, opacity=0.85).encode(
        x=alt.X("Cost", scale=alt.Scale(zero=False), title="Cost (£M)"),
        y=alt.Y(xpts_col, scale=alt.Scale(zero=False), title="Custom Projected Points"),
        color=alt.Color("Position", scale=alt.Scale(range=POSITION_COLOR_RANGE)),
        tooltip=["Player" if "Player" in data.columns else "Last Name", "Team", "Cost", xpts_col],
    ).interactive().properties(height=380)
    return _apply_dark_theme(chart)


def create_pizza_chart(player_data: pd.Series, position_data: pd.DataFrame) -> plt.Figure:
    params = ["Cost", "Total Points", "Value (Pts/Cost)"]
    percentiles = []

    for p in params:
        pct = stats.percentileofscore(position_data[p].dropna(), player_data[p])
        percentiles.append(pct)

    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    percentiles += percentiles[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_PITCH)
    ax.set_facecolor(COLOR_PITCH)

    ax.fill(angles, percentiles, color=COLOR_ACCENT, alpha=0.4)
    ax.plot(angles, percentiles, color=COLOR_ACCENT, linewidth=2)

    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color=COLOR_TEXT, size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params, color=COLOR_TEXT, size=10)

    ax.spines["polar"].set_color(COLOR_LINE)
    ax.tick_params(colors=COLOR_TEXT)
    ax.set_title(f"Percentile Radar: {player_data['Last Name']}", color=COLOR_TEXT, fontsize=12, pad=20)

    return fig