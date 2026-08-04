"""
Chart rendering for the FPL Market Analysis tab: Altair interactive charts
and a Matplotlib percentile radar for individual player profiling.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import altair as alt

from src.config import COLOR_PITCH, COLOR_ACCENT, COLOR_TEXT, COLOR_LINE

POSITION_COLOR_RANGE = ["#e0b04f", "#4f8cf0", "#3fb27f", "#e2685f"]  # GKP DEF MID FWD

# Club identity colors — approximate brand primaries, not official trademarks.
# Unknown clubs fall back to a neutral slate.
TEAM_COLORS = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#DA291C",
    "Brentford": "#E30613",
    "Brighton": "#0057B8",
    "Burnley": "#6C1D45",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Ipswich": "#3A64A3",
    "Leeds": "#FFCD00",
    "Leicester": "#003090",
    "Liverpool": "#C8102E",
    "Luton": "#F78F1E",
    "Man City": "#6CABDD",
    "Man Utd": "#DA291C",
    "Newcastle": "#241F20",
    "Norwich": "#00A650",
    "Nott'm Forest": "#DD0000",
    "Sheffield Utd": "#EE2737",
    "Southampton": "#D71920",
    "Spurs": "#132257",
    "Sunderland": "#EB172B",
    "Watford": "#FBEE23",
    "West Brom": "#122F67",
    "West Ham": "#7A263A",
    "Wolves": "#FDB913",
}

def get_contrasting_text_color(hex_color: str) -> str:
    """Returns '#000000' or '#ffffff', whichever reads better on the
    given background, using the standard relative-luminance heuristic."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#ffffff"
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.55 else "#ffffff"

def style_team_column(df: pd.DataFrame, team_col: str = "Team"):
    """Returns a pandas Styler that colors ONLY the Team cell background
    using each club's identity color, with auto-contrast text. Unlisted
    clubs get a neutral fallback so no club ever breaks the table."""
    def _style_row(row):
        bg = TEAM_COLORS.get(row.get(team_col, ""), "#3a3f4a")
        text = get_contrasting_text_color(bg)
        style = f"background-color: {bg}; color: {text}; font-weight: 600;"
        return [style if col == team_col else "" for col in row.index]
    return df.style.apply(_style_row, axis=1)

def _apply_dark_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure(background=COLOR_PITCH).configure_axis(
        labelColor=COLOR_TEXT, titleColor=COLOR_TEXT, gridColor="#2a2f3a", grid=True
    ).configure_legend(labelColor=COLOR_TEXT, titleColor=COLOR_TEXT)

def create_scatter_plot(data: pd.DataFrame) -> list:
    """Cost vs Total Points scatter.
    Returns a list of charts with strictly enforced global domains for consistency,
    increased height to prevent cropping, and explicit detail mapping to prevent aggregation."""
    df_chart = data.copy()
    
    df_chart["Ownership_Pct"] = pd.to_numeric(df_chart["Ownership_Pct"], errors="coerce").fillna(0.0)
    df_chart["Cost"] = pd.to_numeric(df_chart["Cost"], errors="coerce").fillna(0.0)
    df_chart["Total Points"] = pd.to_numeric(df_chart["Total Points"], errors="coerce").fillna(0.0)

    min_cost, max_cost = df_chart["Cost"].min(), df_chart["Cost"].max()
    min_pts, max_pts = df_chart["Total Points"].min(), df_chart["Total Points"].max()
    
    x_domain = [max(3.5, min_cost - 0.5), max_cost + 0.5]
    y_domain = [min_pts - 10, max_pts + 15]

    charts = []
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_data = df_chart[df_chart["Position"] == pos]
        if pos_data.empty:
            continue

        chart = alt.Chart(pos_data).mark_circle(opacity=0.8).encode(
            x=alt.X(
                "Cost:Q", 
                scale=alt.Scale(domain=x_domain, clamp=True),
                title="Cost (£M)"
            ),
            y=alt.Y(
                "Total Points:Q", 
                scale=alt.Scale(domain=y_domain, clamp=True),
                title="Total Points"
            ),
            color=alt.Color(
                "Position:N",
                scale=alt.Scale(domain=["GKP", "DEF", "MID", "FWD"], range=POSITION_COLOR_RANGE),
                legend=None,
            ),
            size=alt.Size(
                "Ownership_Pct:Q",
                scale=alt.Scale(domain=[0, 100], range=[80, 500]),
                legend=None,
            ),
            detail=[alt.Detail("First Name:N"), alt.Detail("Last Name:N"), alt.Detail("Team:N")],
            tooltip=[
                alt.Tooltip("First Name:N"), alt.Tooltip("Last Name:N"),
                alt.Tooltip("Team:N"), alt.Tooltip("Position:N"),
                alt.Tooltip("Cost:Q", format="£.1f"),
                alt.Tooltip("Total Points:Q"),
                alt.Tooltip("Value (Pts/Cost):Q", format=".2f"),
                alt.Tooltip("Ownership_Pct:Q", title="Ownership %", format=".1f"),
            ]
        ).properties(
            height=280,
            title=alt.TitleParams(text=f"{pos}", color=COLOR_TEXT, anchor="start", fontSize=15, dy=-10)
        )
        
        charts.append(_apply_dark_theme(chart))

    if not charts:
        empty = alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No data available"))
        return [_apply_dark_theme(empty)]

    return charts

def create_team_bar_chart(data: pd.DataFrame) -> alt.Chart:
    """Total points accumulated per team — full-width horizontal bar,
    colored by official club identity color where available."""
    team_colors_list = [TEAM_COLORS.get(t, "#4f8cf0") for t in sorted(data["Team"].unique())]

    bar_chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("sum(Total Points):Q", title="Accumulated Points"),
        y=alt.Y("Team:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color(
            "Team:N",
            scale=alt.Scale(
                domain=sorted(data["Team"].unique()),
                range=team_colors_list,
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Team:N"),
            alt.Tooltip("sum(Total Points):Q", title="Total Points"),
        ],
    ).properties(height=max(550, len(data["Team"].unique()) * 30)).interactive()
    return _apply_dark_theme(bar_chart)

def create_xpts_vs_cost_chart(data: pd.DataFrame, xpts_col: str = "Custom_xPts") -> alt.Chart:
    """Custom xPts projection vs Cost scatter for spotting undervalued players."""
    player_col = "Player" if "Player" in data.columns else "Last Name"
    chart = alt.Chart(data).mark_circle(opacity=0.85).encode(
        x=alt.X("Cost:Q", scale=alt.Scale(zero=False), title="Cost (£M)"),
        y=alt.Y(f"{xpts_col}:Q", scale=alt.Scale(zero=False), title="Custom Projected Points"),
        color=alt.Color(
            "Position:N",
            scale=alt.Scale(
                domain=["GKP", "DEF", "MID", "FWD"],
                range=POSITION_COLOR_RANGE,
            ),
        ),
        size=alt.value(80),
        tooltip=[
            alt.Tooltip(f"{player_col}:N"),
            alt.Tooltip("Team:N"),
            alt.Tooltip("Cost:Q", format="£.1f"),
            alt.Tooltip(f"{xpts_col}:Q", format=".2f"),
        ],
    ).properties(height=480).interactive()
    return _apply_dark_theme(chart)

def create_pizza_chart(player_data: pd.Series, position_data: pd.DataFrame) -> plt.Figure:
    params = ["Cost", "Total Points", "Value (Pts/Cost)"]
    percentiles = [
        stats.percentileofscore(position_data[p].dropna(), player_data[p])
        for p in params
    ]

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