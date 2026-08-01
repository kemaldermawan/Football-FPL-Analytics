import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import altair as alt

def create_scatter_plot(data: pd.DataFrame) -> alt.Chart:
    scatter_chart = alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X('Cost', scale=alt.Scale(zero=False)),
        y=alt.Y('Total Points', scale=alt.Scale(zero=False)),
        color='Position',
        tooltip=['First Name', 'Last Name', 'Team', 'Cost', 'Total Points', 'Value (Pts/Cost)']
    ).interactive()
    return scatter_chart

def create_team_bar_chart(data: pd.DataFrame) -> alt.Chart:
    bar_chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('sum(Total Points):Q', title='Accumulated Points'),
        y=alt.Y('Team:N', sort='-x', title='Team'),
        color=alt.Color('Team:N', legend=None),
        tooltip=['Team', 'sum(Total Points)']
    ).interactive()
    return bar_chart

def create_pizza_chart(player_data: pd.Series, position_data: pd.DataFrame) -> plt.Figure:
    params = ['Cost', 'Total Points', 'Value (Pts/Cost)']
    percentiles = []
    
    for p in params:
        pct = stats.percentileofscore(position_data[p].dropna(), player_data[p])
        percentiles.append(pct)
        
    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    percentiles += percentiles[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#22312b')
    ax.set_facecolor('#22312b')
    
    ax.fill(angles, percentiles, color='#69ea82', alpha=0.5)
    ax.plot(angles, percentiles, color='#69ea82', linewidth=2)
    
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], color='white', size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params, color='white', size=10)
    
    ax.spines['polar'].set_color('white')
    ax.tick_params(colors='white')
    ax.set_title(f"Percentile Radar: {player_data['Last Name']}", color="white", fontsize=12, pad=20)
    
    return fig