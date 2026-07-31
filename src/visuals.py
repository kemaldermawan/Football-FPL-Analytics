import altair as alt
import pandas as pd

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