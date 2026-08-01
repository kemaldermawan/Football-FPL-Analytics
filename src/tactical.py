import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import os

def draw_pass_network():
    file_path = os.path.join('data', 'epl_raw.parquet')
    if not os.path.exists(file_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
        return fig
        
    shots_df = pd.read_parquet(file_path, engine='pyarrow')
    
    shots_df['X'] = shots_df['X'] * 120
    shots_df['Y'] = shots_df['Y'] * 80
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(8, 5))
    
    goals = shots_df[shots_df['result'] == 'Goal']
    misses = shots_df[shots_df['result'] != 'Goal']
    
    pitch.scatter(misses['X'], misses['Y'], ax=ax, s=100, color='#ea6969', edgecolors='black', alpha=0.6, label='Miss')
    pitch.scatter(goals['X'], goals['Y'], ax=ax, s=200, color='#69ea82', edgecolors='black', zorder=2, label='Goal')
    
    ax.set_title("Tactical Spatial Analysis (Actual Shots Data)", color='white', fontsize=14)
    ax.legend(loc='lower left')
    fig.patch.set_facecolor('#22312b')
    
    return fig