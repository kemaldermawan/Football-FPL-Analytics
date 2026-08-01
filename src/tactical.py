import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import os
import streamlit as st

def draw_pass_network():
    file_path = os.path.join('data', 'epl_raw.parquet')
    if not os.path.exists(file_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
        return fig
        
    shots_df = pd.read_parquet(file_path, engine='pyarrow')
    
    possible_x = ['X', 'x', 'location_x', 'shot_x', 'start_x']
    possible_y = ['Y', 'y', 'location_y', 'shot_y', 'start_y']
    
    x_col = next((col for col in possible_x if col in shots_df.columns), None)
    y_col = next((col for col in possible_y if col in shots_df.columns), None)
    
    if not x_col or not y_col:
        st.error(f"Coordinate columns not found. System detected the following schema: {shots_df.columns.tolist()}")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Coordinate resolution error', ha='center', va='center', color='red')
        return fig

    # Konversi tipe data untuk menghindari galat operasi matematis pada string
    shots_df[x_col] = pd.to_numeric(shots_df[x_col], errors='coerce') * 120
    shots_df[y_col] = pd.to_numeric(shots_df[y_col], errors='coerce') * 80
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(8, 5))
    
    # Deteksi dinamis untuk kolom hasil tembakan
    res_col = next((col for col in ['result', 'Result', 'outcome'] if col in shots_df.columns), None)
    
    if res_col:
        goals = shots_df[shots_df[res_col] == 'Goal']
        misses = shots_df[shots_df[res_col] != 'Goal']
    else:
        goals = pd.DataFrame(columns=shots_df.columns)
        misses = shots_df
    
    pitch.scatter(misses[x_col], misses[y_col], ax=ax, s=100, color='#ea6969', edgecolors='black', alpha=0.6, label='Miss')
    pitch.scatter(goals[x_col], goals[y_col], ax=ax, s=200, color='#69ea82', edgecolors='black', zorder=2, label='Goal')
    
    ax.set_title("Tactical Spatial Analysis (Actual Shots Data)", color='white', fontsize=14)
    ax.legend(loc='lower left')
    fig.patch.set_facecolor('#22312b')
    
    return fig