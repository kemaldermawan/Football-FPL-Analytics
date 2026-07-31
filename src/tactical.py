import matplotlib.pyplot as plt
from mplsoccer import Pitch

def draw_pass_network():
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(8, 5))
    
    x_coords = [10, 30, 30, 30, 30, 50, 50, 50, 70, 70, 70]
    y_coords = [40, 20, 60, 40, 80, 40, 20, 60, 40, 20, 60]
    
    pitch.scatter(x_coords, y_coords, ax=ax, s=250, color='#ea6969', edgecolors='black', zorder=2)
    
    pitch.lines(x_coords[0], y_coords[0], x_coords[3], y_coords[3], ax=ax, lw=2, color='white', zorder=1, alpha=0.7)
    pitch.lines(x_coords[3], y_coords[3], x_coords[5], y_coords[5], ax=ax, lw=3, color='white', zorder=1, alpha=0.8)
    
    ax.set_title("Tactical Spatial Analysis (Simulation)", color='white', fontsize=14)
    fig.patch.set_facecolor('#22312b')
    
    return fig