#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file.
df = pd.read_csv("pattern_formation_times.csv")

# Separate points where a pattern was detected (t_pattern != 10) and not detected.
df_pattern = df[df.t_pattern != 50]
df_no_pattern = df[df.t_pattern == 50]

plt.figure(figsize=(12,7))

# Plot runs with pattern formation; set the color scale from 0 to 10.
sc = plt.scatter(
    df_pattern['ratio'],
    df_pattern['beta'],
    c=df_pattern['t_pattern'],
    cmap='YlOrRd',
    s=45,
    edgecolor='k',
    vmin=0,          # Minimum of color scale
    vmax=4,        # Maximum of color scale
    label='Pattern formed (t < 50)'
)

# Plot runs with no pattern formation in a distinct color and marker.
plt.scatter(
    df_no_pattern['ratio'],
    df_no_pattern['beta'],
    color='red',
    marker='x',
    s=45,
    label='No pattern formed'
)

plt.xlabel(r'$\frac{D_{\alpha}}{D_{\beta}}$')
plt.xscale('log')
plt.ylabel('Beta')
plt.title('Time to Pattern Formation in the Reaction-Diffusion System')

# Create colorbar (matching the scatter plot above).
cbar = plt.colorbar(sc)
cbar.set_label('Time to Pattern Formation (s)')

# Position the legend and note outside the main axes.
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0)
note = r"Note: ${\alpha} = 0.02$"
plt.text(
    1.21, 0.85, note,
    fontsize=10,
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
)



plt.tight_layout()
plt.savefig("pattern_formation_heatmap_beta_diffusion.png", dpi=400, bbox_inches='tight')
plt.show()
