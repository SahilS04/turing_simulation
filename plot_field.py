#!/usr/bin/env python3
"""
plot_field.py

This script reads CSV files representing simulation snapshots (e.g., "A_t*.csv")
and creates an MP4 video showing the evolution of the heatmap.
A consistent colorbar is added to every frame (with constant limits across the video),
and the frame is rendered via an in‑memory PNG buffer.
 
Usage:
    python plot_field.py

Requirements:
    - numpy
    - matplotlib
    - imageio
    - imageio-ffmpeg (install via pip)
"""

import glob
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio

def read_csv_field(filename):
    """Read a 2D field from a CSV file."""
    try:
        data = np.loadtxt(filename, delimiter=',')
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        data = None
    return data

def create_frame(data, global_min, global_max, idx, total_frames, dpi=100):
    """
    Create a plot (with a constant color scale) from a 2D field,
    save it to an in-memory PNG buffer, and return it as an RGB image array.

    Parameters:
        data         : 2D NumPy array to plot.
        global_min   : Global minimum to set the colormap lower limit.
        global_max   : Global maximum to set the colormap upper limit.
        idx          : Current frame index.
        total_frames : Total number of frames (used for title).
        dpi          : Resolution for saved figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot the heatmap with fixed color limits.
    im = ax.imshow(data, vmin=global_min, vmax=global_max,
                   cmap='viridis', interpolation='nearest', origin='lower')
    # Set title (including current frame number)
    ax.set_title(f"Activator Evolution (Frame {idx+1}/{total_frames})", fontsize=12)
    ax.axis('off')
    # Add a colorbar that uses the fixed normalization.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout(pad=1.0)
    
    # Save the figure to an in-memory PNG using BytesIO.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    # Read the PNG buffer as an image (RGB uint8 array).
    frame = imageio.imread(buf)
    buf.close()
    return frame

def main():
    # Get a sorted list of CSV files matching the pattern.
    csv_files = sorted(glob.glob("A_t*.csv"))
    if not csv_files:
        print("No CSV files found. Ensure simulation CSVs are in this directory.")
        return

    # Read in all fields and determine the global data range.
    fields = []
    global_min = np.inf
    global_max = -np.inf
    for filename in csv_files:
        data = read_csv_field(filename)
        if data is not None:
            fields.append(data)
            global_min = min(global_min, np.min(data))
            global_max = max(global_max, np.max(data))
    print(f"Global range for activator: {global_min:.4f} to {global_max:.4f}")

    # Create a list of frames (one per field) using the in-memory buffer method.
    frames = []
    total_frames = len(fields)
    for idx, data in enumerate(fields):
        frame = create_frame(data, global_min, global_max, idx, total_frames)
        frames.append(frame)
    
    # Write all frames into an MP4 video using imageio.
    video_filename = "alpha2_beta5_diff100_big_random.mp4"
    fps = 50  # Adjust frames per second as needed.
    with imageio.get_writer(video_filename, fps=fps, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved to {video_filename}")

if __name__ == "__main__":
    main()
