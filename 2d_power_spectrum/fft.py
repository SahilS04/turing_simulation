#!/usr/bin/env python3
"""
fft.py

Usage:
  python fft.py

This script:
  - Loads every CSV snapshot for the activator field A with filenames matching A_t*.csv.
  - Computes the 2D FFT and its radially averaged power spectrum.
  - Determines global x (radial frequency) and y (log-scale average power) axis limits so that these remain constant across plots.
  - Creates a line plot for each CSV showing the radially averaged power spectrum on a log scale (with a title indicating the source file).
  - Compiles the radially averaged plots into an MP4 video showing how the spectrum evolves over time.
  - Saves a PNG image of the power spectrum from the last time step.
  - Additionally, creates a 2D plot of the full power spectrum (using logarithmic scaling), compiles these plots into an MP4 video of the 2D evolution,
    and saves a PNG image of the final 2D spectrum.
    
Requires: numpy, matplotlib, imageio, imageio-ffmpeg
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio

def load_field(filename):
    """Load the CSV file into a 2D array."""
    return np.loadtxt(filename, delimiter=",")

def radial_average_power_spectrum(A, nbins=100):
    """
    Compute 2D FFT of A and then compute its radially averaged power spectrum.

    Returns:
      r_centers : array of radial frequency bin centers.
      P_rad     : radially averaged power in each bin.
      power2d   : 2D power spectrum.
    """
    # Remove DC offset
    A0 = A - np.mean(A)
    F = np.fft.fft2(A0)
    Fshift = np.fft.fftshift(F)
    power2d = np.abs(Fshift)**2

    ny, nx = A.shape
    # Create frequency coordinate arrays (using normalized frequencies)
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx, ky)
    R = np.sqrt(KX**2 + KY**2)

    # Flatten arrays for binning
    R_flat = R.ravel()
    P_flat = power2d.ravel()

    rmax = np.max(R_flat)
    rbins = np.linspace(0, rmax, nbins+1)
    P_rad = np.zeros(nbins)
    r_centers = np.zeros(nbins)
    for i in range(nbins):
        r1, r2 = rbins[i], rbins[i+1]
        mask = (R_flat >= r1) & (R_flat < r2)
        if np.any(mask):
            P_rad[i] = np.mean(P_flat[mask])
        r_centers[i] = 0.5 * (r1 + r2)
    return r_centers, P_rad, power2d

def capture_frame(fig):
    """
    Capture the current state of the figure as an RGB image array.
    This function uses buffer_rgba (recommended in newer Matplotlib versions)
    and then discards the alpha channel.
    """
    fig.canvas.draw()  # Render the canvas
    rgba = np.array(fig.canvas.buffer_rgba())
    return rgba[..., :3]

def main():
    # Define CSV file pattern and list all matching files.
    csv_pattern = "A_t*.csv"
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        print(f"No CSV files found for pattern '{csv_pattern}'. Check your working directory.")
        sys.exit(1)

    print("CSV files found for analysis:")
    for csv_file in csv_files:
        print(f"  {csv_file}")

    # First pass: process each file to compute spectra and determine global axis limits.
    # For the radially averaged plots, we now work with log10(P_rad + 1).
    data_list = []
    global_ymin = np.inf
    global_ymax = -np.inf
    global_xmin = np.inf
    global_xmax = -np.inf
    global_2d_min = np.inf
    global_2d_max = -np.inf

    for csv_file in csv_files:
        print(f"Processing {csv_file} ...")
        A = load_field(csv_file)
        r, P_rad, power2d = radial_average_power_spectrum(A)
        # Compute the log-scale version for the radially averaged power spectrum.
        P_rad_log = np.log10(P_rad + 1)
        data_list.append((csv_file, r, P_rad_log, power2d))
        global_ymin = min(global_ymin, np.min(P_rad_log))
        global_ymax = max(global_ymax, np.max(P_rad_log))
        global_xmin = min(global_xmin, np.min(r))
        global_xmax = max(global_xmax, np.max(r))
        
        # Use log10 scaling for the 2D power spectrum.
        log_power2d = np.log10(power2d + 1)
        global_2d_min = min(global_2d_min, np.min(log_power2d))
        global_2d_max = max(global_2d_max, np.max(log_power2d))

    print(f"Global x-axis limits: ({global_xmin:.2f}, {global_xmax:.2f})")
    print(f"Global y-axis limits (log scale): ({global_ymin:.2e}, {global_ymax:.2e})")
    print(f"Global 2D log(power) limits: ({global_2d_min:.2f}, {global_2d_max:.2f})")

    # Set constant axis limits for the radially averaged plots.
    x_limits = (global_xmin, global_xmax)
    y_limits = (global_ymin, global_ymax)

    # ------------------------
    # Create frames for the radially averaged power spectrum using a single figure.
    fig_line, ax_line = plt.subplots(figsize=(6, 4))
    line, = ax_line.plot([], [], 'o-', markersize=3)
    ax_line.set_xlabel("Radial frequency index")
    # Update y-axis label to indicate a log scale.
    ax_line.set_ylabel("log10(Average Power)")
    ax_line.grid(True)
    ax_line.set_xlim(x_limits)
    ax_line.set_ylim(y_limits)
    fig_line.tight_layout(pad=0.5)

    frames = []
    last_frame = None
    for (csv_file, r, P_rad_log, power2d) in data_list:
        line.set_data(r, P_rad_log)
        ax_line.set_title(f"Radially Averaged Power Spectrum (log scale)\nSource: {os.path.basename(csv_file)}", fontsize=10)
        # Capture the current frame
        frame = capture_frame(fig_line)
        frames.append(frame)
        last_frame = frame  # update last frame
    plt.close(fig_line)

    # Create an MP4 video from the radially averaged plot frames.
    video_filename = "radial_alpha2_beta5_diff100_big_random.mp4"
    fps = 50  # Adjust frames per second as needed
    print(f"Writing video to {video_filename} ...")
    with imageio.get_writer(video_filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved to {video_filename}")

    # Save the last radially averaged spectrum as a PNG.
    png_filename = "radial_alpha2_beta5_diff100_big_random.png"
    imageio.imwrite(png_filename, last_frame)
    print(f"Last time step spectrum saved to {png_filename}")

    # ------------------------
    # Create frames for the 2D power spectrum.
    fig_im, ax_im = plt.subplots(figsize=(6, 5))
    # Initialize with the first frame data.
    initial_log_power = np.log10(data_list[0][3] + 1)
    im = ax_im.imshow(initial_log_power, origin='lower', aspect='auto', cmap='inferno', vmin=global_2d_min, vmax=global_2d_max)
    cbar = fig_im.colorbar(im, ax=ax_im)
    cbar.set_label("log10(Power)")
    ax_im.set_xlabel("k_x index")
    ax_im.set_ylabel("k_y index")
    fig_im.tight_layout(pad=0.5)

    frames_2d = []
    last_frame_2d = None
    for (csv_file, r, P_rad_log, power2d) in data_list:
        log_power = np.log10(power2d + 1)
        im.set_data(log_power)
        ax_im.set_title(f"2D Power Spectrum\nSource: {os.path.basename(csv_file)}", fontsize=10)
        frame2d = capture_frame(fig_im)
        frames_2d.append(frame2d)
        last_frame_2d = frame2d
    plt.close(fig_im)

    # Create an MP4 video from the 2D power spectrum frames.
    video_filename_2d = "power2d_alpha2_beta5_diff100_big_random.mp4"
    print(f"Writing video to {video_filename_2d} ...")
    with imageio.get_writer(video_filename_2d, fps=fps) as writer:
        for frame2d in frames_2d:
            writer.append_data(frame2d)
    print(f"2D power spectrum video saved to {video_filename_2d}")

    # Save the last 2D power spectrum frame as a PNG.
    png_filename_2d = "power2d_alpha2_beta5_diff100_big_random.png"
    imageio.imwrite(png_filename_2d, last_frame_2d)
    print(f"Last time step 2D spectrum saved to {png_filename_2d}")

if __name__ == "__main__":
    main()
