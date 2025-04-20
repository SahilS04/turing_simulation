#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import glob
import os
import scipy.stats as stats

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def compute_fft(field):
    """
    Compute the 2D FFT of the field, shift zero frequency to the center,
    and return the squared magnitude (power spectrum).
    """
    fft_field = np.fft.fft2(field)
    fft_field = np.fft.fftshift(fft_field)
    power_spec = np.abs(fft_field)**2
    return power_spec

def radial_average(power, nbins=50):
    """
    Compute the radial average of a 2D power spectrum.
    Uses a binned-statistic approach similar to fft.py.
    
    Returns:
      r_bin: bin centers (radial distances)
      rad_avg: mean power in each bin
    """
    ny, nx = power.shape
    y, x = np.indices((ny, nx))
    center = (ny // 2, nx // 2)
    # Compute radial distances from the center.
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    # Create bins between 0 and the maximum radius.
    bins = np.linspace(0, r.max(), nbins+1)
    rad_avg, bin_edges, _ = stats.binned_statistic(r.ravel(), power.ravel(),
                                                   statistic='mean', bins=bins)
    # Compute bin centers.
    r_bin = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return r_bin, rad_avg

# -------------------------------------------------------
# Gather CSV Files
# -------------------------------------------------------
# Expect files of the form "A_tXXXXX.csv" (with zero padding)
files = sorted(glob.glob("A_t*.csv"))
if not files:
    print("No CSV files found matching pattern 'A_t*.csv'. Exiting.")
    exit()
total_frames = len(files)

# -------------------------------------------------------
# Pre-compute Global Limits by processing all CSV files
# -------------------------------------------------------
# Initialize the global limits.
global_field_min = np.inf
global_field_max = -np.inf
global_fft_min   = np.inf
global_fft_max   = -np.inf
global_std_min   = np.inf
global_std_max   = -np.inf
global_rad_min   = np.inf
global_rad_max   = -np.inf
std_vals = []  # For standard deviation of each field

# Loop over all CSV files to compute the global limits.
for filename in files:
    try:
        field = np.loadtxt(filename, delimiter=',')
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    # Field (activator) limits.
    fmin = field.min()
    fmax = field.max()
    global_field_min = min(global_field_min, fmin)
    global_field_max = max(global_field_max, fmax)

    # Standard deviation for this field.
    std_val = np.std(field)
    std_vals.append(std_val)
    global_std_min = min(global_std_min, std_val)
    global_std_max = max(global_std_max, std_val)

    # Compute FFT power spectrum.
    power = compute_fft(field)
    # For log scaling the power, ignore zeros.
    nonzero = power[power > 0]
    if nonzero.size > 0:
        global_fft_min = min(global_fft_min, nonzero.min())
    global_fft_max = max(global_fft_max, power.max())

    # Compute radial average from the FFT power spectrum.
    _, rad_avg = radial_average(power, nbins=50)
    global_rad_min = min(global_rad_min, rad_avg.min())
    global_rad_max = max(global_rad_max, rad_avg.max())

# If the global minimum for the FFT is zero (or extremely small), adjust it.
if global_fft_min <= 0:
    global_fft_min = 1e-2

# For debugging: print global limits.
print("Global Field Limits: ", global_field_min, global_field_max)
print("Global FFT Limits:   ", global_fft_min, global_fft_max)
print("Global Std Dev Limits:", global_std_min, global_std_max)
print("Global Radial Avg Limits:", global_rad_min, global_rad_max)

# Also precompute a radial bins vector from one sample (all files share the same grid).
sample_field = np.loadtxt(files[0], delimiter=',')
sample_power = compute_fft(sample_field)
r_bin_sample, _ = radial_average(sample_power, nbins=50)

# -------------------------------------------------------
# Set Up the Figure and Axes
# -------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Axes assignments
ax_std    = axs[0, 0]  # Top-left: standard deviation time series
ax_radial = axs[0, 1]  # Top-right: radially averaged power spectrum
ax_field  = axs[1, 0]  # Bottom-left: activator field display
ax_fft    = axs[1, 1]  # Bottom-right: 2D FFT power spectrum

# Initial image display using the sample data.
field_im = ax_field.imshow(sample_field, 
                           vmin=global_field_min, vmax=global_field_max,
                           cmap='viridis')
power_spec_sample = compute_fft(sample_field)
fft_im = ax_fft.imshow(power_spec_sample, 
                       norm=LogNorm(vmin=1e-1, vmax=global_fft_max),
                       cmap='inferno')

# Initialize line objects for the time series (standard deviation)
# and the radial average plot.
std_line, = ax_std.plot([], [], 'b-', lw=2)
radial_line, = ax_radial.plot([], [], 'r-', lw=2)

# Setting titles for clarity.
ax_field.set_title("Activator Field")
ax_fft.set_title("2D Power Spectrum")
ax_std.set_title("Standard Deviation of Activator Field")
ax_radial.set_title("Radially Averaged Power Spectrum")

# Set fixed axis limits based on the global limits.
ax_std.set_xlim(0, total_frames)
# You may give a bit of margin to the std dev plot.
ax_std.set_ylim(0, global_std_max * 1.1)
ax_radial.set_yscale('log')
ax_radial.set_ylim(global_rad_min, global_rad_max)
# The x-axis for radial average is based on the r_bin from the sample.
ax_radial.set_xlim(r_bin_sample[0], r_bin_sample[-1])

# Add colorbars for the image plots.
cbar_field = fig.colorbar(field_im, ax=ax_field)
cbar_field.set_label("Concentration")
cbar_fft = fig.colorbar(fft_im, ax=ax_fft)
cbar_fft.set_label("Power Spectrum Intensity")

# Data container for evolving standard deviation time series.
std_values = []

# -------------------------------------------------------
# Animation Functions
# -------------------------------------------------------
def init():
    """Initialization function for the animation."""
    std_line.set_data([], [])
    radial_line.set_data([], [])
    return field_im, fft_im, std_line, radial_line

def update(frame):
    """
    Update function for frame 'frame' (an index into the CSV files):
      - Loads the CSV data for the current timestep.
      - Updates the activator field display.
      - Computes and updates the 2D FFT display.
      - Updates the evolving standard deviation time series.
      - Computes and updates the radially averaged power spectrum.
    """
    filename = files[frame]
    field = np.loadtxt(filename, delimiter=',')
    
    # Update activator field image.
    field_im.set_data(field)
    
    # Compute FFT and update 2D power spectrum.
    power_spec = compute_fft(field)
    fft_im.set_data(power_spec)
    
    # Update standard deviation time series.
    current_std = np.std(field)
    std_values.append(current_std)
    x_data = np.arange(len(std_values))
    std_line.set_data(x_data, std_values)
    ax_std.set_title(f"Std Dev (Frame {frame+1}/{total_frames})")
    
    # Compute radially averaged power spectrum.
    r_bin, rad_avg = radial_average(power_spec, nbins=50)
    radial_line.set_data(r_bin, rad_avg)
    
    return field_im, fft_im, std_line, radial_line

# Create the animation.
ani = animation.FuncAnimation(fig, update, frames=total_frames,
                              init_func=init, interval=200, blit=False)

# Save the animation as a video.
ani.save('full_videos/full_a1_-2_b5_d100.mp4', writer='ffmpeg', fps=50)
plt.show()
