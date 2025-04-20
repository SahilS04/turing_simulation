#!/usr/bin/env python3
"""
plot_std_sweep.py

This script reads CSV files generated from the parameter sweep
(e.g. "std_sweep_beta_0.90.csv", "std_sweep_beta_1.00.csv", etc.),
each file containing three columns: time, sigma_A, sigma_B.
It then produces two plots:
  1. Time evolution of sigma_A (activator std dev) with curves for each beta.
  2. Time evolution of sigma_B (inhibitor std dev) with curves for each beta.

Requires: numpy, pandas, matplotlib, glob, os
Usage:
    python plot_std_sweep.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

def extract_beta(filename):
    # Expect filename format like std_sweep_beta_0.90.csv
    pattern = r"std_sweep_diff_([\d\.]+)\.csv"
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    else:
        return None

def main():
    file_pattern = "std_sweep_diff_*.csv"
    csv_files = sorted(glob.glob(file_pattern))
    if not csv_files:
        print("No CSV files found matching pattern:", file_pattern)
        return

    # Prepare lists for plotting.
    plt.figure(figsize=(8,6))
    for fname in csv_files:
        beta_val = extract_beta(os.path.basename(fname))
        if beta_val is None:
            continue
        # Read CSV using pandas:
        df = pd.read_csv(fname)
        plt.plot(df['time_step'], df['sigma_A'], label=f"DA/DB = {beta_val:.2f}")
    plt.xlabel("Time")
    plt.ylabel("σ_A (Activator Std. Dev)")
    plt.title("Time Evolution of Activator Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("std_evolution_activator_diffusion_high_DA.png", dpi=300)
    plt.show()

    # Second plot for inhibitor B
    plt.figure(figsize=(8,6))
    for fname in csv_files:
        beta_val = extract_beta(os.path.basename(fname))
        if beta_val is None:
            continue
        df = pd.read_csv(fname)
        plt.plot(df['time_step'], df['sigma_B'], label=f"DA/DB = {beta_val:.2f}")
    plt.xlabel("Time")
    plt.ylabel("σ_B (Inhibitor Std. Dev)")
    plt.title("Time Evolution of Inhibitor Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("std_evolution_inhibitor_diffusion_high_DA.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
