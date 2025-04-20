#!/usr/bin/env python3
# compare_with_theory.py  – refined theory/simulation check

import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, glob, os
from hopf_turing import critical_mode, lambda_plus   # theory funcs

# ---------------- user parameters ---------------- #
ALPHA, BETA = 0.1, 0.5
DA, DB      = 100.0, 1.0
snap_pattern = "A_t*.csv"        # produced by sweep_solver.cpp
timeseries   = "timeseries.csv"
site         = (0, 0)            # spatial point for temporal FFT
# -------------------------------------------------- #

# --- load last field snapshot ---
snap_name = sorted(glob.glob(snap_pattern))[-1]
A = np.loadtxt(snap_name, delimiter=",")
NY, NX = A.shape
Lx = NX                     # dx = 1 → domain length = NX
print(f"Loaded {snap_name}  (grid {NX}×{NY})")

# --- dominant spatial k ---
Kx = 2*np.pi*np.fft.fftfreq(NX, d=1.0)
Ky = 2*np.pi*np.fft.fftfreq(NY, d=1.0)
P  = np.abs(np.fft.fft2(A))**2
P[0, 0] = 0                 # remove mean
j, i = np.unravel_index(np.argmax(P), P.shape)
k_exp = np.hypot(Kx[i], Ky[j])
print(f"dominant k_exp = {k_exp:.4f}")

# --- growth rate σ from std‑dev time series ---
ts = pd.read_csv(timeseries)
# find first local minimum in std‑dev after t=0
stdv = ts["stdA"].values
idx_min = np.argmin(stdv[:40])
# take next 200 points
#mask = stdv[idx_min+1:] > stdv[idx_min:-1]
linear_idx = np.arange(idx_min+1, idx_min+1+200)
t_lin = ts["t"].values[linear_idx]
lnA  = np.log(stdv[linear_idx])
sigma_exp, _ = np.polyfit(t_lin, lnA, 1)
print(f"growth rate σ_exp = {sigma_exp:.4f}")

# --- temporal frequency ω from a single site ---
A_site = np.loadtxt(timeseries, delimiter=",", skiprows=1, usecols=2)
dt = ts["t"].values[1] - ts["t"].values[0]
freqs = np.fft.fftfreq(len(A_site), d=dt)
spec  = np.fft.fft(A_site - A_site.mean())
spec[0] = 0
idx = np.argmax(np.abs(spec))
omega_exp = 2*np.pi*abs(freqs[idx])
print(f"temporal frequency ω_exp ≈ {omega_exp:.4f}")

# --- theoretical values ---
k_c = critical_mode(ALPHA, BETA, DA, DB)
lam = lambda_plus(ALPHA, BETA, DA, DB, k_c)
print(f"theory: k_c = {k_c:.4f}, Re λ = {lam.real:.4f}, Im λ = {lam.imag:.4f}")

# --- diagnostic plots ---
plt.figure(figsize=(12,4))

# std‑dev growth
plt.subplot(1,2,1)
plt.semilogy(ts["t"], ts["stdA"], '.-', label="sim")
plt.semilogy(t_lin, np.exp(sigma_exp*t_lin+lnA[0]-sigma_exp*t_lin[0]),
             'r--', label=f"fit σ={sigma_exp:.3f}")
plt.xlabel("time"); plt.ylabel("std(A)")
plt.title("growth curve")
plt.legend()

# spatial power spectrum (azimuthal)
# ---- spatial power spectrum (azimuthal) ----
plt.subplot(1, 2, 2)
k_r   = np.hypot(*np.meshgrid(Kx, Ky))
bins  = np.linspace(0, k_r.max(), 50)      # 49 radial bins
bin_idx = np.digitize(k_r.ravel(), bins)   # 1‥len(bins)
P_flat  = P.ravel()
spec_rad = np.bincount(bin_idx, P_flat, minlength=len(bins)+1)[1:-1]  # drop under/overflow
bin_cent = 0.5 * (bins[1:] + bins[:-1])

plt.plot(bin_cent, spec_rad, '.-')
plt.axvline(k_exp, color='k', ls='--', label=f"$k_\\mathrm{{exp}}={k_exp:.2f}$")
plt.axvline(k_c,   color='r', ls=':',  label=f"$k_c={k_c:.2f}$")
plt.xlabel("k"); plt.ylabel("P(k)"); plt.title("radial spectrum"); plt.legend()


plt.tight_layout(); plt.savefig("comparison_diagnostics.png", dpi=200)
print("Saved comparison_diagnostics.png")
