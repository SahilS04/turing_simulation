#!/usr/bin/env python3
"""
simulate_regime.py
------------------
Batch runner + analysis for Schnakenberg regimes:
  • Runs fast_solver over (α,β) grid
  • Loads stdA time‑series and final field snapshot
  • Extracts σ_exp, ω_exp, k_exp
  • Compares to theory (k_c, Re λ, Im λ)
  • Classifies into 4 regimes robustly
  • Plots overlay and writes JSON summary
  • Cleans up run folders
"""
import os
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from hopf_turing import classify, lambda_plus, k_critical

# ---------------- user settings ---------------- #
D_RATIO = 100.0
DA, DB  = D_RATIO, 1.0
SAMPLES = 50
alpha_rng = (-0.6, 0.6)
beta_rng  = (-2.0, 2.0)
GRID_RES  = 120

EXE       = os.path.abspath("fast_solver")  # compiled solver
OUT_INT   = 30
DT        = 0.01
WINDOW    = 50      # last WINDOW outputs for slope & FFT
TH_SIG    = 5e-3    # slope threshold
OSC_THR   = 0.1     # FFT power ratio threshold
# ------------------------------------------------ #

# 1) Build analytic regime map
alphas = np.linspace(*alpha_rng, GRID_RES)
betas  = np.linspace(*beta_rng,  GRID_RES)
analytic = np.empty((GRID_RES, GRID_RES), dtype=int)
for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        kc  = k_critical(a, b, DA, DB)
        lam = lambda_plus(a, b, DA, DB, kc)
        analytic[i, j] = classify(lam)

# 2) Sample points in each regime
rng    = np.random.default_rng(0)
points = []
for regime in range(4):
    idxs = np.column_stack(np.where(analytic == regime))
    pick = rng.choice(len(idxs), size=min(SAMPLES, len(idxs)), replace=False)
    for k in pick:
        i, j = idxs[k]
        points.append((alphas[j], betas[i], regime))
print(f"Total runs: {len(points)}")

# Helper: radial average of 2D FFT power
def radial_average_fft(A):
    N, M = A.shape
    F  = np.fft.fft2(A - A.mean())
    P  = np.abs(F)**2
    kx = 2*np.pi*np.fft.fftfreq(M, d=1.0)
    ky = 2*np.pi*np.fft.fftfreq(N, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).ravel()
    P_flat = P.ravel()
    nbins = min(N, M)//2
    bins = np.linspace(0, K.max(), nbins+1)
    radial = np.zeros(nbins)
    for m in range(nbins):
        mask = (K >= bins[m]) & (K < bins[m+1])
        radial[m] = P_flat[mask].mean() if np.any(mask) else 0
    centers = 0.5*(bins[:-1] + bins[1:])
    return centers, radial

# 3) Classify one run by slope & FFT of stdA
def classify_run(alpha, beta):
    tag = f"a{alpha:+.2f}_b{beta:+.2f}"
    os.makedirs(tag, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tag)

    # run solver
    subprocess.run([EXE, str(alpha), str(beta), str(DA), str(DB)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # load stdA time-series
    ts = pd.read_csv("timeseries.csv")
    t  = ts["t"].values[-WINDOW:]
    s  = ts["stdA"].values[-WINDOW:]
    # guard against log(0)
    s_safe = np.clip(s, 1e-6, None)
    # fit slope of ln(stdA)
    m = np.polyfit(t, np.log(s_safe), 1)[0]

    # detrend envelope & FFT to detect oscillations
    detr = s * np.exp(-m * t)
    spec = np.abs(np.fft.rfft(detr - detr.mean()))**2
    osc  = spec[1:].max()/spec[0] > OSC_THR

    # load final field & compute k_exp
    A_end = np.loadtxt("A_final.csv", delimiter=",")
    k_bins, Pk = radial_average_fft(A_end)
    k_exp = k_bins[np.nanargmax(Pk)]

    # theoretical for completeness (not used in classification)
    kc  = k_critical(alpha, beta, DA, DB)
    lam = lambda_plus(alpha, beta, DA, DB, kc)
    sigma_th, omega_th = lam.real, lam.imag

    # decision table
    if abs(m) < TH_SIG and not osc:
        regime = 0  # stable
    elif m < -TH_SIG and osc:
        regime = 1  # damped oscillation
    elif m >  TH_SIG and osc:
        regime = 2  # growing oscillation
    elif m >  TH_SIG and not osc:
        regime = 3  # static growth
    else:
        regime = 4  # ambiguous

    os.chdir(cwd)
    return regime, m, osc, k_exp, kc, sigma_th, omega_th

# 4) Run all simulations
results = []
for a, b, rt in tqdm(points):
    rs, m, osc, k_exp, kc, st, ot = classify_run(a, b)
    results.append((a, b, rt, rs, m, osc, k_exp, kc, st, ot))

# save JSON
with open("regime_validation.json", "w") as f:
    json.dump(results, f, indent=2)

# 5) Plot overlay
cmap    = ListedColormap(["#1f497d", "#7eb6ff", "#ffb366", "#d62728", "#888888"])
markers = ["o", "s", "^", "x", "+"]
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(analytic, origin="lower",
          extent=[*alpha_rng, *beta_rng],
          aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5)
for a, b, rt, rs, *_ in results:
    edge = "yellow" if rs != rt else "black"
    ax.plot(a, b, markers[rs], mfc="none", mec=edge, ms=6, lw=1)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$")
ax.set_title("Simulation‑validated regimes (yellow edge = mismatch)")
plt.tight_layout()
plt.savefig("regime_overlay.png", dpi=300)

# 6) Clean up run folders
for d in os.listdir("."):
    if d.startswith("a+") or d.startswith("a-"):
        shutil.rmtree(d, ignore_errors=True)

print("Done. Overlay saved to regime_overlay.png")
