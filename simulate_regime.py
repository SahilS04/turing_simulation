#!/usr/bin/env python3
"""
simulate_regime.py
----------------------
Batch‑runner + analysis:
  • Runs fast_solver_new over a grid of (alpha,beta).
  • Loads timeseries.csv and A_final.csv.
  • Extracts σ_exp, ω_exp, k_exp.
  • Compares to theory (k_c, Re λ, Im λ).
  • Classifies into 4 regimes robustly.
  • Plots overlay and writes JSON.
  • Cleans up run folders.
"""
import os, json, shutil, subprocess
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from hopf_turing import critical_mode, lambda_plus, lambda_max_complex

# ----------------- user settings -----------------
D_RATIO = 100.0
DA, DB  = D_RATIO, 1.0
SAMPLES = 2
alpha_rng = (-0.6, 0.6)
beta_rng  = (-2.0, 2.0)
GRID_RES  = 120

EXE = os.path.abspath("fast_solver")   # compiled C++ binary

# analysis parameters
OUT_INT = 30                          # must match solver
DT      = 0.01
WINDOW  = 50                          # last 50 points ≃ 50*DT*OUT_INT time
TH_SIG  = 5e-3                        # slope threshold
OSC_THR = 0.1                         # FFT power ratio threshold
# -------------------------------------------------

# build analytic background
alphas = np.linspace(*alpha_rng, GRID_RES)
betas  = np.linspace(*beta_rng,  GRID_RES)
analytic = np.empty((GRID_RES, GRID_RES), dtype=int)
for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        lam = lambda_max_complex(a, b, DA, DB)
        # 0=stable,1=dampedOsc,2=growOsc,3=growStat
        analytic[i,j] = (lam.real>0)*1 + (abs(lam.imag)>1e-6)*((lam.real>0)*1 + (lam.real<0)*0)

# sample points
rng = np.random.default_rng(0)
points = []
for r in range(4):
    idxs = np.column_stack(np.where(analytic==r))
    choice = rng.choice(len(idxs), size=min(SAMPLES,len(idxs)), replace=False)
    for i,j in idxs[choice]:
        points.append((alphas[j], betas[i], r))

print(f"Total runs: {len(points)}")

# helpers
def radial_average_fft(A):
    N, M = A.shape
    # 2D FFT power
    F = np.fft.fft2(A - A.mean())
    P = np.abs(F)**2
    # freq grids
    kx = 2*np.pi*np.fft.fftfreq(M, d=1.0)
    ky = 2*np.pi*np.fft.fftfreq(N, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).ravel()
    P_flat = P.ravel()
    nbins = min(N, M)//2
    bins = np.linspace(0, K.max(), nbins+1)
    radial = np.zeros(nbins)
    for m in range(nbins):
        mask = (K>=bins[m]) & (K<bins[m+1])
        radial[m] = P_flat[mask].mean() if np.any(mask) else 0
    centers = 0.5*(bins[:-1]+bins[1:])
    return centers, radial

def classify_run(alpha, beta):
    tag = f"a{alpha:+.2f}_b{beta:+.2f}"
    os.makedirs(tag, exist_ok=True)
    os.chdir(tag)
    # 1) run solver
    subprocess.run([EXE, str(alpha), str(beta), str(DA), str(DB)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 2) load timeseries
    df = pd.read_csv("timeseries.csv")
    t   = df["t"].values
    s   = df["stdA"].values
    # take last WINDOW points
    t0, s0 = t[-WINDOW:], s[-WINDOW:]
    # 3) fit log-linear slope
    m = np.polyfit(t0, np.log(s0), 1)[0]
    # 4) oscillation test on detrended envelope
    detr = s0 * np.exp(-m*t0)
    spec = np.abs(np.fft.rfft(detr - detr.mean()))**2
    osc  = spec[1:].max()/spec[0] > OSC_THR
    # 5) spatial k from final A
    A_fin = np.loadtxt("A_final.csv", delimiter=",")
    k_bins, Pk = radial_average_fft(A_fin)
    k_exp = k_bins[np.nanargmax(Pk)]
    # 6) theoretical
    kc = critical_mode(alpha, beta, DA, DB)
    lam= lambda_plus(alpha, beta, DA, DB)
    sigma_th, omega_th = lam.real, lam.imag
    # 7) classification
    if abs(m) < TH_SIG and not osc:
        regime = 0        # stable
    elif m < -TH_SIG and osc:
        regime = 1        # damped oscillation
    elif m >  TH_SIG and osc:
        regime = 2        # growing oscillation
    elif m >  TH_SIG and not osc:
        regime = 3        # static growth
    else:
        regime = 4        # ambiguous

    os.chdir("..")
    return regime, m, osc, k_exp, kc, sigma_th, omega_th

# run all
results = []
for a,b,rt in tqdm(points):
    rs, m, osc, k_exp, kc, st, ot = classify_run(a,b)
    results.append((a,b,rt, rs, m, osc, k_exp, kc, st, ot))

# save JSON
with open("regime_validation.json","w") as f:
    json.dump(results, f, indent=2)

# plot overlay
cmap    = ListedColormap(["#1f497d","#7eb6ff","#ffb366","#d62728","#999999"])
markers = ["o","s","^","x","+"]    # one for each of the 5 classes (0–4)

fig, ax = plt.subplots(figsize=(7,5))
ax.imshow(analytic,
          origin='lower',
          extent=[alpha_rng[0], alpha_rng[1],
                  beta_rng[0],  beta_rng[1]],
          aspect='auto',
          cmap=cmap, vmin=-0.5, vmax=4.5)

# corrected unpacking here:
for a, b, rt, rs, *rest in results:
    edge = "yellow" if rs != rt else "black"
    ax.plot(a, b,
            markers[rs],
            mfc="none", mec=edge,
            ms=6, lw=1)

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_title("Simulation‑validated regimes (yellow edge = mismatch)")
plt.tight_layout()
plt.savefig("regime_overlay.png", dpi=300)

# clean up
for d in os.listdir("."):
    if d.startswith("a+") or d.startswith("a-"):
        shutil.rmtree(d, ignore_errors=True)

print("Done! Overlay → regime_overlay.png")
