#!/usr/bin/env python3
"""
simulate_regime_parallel.py
---------------------------
• Spawns solver runs in parallel using ProcessPoolExecutor
• Uses independent working dirs so no file collisions
• Collects results, plots overlay, then cleans up
"""
import os, json, shutil, subprocess
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from hopf_turing import classify, lambda_plus, k_critical

# ────── User settings ────── #
D_RATIO   = 100.0
DA, DB    = D_RATIO, 1.0
SAMPLES   = 50
alpha_rng = (-0.6, 0.6)
beta_rng  = (-2.0, 2.0)
GRID_RES  = 120
EXE        = os.path.abspath("fast_solver")  # solver binary
WINDOW     = 50
TH_SIG     = 5e-3
OSC_THR    = 0.1
MAX_WORKERS = None   # None => os.cpu_count()
# ─────────────────────────── #

# Build analytic map
alphas = np.linspace(*alpha_rng, GRID_RES)
betas  = np.linspace(*beta_rng,  GRID_RES)
analytic = np.empty((GRID_RES, GRID_RES), int)
for i,b in enumerate(betas):
    for j,a in enumerate(alphas):
        kc = k_critical(a, b, DA, DB)
        lam = lambda_plus(a, b, DA, DB, kc)
        analytic[i,j] = classify(lam)

# Sample points
rng = np.random.default_rng(0)
points = []
for regime in range(4):
    idxs = np.column_stack(np.where(analytic == regime))
    pick = rng.choice(len(idxs), size=min(SAMPLES,len(idxs)), replace=False)
    for k in pick:
        i,j = idxs[k]
        points.append((alphas[j], betas[i], regime))
print(f"Total runs: {len(points)}")

# Helper: run a single (α,β) in its own dir and parse results
def worker(args):
    alpha, beta, rt = args
    tag = f"a{alpha:+.2f}_b{beta:+.2f}"
    os.makedirs(tag, exist_ok=True)
    # run solver in that dir
    subprocess.run([EXE, str(alpha), str(beta), str(DA), str(DB)],
                   cwd=tag, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    # load stdA time‑series
    df = pd.read_csv(os.path.join(tag, "timeseries.csv"))
    t = df["t"].values[-WINDOW:]
    s = df["stdA"].values[-WINDOW:]
    s_safe = np.clip(s, 1e-6, None)
    m = np.polyfit(t, np.log(s_safe), 1)[0]
    detr = s * np.exp(-m*t)
    spec = np.abs(np.fft.rfft(detr - detr.mean()))**2
    dc = spec[0] if spec[0] > 0 else 1e-12
    osc = (spec[1:].max() / dc) > OSC_THR
    # load final field → k_exp
    A_end = np.loadtxt(os.path.join(tag, "A_final.csv"), delimiter=",")
    # radial average of FFT power
    F  = np.fft.fft2(A_end - A_end.mean())
    P  = np.abs(F)**2
    kx = 2*np.pi*np.fft.fftfreq(A_end.shape[1])
    ky = 2*np.pi*np.fft.fftfreq(A_end.shape[0])
    KX,KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).ravel()
    P_flat = P.ravel()
    bins = np.linspace(0, K.max(), min(A_end.shape)//2 + 1)
    radial = np.array([P_flat[(K>=bins[i])&(K<bins[i+1])].mean() 
                       for i in range(len(bins)-1)])
    centers = 0.5*(bins[:-1] + bins[1:])
    k_exp = centers[np.nanargmax(radial)]
    # theory
    kc  = k_critical(alpha, beta, DA, DB)
    lam = lambda_plus(alpha, beta, DA, DB, kc)
    sigma_th, omega_th = lam.real, lam.imag
    # classify
    if abs(m) < TH_SIG and not osc:
        rs = 0
    elif m < -TH_SIG and osc:
        rs = 1
    elif m >  TH_SIG and osc:
        rs = 2
    elif m >  TH_SIG and not osc:
        rs = 3
    else:
        rs = 4
    return (alpha, beta, rt, rs, m, osc, k_exp, kc, sigma_th, omega_th, tag)

# Dispatch in parallel
results = []
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
    futures = { exe.submit(worker, p): p for p in points }
    for fut in tqdm(as_completed(futures), total=len(points)):
        results.append(fut.result())

# Clean up run dirs and collate
json_results = []
for (a,b,rt,rs,m,osc,k_exp,kc,st,ot,tag) in results:
    json_results.append((a,b,rt,rs,m,osc,k_exp,kc,st,ot))
    shutil.rmtree(tag)

# save JSON
with open("regime_validation.json","w") as f:
    json.dump(json_results, f, indent=2)

# Plot overlay
cmap    = ListedColormap(["#1f497d","#7eb6ff","#ffb366","#d62728","#888888"])
markers = ["o","s","^","x","+"]
fig,ax = plt.subplots(figsize=(7,5))
ax.imshow(analytic, origin="lower",
          extent=[*alpha_rng,*beta_rng], aspect="auto",
          cmap=cmap, vmin=-0.5, vmax=4.5)
for (a,b,rt,rs, *_) in json_results:
    edge = "yellow" if rs!=rt else "black"
    ax.plot(a, b, markers[rs], mfc="none", mec=edge, ms=6, lw=1)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$")
ax.set_title("Simulation‑validated regimes (parallel)")
plt.tight_layout(); plt.savefig("regime_overlay.png", dpi=300)

print("Done! regime_overlay.png")
