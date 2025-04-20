#!/usr/bin/env python3
""" classify regimes using std‑dev growth and single‑site FFT """
import numpy as np, subprocess, os, json, shutil, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from hopf_turing import classify, lambda_max_complex

# ---------- settings ----------
D_RATIO = 100.0
DA, DB  = D_RATIO, 1.0
SAMPLES = 4
alpha_rng=(-0.6,0.6)
beta_rng =(-2.0,2.0)
GRID=120
EXE = os.path.abspath("fast_solver")
WIN=60           # last 60 outputs
EPS=1e-3
OSC_THR=0.1
# ------------------------------

# analytic map
alph=np.linspace(*alpha_rng,GRID)
bet =np.linspace(*beta_rng ,GRID)
ana = np.empty((GRID,GRID),int)
for i,b in enumerate(bet):
    for j,a in enumerate(alph):
        ana[i,j]=classify(lambda_max_complex(a,b,DA,DB))

# random points per regime
rng=np.random.default_rng(0)
pts=[]
for r in range(4):
    idx=np.column_stack(np.where(ana==r))
    pick=rng.choice(len(idx),size=min(SAMPLES,len(idx)),replace=False)
    for i,j in idx[pick]: pts.append((alph[j],bet[i],r))

def decide(csv:str):
    df=pd.read_csv(csv)
    if len(df)<WIN: return 4
    t   = df["t"].values[-WIN:]
    std = df["stdA"].values[-WIN:]
    site=df["A_site"].values[-WIN:]

    # growth / decay
    m,_ = np.polyfit(t,np.log(std),1)
    ratio=std[-1]/std[0]

    # oscillation flag using single site
    y = site - site.mean()
    P = np.abs(np.fft.rfft(y))**2
    osc = P[1:].max()/P[0] > OSC_THR

    if abs(m)<EPS and 0.5<ratio<2 and not osc: return 0
    if m<-EPS and ratio<0.8 and osc:            return 1
    if m> EPS and ratio>1.2 and osc:            return 2
    if m> EPS and ratio>1.2 and not osc:        return 3
    return 4

runs,dirs=[],[]
for a,b,r in tqdm(pts):
    tag=f"a{a:+.2f}_b{b:+.2f}"; os.makedirs(tag,exist_ok=True); dirs.append(tag)
    os.chdir(tag)
    subprocess.run([EXE,f"{a}",f"{b}",f"{DA}",f"{DB}"],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    rs=decide("timeseries.csv")
    runs.append((a,b,r,rs))
    os.chdir("..")

json.dump(runs,open("regime_validation.json","w"),indent=2)

# ---------- plot ----------
cmap=ListedColormap(["#1f497d","#7eb6ff","#ffb366","#d62728","#aaaaaa"])
marks=["o","s","^","x","+"]
fig,ax=plt.subplots(figsize=(7,5))
ax.imshow(ana,origin='lower',extent=[*alpha_rng,*beta_rng],
          aspect='auto',cmap=cmap,vmin=-0.5,vmax=4.5)
for a,b,rt,rs in runs:
    edge="yellow" if rs!=rt else "black"
    ax.plot(a,b,marks[rs],mfc='none',mec=edge,ms=6,lw=1)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$")
ax.set_title("Simulation‑validated regimes  (yellow edge = mismatch)")
plt.tight_layout(); plt.savefig("regime_overlay.png",dpi=300)

for d in dirs: shutil.rmtree(d,ignore_errors=True)
print("finished; overlay saved")
