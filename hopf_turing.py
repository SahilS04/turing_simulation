# hopf_turing_analysis.py  – refined version
"""
Refinements
===========
• Accurate critical wavenumber `k_c` via  Brent root on d(Re λ)/dk.
• Robust Complex‑Ginzburg–Landau coefficients using left/right eigenvectors.
• Regime map now shows two contours:
      – black  : Re λ = 0  (instability threshold)
      – white  : Im λ = 0  (Hopf–Turing switch)
  so all four regions are visually separated.
• Colour legend unchanged (dark‑blue, light‑blue, orange, red).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import brentq

# ---- kinetics ----

def steady_state(alpha, beta):
    """
    Compute homogeneous steady states A0, B0:
        A0 = alpha + beta
        B0 = beta/(alpha+beta)^2
    """
    A0 = alpha + beta
    B0 = beta / (alpha + beta) ** 2
    return A0, B0


def jacobian(alpha, beta):
    """
    Jacobian of reaction kinetics at (A0,B0).
    """
    A0, B0 = steady_state(alpha, beta)
    fu = -1 + 2 * A0 * B0
    fv = A0 ** 2
    gu = -2 * A0 * B0
    gv = 0.0
    return fu, fv, gu, gv

# ---- dispersion ----

def lambda_plus(alpha, beta, DA, DB, k):
    """
    Larger eigenvalue λ₊ of the linearized system at wavenumber k.
    """
    fu, fv, gu, gv = jacobian(alpha, beta)
    tr0 = fu + gv
    det0 = fu * gv - fv * gu
    k2 = k * k
    tr = tr0 - (DA + DB) * k2
    det = det0 - (DA * gv + DB * fu) * k2 + DA * DB * k2 ** 2
    disc = tr * tr - 4 * det
    return 0.5 * (tr + np.sqrt(disc + 0j))


def critical_mode(alpha, beta, DA, DB):
    """
    Find k_c that maximizes Re λ⁺ via Brent's method on derivative.
    """
    def deriv(k):
        eps = 1e-6
        lam = lambda_plus(alpha, beta, DA, DB, k + 1j * eps)
        return lam.imag / eps

    try:
        return brentq(deriv, 1e-6, 2.0, maxiter=200)
    except ValueError:
        return 0.0


def lambda_max_complex(alpha, beta, DA, DB):
    """
    Return λ₊(k_c) at the most unstable wavenumber.
    """
    kc = critical_mode(alpha, beta, DA, DB)
    return lambda_plus(alpha, beta, DA, DB, kc)

# Aliases for simulation script compatibility
k_critical = critical_mode
lambda_max = lambda_max_complex

# ---- CGL coefficients ----

def cgl_coefficients(alpha, beta, DA, DB):
    """
    Compute Complex Ginzburg–Landau coefficients:
      returns (k_c, λ_c, D_complex, cubic_sign)
    """
    kc = critical_mode(alpha, beta, DA, DB)
    # linear operator L = J - D k_c^2
    fu, fv, gu, gv = jacobian(alpha, beta)
    L = np.array([[fu - DA*kc*kc, fv], [gu, gv - DB*kc*kc]], dtype=complex)
    # right eigenvectors & values
    eigvals, VR = np.linalg.eig(L)
    idx = np.argmax(eigvals.real)
    lam_c = eigvals[idx]
    v = VR[:, idx]
    # left eigenvector from transposed conjugate
    _, VL = np.linalg.eig(L.T.conj())
    w = VL[:, idx]
    w = w / (w @ v)  # normalize
    # complex diffusion projection
    Dmat = np.diag([-DA, -DB])
    D_complex = w @ (Dmat @ v)
    # cubic proxy sign
    cubic_sign = np.sign(alpha + beta)
    return kc, lam_c, D_complex, cubic_sign

# ---- regime classification & plotting ----

def classify(lam, imag_eps=1e-4):
    """
    Linear regime from λ:
      0 = stable (Re<0, Im≈0)
      1 = damped osc (Re<0, Im≠0)
      2 = growing osc (Re>0, Im≠0)
      3 = static growth (Re>0, Im≈0)
    """
    if lam.real > 0:
        return 2 if abs(lam.imag) > imag_eps else 3
    else:
        return 1 if abs(lam.imag) > imag_eps else 0

_cmap = ListedColormap(["#1f497d", "#7eb6ff", "#ffb366", "#d62728"])


def plot_regime_maps(d_ratio_list=(0.01, 0.03, 0.05, 0.07, 0.09, 0.11),
                     alpha_range=(-2, 2), beta_range=(-5, 5), res=240):
    alphas = np.linspace(alpha_range[0], alpha_range[1], res)
    betas  = np.linspace(beta_range[0], beta_range[1], res)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, d in zip(axes, d_ratio_list):
        grid = np.empty((res, res), dtype=int)
        ReZ = np.zeros_like(grid, dtype=float)
        ImZ = np.zeros_like(grid, dtype=float)
        for i, b in enumerate(betas):
            for j, a in enumerate(alphas):
                lam = lambda_max_complex(a, b, d, 1.0)
                grid[i, j] = classify(lam)
                ReZ[i, j], ImZ[i, j] = lam.real, lam.imag

        ax.imshow(grid, origin="lower", extent=[*alpha_range, *beta_range],
                  aspect='auto', cmap=_cmap, vmin=-0.5, vmax=3.5)
        # Re λ = 0 contour (black)
        ax.contour(alphas, betas, ReZ, levels=[0], colors='k', linewidths=1)
        # Im λ = 0 contour (white)
        ax.contour(alphas, betas, ImZ, levels=[0], colors='w', linewidths=1)
        ax.set_title(fr"$D_A/D_B$ = {d}")
        ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$")

    handles = [plt.Line2D([0], [0], marker='s', ms=10, linestyle='', color=_cmap(i),
                           label=lbl)
               for i, lbl in enumerate(["stable", "damped osc", "growing osc", "growing stat"])]
    fig.legend(handles=handles, loc='upper center', ncol=4)
    fig.suptitle("Linear regimes (Re & Im λ) in (α, β) space", y=0.93)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig("small_regime_alpha_beta_maps.png", dpi=300)

# ---- main ----

def main():
    plot_regime_maps()
    # example coefficients
    alpha, beta, d = 0.59, 1.9, 100
    kc, lam_c, Dcomp, sign_c = cgl_coefficients(alpha, beta, d, 1.0)
    print(f"Point (α={alpha}, β={beta}, d={d}): k_c = {kc:.4f}, λ={lam_c}")
    print(f"   D = {Dcomp.real:.3f} + {Dcomp.imag:.3f} i,  cubic sign≈{sign_c}")

if __name__ == "__main__":
    main()
