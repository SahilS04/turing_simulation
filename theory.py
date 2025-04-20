# theory_analysis.py
"""
Analytic toolkit for the Schnakenberg reaction–diffusion system.
Updated colour‑bars:
* All diverging maps use a symmetric range around 0 (white at λ = 0).
* Colour‑bars are placed outside the plotting area with ample padding.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle

# ---------------- core math ---------------- #

def steady_state(alpha, beta):
    A0 = alpha + beta
    B0 = beta / (alpha + beta) ** 2
    return A0, B0


def jacobian_elements(A0, B0):
    fu = -1 + 2 * A0 * B0
    fv = A0 ** 2
    gu = -2 * A0 * B0
    gv = -A0 ** 2
    return fu, fv, gu, gv


def dispersion_relation(alpha, beta, DA, DB, k):
    A0, B0 = steady_state(alpha, beta)
    fu, fv, gu, gv = jacobian_elements(A0, B0)
    trace0 = fu + gv
    det0 = fu * gv - fv * gu
    k2 = k * k
    trace = trace0 - (DA + DB) * k2
    det = det0 - (DA * gv + DB * fu) * k2 + DA * DB * k2 ** 2
    disc = np.maximum(trace ** 2 - 4 * det, 0.0)
    return 0.5 * (trace + np.sqrt(disc))


def lambda_max(alpha, beta, DA, DB):
    k = np.linspace(0, 2, 800)
    return dispersion_relation(alpha, beta, DA, DB, k).max()

# ---------------- plotting helpers ---------------- #

_cmap = plt.cm.coolwarm


def _sym_norm(data):
    max_abs = np.abs(data).max()
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def plot_dispersion_curves(alpha, beta, DA_list, DB=1.0):
    k = np.linspace(0, 2, 800)
    fig, ax = plt.subplots(figsize=(8, 6))
    for DA, col in zip(DA_list, cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])):
        ax.plot(k, dispersion_relation(alpha, beta, DA, DB, k), label=f"D_A={DA}")
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Growth rate $\lambda_+(k)$")
    ax.set_title(fr"Dispersion relation, $\alpha$={alpha}, $\beta$={beta}, $D_B$={DB}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("dispersion_curves.png", dpi=300)


def plot_lambda_vs_dratio(alpha, beta, DB=1.0, dratio_range=(1e-2, 1e1), n=400):
    d_vals = np.logspace(np.log10(dratio_range[0]), np.log10(dratio_range[1]), n)
    lam_vals = [lambda_max(alpha, beta, d * DB, DB) for d in d_vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d_vals, lam_vals)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.axvline(1., color='k', ls='--', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r"Diffusion ratio $D_A/D_B$")
    ax.set_ylabel(r"$\max_k\,\lambda_+(k)$")
    ax.set_title(fr"Max growth rate vs diffusion ratio, $\alpha$={alpha}, $\beta$={beta}, $D_B$={DB}")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("lambda_vs_dratio.png", dpi=300)


from matplotlib.colors import ListedColormap, BoundaryNorm

# simple two‑colour map: blue for λ<0, red for λ>0
_binary_cmap = ListedColormap(["royalblue", "firebrick"])  # two colours
_binary_norm = BoundaryNorm([-1.5, 0, 1.5], ncolors=_binary_cmap.N)


def plot_lambda_plane(alpha, beta,
                      DA_range=(1e-1, 10), DB_range=(1e-1, 10),
                      res=140):
    DA_vals = np.logspace(np.log10(DA_range[0]), np.log10(DA_range[1]), res)
    DB_vals = np.logspace(np.log10(DB_range[0]), np.log10(DB_range[1]), res)
    lam_sign = np.empty((res, res))
    for i, DA in enumerate(DA_vals):
        for j, DB in enumerate(DB_vals):
            lam = lambda_max(alpha, beta, DA, DB)
            lam_sign[j, i] = np.sign(lam)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(DA_vals, DB_vals, lam_sign,
                levels=[-1.5, 0, 1.5], cmap=_binary_cmap, norm=_binary_norm)
    cs = ax.contour(DA_vals, DB_vals, lam_sign,
                    levels=[0], colors='k'); ax.clabel(cs, fmt={0: 'λ = 0'})
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$D_A$"); ax.set_ylabel(r"$D_B$")
    ax.set_title(fr"Stability map in $(D_A,D_B)$ plane  ($\alpha$={alpha}, $\beta$={beta})")
    ax.grid(alpha=0.3, which='both', ls=':')
    fig.tight_layout(); fig.savefig("lambda_plane.png", dpi=300)


def plot_alpha_beta_fields(d_ratio_list=(1e-8, 0.1, 1, 10, 100, 1e8),
                           alpha_range=(-0.6, 0.6),
                           beta_range=(-2.0, 2.0),
                           res=120):

    alphas = np.linspace(alpha_range[0], alpha_range[1], res)
    betas  = np.linspace(beta_range[0], beta_range[1], res)

    lam_sign_fields = []
    for d_ratio in d_ratio_list:
        lam_sign = np.array([[np.sign(lambda_max(a, b, d_ratio, 1.0))
                              for a in alphas] for b in betas])
        lam_sign_fields.append(lam_sign)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, sign_grid, d_ratio in zip(axes, lam_sign_fields, d_ratio_list):
        ax.contourf(alphas, betas, sign_grid,
                    levels=[-1.5, 0, 1.5], cmap=_binary_cmap, norm=_binary_norm)
        cs = ax.contour(alphas, betas, sign_grid, levels=[0], colors='k')
        ax.clabel(cs, fmt={0: 'λ = 0'})
        ax.set_title(fr"$D_A/D_B$ = {d_ratio}", y=1)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")

    fig.suptitle(r"Stability region (blue: λ<0, red: λ>0) in $(\alpha, \beta)$ space", y=0.99)
    fig.tight_layout(); fig.savefig("lambda_alpha_beta_fields.png", dpi=300)

# ---------------- main ---------------- #

def main():
    alpha = 0.1; beta = 0.9
    plot_dispersion_curves(alpha, beta, [0.01, 0.1, 1.0, 10.0, 100.0])
    plot_lambda_vs_dratio(alpha, beta)
    plot_lambda_plane(alpha, beta)
    plot_alpha_beta_fields()
    print("Figures generated.")


if __name__ == "__main__":
    main()
