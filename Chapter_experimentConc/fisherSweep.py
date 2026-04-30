"""
Cramér–Rao bound analysis for gravitational phase estimation.

Signal model:  lambda(kx) = a0 + a1*cos(kx) + a2*sin(kx)
  a0 = lambda_0,  a1 = lambda_0*C*cos(phi_g),  a2 = -lambda_0*C*sin(phi_g)
Unknowns: theta = (a0, a1, a2)

Fisher matrix:  I_ab = T * sum_j p_j * d_a*lambda(kx_j) * d_b*lambda(kx_j) / lambda(kx_j)
C-optimal criterion: minimise Var(phi_g) = grad_phi^T * I^{-1} * grad_phi
D-optimal criterion: minimise det(I^{-1})
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

plt.style.use("dark_background")
rng = np.random.default_rng(42)

# ── Parameters ────────────────────────────────────────────────────────────────
lambda_0   = 1.0    # mean photon number scale
C          = 0.4    # fringe contrast  (0 < C < 1)
phi_g      = 0.0119    # true gravitational phase [rad]
T          = 1.0    # total measurement time
GRID       = 200    # colorplot resolution
N_RESTARTS = 40     # random restarts per optimisation
# ──────────────────────────────────────────────────────────────────────────────

# ── Derived signal parameters ─────────────────────────────────────────────────
a0 = lambda_0
a1 = lambda_0 * C * np.cos(phi_g)
a2 = -lambda_0 * C * np.sin(phi_g)
r2 = a1**2 + a2**2

# Gradient of phi_g = -atan2(a2, a1) w.r.t. theta = (a0, a1, a2)
grad_phi = np.array([0.0, a2 / r2, -a1 / r2])   # shape (3,)


# ── Core functions ────────────────────────────────────────────────────────────
def signal(kx):
    """lambda(kx)"""
    return a0 + a1 * np.cos(kx) + a2 * np.sin(kx)


def design_matrix(kx):
    """(3, n) array of partial derivatives [d_a0, d_a1, d_a2] at each kx."""
    return np.array([np.ones_like(kx), np.cos(kx), np.sin(kx)])


def build_fisher(positions, weights):
    """
    3×3 Fisher information matrix.
    positions : (n,)  kx values
    weights   : (n,)  time fractions, must sum to 1
    """
    L  = signal(positions)                 # (n,)
    G  = design_matrix(positions)          # (3, n)
    wL = weights / L                       # (n,)
    return T * (G * wL) @ G.T             # (3, 3)


def compute_crb(positions, weights):
    """
    Returns (var_phi, det_Iinv, Iinv) or (inf, inf, None) if Fisher is singular.
    var_phi   = grad_phi^T I^{-1} grad_phi   [C-optimality]
    det_Iinv  = det(I^{-1})                  [D-optimality]
    """
    I = build_fisher(positions, weights)
    if np.linalg.det(I) < 1e-20 or np.linalg.cond(I) > 1e6:
        return np.inf, np.inf, None
    Iinv    = np.linalg.inv(I)
    var_phi = float(grad_phi @ Iinv @ grad_phi)
    det_Iinv = float(np.linalg.det(Iinv))
    return var_phi, det_Iinv, Iinv


# ── Vectorised grid evaluation (n=3, optimised over x1, equal weights) ────────
GRID_X1 = 50   # coarser x1 grid; minimum is taken over x1 for each (x2, x3)

def grid_sweep_n3():
    """
    Returns (sigma_grid, detIinv_grid, x1_opt_grid) each shape (GRID, GRID),
    where sigma/detIinv are the minimum over x1 at each (x2, x3).

    Uses equal weights p_j = 1/3.
    x1 is swept on a GRID_X1-point grid; x2, x3 on GRID points.
    Total Fisher evaluations: GRID_X1 * GRID^2 (fully vectorised).
    """
    x_arr  = np.linspace(0, 2 * np.pi, GRID,    endpoint=False)
    x1_arr = np.linspace(0, 2 * np.pi, GRID_X1, endpoint=False)

    # Axes: (x1, x2, x3) with shape (GRID_X1, GRID, GRID)
    X1, X2, X3 = np.meshgrid(x1_arr, x_arr, x_arr, indexing="ij")

    # positions: (3, GRID_X1, GRID, GRID)
    pos = np.stack([X1, X2, X3], axis=0)
    L   = signal(pos)                                        # (3, G1, G, G)
    G   = design_matrix(pos)                                 # (3, 3, G1, G, G)
    w   = np.full(3, 1/3)
    wL  = w[:, None, None, None] / L                         # (3, G1, G, G)

    # Fisher[a,b,i,j,k] = T * sum_p G[a,p,...]*G[b,p,...]*wL[p,...]
    Fisher = T * np.einsum("apijk,bpijk->abijk", G, G * wL[None])  # (3,3,G1,G,G)

    N_tot = GRID_X1 * GRID * GRID
    F     = Fisher.reshape(3, 3, N_tot).transpose(2, 0, 1)   # (N_tot, 3, 3)

    det  = np.linalg.det(F)
    cond = np.linalg.cond(F)
    valid = (det > 1e-20) & (cond < 1e6)

    sigma_flat   = np.full(N_tot, np.inf)
    detIinv_flat = np.full(N_tot, np.inf)

    if valid.any():
        Finv = np.linalg.inv(F[valid])
        sigma_flat[valid]   = np.sqrt(np.einsum("i,mij,j->m", grad_phi, Finv, grad_phi))
        detIinv_flat[valid] = np.linalg.det(Finv)

    # shape: (GRID_X1, GRID, GRID) — minimise over x1 axis
    sigma_3d   = sigma_flat.reshape(GRID_X1, GRID, GRID)
    detIinv_3d = detIinv_flat.reshape(GRID_X1, GRID, GRID)

    x1_idx_opt   = np.argmin(sigma_3d, axis=0)              # (GRID, GRID)
    sigma_grid   = sigma_3d[x1_idx_opt,
                            np.arange(GRID)[:, None],
                            np.arange(GRID)[None, :]]
    detIinv_grid = detIinv_3d[x1_idx_opt,
                              np.arange(GRID)[:, None],
                              np.arange(GRID)[None, :]]
    x1_opt_grid  = x1_arr[x1_idx_opt]

    return sigma_grid, detIinv_grid, x1_opt_grid, x_arr


# ── Joint position + weight optimisation ─────────────────────────────────────
def optimise_design(n):
    """
    Jointly optimise n measurement positions and n weights
    to minimise Var(phi_g)  [C-optimal].

    Free parameters: [x_1, ..., x_n, p_1, ..., p_{n-1}]
      p_n = 1 - sum(p_1..p_{n-1})   (enforced by constraint)

    Returns dict with keys: positions, weights, var_phi, det_Iinv, Iinv
    """
    best = {"var_phi": np.inf}
    W_MIN = 0.02   # minimum weight per point

    def unpack(params):
        pos    = params[:n] % (2 * np.pi)
        w_free = params[n:]
        w_last = 1.0 - np.sum(w_free)
        w      = np.append(w_free, w_last)
        return pos, w

    def objective(params):
        pos, w = unpack(params)
        if np.any(w < W_MIN):
            return 1e10
        var, _, _ = compute_crb(pos, w)
        return var

    # Bounds: positions free in [0, 2pi], free weights in [W_MIN, 1-W_MIN]
    bounds = ([(0, 2 * np.pi)] * n +
              [(W_MIN, 1 - (n-1)*W_MIN)] * (n - 1))

    for _ in range(N_RESTARTS):
        x0_pos = rng.uniform(0, 2 * np.pi, n)
        x0_w   = rng.dirichlet(np.ones(n))          # sums to 1
        x0     = np.concatenate([x0_pos, x0_w[:-1]])

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 2000})

        if res.fun < best["var_phi"]:
            pos, w = unpack(res.x)
            if np.all(w >= W_MIN / 2):
                var, det_Iinv, Iinv = compute_crb(pos, w)
                best = {"var_phi": var, "det_Iinv": det_Iinv,
                        "Iinv": Iinv, "positions": pos, "weights": w}

    return best


# ── Optimise phase offset of evenly-spaced design ────────────────────────────
def optimise_even_spacing(n):
    """
    Fix spacing to 2π/n, equal weights 1/n.
    Optimise only the phase offset φ₀ ∈ [0, 2π/n) (one period by symmetry).
    Returns same dict format as optimise_design.
    """
    spacing = 2 * np.pi / n
    w = np.full(n, 1.0 / n)

    phi_arr = np.linspace(0, spacing, 500, endpoint=False)
    best_var = np.inf
    best_phi0 = 0.0
    for phi0 in phi_arr:
        pos = phi0 + np.arange(n) * spacing
        var, _, _ = compute_crb(pos, w)
        if var < best_var:
            best_var = var
            best_phi0 = phi0

    pos_opt = best_phi0 + np.arange(n) * spacing
    var, det_Iinv, Iinv = compute_crb(pos_opt, w)
    return {"var_phi": var, "det_Iinv": det_Iinv, "Iinv": Iinv,
            "positions": pos_opt, "weights": w, "phi0": best_phi0}


# ── Run ───────────────────────────────────────────────────────────────────────
print(f"Running {GRID_X1}×{GRID}×{GRID} grid sweep for n=3 landscape (optimised over x1)...")
sigma_grid, detIinv_grid, x1_opt_grid, x_arr = grid_sweep_n3()

print("Optimising designs for n = 3, 4, 5...")
results = {}
for n in (3, 4, 5):
    print(f"  n={n} ...", end=" ", flush=True)
    results[n] = optimise_design(n)
    print(f"  σ(φ_g) = {np.sqrt(results[n]['var_phi']):.6f} rad")

print("Optimising even-spacing phase offset for n = 3, 4, 5...")
results_even = {}
for n in (3, 4, 5):
    results_even[n] = optimise_even_spacing(n)
    print(f"  n={n}  φ₀={results_even[n]['phi0']/np.pi:.4f}π  "
          f"σ={np.sqrt(results_even[n]['var_phi']):.6f} rad")


# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"SUMMARY   λ₀={lambda_0}, C={C}, φ_g={phi_g} rad, T={T}")
print("="*70)
print(f"{'n':>3}  {'σ(φ_g) [rad]':>14}  {'Var(φ_g)':>12}  "
      f"{'det(I⁻¹)':>14}  {'positions [×π]':>30}  {'weights':>20}")
print("-"*70)
for n, res in results.items():
    pos_str = "  ".join(f"{p/np.pi:.3f}" for p in sorted(res["positions"]))
    w_str   = "  ".join(f"{wi:.3f}" for wi in
                        [res["weights"][i] for i in np.argsort(res["positions"])])
    print(f"{n:>3}  {np.sqrt(res['var_phi']):>14.6f}  "
          f"{res['var_phi']:>12.6f}  "
          f"{res['det_Iinv']:>14.4e}  "
          f"{pos_str:>30}  {w_str:>20}")
print("="*70)
print("\nEven-spacing (phase-optimised) vs C-optimal:")
print(f"{'n':>3}  {'σ even [rad]':>14}  {'σ C-opt [rad]':>14}  {'gain [%]':>10}")
print("-"*50)
for n in (3, 4, 5):
    s_even = np.sqrt(results_even[n]["var_phi"])
    s_copt = np.sqrt(results[n]["var_phi"])
    gain   = (s_even / s_copt - 1) * 100
    print(f"{n:>3}  {s_even:>14.6f}  {s_copt:>14.6f}  {gain:>+10.2f}%")
print("="*70)


# ── Figure 1: landscape colorplot (n=3, equal weights, x1 optimised) ─────────
title_block = (rf"$\lambda_0={lambda_0},\ C={C},\ \phi_g={phi_g}$ rad,  $T={T}$"
               rf"  —  $n=3$, equal weights, $x_1$ optimised ({GRID_X1}-pt grid)")

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5.5))
fig1.suptitle(title_block, fontsize=11)

# Left: σ(φ_g)  [C-optimal landscape, min over x1]
ax = axes1[0]
im = ax.contourf(x_arr/np.pi, x_arr/np.pi, sigma_grid, levels=60, cmap="viridis")
fig1.colorbar(im, ax=ax, label=r"$\sigma(\hat\phi_g)$  [rad]")
idx = np.unravel_index(np.nanargmin(np.where(np.isfinite(sigma_grid), sigma_grid, np.nan)),
                       sigma_grid.shape)
ax.scatter(x_arr[idx[0]]/np.pi, x_arr[idx[1]]/np.pi,
           color="red", s=80, zorder=5,
           label=rf"min $\sigma={sigma_grid[idx]:.4f}$ rad")
ax.set_xlabel(r"$x_2\,/\,\pi$");  ax.set_ylabel(r"$x_3\,/\,\pi$")
ax.set_title(r"$\min_{x_1}\,\sigma(\hat\phi_g)$  — C-optimality")
ax.legend(frameon=False, fontsize=9)

# Middle: det(I⁻¹)  [D-optimal landscape]
ax = axes1[1]
im2 = ax.contourf(x_arr/np.pi, x_arr/np.pi,
                  np.log10(np.where(detIinv_grid > 0, detIinv_grid, np.nan)),
                  levels=60, cmap="plasma")
fig1.colorbar(im2, ax=ax, label=r"$\log_{10}\,\det(\mathcal{I}^{-1})$")
idx_d = np.unravel_index(np.nanargmin(detIinv_grid), detIinv_grid.shape)
ax.scatter(x_arr[idx_d[0]]/np.pi, x_arr[idx_d[1]]/np.pi,
           color="cyan", s=80, zorder=5, label="D-opt min")
ax.set_xlabel(r"$x_2\,/\,\pi$");  ax.set_ylabel(r"$x_3\,/\,\pi$")
ax.set_title(r"$\log_{10}\det(\mathcal{I}^{-1})$  — D-optimality")
ax.legend(frameon=False, fontsize=9)

# Right: optimal x1 value achieving the C-optimal minimum
ax = axes1[2]
im3 = ax.contourf(x_arr/np.pi, x_arr/np.pi, x1_opt_grid/np.pi, levels=60, cmap="twilight")
fig1.colorbar(im3, ax=ax, label=r"$x_1^*\,/\,\pi$")
ax.scatter(x_arr[idx[0]]/np.pi, x_arr[idx[1]]/np.pi,
           color="red", s=80, zorder=5)
ax.set_xlabel(r"$x_2\,/\,\pi$");  ax.set_ylabel(r"$x_3\,/\,\pi$")
ax.set_title(r"Optimal $x_1^*/\pi$ (C-criterion)")

plt.tight_layout()
fig1.savefig("fisherSweep_landscape.pdf")


# ── Figure 2: optimal designs for n = 3, 4, 5 ────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), subplot_kw={"projection": "polar"})
fig2.suptitle(
    rf"C-optimal measurement designs  —  $\lambda_0={lambda_0}$, $C={C}$, "
    rf"$\phi_g={phi_g}$ rad", fontsize=11)

colors_n = {3: "C2", 4: "C1", 5: "C3"}
for ax, (n, res) in zip(axes2, results.items()):
    pos = res["positions"]
    w   = res["weights"]
    order = np.argsort(pos)
    pos_s, w_s = pos[order], w[order]

    # bars proportional to weight, positioned at kx angle
    bars = ax.bar(pos_s, w_s, width=0.25, color=colors_n[n], alpha=0.85,
                  bottom=0, align="center")
    # mark phi_g
    ax.axvline(phi_g, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(
        rf"$n={n}$,  $\sigma={np.sqrt(res['var_phi']):.4f}$ rad"
        "\n"
        rf"$\det(\mathcal{{I}}^{{-1}})={res['det_Iinv']:.3e}$",
        fontsize=10, pad=12)
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$",
                        r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$"],
                       fontsize=8)
    ax.set_ylim(0, max(w_s) * 1.3)
    ax.set_ylabel("weight", labelpad=30, fontsize=9)

    # annotate each bar with weight value
    for p, wi in zip(pos_s, w_s):
        ax.text(p, wi + 0.01, f"{wi:.2f}", ha="center", va="bottom",
                fontsize=8, color="white")

plt.tight_layout()
fig2.savefig("fisherSweep_optimal_designs.pdf")


# ── Figure 3: σ(φ_g) vs n and weight sensitivity ─────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
fig3.suptitle(
    rf"Precision scaling  —  $\lambda_0={lambda_0}$, $C={C}$, $\phi_g={phi_g}$ rad",
    fontsize=11)

# Panel A: σ vs T for each n
ax = axes3[0]
T_arr = np.logspace(-2, 4, 300)
for n, res in results.items():
    pos, w = res["positions"], res["weights"]
    sigs = []
    for Tv in T_arr:
        L = signal(pos); G = design_matrix(pos)
        F = Tv * (G * (w / L)) @ G.T
        try:
            Finv = np.linalg.inv(F)
            sigs.append(np.sqrt(grad_phi @ Finv @ grad_phi))
        except np.linalg.LinAlgError:
            sigs.append(np.inf)
    ax.loglog(T_arr, sigs, label=rf"$n={n}$  (C-opt)", color=colors_n[n])

# reference: 1/√T slope
T_ref = T_arr
ax.loglog(T_ref, 1/np.sqrt(T_ref) * 0.5, "w--", linewidth=0.8, alpha=0.5,
          label=r"$\propto T^{-1/2}$")
ax.set_xlabel(r"$T$  [a.u.]");  ax.set_ylabel(r"$\sigma(\hat\phi_g)$  [rad]")
ax.set_title(r"Precision vs total time")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, which="both", linestyle=":", linewidth=0.5)

# Panel B: σ vs C for each n (at their respective optimal designs, reoptimised per C)
ax = axes3[1]
C_arr = np.linspace(0.05, 0.95, 60)

for n in (3, 4, 5):
    sig_C = []
    for Cv in C_arr:
        a1c = lambda_0 * Cv * np.cos(phi_g)
        a2c = -lambda_0 * Cv * np.sin(phi_g)
        r2c = a1c**2 + a2c**2
        gp  = np.array([0.0, a2c/r2c, -a1c/r2c])

        # use optimal positions from C=0.4 run but evaluate at this C
        pos = results[n]["positions"]
        w   = results[n]["weights"]
        L   = lambda_0 + a1c * np.cos(pos) + a2c * np.sin(pos)
        G   = design_matrix(pos)
        F   = T * (G * (w / L)) @ G.T
        try:
            Finv = np.linalg.inv(F)
            sig_C.append(np.sqrt(gp @ Finv @ gp))
        except np.linalg.LinAlgError:
            sig_C.append(np.inf)

    ax.semilogy(C_arr, sig_C, label=rf"$n={n}$", color=colors_n[n])

ax.set_xlabel(r"$C$  (contrast)");  ax.set_ylabel(r"$\sigma(\hat\phi_g)$  [rad]")
ax.set_title(r"Precision vs contrast (positions fixed at $C=0.4$ optimum)")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, which="both", linestyle=":", linewidth=0.5)

plt.tight_layout()
fig3.savefig("fisherSweep_scaling.pdf")

# ── Figure 4: λ(kx) with C-optimal and even-spacing points overlaid ──────────
kx_fine  = np.linspace(0, 2 * np.pi, 1000)
lam_fine = signal(kx_fine)
lam_range = lam_fine.max() - lam_fine.min()

# Top row: signal + points (3 panels for n=3,4,5)
# Bottom row: Var(φ_g) comparison bar chart (one panel spanning all)
fig4 = plt.figure(figsize=(16, 9))
gs   = fig4.add_gridspec(2, 3, hspace=0.45, wspace=0.25,
                          height_ratios=[1.6, 1])
fig4.suptitle(
    rf"$\lambda(kx)$ with measurement designs  —  "
    rf"$\lambda_0={lambda_0}$, $C={C}$, $\phi_g={phi_g}$ rad",
    fontsize=12)

axes_top = [fig4.add_subplot(gs[0, i]) for i in range(3)]

def draw_design_on_ax(ax, n, res, res_ev, lam_fine, lam_range):
    pos_c  = res["positions"];    w_c  = res["weights"]
    pos_e  = res_ev["positions"]; w_e  = res_ev["weights"]
    lam_c  = signal(pos_c);       lam_e = signal(pos_e)

    ax.plot(kx_fine / np.pi, lam_fine, color="white", linewidth=1.5, zorder=1)

    # Even-spacing points (triangles, orange)
    for xi, li, wi in zip(pos_e, lam_e, w_e):
        ax.vlines(xi / np.pi, 0, li, colors="C1", linewidth=0.8,
                  linestyles="dashed", zorder=2)
    ax.scatter(pos_e / np.pi, lam_e,
                      s=900 * w_e, marker="^",
                      color="C1", zorder=4, edgecolors="white",
                      linewidths=0.7, label=r"even-$\Delta$, opt $\phi_0$")
    for xi, li, wi in zip(pos_e[np.argsort(pos_e)],
                          lam_e[np.argsort(pos_e)],
                          w_e[np.argsort(pos_e)]):
        ax.text(xi / np.pi, li + 0.04 * lam_range,
                f"{wi:.2f}", ha="center", va="bottom",
                fontsize=8, color="C1")

    # C-optimal points (circles, green)
    for xi, li in zip(pos_c, lam_c):
        ax.vlines(xi / np.pi, 0, li, colors="C2", linewidth=0.8,
                  linestyles="dotted", zorder=2)
    sc_c = ax.scatter(pos_c / np.pi, lam_c,
                      s=900 * w_c, marker="o",
                      c=w_c, cmap="Greens", vmin=0, vmax=max(w_c) * 1.5,
                      zorder=5, edgecolors="white", linewidths=0.7,
                      label="C-optimal")
    for xi, li, wi in zip(pos_c[np.argsort(pos_c)],
                          lam_c[np.argsort(pos_c)],
                          w_c[np.argsort(pos_c)]):
        ax.text(xi / np.pi, li - 0.08 * lam_range,
                f"{wi:.2f}", ha="center", va="top",
                fontsize=8, color="C2")

    s_c  = np.sqrt(res["var_phi"])
    s_e  = np.sqrt(res_ev["var_phi"])
    gain = (s_e / s_c - 1) * 100
    ax.set_title(
        rf"$n={n}$  |  C-opt $\sigma={s_c:.4f}$,  even $\sigma={s_e:.4f}$ rad"
        "\n"
        rf"even/C-opt gain: ${gain:+.1f}\%$",
        fontsize=9)
    ax.set_xlabel(r"$kx\,/\,\pi$")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, lam_fine.max() * 1.25)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return sc_c

for ax, n in zip(axes_top, (3, 4, 5)):
    sc = draw_design_on_ax(ax, n, results[n], results_even[n], lam_fine, lam_range)

axes_top[0].set_ylabel(r"$\lambda(kx)$")

# Bottom row: Var(φ_g) comparison across n
ax_bar = fig4.add_subplot(gs[1, :])
n_vals  = [3, 4, 5]
var_c   = [results[n]["var_phi"]      for n in n_vals]
var_e   = [results_even[n]["var_phi"] for n in n_vals]
x_pos   = np.arange(len(n_vals))
width   = 0.35

bars_c = ax_bar.bar(x_pos - width/2, var_c, width, label="C-optimal",
                    color="C2", alpha=0.85)
bars_e = ax_bar.bar(x_pos + width/2, var_e, width,
                    label=r"Even-spacing (opt $\phi_0$)",
                    color="C1", alpha=0.85)

# annotate percentage overhead
for xp, vc, ve in zip(x_pos, var_c, var_e):
    gain_pct = (np.sqrt(ve) / np.sqrt(vc) - 1) * 100
    ax_bar.text(xp + width/2, ve * 1.04, rf"$+{gain_pct:.1f}\%$",
                ha="center", va="bottom", fontsize=9, color="C1")

ax_bar.set_yscale("log")
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([rf"$n={n}$" for n in n_vals], fontsize=11)
ax_bar.set_ylabel(r"$\mathrm{Var}(\hat\phi_g)$  [rad²]")
ax_bar.set_title(r"C-optimal vs even-spacing: $\mathrm{Var}(\hat\phi_g)$ comparison")
ax_bar.legend(frameon=False, fontsize=10)
ax_bar.grid(True, axis="y", linestyle=":", linewidth=0.5)

fig4.savefig("fisherSweep_signal_points.pdf")

plt.show()
