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
phi_g      = 0.5    # true gravitational phase [rad]
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


# ── Vectorised grid evaluation (n=3, x1=0 fixed, equal weights) ──────────────
def grid_sweep_n3():
    """
    Returns (sigma_grid, detIinv_grid) each shape (GRID, GRID),
    sweeping x2 and x3 with x1=0 and equal weights.
    """
    x_arr = np.linspace(0, 2 * np.pi, GRID, endpoint=False)
    X2, X3 = np.meshgrid(x_arr, x_arr, indexing="ij")     # (GRID, GRID)

    # positions: (3, GRID, GRID), weights: uniform 1/3
    pos = np.stack([np.zeros_like(X2), X2, X3], axis=0)   # (3, GRID, GRID)
    L   = signal(pos)                                       # (3, GRID, GRID)
    G   = design_matrix(pos)                               # (3, 3, GRID, GRID)
    w   = np.full(3, 1/3)
    wL  = w[:, None, None] / L                             # (3, GRID, GRID)

    # Fisher[a,b,i,j] = T * sum_k G[a,k,i,j] * G[b,k,i,j] * wL[k,i,j]
    Fisher = T * np.einsum("akij,bkij->abij", G, G * wL[None])  # (3,3,GRID,GRID)

    N  = GRID * GRID
    F  = Fisher.reshape(3, 3, N).transpose(2, 0, 1)        # (N, 3, 3)

    det  = np.linalg.det(F)
    cond = np.linalg.cond(F)
    valid = (det > 1e-20) & (cond < 1e6)

    sigma_flat   = np.full(N, np.nan)
    detIinv_flat = np.full(N, np.nan)

    if valid.any():
        Finv = np.linalg.inv(F[valid])                     # (M, 3, 3)
        sigma_flat[valid]   = np.sqrt(np.einsum("i,mij,j->m", grad_phi, Finv, grad_phi))
        detIinv_flat[valid] = np.linalg.det(Finv)

    return (sigma_flat.reshape(GRID, GRID),
            detIinv_flat.reshape(GRID, GRID),
            x_arr)


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


# ── Run ───────────────────────────────────────────────────────────────────────
print("Running 200×200 grid sweep for n=3 landscape...")
sigma_grid, detIinv_grid, x_arr = grid_sweep_n3()

print("Optimising designs for n = 3, 4, 5...")
results = {}
for n in (3, 4, 5):
    print(f"  n={n} ...", end=" ", flush=True)
    results[n] = optimise_design(n)
    print(f"  σ(φ_g) = {np.sqrt(results[n]['var_phi']):.6f} rad")


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


# ── Figure 1: landscape colorplot (n=3, equal weights) ───────────────────────
title_block = (rf"$\lambda_0={lambda_0},\ C={C},\ \phi_g={phi_g}$ rad,  $T={T}$"
               r"  —  $n=3$, equal weights, $x_1=0$")

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5))
fig1.suptitle(title_block, fontsize=11)

# Left: σ(φ_g)  [C-optimal landscape]
ax = axes1[0]
im = ax.contourf(x_arr/np.pi, x_arr/np.pi, sigma_grid, levels=60, cmap="viridis")
fig1.colorbar(im, ax=ax, label=r"$\sigma(\hat\phi_g)$  [rad]")

# mark global minimum
idx = np.unravel_index(np.nanargmin(sigma_grid), sigma_grid.shape)
ax.scatter(x_arr[idx[0]]/np.pi, x_arr[idx[1]]/np.pi,
           color="red", s=80, zorder=5,
           label=rf"min $\sigma={np.nanmin(sigma_grid):.4f}$ rad")
ax.set_xlabel(r"$x_2\,/\,\pi$");  ax.set_ylabel(r"$x_3\,/\,\pi$")
ax.set_title(r"$\sigma(\hat\phi_g)$  — C-optimality")
ax.legend(frameon=False, fontsize=9)

# Right: det(I⁻¹)  [D-optimal landscape]
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

plt.show()
