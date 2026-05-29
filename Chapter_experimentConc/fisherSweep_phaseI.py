"""
Data-taking strategy for measuring fringe contrast C.

Signal: λ(kx) = λ₀·(1 + C·cos(kx − φ))
        = a₀ + a₁·cos(kx) + a₂·sin(kx)
Unknowns: θ = (a₀, a₁, a₂)
Derived:  C = √(a₁²+a₂²)/a₀,   φ = atan2(−a₂, a₁)

WHY EVEN SPACING IS THE NATURAL STARTING POINT
───────────────────────────────────────────────
Without any prior on φ you cannot compute grad_C = [-C/λ₀, cosφ/λ₀, -sinφ/λ₀],
so a "C-optimal" design (which needs φ) is unavailable from the outset.  The
Fisher matrix for (a₀, a₁, a₂) with n evenly-spaced equal-weight positions is
diagonal for the Poisson model when C→0 (all cross-terms vanish by trig
identities), and it is φ-invariant by rotational symmetry.  Even spacing is
therefore the minimax-optimal design over unknown φ for this model.

TWO STRATEGIES COMPARED
────────────────────────
  EVEN     — n = 3, 4, 5, 7 points at kx_j = j·2π/n, equal time.
             No prior needed.  φ-invariant.  The practical baseline.
             For n = 3 this is identical to the adaptive strategy.

  ADAPTIVE — Start with 3 even points → Poisson MLE for (λ₀, C, φ) →
             greedily add each remaining point at the kx that minimises
             Var(C) given the current MLE estimate of φ.
             Practical: uses only data you have already taken.
             Identical to EVEN for n = 3 (no adaptation step possible).

PRACTICAL TAKE-AWAY
───────────────────
• n = 3 even points are the minimum to estimate (λ₀, C, φ).
• σ(C) ∝ 1/√T_total.  Number of positions n barely affects σ(C) — all
  n ≥ 3 evenly-spaced designs give approximately the same CRB at fixed T,
  because the Poisson Fisher information converges rapidly to its
  continuous limit (n→∞).  The dominant lever is observation TIME.
• Adaptive outperforms even spacing only when C is large enough that the
  MLE estimate of φ after 3 points is reliable (C·√(λ₀·T) ≳ few).
• For λ₀ = 0.02 Hz regime see the time-to-significance figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar

plt.style.use("dark_background")
rng = np.random.default_rng(0)

# ── Parameters ────────────────────────────────────────────────────────────────
LAMBDA_0  = 0.02    # Hz  (mean event rate; λ₀ ± 0.3 Hz prior — use as operating point)
PHI_TRUE  = 0.0     # reference phase for simulation (even designs are φ-invariant;
                    # adaptive uses estimated φ̂ from MLE, not this)
T_UNIT    = 1.0     # Fisher computed at T = 1 s;  CRB ∝ 1/T,  σ(C) ∝ 1/√T

N_MC      = 300     # Monte Carlo draws for even-spacing phase-offset sweep
N_ADAPT   = 80      # Poisson simulation runs for adaptive strategy
T_ADAPT   = 3600.0  # [s] simulated total time for Poisson draws in adaptive runs
                    # (controls MLE quality; reported σ is always at T_UNIT = 1 s)

n_vals     = [3, 4, 5, 7]
C_arr      = np.linspace(0.04, 0.88, 35)    # contrast grid
C_sim_vals = [0.05, 0.15, 0.30, 0.50]       # C values for adaptive simulation

# ── Core functions ────────────────────────────────────────────────────────────
def signal(kx, lam0, C, phi):
    return lam0 * (1.0 + C * np.cos(kx - phi))

def design_matrix(kx):
    return np.array([np.ones_like(kx), np.cos(kx), np.sin(kx)])

def build_fisher(positions, weights, lam0, C, phi):
    L = signal(positions, lam0, C, phi)
    G = design_matrix(positions)
    return T_UNIT * (G * (weights / L)) @ G.T

def grad_C_fn(lam0, C, phi):
    """∂C/∂(a₀,a₁,a₂) — requires knowing φ.  Used only for adaptive (estimated φ)."""
    return np.array([-C / lam0, np.cos(phi) / lam0, -np.sin(phi) / lam0])

def var_C_given_phi(positions, weights, lam0, C, phi):
    """Var(C) = grad_C^T I⁻¹ grad_C.  Requires φ — used in adaptive with estimated φ."""
    try:
        F = build_fisher(positions, weights, lam0, C, phi)
        if np.linalg.det(F) < 1e-30 or np.linalg.cond(F) > 1e8:
            return np.inf
        Finv = np.linalg.inv(F)
        gC   = grad_C_fn(lam0, C, phi)
        return float(gC @ Finv @ gC)
    except Exception:
        return np.inf

def var_C_no_prior(positions, weights, lam0, C, n_phi=60):
    """
    φ-averaged Var(C) — the right metric when φ is unknown.
    Integrates over φ ∈ [0, 2π] with uniform prior.
    For symmetric (even-spacing) designs this is very close to the
    per-φ value because the design is nearly φ-invariant.
    """
    phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    vals = [var_C_given_phi(positions, weights, lam0, C, phi) for phi in phi_arr]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else np.inf

def hours_to_nsigma(var_C_unit, C_true, n_sigma):
    """Hours so that C_true / σ(C) ≥ n_sigma, given Var(C) at T = 1 s."""
    if not np.isfinite(var_C_unit) or C_true <= 0:
        return np.inf
    return n_sigma**2 * var_C_unit / C_true**2 / 3600.0

# ── STRATEGY 1: even spacing ──────────────────────────────────────────────────
def even_spacing_stats(n, lam0, C, n_phi=120):
    """
    For n evenly-spaced equal-weight points, return (mean, min, max) Var(C)
    over all grid phase offsets — captures the φ-dependence at finite C.
    """
    w   = np.full(n, 1.0 / n)
    spc = 2 * np.pi / n
    phi0_arr = np.linspace(0, spc, n_phi, endpoint=False)
    # For each grid offset phi0, average Var(C) over true φ
    vals = []
    for phi0 in phi0_arr:
        pos = phi0 + np.arange(n) * spc
        v   = var_C_no_prior(pos, w, lam0, C)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.inf, np.inf, np.inf
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))

# ── STRATEGY 2: adaptive ──────────────────────────────────────────────────────
def mle_fit_poisson(positions, counts, exposures):
    """Poisson MLE for (a₀, a₁, a₂) → (λ₀_hat, C_hat, φ_hat)."""
    def neg_ll(params):
        a0, a1, a2 = params
        lam_arr = a0 + a1 * np.cos(positions) + a2 * np.sin(positions)
        if np.any(lam_arr <= 0):
            return 1e10
        return -np.sum(counts * np.log(lam_arr) - lam_arr * exposures)

    a0_init = max(np.sum(counts) / max(np.sum(exposures), 1e-9), 1e-9)
    best_fun, best_x = np.inf, [a0_init, 0.0, 0.0]
    for _ in range(10):
        x0 = [a0_init * rng.uniform(0.3, 3.0),
               rng.normal(0, a0_init * 0.4),
               rng.normal(0, a0_init * 0.4)]
        try:
            r = minimize(neg_ll, x0, method='L-BFGS-B',
                         bounds=[(1e-9, None), (None, None), (None, None)],
                         options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 500})
            if r.fun < best_fun:
                best_fun, best_x = r.fun, r.x.copy()
        except Exception:
            pass
    a0, a1, a2 = best_x
    a0 = max(a0, 1e-9)
    return a0, np.sqrt(a1**2 + a2**2) / a0, np.arctan2(-a2, a1)


def next_adaptive_kx(current_pos, lam0_hat, C_hat, phi_hat):
    """
    Next measurement kx that minimises Var(C) given current MLE estimates.
    This is C-optimal conditioned on estimated φ — the adaptive step.
    """
    n_next = len(current_pos) + 1
    w_next = np.full(n_next, 1.0 / n_next)
    C_use  = max(C_hat, 1e-3)

    def obj(kx):
        pos = np.append(current_pos, kx % (2 * np.pi))
        return var_C_given_phi(pos, w_next, lam0_hat, C_use, phi_hat)

    kx_grid = np.linspace(0, 2 * np.pi, 100)
    vals    = [obj(kx) for kx in kx_grid]
    kx0     = kx_grid[np.argmin(vals)]
    try:
        r = minimize_scalar(obj, bounds=(kx0 - 0.5, kx0 + 0.5), method='bounded')
        return r.x % (2 * np.pi)
    except Exception:
        return kx0


def run_adaptive(n_total, lam0_true, C_true, phi_true):
    """
    Single adaptive run:
      • Phase 1: 3 evenly-spaced points (T_ADAPT/n_total seconds each)
      • Phase 2: MLE after each point → add next point at estimated-φ C-optimal kx
    Returns φ-averaged Var(C) at T=1 s after each cumulative point total (3, 4, …, n_total).
    """
    t_pt  = T_ADAPT / n_total
    n_ini = 3
    pos   = np.linspace(0, 2 * np.pi, n_ini, endpoint=False)
    exp   = np.full(n_ini, t_pt)
    cnts  = rng.poisson(signal(pos, lam0_true, C_true, phi_true) * exp)

    w = np.full(n_ini, 1.0 / n_ini)
    sigs = [np.sqrt(var_C_no_prior(pos, w, lam0_true, C_true))]

    for _ in range(n_ini, n_total):
        try:
            lam0h, Ch, phih = mle_fit_poisson(pos, cnts, exp)
        except Exception:
            lam0h, Ch, phih = lam0_true, C_true, phi_true

        kx_next  = next_adaptive_kx(pos, lam0h, Ch, phih)
        cnt_next = rng.poisson(
            signal(np.array([kx_next]), lam0_true, C_true, phi_true)[0] * t_pt)
        pos  = np.append(pos, kx_next)
        exp  = np.append(exp, t_pt)
        cnts = np.append(cnts, cnt_next)

        w = np.full(len(pos), 1.0 / len(pos))
        sigs.append(np.sqrt(var_C_no_prior(pos, w, lam0_true, C_true)))

    return np.array(sigs)   # length n_total − 2 (n_pts = 3, 4, …, n_total)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
print("Computing even-spacing CRBs …")

# arrays indexed [n_idx, C_idx]
var_even_mean  = np.full((len(n_vals), len(C_arr)), np.inf)
var_even_best  = np.full_like(var_even_mean, np.inf)
var_even_worst = np.full_like(var_even_mean, np.inf)

for ni, n in enumerate(n_vals):
    print(f"  n={n}", end="  ", flush=True)
    for ci, C in enumerate(C_arr):
        vm, vb, vw = even_spacing_stats(n, LAMBDA_0, C)
        var_even_mean[ni, ci]  = vm
        var_even_best[ni, ci]  = vb
        var_even_worst[ni, ci] = vw
    print("done")

print(f"\nRunning adaptive simulations ({N_ADAPT} runs, T_adapt = {int(T_ADAPT)} s) …")
# adapt_results[n][C_sim] = array (N_ADAPT, n−2)
adapt_results = {n: {} for n in n_vals}
for n in n_vals:
    for C_sim in C_sim_vals:
        print(f"  n={n}, C={C_sim:.2f}", end="  ", flush=True)
        runs = [run_adaptive(n, LAMBDA_0, C_sim, PHI_TRUE) for _ in range(N_ADAPT)]
        adapt_results[n][C_sim] = np.array(runs)
        print("done")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"SUMMARY   λ₀ = {LAMBDA_0} Hz   (σ(C) at T = 1 s, scales as T^(-1/2))")
print("=" * 65)
for C_ref in [0.05, 0.10, 0.20, 0.30, 0.50]:
    ci = np.argmin(np.abs(C_arr - C_ref))
    print(f"\n  C = {C_ref:.2f}")
    print(f"  {'n':>3}  {'σ even (φ-avg)':>16}  "
          f"{'T_3σ [h]':>10}  {'T_5σ [h]':>10}")
    print("  " + "-" * 45)
    for ni, n in enumerate(n_vals):
        se = np.sqrt(var_even_mean[ni, ci])
        h3 = hours_to_nsigma(var_even_mean[ni, ci], C_ref, 3)
        h5 = hours_to_nsigma(var_even_mean[ni, ci], C_ref, 5)
        print(f"  {n:>3}  {se:>16.5f}  {h3:>10.2f}  {h5:>10.2f}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Signal geometry + Fisher information density
# Explains WHY certain positions carry more information about C
# ─────────────────────────────────────────────────────────────────────────────
C_geo, phi_geo = 0.40, 0.0
kx_fine = np.linspace(0, 2 * np.pi, 500)
lam_fine = signal(kx_fine, LAMBDA_0, C_geo, phi_geo)

# Fisher information density for C at each kx (marginalised — approximate)
# Full marginal requires matrix inversion; use the a₁ component as a proxy:
# f_a1(kx) = cos²(kx) / λ(kx)  — dominant term when φ ≈ 0
fisher_density = np.cos(kx_fine - phi_geo)**2 / lam_fine  # ∝ info about C amplitude

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig1.suptitle(
    rf"Signal geometry and Fisher information density"
    "\n"
    rf"$\lambda(kx)=\lambda_0(1+C\cos(kx-\phi))$,  "
    rf"$\lambda_0={LAMBDA_0}$ Hz,  $C={C_geo}$,  $\phi=0$",
    fontsize=11)

# ── top: signal + measurement positions ──────────────────────────────────────
ax1a.plot(kx_fine / np.pi, lam_fine, 'w-', lw=1.8, label=r'$\lambda(kx)$')

markers = {'Even n=3': ('C0', 'o', np.linspace(0, 2*np.pi, 3, endpoint=False)),
           'Even n=5': ('C2', 's', np.linspace(0, 2*np.pi, 5, endpoint=False))}

for label, (col, mk, pos) in markers.items():
    lam_pts = signal(pos, LAMBDA_0, C_geo, phi_geo)
    ax1a.scatter(pos / np.pi, lam_pts, s=80, marker=mk, color=col,
                 zorder=5, label=label)
    for p, lp in zip(pos, lam_pts):
        ax1a.vlines(p / np.pi, 0, lp, colors=col, lw=0.8, ls='--', alpha=0.6)

ax1a.set_ylabel(r'$\lambda(kx)$  [Hz]')
ax1a.set_ylim(bottom=0)
ax1a.legend(frameon=False, fontsize=9)
ax1a.grid(True, linestyle=':', linewidth=0.5)

# ── bottom: Fisher info density for C ────────────────────────────────────────
ax1b.plot(kx_fine / np.pi, fisher_density / fisher_density.max(),
          color='C1', lw=1.8,
          label=r'$\cos^2(kx-\phi)/\lambda(kx)$ — info density for $C$')
ax1b.fill_between(kx_fine / np.pi,
                  fisher_density / fisher_density.max(),
                  alpha=0.2, color='C1')
ax1b.axhline(1.0, color='gray', lw=0.6, ls='--', alpha=0.5)
ax1b.set_xlabel(r'$kx\,/\,\pi$')
ax1b.set_ylabel('Information density\n(normalised)')
ax1b.set_ylim(0, 1.15)
ax1b.legend(frameon=False, fontsize=9)
ax1b.grid(True, linestyle=':', linewidth=0.5)
ax1b.set_xticks(np.linspace(0, 2, 9))
ax1b.text(0.01, 0.88,
    "Most information about C\nnear the signal minimum\n(Poisson: more surprise per count).",
    transform=ax1b.transAxes, fontsize=8, color='C1',
    va='top', bbox=dict(boxstyle='round', fc='black', alpha=0.5))

plt.tight_layout()
fig1.savefig("fisherSweep_C_geometry.pdf", bbox_inches="tight")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: σ(C) vs C — all n on one plot to show n barely matters
#
# KEY POINT: all n ≥ 3 evenly-spaced designs give nearly identical σ(C) at
# fixed total time T.  The Poisson Fisher info converges to its continuous
# limit fast — n=3 is already within ~10-20% of n→∞.  The dominant lever
# for improving σ(C) is observation TIME, not number of positions.
# ─────────────────────────────────────────────────────────────────────────────
colors_n = {3: 'C0', 4: 'C1', 5: 'C2', 7: 'C3'}

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.suptitle(
    r"Even-spacing: $\sigma(C)$ at fixed total time $T$  ($\lambda_0=$"
    + f"{LAMBDA_0}" + r" Hz)"
    "\n"
    r"LEFT — all $n$ at $T=1$ s: curves nearly identical $\Rightarrow$ $n$ barely matters."
    r"  RIGHT — $\sigma(C)$ vs $T$ for $n=3$ at selected $C$.",
    fontsize=11)

# ── left: σ(C) vs C for all n at T=1 s ──────────────────────────────────────
analytic_floor = np.sqrt(2.0 / (LAMBDA_0 * T_UNIT))

for ni, n in enumerate(n_vals):
    ax2a.semilogy(C_arr, np.sqrt(var_even_mean[ni]),
                  '-', color=colors_n[n], lw=2.0, label=rf"$n={n}$ (φ-avg)")
    ax2a.fill_between(C_arr,
                      np.sqrt(var_even_best[ni]),
                      np.sqrt(var_even_worst[ni]),
                      alpha=0.10, color=colors_n[n])

ax2a.axhline(analytic_floor, color='gray', lw=1.0, ls='--',
             label=r'$\sqrt{2/(\lambda_0 T)}$  ($C\!\to\!0$,  $n\!\to\!\infty$)')
ax2a.set_xlabel("Contrast  $C$")
ax2a.set_ylabel(r"$\sigma(C)$  [at $T=1$ s]")
ax2a.legend(frameon=False, fontsize=9)
ax2a.grid(True, which='both', linestyle=':', linewidth=0.5)
ax2a.set_xlim(C_arr[0], C_arr[-1])
ax2a.text(0.97, 0.97,
    "Shaded band = phase-offset spread\nLines nearly overlap → n barely matters",
    transform=ax2a.transAxes, fontsize=8, color='white',
    ha='right', va='top', bbox=dict(boxstyle='round', fc='black', alpha=0.6))

# ── right: σ(C) vs T (hours) for n=3, selected C values ─────────────────────
T_hours = np.logspace(-2, 3, 300)          # 0.01 h to 1000 h
T_sec   = T_hours * 3600.0
C_show  = [0.05, 0.10, 0.20, 0.50]
cols_C  = ['C3', 'C1', 'C2', 'C0']

ni3 = n_vals.index(3)
for C_ref, col in zip(C_show, cols_C):
    ci = np.argmin(np.abs(C_arr - C_ref))
    # σ(C) at T = var(T=1s) / T_actual  →  σ = sqrt(var_1s) / sqrt(T_sec)
    sig_T = np.sqrt(var_even_mean[ni3, ci]) / np.sqrt(T_sec)
    ax2b.loglog(T_hours, sig_T, '-', color=col, lw=1.8, label=rf"$C={C_ref}$")
    # mark 3σ and 5σ detection thresholds: σ(C) = C/Nσ
    for Nsig, ls in [(3, '--'), (5, ':')]:
        T_thresh = var_even_mean[ni3, ci] * Nsig**2 / C_ref**2 / 3600.0
        ax2b.axvline(T_thresh, color=col, lw=0.8, ls=ls, alpha=0.6)

ax2b.plot([], [], 'w--', lw=1.0, label=r'$3\sigma$ threshold (vertical)')
ax2b.plot([], [], 'w:',  lw=1.0, label=r'$5\sigma$ threshold (vertical)')
for h_ref, lbl in [(1, '1 h'), (24, '1 day'), (168, '1 week')]:
    ax2b.axvline(h_ref, color='gray', lw=0.5, ls='-.', alpha=0.4)
    ax2b.text(h_ref * 1.08, 3e-3, lbl,
              color='gray', fontsize=7, rotation=90, va='bottom')

ax2b.set_xlabel("Total observation time  [hours]")
ax2b.set_ylabel(r"$\sigma(C)$  (uncertainty on contrast)")
ax2b.set_title(r"$n=3$ even spacing — $\sigma(C) \propto T^{-1/2}$")
ax2b.legend(frameon=False, fontsize=9)
ax2b.grid(True, which='both', linestyle=':', linewidth=0.5)

plt.tight_layout()
fig2.savefig("fisherSweep_C_sigma_vs_C.pdf", bbox_inches="tight")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Time to 3σ / 5σ in hours — the practical output
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))
fig3.suptitle(
    rf"Time to significance  ($\lambda_0 = {LAMBDA_0}$ Hz)"
    "\n"
    r"$T_{N\sigma} = N^2 \cdot \mathrm{Var}(C,\,T{=}1\,\mathrm{s}) \;/\; C^2$"
    r"   [hours]",
    fontsize=11)

for axi, (n_sig, ax) in enumerate(zip([3, 5], axes3)):
    for ni, n in enumerate(n_vals):
        col = colors_n[n]
        h_even = [hours_to_nsigma(var_even_mean[ni, ci], C, n_sig)
                  for ci, C in enumerate(C_arr)]
        ax.semilogy(C_arr, h_even, '-', color=col, lw=2.0,
                    label=rf"$n={n}$")

    for h_ref, lbl in [(1, '1 h'), (24, '1 day'), (168, '1 week'), (720, '1 month')]:
        ax.axhline(h_ref, color='gray', lw=0.6, ls='-.', alpha=0.45)
        ax.text(C_arr[-1] * 0.99, h_ref * 1.18, lbl,
                color='gray', ha='right', fontsize=8)

    ax.set_title(rf"${n_sig}\sigma$ detection of $C \neq 0$")
    ax.set_xlabel("Contrast  $C$")
    ax.set_ylabel("Required time  [hours]")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax.set_xlim(C_arr[0], C_arr[-1])

plt.tight_layout()
fig3.savefig("fisherSweep_C_time_to_significance.pdf", bbox_inches="tight")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Adaptive benefit — σ(C) ratio (even / adaptive) and required-time
# reduction vs C, for each n > 3.
#
# For each (C_true, n), the adaptive simulation ran N_ADAPT Poisson runs with
# T_ADAPT total seconds; the FINAL σ(C) (at T=1 s) uses the adaptively-chosen
# positions vs the n-point even-spacing σ(C).
# Ratio > 1  means adaptation improved precision at fixed total time.
# ─────────────────────────────────────────────────────────────────────────────

# Collect final σ(C) from adaptive runs
#   adapt_results[n][C_sim] has shape (N_ADAPT, n-2);  last column = final step
n_vals_adapt = [n for n in n_vals if n > 3]   # n=3 has no adaptive step

fig4, axes4 = plt.subplots(2, len(n_vals_adapt), figsize=(14, 8),
                            sharey='row', sharex=True)
fig4.suptitle(
    rf"Adaptive benefit vs even spacing  ($\lambda_0={LAMBDA_0}$ Hz, "
    rf"$T_\mathrm{{adapt}}={int(T_ADAPT/3600)}$ h per simulation, {N_ADAPT} runs)"
    "\n"
    r"TOP: $\sigma_\mathrm{even}/\sigma_\mathrm{adaptive}$ — ratio > 1 means adaptation wins."
    r"  BOTTOM: % time saving at fixed $\sigma$ target.",
    fontsize=11)

for ni_a, n in enumerate(n_vals_adapt):
    ax_top = axes4[0, ni_a]
    ax_bot = axes4[1, ni_a]

    ratios_med, ratios_q25, ratios_q75 = [], [], []
    saving_med, saving_q25, saving_q75 = [], [], []

    for C_sim in C_sim_vals:
        runs = adapt_results[n][C_sim]        # (N_ADAPT, n-2)
        sig_adapt_runs = runs[:, -1]          # final σ(C) across MC runs

        vm, _, _ = even_spacing_stats(n, LAMBDA_0, C_sim)
        sig_even_val = np.sqrt(vm)

        ratio = sig_even_val / sig_adapt_runs           # >1 if adaptive wins
        # time saving: T_even/T_adapt - 1 = ratio² - 1  (since σ∝1/√T)
        time_save_pct = (ratio**2 - 1.0) * 100.0

        ratios_med.append(np.median(ratio))
        ratios_q25.append(np.percentile(ratio, 25))
        ratios_q75.append(np.percentile(ratio, 75))
        saving_med.append(np.median(time_save_pct))
        saving_q25.append(np.percentile(time_save_pct, 25))
        saving_q75.append(np.percentile(time_save_pct, 75))

    x = np.arange(len(C_sim_vals))

    # top: ratio
    ax_top.bar(x, ratios_med, color=colors_n[n], alpha=0.75, width=0.5)
    ax_top.errorbar(x, ratios_med,
                    yerr=[np.array(ratios_med) - np.array(ratios_q25),
                          np.array(ratios_q75) - np.array(ratios_med)],
                    fmt='none', color='white', capsize=4, lw=1.2)
    ax_top.axhline(1.0, color='white', lw=1.0, ls='--', alpha=0.7,
                   label='No benefit')
    ax_top.set_title(rf"$n={n}$", fontsize=11)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"$C={c}$" for c in C_sim_vals], fontsize=9)
    if ni_a == 0:
        ax_top.set_ylabel(r"$\sigma_\mathrm{even}\,/\,\sigma_\mathrm{adapt}$"
                          "\n(ratio > 1 = adaptive wins)")
    ax_top.grid(True, axis='y', linestyle=':', linewidth=0.5)
    ax_top.legend(frameon=False, fontsize=8)

    # bottom: % time saving
    cols_bar = ['C3' if s > 0 else 'C1' for s in saving_med]
    ax_bot.bar(x, saving_med, color=cols_bar, alpha=0.75, width=0.5)
    ax_bot.errorbar(x, saving_med,
                    yerr=[np.array(saving_med) - np.array(saving_q25),
                          np.array(saving_q75) - np.array(saving_med)],
                    fmt='none', color='white', capsize=4, lw=1.2)
    ax_bot.axhline(0.0, color='white', lw=1.0, ls='--', alpha=0.7)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"$C={c}$" for c in C_sim_vals], fontsize=9)
    ax_bot.set_xlabel("True contrast  $C$")
    if ni_a == 0:
        ax_bot.set_ylabel("Equivalent time saving [%]\n"
                          r"$((\sigma_\mathrm{even}/\sigma_\mathrm{adapt})^2-1)\times100$")
    ax_bot.grid(True, axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout()
fig4.savefig("fisherSweep_C_adaptive.pdf", bbox_inches="tight")

plt.show()
