"""
Steady-state amplitude response: base-excited mass-spring system
with nonlinear (quadratic velocity) damping vs linear damping.

Equation of motion in relative coordinates z = x - y,
base excitation y = a sin(ωt):

    m z'' + k z + c |z'| z' = m a ω² sin(ωt)   [nonlinear]
    m z'' + k z + c_lin z'  = m a ω² sin(ωt)   [linear]

Equivalent linearisation for the quadratic damper:
    c_equiv(Z, ω) = (8/3π) c Z ω

giving an implicit equation for the relative amplitude Z(ω)
that is solved numerically at each frequency.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import brentq

# ── system parameters ─────────────────────────────────────────────────────────
m = 1.0          # kg
k = 1.0          # N/m
a = 1.0          # base excitation amplitude (m)

omega_0 = np.sqrt(k / m)   # natural frequency (rad/s)

# ── frequency sweep ───────────────────────────────────────────────────────────
omega = np.logspace(np.log10(0.1 * omega_0), np.log10(3.5 * omega_0), 2000)

# ── nonlinear damping coefficients to sweep ───────────────────────────────────
c_nl_values = [0.05, 0.2, 0.5, 1.5, 4.0]      # nonlinear c  [N·s²/m²]

# ── linear damping coefficients for comparison ────────────────────────────────
# Choose values that produce broadly similar peak suppression to the nl cases
c_lin_values = [0.02, 0.08, 0.2, 0.6, 1.5]    # linear c_lin [N·s/m]

# ── amplitude formula (shared structure) ──────────────────────────────────────
def Z_over_a(omega_val, c_eq):
    """
    Standard base-excitation relative-amplitude ratio for a given
    (possibly equivalent) linear damping coefficient c_eq.

        Z/a = sqrt(k² + (c_eq ω)²) / sqrt((k - m ω²)² + (c_eq ω)²)
    """
    num = np.sqrt(k**2 + (c_eq * omega_val)**2)
    den = np.sqrt((k - m * omega_val**2)**2 + (c_eq * omega_val)**2)
    return num / den


# ── nonlinear solve: implicit equation for Z ─────────────────────────────────
def solve_nonlinear(c_nl, omega_arr):
    """
    At each frequency find Z satisfying:
        Z = a · Z_over_a(ω, c_equiv(Z, ω))
    where  c_equiv = (8/3π) c_nl Z ω

    Rearranged to root form:
        F(Z) = Z - a · Z_over_a(ω, (8/3π) c_nl Z ω) = 0

    Strategy:
      • Use the linear solution with c_lin = 0 (undamped) as initial bracket
      • Bracket in [Z_lo, Z_hi] and use Brent's method for robustness
    """
    Z_out = np.empty_like(omega_arr)
    coeff = 8.0 / (3.0 * np.pi)

    for i, w in enumerate(omega_arr):
        def F(Z):
            if Z <= 0:
                return -a * Z_over_a(w, 0.0)   # force sign at zero
            c_eq = coeff * c_nl * Z * w
            return Z - a * Z_over_a(w, c_eq)

        # undamped amplitude as a generous upper bracket
        Z_undamped = a * abs(k / (k - m * w**2 + 1e-30))
        Z_hi = max(Z_undamped * 2.0, 20.0 * a)
        Z_lo = 1e-9 * a

        try:
            Z_out[i] = brentq(F, Z_lo, Z_hi, xtol=1e-12, rtol=1e-10)
        except ValueError:
            # fallback: if brackets have same sign, evaluate at midpoint
            Z_out[i] = (Z_lo + Z_hi) / 2.0

    return Z_out


# ── compute curves ────────────────────────────────────────────────────────────
nl_curves  = {c: solve_nonlinear(c, omega)    for c in c_nl_values}
lin_curves = {c: a * Z_over_a(omega, c)       for c in c_lin_values}

# ── plotting ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":         False,
    "mathtext.fontset":    "cm",
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.titlesize":      11,
    "legend.fontsize":     9,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "axes.linewidth":      0.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":          150,
})

fig, ax = plt.subplots(figsize=(9, 5.5))

# nonlinear curves — blue palette
nl_colors = plt.cm.Blues(np.linspace(0.4, 0.92, len(c_nl_values)))
for color, c in zip(nl_colors, c_nl_values):
    ax.semilogy(omega / omega_0, nl_curves[c] / a,
                color=color, lw=2.0,
                label=rf'NL  $c={c}$')

# linear curves — red/orange palette, dashed
lin_colors = plt.cm.Oranges(np.linspace(0.4, 0.92, len(c_lin_values)))
for color, c in zip(lin_colors, c_lin_values):
    ax.semilogy(omega / omega_0, lin_curves[c] / a,
                color=color, lw=1.6, ls='--',
                label=rf'Lin $c_{{\rm lin}}={c}$')

# resonance marker
ax.axvline(1.0, color='0.6', lw=0.8, ls=':', zorder=0)
ax.text(1.01, ax.get_ylim()[0] * 1.5 if ax.get_ylim()[0] > 0 else 0.05,
        r'$\omega_0$', color='0.55', fontsize=10, va='bottom')

ax.set_xlabel(r'Frequency ratio  $\omega / \omega_0$')
ax.set_ylabel(r'Amplitude ratio  $Z / a$')
ax.set_title(
    'Base-excited mass–spring: nonlinear (quadratic) vs linear damping\n'
    r'$m\ddot{z} + kz + c|\dot{z}|\dot{z} = ma\omega^2\sin\omega t$'
    r'   —   equiv. linearisation  $c_{\rm equiv} = \frac{8}{3\pi}\,c\,Z\,\omega$',
    fontsize=10)

ax.set_xlim(omega[0] / omega_0, omega[-1] / omega_0)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

# two-column legend: nonlinear left, linear right
handles, labels = ax.get_legend_handles_labels()
n_nl = len(c_nl_values)
leg = ax.legend(handles, labels,
                ncol=2, fontsize=8.5,
                loc='upper right',
                framealpha=0.88,
                title='Nonlinear (solid)  |  Linear (dashed)',
                title_fontsize=8)

# annotation explaining equivalent linearisation
ax.text(0.02, 0.04,
        r'$c_{\rm equiv} \propto Z\omega$  '
        '→ stronger damping at large $Z$\n'
        'Peak shifts and flattens with increasing $c$',
        transform=ax.transAxes, fontsize=8.5,
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85, ec='0.7'))

ax.grid(which='major', color='0.88', lw=0.6)
ax.grid(which='minor', color='0.93', lw=0.4)

fig.tight_layout()
fig.savefig('magneticDamping.pdf', bbox_inches='tight')
fig.savefig('magneticDamping.png', dpi=200, bbox_inches='tight')
print("Saved magneticDamping.pdf / .png")
plt.show()
