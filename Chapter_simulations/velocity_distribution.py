"""
Velocity distribution for LEMING Muonium gravity interferometer
────────────────────────────────────────────────────────────────
Source:         Gaussian  v₀ = 2175 m/s,  σ_v = 70 m/s
Interferometer: L_tot = 19.2 mm  (3 gratings, 2 equal gaps of 9.6 mm)
Grating period: d = 100 nm
Lifetime filter: P(v) = exp(-L_tot / (v tau))

Key output: shift of the mean  Delta_v = <v>_det - v0
            and corresponding gravitational phase shift Delta_phi
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, LogLocator
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({
    "text.usetex":         False,  # set True on your local machine
    "mathtext.fontset":    "cm",
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.titlesize":      11,
    "legend.fontsize":     9.5,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "axes.linewidth":      0.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":          150,
})

# ── physical parameters ───────────────────────────────────────────────────────
tau     = 2.196981e-6       # s    muon mean lifetime
g_acc   = 9.80665           # m/s²

v0      = 2175.0            # m/s  centre of source Gaussian
sigma_v = 70.0              # m/s  1-sigma width

L_tot   = 4*tau*v0           # m    total interferometer length
L_gap   = L_tot / 2        # m    one gap (between adjacent gratings)
d       = 100e-9            # m    grating period

# ── velocity grid: ±8σ around v0, very fine ──────────────────────────────────
v_lo  = max(1.0, v0 - 8*sigma_v)
v_hi  =          v0 + 8*sigma_v
v     = np.linspace(v_lo, v_hi, 500_000)

def trapN(f, x):
    return np.trapezoid(f, x)

# ── distributions ─────────────────────────────────────────────────────────────
A_src  = np.exp(-0.5 * ((v - v0)/sigma_v)**2)   # Gaussian
P_surv = np.exp(-L_tot / (v * tau))              # survival prob.
A_det  = A_src * P_surv                          # detected

# Normalise to unit area
A_src_n = A_src / trapN(A_src, v)
A_det_n = A_det / trapN(A_det, v)

# ── means and widths ──────────────────────────────────────────────────────────
v_mean_src = trapN(v * A_src_n, v)       # = v0 by symmetry
v_mean_det = trapN(v * A_det_n, v)
Delta_v    = v_mean_det - v_mean_src     # the key number  [m/s]

sigma_src  = np.sqrt(trapN((v - v_mean_src)**2 * A_src_n, v))
sigma_det  = np.sqrt(trapN((v - v_mean_det)**2 * A_det_n, v))

eff = trapN(A_det, v) / trapN(A_src, v)

# ── gravitational phase  phi(v) = g (L_gap/v)^2 / d  [grat. periods] ──
def phi(vel):
    return g_acc * (L_gap / vel)**2 / d

phi_v = phi(v)

phi_mean_src = trapN(phi_v * A_src_n, v)
phi_mean_det = trapN(phi_v * A_det_n, v)
Delta_phi    = phi_mean_det - phi_mean_src   # [grating periods]

# Exact complex visibility
C_src = trapN(A_src_n * np.exp(2j*np.pi * phi_v), v)
C_det = trapN(A_det_n * np.exp(2j*np.pi * phi_v), v)
arg_shift = np.degrees(np.angle(C_det)) - np.degrees(np.angle(C_src))

# ── print results ─────────────────────────────────────────────────────────────
sep = "=" * 64
print(sep)
print("  LEMING: lifetime-induced mean velocity shift")
print(sep)
print(f"  Source:              Gaussian, v0 = {v0:.1f} m/s,  sigma = {sigma_v:.1f} m/s")
print(f"  Muon lifetime:       tau = {tau*1e6:.4f} us")
print(f"  L_tot = {L_tot*1e3:.1f} mm,  L_gap = {L_gap*1e3:.1f} mm,  d = {d*1e9:.0f} nm")
print(sep)
print(f"  <v>_src              = {v_mean_src:.6f} m/s  (sigma = {sigma_src:.4f} m/s)")
print(f"  <v>_det              = {v_mean_det:.6f} m/s  (sigma = {sigma_det:.4f} m/s)")
print(f"  Delta<v>  (shift)    = {Delta_v:+.6f} m/s")
print(f"                       = {Delta_v/sigma_v*100:+.4f} % of sigma_v")
print(f"                       = {Delta_v/v0*100:+.6f} % of v0")
print(f"  Detection efficiency = {eff*100:.4f} %")
print(sep)
P_v0 = float(np.exp(-L_tot/(v0*tau)))
print(f"  P_surv(v0)           = {P_v0*100:.4f} %  (at beam centre)")
print(sep)
print(f"  phi(v0)              = {phi(v0):.6f}  grating periods")
print(f"  <phi>_src            = {phi_mean_src:.6f}  grating periods")
print(f"  <phi>_det            = {phi_mean_det:.6f}  grating periods")
print(f"  Delta<phi>           = {Delta_phi:+.8f}  grating periods")
print(f"                       = {Delta_phi*360:+.6f} deg")
print(f"                       = {Delta_phi*2*np.pi*1e6:+.4f} urad")
print(sep)
print(f"  |C|_src = {abs(C_src):.8f},  arg = {np.degrees(np.angle(C_src)):+.6f} deg")
print(f"  |C|_det = {abs(C_det):.8f},  arg = {np.degrees(np.angle(C_det)):+.6f} deg")
print(f"  Phase offset (arg C_det - arg C_src) = {arg_shift:+.8f} deg")
print(sep)

# ── FIGURE 1: velocity distributions ─────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(13, 4.5), gridspec_kw=dict(wspace=0.38))
ax1, ax2, ax3 = axes

dv = v - v0   # x-axis: offset from v0 in m/s

# (a) distributions
ax1.fill_between(dv, A_src_n * sigma_v, alpha=0.20, color='C0')
ax1.plot(dv, A_src_n * sigma_v, 'C0', lw=2,
         label=r'Source $\mathcal{N}(v_0,\sigma_v^2)$')
ax1.fill_between(dv, A_det_n * sigma_v, alpha=0.25, color='C3')
ax1.plot(dv, A_det_n * sigma_v, 'C3', lw=2,
         label=r'Detected (lifetime-corr.)')
ax1.axvline(0,       color='C0', lw=1.0, ls='--', alpha=0.7, label=rf'$v_0={v0:.0f}$ m/s')
ax1.axvline(Delta_v, color='C3', lw=1.0, ls='--', alpha=0.7,
            label=rf'$\langle v\rangle_{{\rm det}} = v_0{Delta_v:+.2f}$ m/s')

# double-headed arrow for Delta_v
ymax_a = ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.65
ax1.annotate('', xy=(Delta_v, 0.58), xytext=(0.0, 0.58),
             arrowprops=dict(arrowstyle='<->', color='k', lw=1.3))
ax1.text(Delta_v/2, 0.61,
         rf'$\Delta\langle v\rangle = {Delta_v:+.2f}$ m/s',
         ha='center', va='bottom', fontsize=9)

ax1.set_xlabel(r'$v - v_0$ (m\,s$^{-1}$)')
ax1.set_ylabel(r'$p(v)\,\sigma_v$ (normalised)')
ax1.set_xlim(-5*sigma_v, 5*sigma_v)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.set_title(r'(a) Velocity distributions')
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# (b) survival probability
ax2.semilogy(dv, P_surv, 'm', lw=2.2)
ax2.fill_between(dv, P_surv, P_surv.min()*0.5, alpha=0.13, color='m')
ax2.axvline(0,       color='C0', lw=1.0, ls='--', alpha=0.7, label=r'$v_0$')
ax2.axvline(Delta_v, color='C3', lw=1.0, ls='--', alpha=0.7,
            label=r'$\langle v\rangle_{\rm det}$')
ax2.axhline(P_v0, color='0.55', lw=0.9, ls=':')
ax2.text(-4.6*sigma_v, P_v0 * 1.8,
         rf'$P(v_0)={P_v0*100:.3f}\%$', fontsize=9, color='0.4')
ax2.set_xlabel(r'$v - v_0$ (m\,s$^{-1}$)')
ax2.set_ylabel(r'Survival probability $P(v)$')
ax2.set_xlim(-5*sigma_v, 5*sigma_v)
ax2.legend(loc='lower right', fontsize=9)
ax2.set_title(r'(b) Muon survival $\exp(-L_{\rm tot}/v\tau)$')
ax2.yaxis.set_minor_locator(LogLocator(subs='all', numticks=10))

# (c) difference (redistribution)
diff = (A_det_n - A_src_n) * sigma_v
ax3.fill_between(dv, diff, 0, where=(diff >= 0), alpha=0.30, color='C3',
                 label=r'Excess (fast Mu)')
ax3.fill_between(dv, diff, 0, where=(diff <= 0), alpha=0.30, color='C0',
                 label=r'Shortage (slow Mu)')
ax3.plot(dv, diff, 'k', lw=1.5)
ax3.axvline(0,       color='0.55', lw=0.8, ls='--')
ax3.axhline(0,       color='0.55', lw=0.8)
ax3.set_xlabel(r'$v - v_0$ (m\,s$^{-1}$)')
ax3.set_ylabel(r'$[p_{\rm det}(v) - p_{\rm src}(v)]\,\sigma_v$')
ax3.set_xlim(-5*sigma_v, 5*sigma_v)
ax3.legend(fontsize=9)
ax3.set_title(r'(c) Lifetime-induced redistribution')
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.text(0.03, 0.97,
         rf'$\Delta\langle v\rangle = {Delta_v:+.3f}$ m/s'
         '\n'
         rf'$= {Delta_v/sigma_v*100:+.3f}\%\;\sigma_v$'
         '\n'
         rf'Efficiency $= {eff*100:.3f}\%$',
         transform=ax3.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85, ec='0.65'))

fig1.suptitle(
    rf'LEMING: Gaussian source $v_0={v0:.0f}$ m/s, '
    rf'$\sigma_v={sigma_v:.0f}$ m/s, '
    rf'$L_{{\rm tot}}={L_tot*1e3:.1f}$ mm ($= 4\tau v_0$), '
    rf'$L_{{\rm gap}}={L_gap*1e3:.2f}$ mm, '
    rf'$d={d*1e9:.0f}$ nm, '
    rf'$\tau_\mu={tau*1e6:.4f}\ \mu$s',
    fontsize=11)

# Parameter summary box on panel (a)
param_str = (
    rf'Initial params.:'              '\n'
    rf'$v_0 = {v0:.0f}$ m/s'            '\n'
    rf'$\sigma_v = {sigma_v:.0f}$ m/s'  '\n'
    rf'$L_{{\rm tot}} = {L_tot*1e3:.2f}$ mm'  '\n'
    rf'$L_{{\rm gap}} = {L_gap*1e3:.2f}$ mm'  '\n'
    rf'$d = {d*1e9:.0f}$ nm'            '\n'
    rf'$\tau_\mu = {tau*1e6:.4f}\ \mu$s'
)
ax1.text(0.03, 0.30, param_str,
         transform=ax1.transAxes, fontsize=8.5, va='top',
         bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4ff', alpha=0.90, ec='0.55'))

fig1.savefig(OUT / 'vel_dist.pdf', bbox_inches='tight')
fig1.savefig(OUT / 'vel_dist.png', dpi=200, bbox_inches='tight')
print("\nFigure 1 saved.")

# ── FIGURE 2: phase shift ─────────────────────────────────────────────────────
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.0),
                                 gridspec_kw=dict(wspace=0.38))

# (A) phase distribution  p(phi)
# Change of variables v -> phi:  phi = g(L_gap/v)^2/d  => v = L_gap sqrt(g/(phi d))
phi_lo = phi(v_hi)
phi_hi = phi(v_lo)
phi_arr = np.linspace(phi_lo * 0.998, phi_hi * 1.002, 100_000)
v_of_phi = L_gap * np.sqrt(g_acc / (phi_arr * d))
jac      = v_of_phi / (2 * phi_arr)     # |dv/d phi|

A_src_phi = np.exp(-0.5*((v_of_phi - v0)/sigma_v)**2) * jac
A_det_phi = A_src_phi * np.exp(-L_tot/(v_of_phi * tau))
A_src_phi /= trapN(A_src_phi, phi_arr)
A_det_phi /= trapN(A_det_phi, phi_arr)

phi_mu_src = trapN(phi_arr * A_src_phi, phi_arr)
phi_mu_det = trapN(phi_arr * A_det_phi, phi_arr)
D_phi      = phi_mu_det - phi_mu_src

axA.fill_between(phi_arr, A_src_phi, alpha=0.20, color='C0')
axA.plot(phi_arr, A_src_phi, 'C0', lw=2, label=r'Source')
axA.fill_between(phi_arr, A_det_phi, alpha=0.25, color='C3')
axA.plot(phi_arr, A_det_phi, 'C3', lw=2, label=r'Detected')
axA.axvline(phi_mu_src, color='C0', lw=1.0, ls='--', alpha=0.8,
            label=rf'$\langle\varphi\rangle_{{\rm src}}={phi_mu_src:.5f}$')
axA.axvline(phi_mu_det, color='C3', lw=1.0, ls='--', alpha=0.8,
            label=rf'$\langle\varphi\rangle_{{\rm det}}={phi_mu_det:.5f}$')
axA.set_xlabel(r'Grav. phase $\varphi(v) = g(L_{\rm gap}/v)^2/d$ (grating periods)')
axA.set_ylabel(r'Prob. density (grating periods$^{-1}$)')
axA.legend(fontsize=9)
axA.set_title(r'(a) Phase distribution $p(\varphi)$')
axA.xaxis.set_minor_locator(AutoMinorLocator())
axA.yaxis.set_minor_locator(AutoMinorLocator())
axA.text(0.5, 0.5,
         rf'$\Delta\langle\varphi\rangle = {D_phi:+.6f}$ gr.p.'
         '\n'
         rf'$= {D_phi*360:+.4f}^\circ$'
         '\n'
         rf'$= {D_phi*2*np.pi*1e6:+.3f}\ \mu$rad',
         transform=axA.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85, ec='0.65'))

# (B) Integrated fringe  I(x) = int A(v) cos(2pi(x - phi(v))) dv
x = np.linspace(-0.5, 0.5, 3000)
phi_v_col = phi_v[np.newaxis, :]
x_row     = x[:, np.newaxis]
A_src_row = (A_src_n / trapN(A_src_n, v))[np.newaxis, :]
A_det_row = (A_det_n / trapN(A_det_n, v))[np.newaxis, :]

I_src = np.trapezoid(A_src_n[np.newaxis,:] * np.cos(2*np.pi*(x_row - phi_v_col)), v, axis=1)
I_det = np.trapezoid(A_det_n[np.newaxis,:] * np.cos(2*np.pi*(x_row - phi_v_col)), v, axis=1)

# normalise so amplitude is visible
I_src /= np.max(np.abs(I_src))
I_det /= np.max(np.abs(I_det))

x_pk_src = x[np.argmax(I_src)]
x_pk_det = x[np.argmax(I_det)]
fringe_shift = x_pk_det - x_pk_src

offset = 2.4
axB.plot(x, I_src + offset, 'C0', lw=2,
         label=rf'Source  (peak: $x={x_pk_src:+.5f}\,d$)')
axB.plot(x, I_det,           'C3', lw=2,
         label=rf'Detected (peak: $x={x_pk_det:+.5f}\,d$)')
axB.axvline(x_pk_src, color='C0', lw=0.9, ls=':', alpha=0.8)
axB.axvline(x_pk_det, color='C3', lw=0.9, ls=':', alpha=0.8)
axB.annotate('', xy=(x_pk_det, -0.3), xytext=(x_pk_src, -0.3),
             arrowprops=dict(arrowstyle='<->', color='k', lw=1.3))
axB.text((x_pk_src+x_pk_det)/2, -0.22,
         rf'$\delta x = {fringe_shift:+.5f}\,d = {fringe_shift*360:+.3f}^\circ$',
         ha='center', va='bottom', fontsize=9)
axB.set_xlabel(r'Position $x/d$ (grating periods)')
axB.set_ylabel(r'Intensity (arb., source offset for clarity)')
axB.set_xlim(-0.5, 0.5)
axB.legend(fontsize=9)
axB.set_title(r'(b) Integrated fringe $I(x) = \int   p(v)\cos[2\pi(x/d - \varphi(v))]\,dv$')
axB.xaxis.set_minor_locator(AutoMinorLocator())

axB.text(0.03, 0.97,
         rf'$|C|_{{\rm src}} = {abs(C_src):.6f}$'
         '\n'
         rf'$|C|_{{\rm det}} = {abs(C_det):.6f}$'
         '\n'
         rf'$\Delta\arg C = {arg_shift:+.6f}^\circ$',
         transform=axB.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85, ec='0.65'))

fig2.suptitle(
    rf'Gravitational phase shift from lifetime selection  —  '
    rf'$\varphi(v) = g(L_{{\rm gap}}/v)^2/d$,  '
    rf'$v_0={v0:.0f}$ m/s,  $\sigma_v={sigma_v:.0f}$ m/s,  '
    rf'$L_{{\rm gap}}={L_gap*1e3:.2f}$ mm,  '
    rf'$d={d*1e9:.0f}$ nm,  $\tau_\mu={tau*1e6:.4f}\ \mu$s',
    fontsize=10)

fig2.savefig(OUT / 'phase_shift.pdf', bbox_inches='tight')
fig2.savefig(OUT / 'phase_shift.png', dpi=200, bbox_inches='tight')
print("Figure 2 saved.")

plt.show()
print("\nAll done.")