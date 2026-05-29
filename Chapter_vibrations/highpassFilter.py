"""
Window transfer function for the LEMING measurement window.

H(f) = sin(pi f tau) for f <= 1/(2 tau)   [rises 0 → 1]
     = 1              for f >  1/(2 tau)   [full amplitude, saturated]

The integral formula in the thesis uses [2 sin(pi f tau)]^2 = [2 H(f)]^2.
Above the Nyquist frequency 1/(2 tau) = 50 kHz, vibrations complete at least
one half-cycle inside the window and contribute their full amplitude (H = 1).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── parameters ────────────────────────────────────────────────────────────────
tau    = 10e-6          # measurement window (10 µs)
f_nyq  = 1 / (2 * tau) # 50 kHz — Nyquist of the window, first maximum of H
f_3dB  = 1 / (4 * tau) # 25 kHz — H = 1/sqrt(2), i.e. −3 dB

# ── frequency array ───────────────────────────────────────────────────────────
f = np.logspace(2, 7, 10000)   # 100 Hz → 10 MHz

# ── transfer function (amplitude; normalized so max = 1) ─────────────────────
# Below Nyquist: rises as |sin(pi f tau)|; above: saturated at 1
H = np.where(f <= f_nyq, np.abs(np.sin(np.pi * f * tau)), 1.0)

# low-frequency Taylor approximation: sin(x) ≈ x  →  H ≈ pi f tau
H_lf = np.pi * f * tau

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":          False,
    "mathtext.fontset":     "cm",
    "font.family":          "serif",
    "font.size":            11,
    "axes.labelsize":       12,
    "legend.fontsize":      9.5,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "axes.linewidth":       0.8,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "figure.dpi":           150,
})

fig, ax = plt.subplots(figsize=(6, 4))

# main transfer function
ax.semilogx(f * 1e-3, H,
            color='steelblue', lw=1.8, zorder=3,
            label=r'$H(f) = |\sin(\pi f \tau)|$ (sat. at 1)')

# low-frequency linear approximation (drawn only while H_lf < 1)
mask = H_lf < 1.0
ax.semilogx(f[mask] * 1e-3, H_lf[mask],
            color='gray', lw=1.2, ls='--', zorder=2,
            label=r'Low-$f$ approx. $\pi f\tau$')

# ── reference lines ───────────────────────────────────────────────────────────
# full-amplitude level H = 1
ax.axhline(1.0, color='0.55', lw=0.8, ls=':', zorder=1)
ax.text(0.115, 1.03, r'$H = 1$ (full amplitude)',
        color='0.45', fontsize=8.5,
        transform=ax.get_yaxis_transform(), ha='left')

# 3 dB line at H = 1/sqrt(2)
ax.axhline(1 / np.sqrt(2), color='tomato', lw=0.9, ls=':', zorder=1, alpha=0.85)
ax.text(0.115, 1/np.sqrt(2) + 0.025,
        r'$H = 1/\!\sqrt{2}\;(\mathrm{-3\,dB})$',
        color='tomato', fontsize=8.5,
        transform=ax.get_yaxis_transform(), ha='left')

# vertical: Nyquist / first maximum at 50 kHz
ax.axvline(f_nyq * 1e-3, color='0.55', lw=0.8, ls=':', zorder=1)
ax.text(f_nyq * 1e-3 * 1.06, 0.03,
        r'$\frac{1}{2\tau} = 50\,\mathrm{kHz}$',
        color='0.45', fontsize=8.5, va='bottom')

# vertical: −3 dB corner at 25 kHz
ax.axvline(f_3dB * 1e-3, color='tomato', lw=0.9, ls=':', zorder=1, alpha=0.85)
ax.text(f_3dB * 1e-3 * 0.92, 0.03,
        r'$25\,\mathrm{kHz}$',
        color='tomato', fontsize=8.5, va='bottom', ha='right')

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel(r'Frequency $f$ (kHz)')
ax.set_ylabel(r'Window transfer function $H(f)$')
ax.set_xlim(f[0] * 1e-3, f[-1] * 1e-3)
ax.set_ylim(0, 1.25)

ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'{x:g}'))

ax.legend(loc='lower right')
ax.grid(which='major', color='0.88', lw=0.6)
ax.grid(which='minor', color='0.93', lw=0.4)

fig.tight_layout()
fig.savefig('Chapter_vibrations/highpassFilter.pdf', bbox_inches='tight')
fig.savefig('Chapter_vibrations/highpassFilter.png', dpi=200, bbox_inches='tight')
print("Saved highpassFilter.pdf / .png")
plt.show()
